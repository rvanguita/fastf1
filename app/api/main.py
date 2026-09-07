"""API compatível e versionada para previsão do campeão de Fórmula 1."""

from __future__ import annotations

import os
import time

import mlflow
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

API_PORT = int(os.environ["API_PORT"])
MLFLOW_URI = os.environ["MLFLOW_URI"]
MLFLOW_MODEL_REGISTERED = os.environ["MLFLOW_MODEL_REGISTERED"]
MODEL_CACHE_TTL = float(os.environ.get("MODEL_CACHE_TTL", "300"))
_MODEL_CACHE: dict[str, tuple[float, object]] = {}


def _load_model(model_id: str | None):
    mlflow.set_tracking_uri(MLFLOW_URI)
    models = mlflow.search_registered_models(filter_string=f"name='{model_id}'")
    if not models or not models[0].latest_versions:
        raise LookupError(f"Registered model not found: {model_id}")
    last_version = max(int(version.version) for version in models[0].latest_versions)
    return mlflow.sklearn.load_model(f"models:/{model_id}/{last_version}")


def model_find(model_id: str | None = None):
    cached = _MODEL_CACHE.get(model_id)
    if cached is not None and (time.monotonic() - cached[0]) < MODEL_CACHE_TTL:
        return cached[1]
    try:
        model = _load_model(model_id)
    except Exception:  # noqa: BLE001
        return None
    _MODEL_CACHE[model_id] = (time.monotonic(), model)
    return model


def _find_importance_estimator(model):
    if hasattr(model, "feature_importances_"):
        return model
    for step in getattr(model, "named_steps", {}).values():
        found = _find_importance_estimator(step)
        if found is not None:
            return found
    for calibrated in getattr(model, "calibrated_classifiers_", []):
        found = _find_importance_estimator(getattr(calibrated, "estimator", None))
        if found is not None:
            return found
    child = getattr(model, "estimator", None)
    return (
        _find_importance_estimator(child)
        if child is not None and child is not model
        else None
    )


def _feature_importances(model) -> tuple[list[str], dict[str, float]]:
    names = [str(column) for column in getattr(model, "feature_names_in_", [])]
    estimator = _find_importance_estimator(model)
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return names, {}
    return names, {name: float(value) for name, value in zip(names, importances)}


def _positive_probability(model, frame: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(frame), dtype=float)
    classes = list(getattr(model, "classes_", [0, 1]))
    index = classes.index(1) if 1 in classes else len(classes) - 1
    return probabilities[:, index]


def _groups(frame: pd.DataFrame) -> pd.Series:
    if "prediction_group" in frame:
        return frame["prediction_group"].astype(str)
    return frame["id"].astype(str).str.rsplit("_", n=1).str[0]


def _normalize(scores: np.ndarray, groups: pd.Series) -> np.ndarray:
    values = pd.Series(np.clip(scores, 0, None), index=groups.index)
    totals = values.groupby(groups).transform("sum")
    sizes = groups.groupby(groups).transform("size")
    normalized = values.div(totals.where(totals > 0))
    return normalized.where(totals > 0, 1 / sizes).to_numpy()


def _ensemble_scores(model, frame: pd.DataFrame) -> list[np.ndarray]:
    """Retorna previsões dos membros quando o artefato expõe um ensemble."""
    if hasattr(model, "predict_proba_members"):
        return [np.asarray(values) for values in model.predict_proba_members(frame)]
    members = []
    for calibrated in getattr(model, "calibrated_classifiers_", []):
        try:
            members.append(_positive_probability(calibrated, frame))
        except (AttributeError, ValueError):
            pass
    if members:
        return members
    if getattr(model, "named_steps", None):
        steps = list(model.named_steps.values())
        transformed = frame
        for step in steps[:-1]:
            transformed = step.transform(transformed)
        forest = steps[-1]
        for tree in getattr(forest, "estimators_", []):
            try:
                members.append(_positive_probability(tree, transformed))
            except (AttributeError, ValueError):
                pass
    elif getattr(getattr(model, "estimator", None), "named_steps", None):
        return _ensemble_scores(model.estimator, frame)
    return members


def _prediction_payload(model, frame: pd.DataFrame) -> tuple[dict, dict]:
    missing = [column for column in model.feature_names_in_ if column not in frame]
    if missing:
        raise HTTPException(
            status_code=422, detail=f"missing feature columns: {missing}"
        )
    features = frame[list(model.feature_names_in_)]
    raw = _positive_probability(model, features)
    groups = _groups(frame)
    normalized = _normalize(raw, groups)
    member_scores = _ensemble_scores(model, features)
    if member_scores:
        member_probabilities = np.vstack(
            [_normalize(scores, groups) for scores in member_scores]
        )
        lower = np.quantile(member_probabilities, 0.10, axis=0)
        upper = np.quantile(member_probabilities, 0.90, axis=0)
        method = "ensemble_p10_p90"
    else:
        lower = upper = normalized
        method = "point_estimate"
    predictions = {
        str(identifier): {
            "raw_score": float(raw[i]),
            "probability": float(normalized[i]),
            "lower": float(lower[i]),
            "upper": float(upper[i]),
        }
        for i, identifier in enumerate(frame["id"])
    }
    return predictions, {
        "normalization": "within_prediction_group",
        "interval_method": method,
        "probability_sum_tolerance": 1e-9,
    }


def _model_card(model) -> dict:
    card = dict(getattr(model, "model_card_", {}) or {})
    card.setdefault("status", "experimental")
    card.setdefault("model_name", MLFLOW_MODEL_REGISTERED)
    card.setdefault("evaluations", [])
    card.setdefault("calibration", [])
    card.setdefault(
        "limitations",
        [
            "Não incorpora telemetria, clima ou estratégia de pneus.",
            "Intervalos representam dispersão do ensemble, não garantia estatística.",
        ],
    )
    return card


def _shap_explanations(model, frame: pd.DataFrame, top_n: int) -> dict:
    try:
        import shap
    except ImportError as exc:
        raise HTTPException(status_code=501, detail="SHAP is not installed") from exc
    features = frame[list(model.feature_names_in_)]
    estimator = _find_importance_estimator(model)
    if estimator is None:
        raise HTTPException(
            status_code=422, detail="Model does not expose a tree estimator"
        )
    transformed = features
    pipeline = None
    if getattr(model, "named_steps", None):
        pipeline = model
    elif getattr(model, "calibrated_classifiers_", None):
        candidate = getattr(model.calibrated_classifiers_[0], "estimator", None)
        pipeline = candidate if getattr(candidate, "named_steps", None) else None
    elif getattr(getattr(model, "estimator", None), "named_steps", None):
        pipeline = model.estimator
    if pipeline is not None:
        for step in list(pipeline.named_steps.values())[:-1]:
            transformed = step.transform(transformed)
    explanation = shap.TreeExplainer(estimator)(transformed)
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[:, :, -1]
    base = np.asarray(explanation.base_values)
    if base.ndim > 1:
        base = base[:, -1]
    names = list(model.feature_names_in_)
    output = {}
    for row_index, identifier in enumerate(frame["id"]):
        ranked = np.argsort(np.abs(values[row_index]))[::-1][:top_n]
        output[str(identifier)] = {
            "base_value": float(
                np.ravel(base)[row_index if np.ravel(base).size > 1 else 0]
            ),
            "contributions": [
                {
                    "feature": str(names[i]),
                    "value": float(features.iloc[row_index, i]),
                    "contribution": float(values[row_index, i]),
                }
                for i in ranked
            ],
        }
    return output


class PredictRequest(BaseModel):
    values: list[dict]


class ExplainRequest(PredictRequest):
    top_n: int = Field(default=10, ge=1, le=30)


app = FastAPI(
    title="F1 Driver Champion API",
    description="Predição calibrada e transparente do campeonato de pilotos.",
    version="2.0.0",
)


@app.get("/health_check", tags=["health"])
def health_check():
    return {"status": "OK"}


@app.get("/model_info", tags=["model"])
def model_info():
    model = model_find(MLFLOW_MODEL_REGISTERED)
    if model is None:
        raise HTTPException(status_code=500, detail="Model not found")
    names, importances = _feature_importances(model)
    classes = [
        value.item() if hasattr(value, "item") else value
        for value in getattr(model, "classes_", [])
    ]
    return {
        "n_features": len(names),
        "features": names,
        "importances": importances,
        "classes": classes,
    }


@app.post("/predict", tags=["model"])
def predict(body: PredictRequest):
    """Contrato legado preservado sem alteração de formato."""
    model = model_find(MLFLOW_MODEL_REGISTERED)
    if model is None:
        raise HTTPException(status_code=500, detail="Model not found")
    if not body.values:
        raise HTTPException(status_code=400, detail="No features provided")
    frame = pd.DataFrame(body.values)
    missing = [column for column in model.feature_names_in_ if column not in frame]
    if missing:
        raise HTTPException(
            status_code=422, detail=f"missing feature columns: {missing}"
        )
    probabilities = pd.DataFrame(
        model.predict_proba(frame[list(model.feature_names_in_)]),
        columns=model.classes_,
    )
    probabilities["id"] = frame["id"].copy()
    return {"predictions": probabilities.set_index("id").to_dict(orient="index")}


@app.post("/v1/predict", tags=["model-v1"])
def predict_v1(body: PredictRequest):
    model = model_find(MLFLOW_MODEL_REGISTERED)
    if model is None:
        raise HTTPException(status_code=500, detail="Model not found")
    if not body.values:
        raise HTTPException(status_code=400, detail="No features provided")
    frame = pd.DataFrame(body.values)
    predictions, metadata = _prediction_payload(model, frame)
    metadata.update(
        {"model_name": MLFLOW_MODEL_REGISTERED, "status": _model_card(model)["status"]}
    )
    return {"predictions": predictions, "metadata": metadata}


@app.post("/v1/explain", tags=["model-v1"])
def explain_v1(body: ExplainRequest):
    model = model_find(MLFLOW_MODEL_REGISTERED)
    if model is None:
        raise HTTPException(status_code=500, detail="Model not found")
    if not body.values:
        raise HTTPException(status_code=400, detail="No features provided")
    frame = pd.DataFrame(body.values)
    missing = [column for column in model.feature_names_in_ if column not in frame]
    if missing:
        raise HTTPException(
            status_code=422, detail=f"missing feature columns: {missing}"
        )
    return {"explanations": _shap_explanations(model, frame, body.top_n)}


@app.get("/v1/model-card", tags=["model-v1"])
def model_card_v1():
    model = model_find(MLFLOW_MODEL_REGISTERED)
    if model is None:
        raise HTTPException(status_code=500, detail="Model not found")
    return _model_card(model)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
