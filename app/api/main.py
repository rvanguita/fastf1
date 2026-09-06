import os
import time

import mlflow
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

API_PORT = int(os.environ["API_PORT"])
MLFLOW_URI = os.environ["MLFLOW_URI"]
MLFLOW_MODEL_REGISTERED = os.environ["MLFLOW_MODEL_REGISTERED"]
# Seconds a loaded model is reused before the registry is re-queried. Keeps a
# newly registered version reachable without a restart, while sparing every
# request an MLflow round-trip + deserialization.
MODEL_CACHE_TTL = float(os.environ.get("MODEL_CACHE_TTL", "300"))

# {model_id: (loaded_at_monotonic, model)}
_MODEL_CACHE: dict[str, tuple[float, object]] = {}


def _load_model(model_id: str | None):
    """Fetch the latest registered version of ``model_id`` from MLflow."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    models = mlflow.search_registered_models(filter_string=f"name='{model_id}'")[-1]
    last_version = int(models.latest_versions[-1].version)
    return mlflow.sklearn.load_model(f"models:/{model_id}/{last_version}")


def model_find(model_id: str | None = None):
    cached = _MODEL_CACHE.get(model_id)
    if cached is not None and (time.monotonic() - cached[0]) < MODEL_CACHE_TTL:
        return cached[1]
    try:
        model = _load_model(model_id)
    except Exception:  # noqa: BLE001 - any MLflow failure => "model unavailable"
        return None
    _MODEL_CACHE[model_id] = (time.monotonic(), model)
    return model


def _feature_importances(model) -> tuple[list[str], dict[str, float]]:
    """(feature_names, {name: importance}) from a fitted estimator or Pipeline.

    Walks a ``Pipeline``'s steps for the one exposing ``feature_importances_``
    (the RandomForest), falling back to the model itself. Returns an empty map
    when the estimator has no importances.
    """
    estimator = model
    for step in getattr(model, "named_steps", {}).values():
        if hasattr(step, "feature_importances_"):
            estimator = step
            break

    names = [str(c) for c in getattr(model, "feature_names_in_", [])]
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return names, {}
    return names, {n: float(v) for n, v in zip(names, importances)}


class PredictRequest(BaseModel):
    values: list[dict]


app = FastAPI(
    title="F1 Driver Champion API",
    description="Predicts championship win probability for F1 drivers.",
    version="1.0.0",
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
        c.item() if hasattr(c, "item") else c for c in getattr(model, "classes_", [])
    ]
    return {
        "n_features": len(names),
        "features": names,
        "importances": importances,
        "classes": classes,
    }


@app.post("/predict", tags=["model"])
def predict(body: PredictRequest):
    model = model_find(MLFLOW_MODEL_REGISTERED)
    if model is None:
        raise HTTPException(status_code=500, detail="Model not found")

    if not body.values:
        raise HTTPException(status_code=400, detail="No features provided")

    df = pd.DataFrame(body.values)

    missing = [c for c in model.feature_names_in_ if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422, detail=f"missing feature columns: {missing}"
        )

    X = df[model.feature_names_in_]

    df_proba = pd.DataFrame(model.predict_proba(X), columns=model.classes_)
    df_proba["id"] = df["id"].copy()
    df_proba.set_index("id", inplace=True)

    return {"predictions": df_proba.to_dict(orient="index")}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
