"""Treino temporalmente seguro do modelo de campeão da Fórmula 1.

Executa backtests rolling-origin, calibra apenas com dados passados e registra
um model card junto ao estimador no MLflow. A temporada em andamento nunca é
usada como rótulo de treino.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import mlflow.data
import numpy as np
import pandas as pd
from sklearn import ensemble, impute, metrics, pipeline
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

NON_FEATURES = {"dt_ref", "DriverId", "flChampion", "Year", "id"}


class TemporalCalibratedClassifier:
    """Estimador sklearn-like com calibração aprendida no último ano concluído."""

    def __init__(self, estimator, calibrator, feature_names: list[str]):
        self.estimator = estimator
        self.calibrator = calibrator
        self.feature_names_in_ = np.asarray(feature_names)
        self.classes_ = np.asarray([0, 1])

    def _calibrate(self, raw: np.ndarray) -> np.ndarray:
        return self.calibrator.predict_proba(np.asarray(raw).reshape(-1, 1))[:, 1]

    def predict_proba(self, frame) -> np.ndarray:
        raw = self.estimator.predict_proba(frame)[:, 1]
        positive = self._calibrate(raw)
        return np.column_stack([1 - positive, positive])

    def predict(self, frame) -> np.ndarray:
        return (self.predict_proba(frame)[:, 1] >= 0.5).astype(int)

    def predict_proba_members(self, frame) -> list[np.ndarray]:
        """Probabilidade positiva por árvore para o intervalo de dispersão."""
        steps = list(self.estimator.named_steps.values())
        transformed = frame
        for step in steps[:-1]:
            transformed = step.transform(transformed)
        return [
            self._calibrate(tree.predict_proba(transformed)[:, 1])
            for tree in steps[-1].estimators_
        ]


def prepare_training_frame(
    frame: pd.DataFrame, current_year: int | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Ordena a ABT e remove a temporada incompleta do universo rotulado."""
    current_year = current_year or datetime.now(UTC).year
    out = frame.copy()
    out["dt_ref"] = pd.to_datetime(out["dt_ref"], errors="raise")
    out["Year"] = out["dt_ref"].dt.year
    out = (
        out[out["Year"] < current_year]
        .sort_values(["dt_ref", "DriverId"])
        .reset_index(drop=True)
    )
    features = [column for column in out.columns if column not in NON_FEATURES]
    if out.empty or out["flChampion"].nunique() < 2:
        raise ValueError(
            "Training data must contain completed seasons and both target classes"
        )
    return out, features


def _base_estimator(random_state: int = 42) -> pipeline.Pipeline:
    return pipeline.Pipeline(
        [
            ("imputer", impute.SimpleImputer(strategy="constant", fill_value=-10000)),
            (
                "forest",
                ensemble.RandomForestClassifier(
                    min_samples_leaf=40,
                    n_estimators=400,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _normalize_by_snapshot(scores: np.ndarray, dates: pd.Series) -> np.ndarray:
    values = pd.Series(np.clip(scores, 0, None), index=dates.index)
    totals = values.groupby(dates).transform("sum")
    sizes = dates.groupby(dates).transform("size")
    return values.div(totals.where(totals > 0)).where(totals > 0, 1 / sizes).to_numpy()


def _snapshot_accuracy(test: pd.DataFrame, probabilities: np.ndarray) -> float:
    scored = test[["dt_ref", "DriverId", "flChampion"]].copy()
    scored["probability"] = probabilities
    winners = scored.loc[scored.groupby("dt_ref")["probability"].idxmax()]
    return float(winners["flChampion"].mean())


def _recent_points_baseline(test: pd.DataFrame) -> float:
    candidates = [
        column
        for column in (
            "total_points_r_20",
            "total_points_20",
            "total_points_r_10",
            "total_points_10",
        )
        if column in test
    ]
    if not candidates:
        return 0.0
    field = candidates[0]
    winners = test.loc[test.groupby("dt_ref")[field].idxmax()]
    return float(winners["flChampion"].mean())


def rolling_backtest(
    frame: pd.DataFrame, features: list[str], min_train_seasons: int = 8
) -> tuple[list[dict], pd.DataFrame]:
    """Treina somente no passado e testa uma temporada futura por vez."""
    years = sorted(frame["Year"].unique())
    evaluations: list[dict] = []
    scored_parts = []
    for index, season in enumerate(years):
        if index < min_train_seasons:
            continue
        train = frame[frame["Year"] < season]
        test = frame[frame["Year"] == season].copy()
        if train["flChampion"].nunique() < 2 or test.empty:
            continue
        model = _base_estimator(random_state=int(season))
        model.fit(train[features], train["flChampion"])
        raw = model.predict_proba(test[features])[:, list(model.classes_).index(1)]
        probability = _normalize_by_snapshot(raw, test["dt_ref"])
        binary_probability = np.clip(raw, 1e-6, 1 - 1e-6)
        evaluations.append(
            {
                "season": int(season),
                "top1_accuracy": _snapshot_accuracy(test, probability),
                "baseline_accuracy": _recent_points_baseline(test),
                "roc_auc": float(metrics.roc_auc_score(test["flChampion"], raw)),
                "brier": float(metrics.brier_score_loss(test["flChampion"], raw)),
                "log_loss": float(
                    metrics.log_loss(test["flChampion"], binary_probability)
                ),
                "snapshots": int(test["dt_ref"].nunique()),
            }
        )
        test["raw_score"] = raw
        test["probability"] = probability
        scored_parts.append(
            test[["dt_ref", "DriverId", "flChampion", "raw_score", "probability"]]
        )
    return evaluations, pd.concat(
        scored_parts, ignore_index=True
    ) if scored_parts else pd.DataFrame()


def fit_final_model(frame: pd.DataFrame, features: list[str]):
    """Calibra no último ano e refaz o estimador com todo o histórico concluído."""
    ordered = frame.sort_values(["dt_ref", "DriverId"]).reset_index(drop=True)
    years = sorted(ordered["Year"].unique())
    if len(years) < 2:
        raise ValueError("Temporal calibration requires at least two completed seasons")
    calibration_year = years[-1]
    calibration_train = ordered[ordered["Year"] < calibration_year]
    calibration_test = ordered[ordered["Year"] == calibration_year]
    calibration_estimator = _base_estimator(random_state=calibration_year)
    calibration_estimator.fit(
        calibration_train[features], calibration_train["flChampion"]
    )
    raw = calibration_estimator.predict_proba(calibration_test[features])[:, 1]
    calibrator = LogisticRegression(class_weight="balanced", random_state=42)
    calibrator.fit(raw.reshape(-1, 1), calibration_test["flChampion"])
    final_estimator = _base_estimator()
    final_estimator.fit(ordered[features], ordered["flChampion"])
    return TemporalCalibratedClassifier(final_estimator, calibrator, features)


def build_model_card(
    frame: pd.DataFrame, evaluations: list[dict], scored: pd.DataFrame
) -> dict:
    calibration = []
    if not scored.empty:
        observed, predicted = calibration_curve(
            scored["flChampion"], scored["raw_score"], n_bins=8, strategy="quantile"
        )
        calibration = [
            {"predicted": float(p), "observed": float(o)}
            for p, o in zip(predicted, observed)
        ]
    model_score = (
        np.mean([item["top1_accuracy"] for item in evaluations]) if evaluations else 0
    )
    baseline_score = (
        np.mean([item["baseline_accuracy"] for item in evaluations])
        if evaluations
        else 0
    )
    return {
        "status": "validated"
        if evaluations and model_score > baseline_score
        else "experimental",
        "trained_through": int(frame["Year"].max()),
        "training_seasons": [int(frame["Year"].min()), int(frame["Year"].max())],
        "validation_strategy": "rolling_origin_by_season",
        "probability_calibration": "sigmoid_on_latest_completed_season",
        "baseline": "recent_points_20_sessions",
        "evaluations": evaluations,
        "calibration": calibration,
        "limitations": [
            "Não incorpora telemetria, clima ou estratégia de pneus.",
            "Probabilidades são normalizadas entre os pilotos de cada snapshot.",
            "Intervalos da API representam dispersão do ensemble calibrado.",
        ],
        "generated_at": datetime.now(UTC).isoformat(),
    }


def train_and_log(frame: pd.DataFrame):
    prepared, features = prepare_training_frame(frame)
    evaluations, scored = rolling_backtest(prepared, features)
    model = fit_final_model(prepared, features)
    model.model_card_ = build_model_card(prepared, evaluations, scored)

    mlflow.set_tracking_uri(os.environ["MLFLOW_URI"])
    mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])
    with mlflow.start_run():
        if evaluations:
            latest = evaluations[-1]
            mlflow.log_metrics(
                {
                    f"oot_{key}": value
                    for key, value in latest.items()
                    if key not in {"season", "snapshots"}
                }
            )
        mlflow.log_dict(model.model_card_, "model_card.json")
        mlflow.log_input(
            mlflow.data.from_pandas(
                prepared, name="tb_abt_completed", targets="flChampion"
            ),
            context="training",
        )
        mlflow.sklearn.log_model(
            sk_model=model,
            name="champion_model",
            registered_model_name=os.environ["MLFLOW_MODEL_REGISTERED"],
            input_example=prepared[features].head(3),
            code_paths=[str(Path(__file__).resolve().parents[1] / "src")],
        )
    return model


def main() -> None:
    from src.spark_session import spark_session

    spark = spark_session()
    try:
        frame = (
            spark.read.format("delta")
            .load(f"{os.environ['PATH_SILVER']}/tb_abt")
            .toPandas()
        )
    finally:
        spark.stop()
    train_and_log(frame)


if __name__ == "__main__":
    main()
