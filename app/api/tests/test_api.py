"""Route tests for ``app/api/main.py``.

MLflow is never contacted: ``main.model_find`` is monkeypatched to return a
minimal fake estimator (or ``None``), and the model cache is cleared between
tests.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import main


class FakeModel:
    """Just enough of a fitted sklearn estimator for the routes."""

    feature_names_in_ = np.array(["f1", "f2"])
    classes_ = np.array([0, 1])
    # a Pipeline would nest this on a step; _feature_importances handles both
    feature_importances_ = np.array([0.25, 0.75])

    def predict_proba(self, X):
        return np.tile([0.3, 0.7], (len(X), 1))


@pytest.fixture(autouse=True)
def _clear_model_cache():
    main._MODEL_CACHE.clear()
    yield
    main._MODEL_CACHE.clear()


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def fake_model(monkeypatch):
    monkeypatch.setattr(main, "model_find", lambda *a, **k: FakeModel())
    return FakeModel()


def test_health_check(client):
    resp = client.get("/health_check")
    assert resp.status_code == 200
    assert resp.json() == {"status": "OK"}


def test_predict_happy_path(client, fake_model):
    payload = {
        "values": [
            {"id": "2024-03-10_max", "f1": 1.0, "f2": 2.0},
            {"id": "2024-03-10_lando", "f1": 3.0, "f2": 4.0},
        ]
    }
    resp = client.post("/predict", json=payload)

    assert resp.status_code == 200
    preds = resp.json()["predictions"]
    assert set(preds) == {"2024-03-10_max", "2024-03-10_lando"}
    # inner keys are model.classes_ (0, 1) serialised as JSON string keys
    assert preds["2024-03-10_max"]["1"] == pytest.approx(0.7)
    assert preds["2024-03-10_max"]["0"] == pytest.approx(0.3)


def test_predict_empty_values_returns_400(client, fake_model):
    resp = client.post("/predict", json={"values": []})
    assert resp.status_code == 400
    assert resp.json()["detail"] == "No features provided"


def test_predict_model_not_found_returns_500(client, monkeypatch):
    monkeypatch.setattr(main, "model_find", lambda *a, **k: None)
    resp = client.post("/predict", json={"values": [{"id": "x", "f1": 1, "f2": 2}]})
    assert resp.status_code == 500
    assert resp.json()["detail"] == "Model not found"


def test_predict_missing_feature_column_returns_422(client, fake_model):
    resp = client.post("/predict", json={"values": [{"id": "x", "f1": 1.0}]})
    assert resp.status_code == 422
    assert "f2" in resp.json()["detail"]


def test_predict_values_not_a_list_returns_422(client, fake_model):
    resp = client.post("/predict", json={"values": "not-a-list"})
    assert resp.status_code == 422


def test_predict_v1_normalizes_each_snapshot(client, fake_model):
    response = client.post(
        "/v1/predict",
        json={
            "values": [
                {"id": "2024-01-01_a", "prediction_group": "r1", "f1": 1, "f2": 2},
                {"id": "2024-01-01_b", "prediction_group": "r1", "f1": 3, "f2": 4},
            ]
        },
    )
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert sum(item["probability"] for item in predictions.values()) == pytest.approx(1)
    assert predictions["2024-01-01_a"]["raw_score"] == pytest.approx(0.7)


def test_model_card_has_safe_defaults(client, fake_model):
    body = client.get("/v1/model-card").json()
    assert body["status"] == "experimental"
    assert body["evaluations"] == []


def test_explain_v1_uses_explanation_helper(client, fake_model, monkeypatch):
    monkeypatch.setattr(
        main,
        "_shap_explanations",
        lambda model, frame, top_n: {frame.iloc[0]["id"]: {"contributions": []}},
    )
    response = client.post(
        "/v1/explain",
        json={"values": [{"id": "x", "f1": 1, "f2": 2}], "top_n": 5},
    )
    assert response.status_code == 200
    assert "x" in response.json()["explanations"]


# ── /model_info ─────────────────────────────────────────────────────────────


def test_model_info_happy_path(client, fake_model):
    resp = client.get("/model_info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_features"] == 2
    assert body["features"] == ["f1", "f2"]
    assert body["importances"] == {"f1": pytest.approx(0.25), "f2": pytest.approx(0.75)}
    assert body["classes"] == [0, 1]


def test_model_info_no_model_returns_500(client, monkeypatch):
    monkeypatch.setattr(main, "model_find", lambda *a, **k: None)
    resp = client.get("/model_info")
    assert resp.status_code == 500


def test_model_info_estimator_without_importances(client, monkeypatch):
    class Bare:
        feature_names_in_ = np.array(["a", "b", "c"])
        classes_ = np.array([0, 1])

    monkeypatch.setattr(main, "model_find", lambda *a, **k: Bare())
    body = client.get("/model_info").json()
    assert body["n_features"] == 3
    assert body["importances"] == {}


def test_feature_importances_reads_from_pipeline_step():
    pipe = SimpleNamespace(
        feature_names_in_=np.array(["f1", "f2"]),
        named_steps={"imp": object(), "rf": FakeModel()},
    )
    names, importances = main._feature_importances(pipe)
    assert names == ["f1", "f2"]
    assert importances["f2"] == pytest.approx(0.75)


# ── model cache ─────────────────────────────────────────────────────────────


def test_model_find_caches_within_ttl(monkeypatch):
    calls = {"n": 0}

    def counting_loader(model_id=None):
        calls["n"] += 1
        return FakeModel()

    monkeypatch.setattr(main, "_load_model", counting_loader)
    monkeypatch.setattr(main, "MODEL_CACHE_TTL", 300.0)

    first = main.model_find("f1-champion")
    second = main.model_find("f1-champion")

    assert first is second
    assert calls["n"] == 1


def test_model_find_reloads_after_ttl(monkeypatch):
    calls = {"n": 0}

    def counting_loader(model_id=None):
        calls["n"] += 1
        return FakeModel()

    monkeypatch.setattr(main, "_load_model", counting_loader)
    monkeypatch.setattr(main, "MODEL_CACHE_TTL", 0.0)  # every entry is already stale

    main.model_find("f1-champion")
    main.model_find("f1-champion")

    assert calls["n"] == 2


def test_model_find_returns_none_on_loader_failure(monkeypatch):
    def boom(model_id=None):
        raise RuntimeError("mlflow down")

    monkeypatch.setattr(main, "_load_model", boom)
    assert main.model_find("f1-champion") is None
