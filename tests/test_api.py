"""
Tests for the FastAPI service.

These tests do NOT need a running MLflow server: instead of loading the model
from the registry, we train a tiny pipeline on synthetic data and put it into
the API's model store.
"""

import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from api.main import app, model_store
from api.schemas import CustomerFeatures
from src.data import load_data, split_features_target
from src.features import build_model_pipeline

# The example request shown in Swagger - reused here as a valid customer.
EXAMPLE_CUSTOMER = CustomerFeatures.model_config["json_schema_extra"]["example"]

# Note: TestClient(app) without "with" does not run the start-up event,
# so the API does not try to contact MLflow during the tests.
client = TestClient(app)


@pytest.fixture
def loaded_model(synthetic_csv):
    X, y = split_features_target(load_data(synthetic_csv))
    pipeline = build_model_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    model_store.model = pipeline
    model_store.model_version = "test"
    model_store.model_uri = "models:/credit_default_model@champion"
    yield
    model_store.model = None


def test_root_returns_status():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_returns_expected_fields(loaded_model):
    response = client.post("/predict", json=EXAMPLE_CUSTOMER)
    assert response.status_code == 200
    body = response.json()
    for field in ["prediction", "label", "default_probability", "model_version"]:
        assert field in body
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("high_default_risk", "low_default_risk")
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["model_version"] == "test"


def test_predict_rejects_invalid_input(loaded_model):
    bad_customer = {**EXAMPLE_CUSTOMER, "AGE": 5}  # age must be >= 18
    response = client.post("/predict", json=bad_customer)
    assert response.status_code == 422


def test_predict_returns_503_without_model(monkeypatch):
    model_store.model = None
    monkeypatch.setattr(model_store, "load", lambda: False)
    response = client.post("/predict", json=EXAMPLE_CUSTOMER)
    assert response.status_code == 503
