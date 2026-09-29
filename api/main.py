"""
FastAPI prediction service.

The API loads the model from the MLflow Model Registry:
    1st choice: models:/credit_default_model@champion
    fallback:   the latest registered version of credit_default_model

If the model is not available yet (e.g. you started the API before training),
the API still starts; /predict simply tries to load the model again on the
next request.

Run it with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000     (locally)
    docker compose up -d api                            (with Docker)
Swagger docs: http://localhost:8000/docs
"""

import threading
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient

from api.schemas import CustomerFeatures, HealthResponse, PredictionResponse
from src import config

DISCLAIMER = (
    "Academic decision-support prototype. The score must be reviewed by a human "
    "and must not be used as an automated credit decision."
)


class ModelStore:
    """Holds the loaded model and knows how to (re)load it from MLflow."""

    def __init__(self) -> None:
        self.model = None
        self.model_version = "unknown"
        self.model_uri = ""
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> bool:
        """Try to load the champion model. Returns True on success."""
        with self._lock:
            mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
            client = MlflowClient()
            name = config.REGISTERED_MODEL_NAME

            # 1) Preferred: the model version that has the "champion" alias.
            try:
                version = client.get_model_version_by_alias(name, config.MODEL_ALIAS)
                uri = f"models:/{name}@{config.MODEL_ALIAS}"
                self._set(mlflow.sklearn.load_model(uri), str(version.version), uri)
                return True
            except Exception as error:  # noqa: BLE001
                print(f"[api] Could not load '@{config.MODEL_ALIAS}': {error}")

            # 2) Fallback: the most recent registered version.
            try:
                versions = client.search_model_versions(f"name='{name}'")
                if not versions:
                    print("[api] No registered model yet. Run the trainer first.")
                    return False
                latest = max(versions, key=lambda v: int(v.version))
                uri = f"models:/{name}/{latest.version}"
                self._set(mlflow.sklearn.load_model(uri), str(latest.version), uri)
                return True
            except Exception as error:  # noqa: BLE001
                print(f"[api] Could not load any model version: {error}")
                return False

    def _set(self, model, version: str, uri: str) -> None:
        self.model = model
        self.model_version = version
        self.model_uri = uri
        print(f"[api] Loaded {uri} (version {version})")


model_store = ModelStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Try to load the model at start-up; failure is not fatal.
    model_store.load()
    yield


app = FastAPI(
    title="Credit Default Risk API",
    description=(
        "Predicts the probability that a credit card customer defaults next month. "
        + DISCLAIMER
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict:
    return {
        "project": "Credit Default Risk Prediction with MLOps",
        "status": "running",
        "model_loaded": model_store.is_loaded,
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="healthy", model_loaded=model_store.is_loaded)


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures) -> PredictionResponse:
    # Lazy loading: if training finished after the API started, pick it up now.
    if not model_store.is_loaded and not model_store.load():
        raise HTTPException(
            status_code=503,
            detail="Model not available. Train and register a model first "
            "(docker compose run --rm trainer).",
        )

    # One-row DataFrame with the columns in the order used in training.
    row = pd.DataFrame([customer.model_dump()])[config.RAW_FEATURES]
    probability = float(model_store.model.predict_proba(row)[0, 1])
    prediction = int(probability >= config.CLASSIFICATION_THRESHOLD)

    return PredictionResponse(
        prediction=prediction,
        label="high_default_risk" if prediction == 1 else "low_default_risk",
        default_probability=round(probability, 4),
        threshold=config.CLASSIFICATION_THRESHOLD,
        model_version=model_store.model_version,
        model_uri=model_store.model_uri,
        disclaimer=DISCLAIMER,
    )


@app.post("/reload")
def reload_model() -> dict:
    """Reload the champion model (e.g. after retraining), without restarting."""
    if not model_store.load():
        raise HTTPException(status_code=503, detail="No registered model could be loaded.")
    return {"status": "reloaded", "model_version": model_store.model_version}
