"""
Register the selected model in the MLflow Model Registry.

After training, the best model (highest ROC-AUC) is registered under the name
`credit_default_model` and receives the alias `champion`. The API always loads
`models:/credit_default_model@champion`, so promoting a new model only means
moving the alias - no code change in the API.
"""

import mlflow
from mlflow import MlflowClient

from src import config


def register_best_model(model_uri: str, run_id: str, model_type: str, metrics: dict) -> str:
    """Register a logged model and point the 'champion' alias at it.

    Returns the new model version number (as a string).
    """
    client = MlflowClient()

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=config.REGISTERED_MODEL_NAME,
        tags={"model_type": model_type, "source_run_id": run_id},
    )
    version = str(model_version.version)

    client.update_registered_model(
        name=config.REGISTERED_MODEL_NAME,
        description=(
            "Credit default risk prototype (academic project). Predicts the "
            "probability that a credit card customer defaults next month. "
            "Decision-support only - not an automated credit decision system."
        ),
    )
    client.update_model_version(
        name=config.REGISTERED_MODEL_NAME,
        version=version,
        description=(
            f"{model_type} selected by ROC-AUC = {metrics['roc_auc']:.4f} "
            f"(recall = {metrics['recall']:.4f})."
        ),
    )

    # Aliases are the modern replacement for the deprecated "stages"
    # (Staging / Production). Fall back gracefully on very old MLflow versions.
    try:
        client.set_registered_model_alias(
            name=config.REGISTERED_MODEL_NAME,
            alias=config.MODEL_ALIAS,
            version=version,
        )
        print(
            f"[registry] {config.REGISTERED_MODEL_NAME} version {version} "
            f"is now '@{config.MODEL_ALIAS}'."
        )
    except Exception as error:  # noqa: BLE001 - keep training successful
        print(f"[registry] Could not set alias ({error}). The API will use the latest version.")

    return version
