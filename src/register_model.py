"""
Register the selected model in the MLflow Model Registry - Champion vs Challenger.

After training, the best model of this run (highest ROC-AUC) is registered as a
new version of `credit_default_model`. It is the CHALLENGER. Then we compare it
with the current CHAMPION (the version the API serves):

- no champion yet                      -> the challenger becomes '@champion'
- challenger ROC-AUC > champion ROC-AUC -> promoted: it becomes '@champion'
- otherwise                            -> NOT promoted: it gets '@challenger',
                                          the old champion stays in production

The API always loads `models:/credit_default_model@champion`, so promoting a
model only means moving the alias - no code change in the API. A bad retrain
(e.g. on broken data) can never replace a better model automatically.
"""

import mlflow
from mlflow import MlflowClient

from src import config

CHALLENGER_ALIAS = "challenger"
PROMOTION_METRIC = "roc_auc"
# How much better the challenger must be. 0.0 = any improvement counts;
# a real team could require e.g. +0.005 to avoid swapping models for noise.
MIN_IMPROVEMENT = 0.0


def should_promote(challenger_score: float, champion_score: float | None,
                   min_improvement: float = MIN_IMPROVEMENT) -> bool:
    """The promotion rule: no champion yet, or the challenger is strictly better."""
    if champion_score is None:
        return True
    return challenger_score > champion_score + min_improvement


def get_champion(client: MlflowClient):
    """Return (version, score) of the current champion, or (None, None)."""
    try:
        champion = client.get_model_version_by_alias(
            config.REGISTERED_MODEL_NAME, config.MODEL_ALIAS
        )
    except Exception:  # noqa: BLE001 - no model or no alias yet
        return None, None
    score = client.get_run(champion.run_id).data.metrics.get(PROMOTION_METRIC)
    return str(champion.version), score


def register_best_model(model_uri: str, run_id: str, model_type: str, metrics: dict) -> dict:
    """Register the challenger and decide whether it becomes the champion.

    Returns a dict with the new version, whether it was promoted, and the
    version that is champion after the decision.
    """
    client = MlflowClient()
    name = config.REGISTERED_MODEL_NAME

    # Look at the champion BEFORE registering the new version.
    champion_version, champion_score = get_champion(client)

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=name,
        tags={"model_type": model_type, "source_run_id": run_id},
    )
    version = str(model_version.version)
    challenger_score = metrics[PROMOTION_METRIC]

    client.update_registered_model(
        name=name,
        description=(
            "Credit default risk prototype (academic project). Predicts the "
            "probability that a credit card customer defaults next month. "
            "Decision-support only - not an automated credit decision system. "
            "See MODEL_CARD.md."
        ),
    )

    promoted = should_promote(challenger_score, champion_score)
    if champion_score is None:
        decision = "first model - promoted to champion"
    elif promoted:
        decision = (f"promoted: {PROMOTION_METRIC} {challenger_score:.4f} beats "
                    f"champion v{champion_version} ({champion_score:.4f})")
    else:
        decision = (f"not promoted: {PROMOTION_METRIC} {challenger_score:.4f} does not beat "
                    f"champion v{champion_version} ({champion_score:.4f})")

    client.update_model_version(
        name=name,
        version=version,
        description=(
            f"{model_type}, ROC-AUC = {challenger_score:.4f}, "
            f"recall = {metrics['recall']:.4f}. Decision: {decision}."
        ),
    )
    client.set_model_version_tag(name, version, "promotion_decision", decision)

    if promoted:
        client.set_registered_model_alias(name, config.MODEL_ALIAS, version)
        # The previous challenger (if any) is no longer relevant.
        try:
            client.delete_registered_model_alias(name, CHALLENGER_ALIAS)
        except Exception:  # noqa: BLE001 - there was no challenger
            pass
        champion_after = version
    else:
        client.set_registered_model_alias(name, CHALLENGER_ALIAS, version)
        champion_after = champion_version

    print(f"[registry] Challenger = version {version} ({model_type}). {decision}.")
    print(f"[registry] '@{config.MODEL_ALIAS}' is version {champion_after}.")

    return {"version": version, "promoted": promoted, "champion_version": champion_after,
            "decision": decision}
