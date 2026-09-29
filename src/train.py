"""
Training script: trains two models, tracks them in MLflow, registers the best.

Run it with:
    python -m src.train                      (locally)
    docker compose run --rm trainer          (with Docker)

What happens:
1. Load the data and create a stratified train/test split.
2. Train Logistic Regression and Random Forest. Each one gets its own MLflow Run
   with parameters, metrics, plots, a JSON summary and the model itself.
3. Compare the runs on ROC-AUC and register the winner as
   `credit_default_model` with alias `champion`.
"""

import json
from datetime import datetime, timezone

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src import config
from src.data import load_data, train_test_split_data
from src.download_data import ensure_dataset
from src.evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_comparison,
    plot_roc_curve,
)
from src.fairness import fairness_report, plot_fairness, save_report, summary_metrics
from src.features import build_model_pipeline
from src.register_model import register_best_model

# Types that skops (MLflow's model format) must be told to trust. See the
# comment next to mlflow.sklearn.log_model below.
SKOPS_TRUSTED_TYPES = ["src.features.prepare_features", "sklearn.tree._tree.Tree"]


def get_candidate_models() -> dict:
    """The two baseline models and their hyperparameters.

    class_weight="balanced" gives more importance to the minority class
    (defaulters), which usually increases recall.
    """
    return {
        "logistic_regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        ),
    }


def train_and_log_model(model_name, estimator, X_train, X_test, y_train, y_test) -> dict:
    """Train one model inside its own MLflow Run and log everything."""
    run_artifacts_dir = config.ARTIFACTS_DIR / model_name
    run_artifacts_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=model_name) as run:
        pipeline = build_model_pipeline(estimator)
        pipeline.fit(X_train, y_train)

        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= config.CLASSIFICATION_THRESHOLD).astype(int)
        metrics = compute_metrics(y_test, y_pred, y_proba)

        # ---- Parameters ------------------------------------------------------
        hyperparameters = {
            key: value
            for key, value in estimator.get_params().items()
            if key in {"C", "max_iter", "n_estimators", "max_depth", "min_samples_leaf", "solver"}
        }
        mlflow.log_params(
            {
                "model_type": model_name,
                "random_state": config.RANDOM_STATE,
                "test_size": config.TEST_SIZE,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "stratified_split": True,
                "class_weight": estimator.get_params().get("class_weight"),
                "classification_threshold": config.CLASSIFICATION_THRESHOLD,
                "n_raw_features": len(config.RAW_FEATURES),
                "derived_features": ",".join(config.DERIVED_FEATURES),
                **hyperparameters,
            }
        )

        # ---- Metrics ---------------------------------------------------------
        mlflow.log_metrics(metrics)

        # ---- Plots and summary (artifacts) -----------------------------------
        pretty_name = model_name.replace("_", " ").title()
        plots = [
            plot_confusion_matrix(
                y_test, y_pred, f"Confusion matrix - {pretty_name}",
                run_artifacts_dir / "confusion_matrix.png",
            ),
            plot_roc_curve(
                y_test, y_proba, f"ROC curve - {pretty_name}",
                run_artifacts_dir / "roc_curve.png",
            ),
            plot_precision_recall_curve(
                y_test, y_proba, f"Precision-Recall curve - {pretty_name}",
                run_artifacts_dir / "precision_recall_curve.png",
            ),
        ]
        for plot_path in plots:
            mlflow.log_artifact(str(plot_path), artifact_path="plots")

        # ---- Fairness check (governance evidence) ----------------------------
        # Same metrics, but per group (sex, age band, marriage, education).
        # Ratios below 0.8 are listed in "needs_review" for a human to check.
        fairness = fairness_report(X_test, y_test, y_pred)
        mlflow.log_metrics(summary_metrics(fairness))
        mlflow.log_artifact(
            str(save_report(fairness, run_artifacts_dir / "fairness_report.json")),
            artifact_path="fairness",
        )
        mlflow.log_artifact(
            str(plot_fairness(
                fairness, f"Fairness by group - {pretty_name}",
                run_artifacts_dir / "fairness_by_group.png",
            )),
            artifact_path="fairness",
        )
        mlflow.set_tag("fairness_needs_review", ", ".join(fairness["needs_review"]) or "none")

        summary = {
            "model_type": model_name,
            "run_id": run.info.run_id,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "threshold": config.CLASSIFICATION_THRESHOLD,
            "train_default_rate": float(y_train.mean()),
            "test_default_rate": float(y_test.mean()),
            "fairness_needs_review": fairness["needs_review"],
            "note": "Academic decision-support prototype. Not for automated credit decisions.",
        }
        summary_path = run_artifacts_dir / "run_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        mlflow.log_artifact(str(summary_path), artifact_path="summary")

        # ---- Model -----------------------------------------------------------
        # - code_paths ships our src/ package with the model, because the
        #   pipeline uses our own function src.features.prepare_features.
        # - MLflow 3 saves sklearn models in the safer "skops" format, which
        #   only loads types that are declared as trusted. We created these
        #   models ourselves, so we trust our own function and sklearn's tree
        #   structure (used inside the Random Forest).
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            input_example=X_train.head(3).astype(float),
            code_paths=[str(config.PROJECT_ROOT / "src")],
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

        mlflow.set_tags({"project": "credit-default-mlops", "stage": "baseline"})

        print(f"[train] {model_name:<20} " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        print(f"[fairness] {model_name:<17} needs review: {fairness['needs_review'] or 'none'}")

        return {
            "run_id": run.info.run_id,
            "model_uri": model_info.model_uri,
            "metrics": metrics,
            "y_proba": y_proba,
        }


def main() -> None:
    print(f"[train] MLflow tracking URI: {config.MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    # Downloads from Kaggle only if the CSV is missing and credentials exist.
    ensure_dataset()
    df = load_data()
    print(f"[train] Loaded {len(df):,} rows. Default rate: {df[config.TARGET_COLUMN].mean():.2%}")
    X_train, X_test, y_train, y_test = train_test_split_data(df)

    results = {}
    for model_name, estimator in get_candidate_models().items():
        results[model_name] = train_and_log_model(
            model_name, estimator, X_train, X_test, y_train, y_test
        )

    # ---- Model comparison: highest ROC-AUC wins ------------------------------
    best_name = max(results, key=lambda name: results[name]["metrics"]["roc_auc"])
    best = results[best_name]
    print(f"[train] Best model by ROC-AUC: {best_name} ({best['metrics']['roc_auc']:.4f})")

    client = mlflow.MlflowClient()
    for name, result in results.items():
        client.set_tag(result["run_id"], "selected_as_champion", str(name == best_name).lower())

    # Save a comparison file + a combined ROC chart (useful for the slides).
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison = {
        "selection_metric": "roc_auc",
        "selected_model": best_name,
        "models": {name: r["metrics"] for name, r in results.items()},
    }
    (config.ARTIFACTS_DIR / "model_comparison.json").write_text(json.dumps(comparison, indent=2))
    plot_roc_comparison(results, y_test, config.ARTIFACTS_DIR / "roc_comparison.png")

    # ---- Register the winner -------------------------------------------------
    version = register_best_model(
        model_uri=best["model_uri"],
        run_id=best["run_id"],
        model_type=best_name,
        metrics=best["metrics"],
    )

    print("\n================ MODEL COMPARISON ================")
    print(f"{'metric':<12}" + "".join(f"{name:>22}" for name in results))
    for metric in ["roc_auc", "recall", "precision", "f1", "accuracy"]:
        print(f"{metric:<12}" + "".join(f"{r['metrics'][metric]:>22.4f}" for r in results.values()))
    print("==================================================")
    print(f"Registered '{config.REGISTERED_MODEL_NAME}' version {version} ({best_name}).")
    print("Open MLflow at http://localhost:5000 to explore the runs.")


if __name__ == "__main__":
    main()
