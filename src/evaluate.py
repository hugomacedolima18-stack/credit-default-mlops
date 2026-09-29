"""
Model evaluation: metrics and plots.

We do NOT rely on accuracy alone. About 22% of customers default, so a model
that always says "no default" would already be ~78% accurate while being
useless. ROC-AUC is used to choose the best model; recall, precision and F1
show the trade-off at the chosen threshold.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw plots without a screen (needed inside Docker)
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Return the classification metrics as a dict of floats."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def plot_confusion_matrix(y_true, y_pred, title: str, output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["No default", "Default"],
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def plot_roc_curve(y_true, y_proba, title: str, output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax, name="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random guess")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def plot_precision_recall_curve(y_true, y_proba, title: str, output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax, name="Model")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def plot_roc_comparison(results: dict, y_true, output_path: Path) -> Path:
    """One ROC chart with both models, handy for the slides."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for model_name, result in results.items():
        RocCurveDisplay.from_predictions(
            y_true, result["y_proba"], ax=ax, name=model_name
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random guess")
    ax.set_title("ROC curve - model comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path
