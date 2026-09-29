"""
Fairness check: does the model work equally well for different groups of people?

Credit scoring is a high-risk use case under the EU AI Act, so we do not only
ask "is the model accurate?" but also "is it equally accurate for men and
women, for young and older customers, ...?".

For every sensitive attribute (SEX, AGE band, MARRIAGE, EDUCATION) we split the
test customers into groups and compute, per group:

- recall               of the real defaulters in the group, how many we catch
                       (if lower for one group, we fail to help that group)
- false_positive_rate  of the good customers in the group, how many we wrongly
                       flag (if higher for one group, it is unfairly suspected)
- flag_rate            share of the group we flag as high risk
- actual_default_rate  share of the group that really defaulted (context:
                       flag rates can differ simply because default rates differ)

Then we compare the groups with a ratio = lowest value / highest value.
Rule of thumb ("80% rule"): a ratio below 0.8 means the gap should be reviewed
by a human before the model is used. The check does NOT block training: it
produces evidence for the governance review.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw plots without a screen (needed inside Docker)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Readable names for the dataset codes.
GROUP_NAMES = {
    "SEX": {1: "male", 2: "female"},
    "MARRIAGE": {1: "married", 2: "single", 3: "others"},
    "EDUCATION": {1: "graduate school", 2: "university", 3: "high school", 4: "others"},
}
AGE_BINS = [0, 29, 39, 49, 200]
AGE_LABELS = ["<30", "30-39", "40-49", "50+"]

SENSITIVE_ATTRIBUTES = ["SEX", "AGE", "MARRIAGE", "EDUCATION"]
COMPARED_METRICS = ["recall", "false_positive_rate", "flag_rate"]

DISPARITY_THRESHOLD = 0.8  # the "80% rule"

# Tiny groups give unreliable numbers (e.g. "recall = 0.20" based on only
# 5 defaulters is noise). A group only enters a ratio if the metric is based
# on at least MIN_CASES customers: recall needs 30 defaulters, the false
# positive rate needs 30 good customers, the flag rate needs 30 customers.
MIN_CASES = 30
METRIC_BASE = {
    "recall": "n_defaulters",
    "false_positive_rate": "n_non_defaulters",
    "flag_rate": "n_customers",
}


def group_labels(X: pd.DataFrame, attribute: str) -> pd.Series:
    """Turn the raw column into readable group names (e.g. 2 -> 'female')."""
    if attribute == "AGE":
        return pd.cut(X["AGE"], bins=AGE_BINS, labels=AGE_LABELS).astype(str)

    values = X[attribute]
    # Same grouping of undocumented codes as in src/features.py.
    if attribute == "EDUCATION":
        values = values.replace({0: 4, 5: 4, 6: 4})
    if attribute == "MARRIAGE":
        values = values.replace({0: 3})
    return values.map(GROUP_NAMES[attribute]).fillna("unknown").astype(str)


def _safe_divide(numerator: float, denominator: float):
    return float(numerator / denominator) if denominator > 0 else None


def group_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Confusion-matrix based metrics for one group of customers."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    n = len(y_true)
    return {
        "n_customers": n,
        "n_defaulters": tp + fn,
        "n_non_defaulters": fp + tn,
        "actual_default_rate": _safe_divide(tp + fn, n),
        "flag_rate": _safe_divide(tp + fp, n),
        "recall": _safe_divide(tp, tp + fn),
        "false_positive_rate": _safe_divide(fp, fp + tn),
        "precision": _safe_divide(tp, tp + fp),
    }


def _ratio(values: list) -> float | None:
    """Lowest / highest value. 1.0 = identical groups, lower = bigger gap."""
    values = [v for v in values if v is not None]
    if len(values) < 2 or max(values) == 0:
        return None
    return float(min(values) / max(values))


def fairness_report(X: pd.DataFrame, y_true, y_pred) -> dict:
    """Compute per-group metrics and disparity ratios for every attribute."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    report = {
        "threshold_rule": f"ratio < {DISPARITY_THRESHOLD} -> needs review",
        "min_cases_per_metric": MIN_CASES,
        "attributes": {},
        "needs_review": [],
    }

    for attribute in SENSITIVE_ATTRIBUTES:
        labels = group_labels(X, attribute).to_numpy()
        groups = {}
        present = set(labels)
        # Age bands in natural order (<30, 30-39, ...), other groups alphabetically.
        order = [g for g in AGE_LABELS if g in present] if attribute == "AGE" else sorted(present)
        for group in order:
            mask = labels == group
            groups[group] = group_metrics(y_true[mask], y_pred[mask])

        ratios, excluded = {}, {}
        for metric in COMPARED_METRICS:
            base = METRIC_BASE[metric]
            reliable = {g: m for g, m in groups.items() if m[base] >= MIN_CASES}
            ratios[metric] = _ratio([m[metric] for m in reliable.values()])
            excluded[metric] = sorted(set(groups) - set(reliable))
        flagged = [
            metric
            for metric, ratio in ratios.items()
            if ratio is not None and ratio < DISPARITY_THRESHOLD
        ]
        report["attributes"][attribute] = {
            "groups": groups,
            "ratios": ratios,
            "too_small_to_compare": excluded,
            "metrics_below_threshold": flagged,
        }
        report["needs_review"] += [f"{attribute}:{metric}" for metric in flagged]

    return report


def summary_metrics(report: dict) -> dict:
    """Flat numbers for MLflow, e.g. fairness_sex_recall_ratio = 0.95."""
    metrics = {}
    for attribute, content in report["attributes"].items():
        for metric, ratio in content["ratios"].items():
            if ratio is not None:
                metrics[f"fairness_{attribute.lower()}_{metric}_ratio"] = ratio
    metrics["fairness_checks_below_threshold"] = float(len(report["needs_review"]))
    return metrics


def save_report(report: dict, output_path: Path) -> Path:
    output_path.write_text(json.dumps(report, indent=2))
    return output_path


def plot_fairness(report: dict, title: str, output_path: Path) -> Path:
    """One small bar chart per attribute: recall and false-positive rate per group."""
    attributes = list(report["attributes"])
    fig, axes = plt.subplots(1, len(attributes), figsize=(4.2 * len(attributes), 4.2))
    for ax, attribute in zip(np.atleast_1d(axes), attributes):
        groups = report["attributes"][attribute]["groups"]
        names = list(groups)
        recall = [groups[g]["recall"] or 0 for g in names]
        fpr = [groups[g]["false_positive_rate"] or 0 for g in names]
        x = np.arange(len(names))
        ax.bar(x - 0.2, recall, width=0.4, label="Recall (higher = better)", color="#4C78A8")
        ax.bar(x + 0.2, fpr, width=0.4, label="False positive rate (lower = better)", color="#E45756")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(attribute)
    np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path
