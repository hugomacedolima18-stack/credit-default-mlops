"""Tests for the fairness check (src/fairness.py)."""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import load_data, split_features_target
from src.fairness import (
    SENSITIVE_ATTRIBUTES,
    fairness_report,
    group_labels,
    group_metrics,
    summary_metrics,
)


def test_group_metrics_are_computed_correctly():
    # 2 defaulters (1 caught, 1 missed) and 2 good customers (1 wrongly flagged).
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 1, 0]
    m = group_metrics(y_true, y_pred)
    assert m["n_customers"] == 4
    assert m["recall"] == 0.5
    assert m["false_positive_rate"] == 0.5
    assert m["flag_rate"] == 0.5
    assert m["actual_default_rate"] == 0.5


def test_recall_is_none_when_group_has_no_defaulters():
    m = group_metrics([0, 0, 0], [0, 1, 0])
    assert m["recall"] is None
    assert m["false_positive_rate"] == pytest.approx(1 / 3)


def test_group_labels_are_readable_and_codes_are_grouped():
    X = pd.DataFrame({"SEX": [1, 2], "AGE": [25, 55], "EDUCATION": [6, 1], "MARRIAGE": [0, 1]})
    assert list(group_labels(X, "SEX")) == ["male", "female"]
    assert list(group_labels(X, "AGE")) == ["<30", "50+"]
    assert list(group_labels(X, "EDUCATION")) == ["others", "graduate school"]
    assert list(group_labels(X, "MARRIAGE")) == ["others", "married"]


def test_report_covers_all_sensitive_attributes(synthetic_csv):
    X, y = split_features_target(load_data(synthetic_csv))
    y_pred = (X["PAY_0"] > 0).astype(int)
    report = fairness_report(X, y, y_pred)
    assert set(report["attributes"]) == set(SENSITIVE_ATTRIBUTES)
    for content in report["attributes"].values():
        for ratio in content["ratios"].values():
            assert ratio is None or 0.0 <= ratio <= 1.0


def test_large_gap_is_flagged_for_review():
    # Model catches every male defaulter but no female defaulter -> must be flagged.
    n = 100
    X = pd.DataFrame({col: np.zeros(2 * n, dtype=int) for col in config.RAW_FEATURES})
    X["SEX"] = [1] * n + [2] * n
    X["AGE"] = 35
    X["EDUCATION"] = 2
    X["MARRIAGE"] = 1
    y_true = np.array([1, 0] * n)
    y_pred = np.array([1, 0] * (n // 2) + [0, 0] * (n // 2))
    report = fairness_report(X, y_true, y_pred)
    assert "SEX:recall" in report["needs_review"]
    assert summary_metrics(report)["fairness_sex_recall_ratio"] == 0.0
