"""Tests for data loading and feature preparation."""

import pandas as pd
import pytest

from src import config
from src.data import detect_target_column, load_data, train_test_split_data
from src.features import build_model_pipeline, prepare_features


def test_load_data_works(synthetic_csv):
    df = load_data(synthetic_csv)
    assert len(df) == 300
    assert config.TARGET_COLUMN in df.columns


def test_id_is_removed(synthetic_csv):
    df = load_data(synthetic_csv)
    assert config.ID_COLUMN not in df.columns
    assert config.ID_COLUMN not in config.RAW_FEATURES


@pytest.mark.parametrize(
    "target_name",
    ["default.payment.next.month", "default payment next month", "default_payment_next_month"],
)
def test_target_is_detected_for_common_names(synthetic_df, target_name):
    df = synthetic_df.rename(columns={"default.payment.next.month": target_name})
    assert detect_target_column(df) == target_name


def test_missing_target_raises_error(synthetic_df):
    df = synthetic_df.drop(columns=["default.payment.next.month"])
    with pytest.raises(ValueError):
        detect_target_column(df)


def test_pay_1_is_renamed_to_pay_0(tmp_path, synthetic_df):
    path = tmp_path / "data.csv"
    synthetic_df.rename(columns={"PAY_0": "PAY_1"}).to_csv(path, index=False)
    df = load_data(path)
    assert "PAY_0" in df.columns


def test_stratified_split_keeps_default_rate(synthetic_csv):
    df = load_data(synthetic_csv)
    X_train, X_test, y_train, y_test = train_test_split_data(df)
    assert len(X_test) == pytest.approx(len(df) * config.TEST_SIZE, abs=1)
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.05)


def test_prepare_features_adds_derived_columns_and_groups_codes():
    row = {col: 0 for col in config.RAW_FEATURES}
    row.update({"EDUCATION": 6, "MARRIAGE": 0, "PAY_0": 2, "PAY_2": 1, "BILL_AMT1": 600})
    out = prepare_features(pd.DataFrame([row]))
    assert out.loc[0, "EDUCATION"] == 4
    assert out.loc[0, "MARRIAGE"] == 3
    assert out.loc[0, "NUM_DELAYED_PAYMENTS"] == 2
    assert out.loc[0, "AVG_BILL_AMT"] == 100


def test_pipeline_trains_and_predicts_probabilities(synthetic_csv):
    from sklearn.linear_model import LogisticRegression

    df = load_data(synthetic_csv)
    X_train, X_test, y_train, _ = train_test_split_data(df)
    pipeline = build_model_pipeline(LogisticRegression(max_iter=500))
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_download_is_skipped_when_csv_exists(tmp_path):
    from src.download_data import download_dataset

    existing = tmp_path / "default_credit_card_clients.csv"
    existing.write_text("already here")
    assert download_dataset(existing) == existing
    assert existing.read_text() == "already here"


def test_download_without_credentials_gives_clear_error(tmp_path, monkeypatch):
    from src import download_data

    monkeypatch.setattr(download_data, "kaggle_credentials_available", lambda: False)
    with pytest.raises(RuntimeError, match="Kaggle credentials"):
        download_data.download_dataset(tmp_path / "missing.csv")
