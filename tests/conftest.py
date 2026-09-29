"""
Shared test fixtures.

The tests use a small SYNTHETIC dataset with the same columns as the Kaggle
file, so they run anywhere (including GitHub Actions) without the real data.
"""

import numpy as np
import pandas as pd
import pytest

from src import config


def make_synthetic_dataframe(n_rows: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {
        "ID": np.arange(1, n_rows + 1),
        "LIMIT_BAL": rng.integers(10_000, 500_000, n_rows),
        "SEX": rng.integers(1, 3, n_rows),
        "EDUCATION": rng.integers(0, 7, n_rows),
        "MARRIAGE": rng.integers(0, 4, n_rows),
        "AGE": rng.integers(21, 70, n_rows),
    }
    for col in config.PAY_STATUS_FEATURES:
        data[col] = rng.integers(-2, 4, n_rows)
    for col in config.BILL_FEATURES:
        data[col] = rng.integers(-1_000, 200_000, n_rows)
    for col in config.PAY_AMOUNT_FEATURES:
        data[col] = rng.integers(0, 20_000, n_rows)

    df = pd.DataFrame(data)
    # Make the target depend on payment delays so a model can learn something.
    risk = (df["PAY_0"] > 0).astype(int) + rng.random(n_rows)
    df["default.payment.next.month"] = (risk > 0.9).astype(int)
    return df


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    return make_synthetic_dataframe()


@pytest.fixture
def synthetic_csv(tmp_path, synthetic_df):
    path = tmp_path / "default_credit_card_clients.csv"
    synthetic_df.to_csv(path, index=False)
    return path
