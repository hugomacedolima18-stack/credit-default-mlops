"""
Central configuration for the whole project.

Every path, name and setting lives here, so the training code, the API and the
tests all read the same values. Most settings can be overridden with
environment variables (Docker Compose sets them in docker-compose.yml).
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Project root = the folder that contains src/, api/, data/ ...
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Where the Kaggle CSV is expected. Can be overridden with DATA_PATH.
DATA_PATH = Path(
    os.getenv("DATA_PATH", str(RAW_DATA_DIR / "default_credit_card_clients.csv"))
)

# Local folder where training writes plots and summaries (also logged to MLflow).
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", str(PROJECT_ROOT / "artifacts")))

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
# Inside Docker Compose this is set to http://mlflow:5000.
# When you run Python directly on your machine, the MLflow container is
# reachable at http://localhost:5000.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "credit-default-risk")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "credit_default_model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")

# ---------------------------------------------------------------------------
# Machine learning settings
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2  # 80% train / 20% test
CLASSIFICATION_THRESHOLD = 0.5  # probability above which we say "high risk"

# The target column has different names depending on where the CSV came from.
TARGET_CANDIDATES = [
    "default.payment.next.month",
    "default payment next month",
    "default_payment_next_month",
    "default",
    "Y",
]
# Internal, clean name we rename the target to.
TARGET_COLUMN = "default_next_month"

ID_COLUMN = "ID"

# Categorical variables (codes, not quantities).
CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]

# Repayment status per month (-2 = no consumption, -1 = paid duly,
# 0 = revolving credit, 1..9 = months of payment delay).
PAY_STATUS_FEATURES = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_FEATURES = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMOUNT_FEATURES = [f"PAY_AMT{i}" for i in range(1, 7)]

# The raw input columns the model (and the API) expects.
RAW_FEATURES = (
    ["LIMIT_BAL"]
    + CATEGORICAL_FEATURES
    + ["AGE"]
    + PAY_STATUS_FEATURES
    + BILL_FEATURES
    + PAY_AMOUNT_FEATURES
)

# Simple derived features created in src/features.py.
DERIVED_FEATURES = ["AVG_BILL_AMT", "AVG_PAY_AMT", "NUM_DELAYED_PAYMENTS"]

# Numeric columns after feature creation (everything except the categoricals).
NUMERIC_FEATURES = (
    ["LIMIT_BAL", "AGE"]
    + PAY_STATUS_FEATURES
    + BILL_FEATURES
    + PAY_AMOUNT_FEATURES
    + DERIVED_FEATURES
)
