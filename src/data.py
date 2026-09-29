"""
Data loading and preparation.

Responsibilities:
- find and read the Kaggle CSV
- detect the target column (its name varies between downloads)
- remove the ID column (it is not a predictive feature)
- split the data into a stratified train / test set
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def resolve_data_path(path: Path | str | None = None) -> Path:
    """Return the CSV path to use.

    1. If a path is given (or DATA_PATH is set) and exists, use it.
    2. Otherwise, if data/raw/ contains exactly one CSV file, use that one
       (so the project still works if you forgot to rename the Kaggle file,
       e.g. UCI_Credit_Card.csv).
    """
    candidate = Path(path) if path is not None else config.DATA_PATH
    if candidate.exists():
        return candidate

    csv_files = sorted(candidate.parent.glob("*.csv")) if candidate.parent.exists() else []
    if len(csv_files) == 1:
        print(f"[data] '{candidate.name}' not found, using '{csv_files[0].name}' instead.")
        return csv_files[0]

    raise FileNotFoundError(
        f"Dataset not found at '{candidate}'.\n"
        "Download 'Default of Credit Card Clients Dataset' from Kaggle and save it as:\n"
        "    data/raw/default_credit_card_clients.csv"
    )


def detect_target_column(df: pd.DataFrame) -> str:
    """Find the target column, whatever variant of the name the CSV uses."""
    # Compare in a normalised form: lower case, dots/spaces -> underscores.
    def normalise(name: str) -> str:
        return str(name).strip().lower().replace(".", "_").replace(" ", "_")

    normalised_columns = {normalise(col): col for col in df.columns}
    for candidate in config.TARGET_CANDIDATES:
        key = normalise(candidate)
        if key in normalised_columns:
            return normalised_columns[key]

    raise ValueError(
        "Could not find the target column. Expected one of: "
        f"{config.TARGET_CANDIDATES}. Columns found: {list(df.columns)}"
    )


def load_data(path: Path | str | None = None) -> pd.DataFrame:
    """Load the CSV and return a clean DataFrame.

    The returned DataFrame contains the raw feature columns plus the target,
    renamed to config.TARGET_COLUMN. The ID column is removed.
    """
    csv_path = resolve_data_path(path)
    df = pd.read_csv(csv_path)

    # Some versions call the first repayment column PAY_1 instead of PAY_0.
    if "PAY_0" not in df.columns and "PAY_1" in df.columns:
        df = df.rename(columns={"PAY_1": "PAY_0"})

    target = detect_target_column(df)
    df = df.rename(columns={target: config.TARGET_COLUMN})

    # ID is just a row number: never use it as a feature.
    if config.ID_COLUMN in df.columns:
        df = df.drop(columns=[config.ID_COLUMN])

    missing = [col for col in config.RAW_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"The dataset is missing expected columns: {missing}")

    # Keep only the columns we use, in a fixed order.
    df = df[config.RAW_FEATURES + [config.TARGET_COLUMN]].copy()
    df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].astype(int)
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the input features (X) from the target (y)."""
    X = df[config.RAW_FEATURES]
    y = df[config.TARGET_COLUMN]
    return X, y


def train_test_split_data(df: pd.DataFrame):
    """Stratified split so train and test keep the same share of defaults."""
    X, y = split_features_target(df)
    return train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
