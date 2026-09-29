"""
Download the dataset automatically with the Kaggle API.

Run it with:
    python -m src.download_data                                   (locally)
    docker compose run --rm trainer python -m src.download_data   (with Docker)

You need a free Kaggle account and an API token. Put the credentials in a
`.env` file at the project root (never commit it - it is in .gitignore):

    KAGGLE_API_TOKEN=your_token             # new-style token (recommended)
or
    KAGGLE_USERNAME=your_username           # legacy kaggle.json credentials
    KAGGLE_KEY=your_key

Locally you can instead keep the token in ~/.kaggle/access_token or
~/.kaggle/kaggle.json.

Manually placing the CSV in data/raw/ still works - this is only a shortcut.
"""

import os
import shutil
import tempfile
from pathlib import Path

from src import config

KAGGLE_DATASET = os.getenv("KAGGLE_DATASET", "uciml/default-of-credit-card-clients-dataset")
KAGGLE_FILE_NAME = "UCI_Credit_Card.csv"  # name of the CSV inside the Kaggle dataset


def kaggle_credentials_available() -> bool:
    """True if any of the supported Kaggle credential methods is present."""
    if os.getenv("KAGGLE_API_TOKEN"):
        return True
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return True
    kaggle_dir = Path(os.getenv("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle")))
    return (kaggle_dir / "access_token").exists() or (kaggle_dir / "kaggle.json").exists()


def download_dataset(destination: Path | None = None, force: bool = False) -> Path:
    """Download the CSV from Kaggle and save it as the expected file name."""
    destination = Path(destination or config.DATA_PATH)

    if destination.exists() and not force:
        print(f"[download] Dataset already present: {destination}")
        return destination

    if not kaggle_credentials_available():
        raise RuntimeError(
            "No Kaggle credentials found. Create a token at "
            "https://www.kaggle.com/settings/api and set KAGGLE_API_TOKEN "
            "(or KAGGLE_USERNAME + KAGGLE_KEY) in the .env file."
        )

    # Imported here because the kaggle package tries to authenticate on import.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        print(f"[download] Downloading '{KAGGLE_DATASET}' from Kaggle ...")
        api.dataset_download_files(KAGGLE_DATASET, path=tmp, unzip=True, quiet=False)

        csv_files = list(Path(tmp).rglob("*.csv"))
        if not csv_files:
            raise RuntimeError("The Kaggle download did not contain a CSV file.")
        # Prefer the known file name; otherwise take the only/first CSV.
        source = next((f for f in csv_files if f.name == KAGGLE_FILE_NAME), csv_files[0])
        shutil.copyfile(source, destination)

    print(f"[download] Saved to {destination}")
    return destination


def ensure_dataset() -> None:
    """Called by the training script: download only if the CSV is missing
    AND Kaggle credentials exist. Otherwise do nothing (data.py will explain
    what to do if the file is still missing)."""
    csv_in_raw = list(config.DATA_PATH.parent.glob("*.csv")) if config.DATA_PATH.parent.exists() else []
    if config.DATA_PATH.exists() or csv_in_raw:
        return
    if kaggle_credentials_available():
        download_dataset()


if __name__ == "__main__":
    download_dataset()
