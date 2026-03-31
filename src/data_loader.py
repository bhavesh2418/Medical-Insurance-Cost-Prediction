"""
data_loader.py — Load and validate the raw insurance dataset.
"""

import pandas as pd
from pathlib import Path
from src.config import RAW_DATA_FILE


def load_raw_data(filepath: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """Load raw insurance CSV and run basic validation."""
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at {filepath}\n"
            "Run: python scripts/download_data.py"
        )

    df = pd.read_csv(filepath)
    _validate(df)
    return df


def _validate(df: pd.DataFrame):
    expected_cols = {"age", "sex", "bmi", "children", "smoker", "region", "charges"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    assert len(df) > 0, "Dataset is empty"
    print(f"Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Missing values:\n{df.isnull().sum()}\n")
