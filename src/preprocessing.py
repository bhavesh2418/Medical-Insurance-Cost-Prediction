"""
preprocessing.py — Clean raw data and encode categorical features.
"""

import pandas as pd
from src.config import CLEAN_DATA_FILE


def preprocess(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Clean data, encode categoricals, add engineered features."""
    df = df.copy()

    # ── Drop duplicates ───────────────────────────────────────────────────────
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Dropped {before - len(df)} duplicate rows")

    # ── Standardize column names ──────────────────────────────────────────────
    df.columns = df.columns.str.lower().str.strip()

    # ── Encode binary columns ─────────────────────────────────────────────────
    df["sex"]    = df["sex"].map({"male": 1, "female": 0})
    df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})

    # ── One-hot encode region ─────────────────────────────────────────────────
    df = pd.get_dummies(df, columns=["region"], drop_first=True)

    # ── Feature Engineering ───────────────────────────────────────────────────
    # BMI category (WHO classification)
    df["bmi_category"] = pd.cut(
        df["bmi"],
        bins=[0, 18.5, 24.9, 29.9, float("inf")],
        labels=["Underweight", "Normal", "Overweight", "Obese"]
    ).astype(str)
    df["is_obese"] = (df["bmi"] >= 30).astype(int)

    # Age groups
    df["age_group"] = pd.cut(
        df["age"],
        bins=[17, 30, 45, 60, float("inf")],
        labels=["Young", "Adult", "Middle-Aged", "Senior"]
    ).astype(str)

    # Smoker + Obese interaction (key driver of high charges)
    df["smoker_obese"] = df["smoker"] * df["is_obese"]

    if save:
        df.to_csv(CLEAN_DATA_FILE, index=False)
        print(f"Cleaned data saved to: {CLEAN_DATA_FILE}")

    return df
