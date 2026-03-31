"""
config.py — Central configuration for all paths, constants, and model parameters.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR     = ROOT / "models"
REPORTS_DIR    = ROOT / "reports"
FIGURES_DIR    = ROOT / "reports" / "figures"

# Ensure directories exist
for d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
RAW_DATA_FILE  = DATA_RAW / "insurance.csv"
CLEAN_DATA_FILE = DATA_PROCESSED / "insurance_clean.csv"

# ── Target & Features ─────────────────────────────────────────────────────────
TARGET_COLUMN = "charges"

CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
NUMERICAL_FEATURES   = ["age", "bmi", "children"]
ALL_FEATURES         = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# ── Model Parameters ──────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.2

MODEL_PARAMS = {
    "linear_regression": {},
    "ridge": {"alpha": 1.0},
    "lasso": {"alpha": 1.0},
    "random_forest": {
        "n_estimators": 100,
        "random_state": RANDOM_STATE
    },
    "xgboost": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "random_state": RANDOM_STATE
    },
}
