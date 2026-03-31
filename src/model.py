"""
model.py — Train, evaluate, and persist all models.
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from src.config import (
    TARGET_COLUMN, TEST_SIZE, RANDOM_STATE,
    MODEL_PARAMS, MODELS_DIR
)


MODELS = {
    "Linear Regression":  LinearRegression(**MODEL_PARAMS["linear_regression"]),
    "Ridge Regression":   Ridge(**MODEL_PARAMS["ridge"]),
    "Lasso Regression":   Lasso(**MODEL_PARAMS["lasso"]),
    "Random Forest":      RandomForestRegressor(**MODEL_PARAMS["random_forest"]),
    "XGBoost":            XGBRegressor(**MODEL_PARAMS["xgboost"], verbosity=0),
}


def get_feature_columns(df: pd.DataFrame) -> list:
    drop_cols = [TARGET_COLUMN, "bmi_category", "age_group"]
    return [c for c in df.columns if c not in drop_cols]


def train_all(df: pd.DataFrame) -> pd.DataFrame:
    """Train all models and return a comparison DataFrame."""
    X = df[get_feature_columns(df)]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    results = []
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)
        cv   = cross_val_score(model, X, y, cv=5, scoring="r2").mean()

        results.append({
            "Model":     name,
            "RMSE":      round(rmse, 2),
            "MAE":       round(mae, 2),
            "R2_Score":  round(r2, 4),
            "CV_R2_Mean": round(cv, 4),
        })

        # Save model
        joblib.dump(model, MODELS_DIR / f"{name.lower().replace(' ', '_')}.pkl")
        print(f"{name:22s} | R2={r2:.4f} | RMSE={rmse:,.2f} | MAE={mae:,.2f}")

    results_df = pd.DataFrame(results).sort_values("R2_Score", ascending=False)
    return results_df


def load_model(model_name: str):
    path = MODELS_DIR / f"{model_name.lower().replace(' ', '_')}.pkl"
    return joblib.load(path)


def predict(model_name: str, input_df: pd.DataFrame) -> np.ndarray:
    model = load_model(model_name)
    return model.predict(input_df)
