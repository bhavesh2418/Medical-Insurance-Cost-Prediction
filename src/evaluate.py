"""
evaluate.py — Generate and save evaluation plots for all models.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from src.config import TARGET_COLUMN, TEST_SIZE, RANDOM_STATE, FIGURES_DIR
from src.model import MODELS, get_feature_columns

sns.set_theme(style="whitegrid", palette="muted")


def plot_model_comparison(results_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ["R2_Score", "RMSE", "MAE"]
    colors  = ["steelblue", "tomato", "seagreen"]

    for ax, metric, color in zip(axes, metrics, colors):
        data = results_df.sort_values(metric, ascending=(metric != "R2_Score"))
        ax.barh(data["Model"], data[metric], color=color)
        ax.set_title(f"Model Comparison — {metric}")
        ax.set_xlabel(metric)

    plt.tight_layout()
    path = FIGURES_DIR / "model_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_actual_vs_predicted(df: pd.DataFrame, model_name: str = "Random Forest"):
    from src.model import load_model

    X = df[get_feature_columns(df)]
    y = df[TARGET_COLUMN]
    _, X_test, _, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    model  = load_model(model_name)
    y_pred = model.predict(X_test)
    r2     = r2_score(y_test, y_pred)

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color="steelblue")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Charges ($)")
    plt.ylabel("Predicted Charges ($)")
    plt.title(f"{model_name} — Actual vs Predicted (R²={r2:.4f})")
    plt.tight_layout()
    path = FIGURES_DIR / "actual_vs_predicted.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_feature_importance(df: pd.DataFrame):
    from src.model import load_model

    model    = load_model("Random Forest")
    features = get_feature_columns(df)
    importance = pd.Series(model.feature_importances_, index=features).sort_values()

    plt.figure(figsize=(8, 6))
    importance.plot(kind="barh", color="steelblue")
    plt.title("Feature Importance — Random Forest")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    path = FIGURES_DIR / "feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
