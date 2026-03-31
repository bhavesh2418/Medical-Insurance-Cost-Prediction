"""
main.py — Full pipeline: Load → Preprocess → Train → Evaluate
Run: python main.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader   import load_raw_data
from src.preprocessing import preprocess
from src.model         import train_all
from src.evaluate      import plot_model_comparison, plot_actual_vs_predicted, plot_feature_importance


def main():
    print("=" * 55)
    print("  Medical Insurance Cost Prediction Pipeline")
    print("=" * 55)

    # Step 1: Load
    print("\n[1/4] Loading data...")
    df_raw = load_raw_data()

    # Step 2: Preprocess
    print("\n[2/4] Preprocessing & feature engineering...")
    df_clean = preprocess(df_raw, save=True)
    print(f"Final shape: {df_clean.shape}")

    # Step 3: Train
    print("\n[3/4] Training models...")
    results = train_all(df_clean)
    print("\n── Model Results ──────────────────────────────────")
    print(results.to_string(index=False))

    # Step 4: Evaluate
    print("\n[4/4] Generating evaluation plots...")
    plot_model_comparison(results)
    plot_actual_vs_predicted(df_clean)
    plot_feature_importance(df_clean)

    print("\nPipeline complete.")
    print(f"Plots saved in: reports/figures/")
    print(f"Models saved in: models/")


if __name__ == "__main__":
    main()
