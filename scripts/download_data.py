"""
Script: download_data.py
Purpose: Download the Medical Insurance dataset from Kaggle
Dataset: https://www.kaggle.com/datasets/mirichoi0218/insurance
"""

import os
import sys
import zipfile
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"


def download_dataset():
    """Download insurance dataset from Kaggle using API credentials."""

    # Set Kaggle credentials from .env
    kaggle_username = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")

    if not kaggle_username or not kaggle_key:
        print("ERROR: KAGGLE_USERNAME and KAGGLE_KEY must be set in .env file.")
        print("Get your API key from: https://www.kaggle.com/settings > API > Create New Token")
        sys.exit(1)

    os.environ["KAGGLE_USERNAME"] = kaggle_username
    os.environ["KAGGLE_KEY"] = kaggle_key

    # Create data/raw directory
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset from Kaggle...")

    import kaggle
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        "mirichoi0218/insurance",
        path=str(DATA_RAW),
        unzip=True
    )

    # Verify
    csv_file = DATA_RAW / "insurance.csv"
    if csv_file.exists():
        print(f"Dataset downloaded successfully: {csv_file}")
        print(f"File size: {csv_file.stat().st_size / 1024:.1f} KB")
    else:
        print("Download may have completed but insurance.csv not found. Check data/raw/")


if __name__ == "__main__":
    download_dataset()
