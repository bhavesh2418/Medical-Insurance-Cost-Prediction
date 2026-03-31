# Medical Insurance Cost Prediction

A complete end-to-end Machine Learning project to predict individual medical insurance charges based on demographic and health factors.

---

## Problem Statement

Healthcare costs vary significantly across individuals. This project builds predictive models to estimate insurance charges using features like age, BMI, smoking status, and region — helping insurers and individuals understand cost drivers.

---

## Dataset

**Source:** [Kaggle — Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)

| Feature    | Type        | Description                          |
|------------|-------------|--------------------------------------|
| age        | Numerical   | Age of the primary beneficiary       |
| sex        | Categorical | Biological sex (male/female)         |
| bmi        | Numerical   | Body Mass Index                      |
| children   | Numerical   | Number of dependents                 |
| smoker     | Categorical | Smoker status (yes/no)               |
| region     | Categorical | US residential area (4 regions)      |
| charges    | Numerical   | **Target** — Medical insurance cost  |

---

## Project Structure

```
Medical-Insurance-Cost-Prediction/
│
├── data/
│   ├── raw/                    # Original dataset (not tracked by git)
│   └── processed/              # Cleaned & encoded data
│
├── notebooks/
│   ├── 01_EDA.ipynb            # Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Model_Training.ipynb
│   └── 04_Model_Evaluation.ipynb
│
├── src/
│   ├── config.py               # Paths, constants, model params
│   ├── data_loader.py          # Load & validate raw data
│   ├── preprocessing.py        # Cleaning + feature engineering
│   ├── model.py                # Train, save, load, predict
│   └── evaluate.py             # Plots & evaluation metrics
│
├── models/                     # Saved trained models (.pkl)
├── reports/
│   └── figures/                # Auto-generated plots
├── scripts/
│   ├── download_data.py        # Kaggle API download script
│   └── github_push.py          # Git commit & push helper
│
├── main.py                     # Full pipeline runner
├── requirements.txt
├── .env.example                # Template for credentials
└── .gitignore
```

---

## Setup & Usage

### 1. Clone & install dependencies
```bash
git clone https://github.com/<your-username>/Medical-Insurance-Cost-Prediction.git
cd Medical-Insurance-Cost-Prediction
pip install -r requirements.txt
```

### 2. Configure credentials
```bash
cp .env.example .env
# Fill in your Kaggle and GitHub credentials in .env
```

### 3. Download dataset
```bash
python scripts/download_data.py
```

### 4. Run the full pipeline
```bash
python main.py
```

---

## Models Evaluated

| Model               | R² Score | RMSE     | Notes                    |
|---------------------|----------|----------|--------------------------|
| Linear Regression   | ~0.75    | ~6,056   | Simple baseline          |
| Ridge Regression    | ~0.86    | ~4,900   | Best R²                  |
| Lasso Regression    | ~0.75    | ~6,050   | Feature selection        |
| Random Forest       | ~0.87    | ~4,841   | Best RMSE                |
| XGBoost             | ~0.88    | ~4,700   | Best overall             |

---

## Key Insights

- **Smoking status** is the single most influential factor — smokers pay 3–4x more
- **BMI × Smoker** interaction creates the highest-cost segment
- **Age** has a strong positive correlation with charges
- Region has minimal impact compared to lifestyle factors

---

## Tech Stack

- **Python 3.10+**
- **Pandas, NumPy** — Data manipulation
- **Scikit-learn** — ML models & metrics
- **XGBoost** — Gradient boosting
- **Matplotlib, Seaborn** — Visualization
- **Joblib** — Model persistence
- **Jupyter Notebook** — Analysis

---

## Author

Built as a portfolio project demonstrating a production-style ML workflow.
