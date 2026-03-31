# Medical Insurance Cost Prediction

An end-to-end Machine Learning project to predict individual medical insurance charges based on demographic and health factors. Follows a structured, production-style workflow: EDA → Feature Engineering → Model Training → Evaluation.

---

## Problem Statement

Healthcare costs vary significantly across individuals. This project builds predictive regression models to estimate insurance charges using features like age, BMI, smoking status, and region — helping insurers and individuals understand the key cost drivers.

---

## Dataset

**Source:** [Kaggle — Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)  
**Size:** 1,338 records | 7 features | No missing values

| Feature    | Type        | Description                          |
|------------|-------------|--------------------------------------|
| age        | Numerical   | Age of the primary beneficiary       |
| sex        | Categorical | Biological sex (male / female)       |
| bmi        | Numerical   | Body Mass Index                      |
| children   | Numerical   | Number of dependents                 |
| smoker     | Categorical | Smoker status (yes / no)             |
| region     | Categorical | US residential area (4 regions)      |
| charges    | Numerical   | **Target** — Medical insurance cost ($) |

---

## Project Structure

```
Medical-Insurance-Cost-Prediction/
│
├── data/
│   ├── raw/                         # Original dataset (gitignored)
│   └── processed/                   # Cleaned & encoded data (gitignored)
│
├── notebooks/
│   ├── 01_EDA.ipynb                 # Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb # Encoding + new feature creation
│   └── 03_Model_Training.ipynb      # Train, compare & evaluate 5 models
│
├── src/
│   ├── config.py                    # Paths, constants, model parameters
│   ├── data_loader.py               # Load & validate raw data
│   ├── preprocessing.py             # Cleaning + feature engineering
│   ├── model.py                     # Train, save, load, predict
│   └── evaluate.py                  # Plots & evaluation metrics
│
├── models/                          # Saved trained models (.pkl, gitignored)
├── reports/
│   ├── model_results.csv            # Final model performance table
│   └── figures/                     # Auto-generated plots
│
├── scripts/
│   ├── download_data.py             # Kaggle API download script
│   └── github_push.py               # Git commit & push helper
│
├── main.py                          # Full pipeline runner
├── requirements.txt
├── .env.example                     # Credentials template
└── .gitignore
```

---

## Workflow

```
Raw Data  →  01_EDA.ipynb  →  02_Feature_Engineering.ipynb  →  03_Model_Training.ipynb  →  main.py (pipeline)
```

Each notebook is self-contained and tells the story of one phase. `main.py` runs the full pipeline end-to-end in one command.

---

## Setup & Usage

### 1. Clone & install
```bash
git clone https://github.com/bhavesh2418/Medical-Insurance-Cost-Prediction.git
cd Medical-Insurance-Cost-Prediction
pip install -r requirements.txt
```

### 2. Configure credentials
```bash
cp .env.example .env
# Fill in KAGGLE_USERNAME, KAGGLE_KEY, GITHUB_TOKEN, etc.
```

### 3. Download dataset
```bash
python scripts/download_data.py
```

### 4. Run full pipeline
```bash
python main.py
```

---

## Exploratory Data Analysis

Key findings from `notebooks/01_EDA.ipynb`:

- **Smoking status** is the dominant cost driver — smokers pay **~$32,000** vs **~$8,400** for non-smokers
- **Age** has a strong positive linear relationship with charges
- **BMI** shows a non-linear effect — charges spike sharply above BMI 30 (obese) for smokers
- **Region and gender** have minimal impact on charges
- Charges are **right-skewed** — a small group of high-risk individuals drives the upper tail

---

## Feature Engineering

New features created in `notebooks/02_Feature_Engineering.ipynb`:

| Feature | Description | Rationale |
|---|---|---|
| `is_obese` | BMI >= 30 (binary) | Captures clinical obesity threshold |
| `bmi_category` | WHO classification | Underweight / Normal / Overweight / Obese |
| `age_group` | Life stage buckets | Young / Adult / Middle-Aged / Senior |
| `smoker_obese` | smoker × is_obese | Highest-cost interaction identified in EDA |

---

## Model Results

Results from `notebooks/03_Model_Training.ipynb` and `reports/model_results.csv`:

| Rank | Model | R² Score | RMSE | MAE | CV R² |
|------|-------|----------|------|-----|-------|
| 1 | **Linear Regression** | **0.9066** | $4,142 | $2,369 | 0.8596 |
| 2 | Lasso Regression | 0.9065 | $4,144 | $2,369 | 0.8596 |
| 3 | Ridge Regression | 0.9062 | $4,151 | $2,376 | 0.8596 |
| 4 | XGBoost | 0.8911 | $4,474 | $2,531 | 0.8385 |
| 5 | Random Forest | 0.8799 | $4,698 | $2,632 | 0.8364 |

**Best Model: Linear Regression** — highest R² (0.9066), lowest RMSE ($4,142), simplest and most interpretable.

> All models achieve R² > 0.88, confirming that the selected features explain most of the variance in insurance charges.

---

## Key Insights

1. **Smoker status alone** explains the majority of variance in charges
2. The **smoker × obese** combination produces the highest-cost segment (~$40,000+ avg)
3. Linear relationships dominate — explaining why Linear Regression outperforms complex ensembles
4. **Feature engineering** (smoker_obese interaction) improved model explainability and signal strength

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| Pandas, NumPy | Data manipulation |
| Scikit-learn | ML models & metrics |
| XGBoost | Gradient boosting |
| Matplotlib, Seaborn | Visualization |
| Joblib | Model persistence |
| Jupyter Notebook | Interactive analysis |
| Kaggle API | Dataset download |

---

## Author

**Bhavesh** — Portfolio project demonstrating a production-style, end-to-end ML workflow.
