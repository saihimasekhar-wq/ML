# Heavy Vehicle Failure Prediction Through Intelligent Sensor Data

A machine-learning project that predicts **Air Pressure System (APS) related failures** in Scania heavy trucks from anonymized sensor and operational data.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/status-completed-brightgreen)

> Course project for **Machine Learning** (B.Tech, Department of Computer Science and Engineering, KL University), A.Y. 2026-27.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models](#models)
- [Results](#results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Key Takeaways](#key-takeaways)
- [Future Work](#future-work)
- [Team](#team)
- [Acknowledgements](#acknowledgements)

---

## Overview

Unexpected component failures in heavy vehicles cause downtime, maintenance delays, higher operating costs and safety risks. The Air Pressure System supports critical functions such as braking and gear changes, so detecting APS problems early is valuable.

This project builds a leakage-safe ML pipeline that:

- handles heavy **missing data** (169 of 170 features have missing values),
- deals with an extreme **class imbalance** (1.81% positive class),
- compares several classifiers using **imbalance-aware metrics**, and
- selects a final model that **prioritizes recall**, because missing a real failure costs more than an unnecessary inspection.

## Problem Statement

Binary classification task. Given anonymized sensor measurements from a heavy vehicle, predict whether the failure is related to the APS component.

| Class | Meaning |
|-------|---------|
| `0` (neg) | Failure not related to APS |
| `1` (pos) | APS-related failure |

## Dataset

**APS Failure at Scania Trucks** dataset ([UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks)).

| Property | Value |
|----------|-------|
| Records | 76,000 |
| Total columns | 171 (170 anonymized sensor features + target `class`) |
| Negative class | 74,625 (98.19%) |
| Positive class | 1,375 (1.81%) |
| Duplicate rows | 0 |
| Columns with missing values | 169 |
| Missing cells | 1,078,695 |

Sensor names are anonymized, so their physical meaning is not available.

## Methodology

```
Dataset → Data Cleaning → Missing-Value Treatment → Encoding → Train/Test Split
        → Model Training → Hyperparameter Tuning → Prediction → Evaluation
```

**Preprocessing**

- Converted the `na` string into proper missing values
- Checked for duplicate rows (none found)
- Median imputation, **learned on training data only**
- Outliers retained, since extreme values may reflect genuine abnormal operating conditions
- Target encoded as `neg` = 0, `pos` = 1
- `StandardScaler` for Logistic Regression only (tree models do not need scaling)
- **Scikit-learn Pipelines** used throughout to prevent data leakage

**Exploratory Data Analysis**

Descriptive statistics, class distribution, missingness analysis, representative sensor distributions, outlier inspection and correlation analysis.

**Feature selection**

No artificial features were created. All 170 sensor features were retained, to avoid discarding low-variance sensors that may still signal abnormal behaviour.

**Experimental setup**

| Component | Configuration |
|-----------|---------------|
| Environment | Jupyter Notebook in Visual Studio Code |
| OS | Windows |
| Python | 3.14.6 |
| Libraries | Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Joblib |
| Split | 80% train / 20% test, stratified |
| Random state | 42 |

## Models

| Model | Purpose | Configuration |
|-------|---------|---------------|
| Logistic Regression | Linear baseline | Balanced class weight, median imputation, standard scaling |
| Decision Tree | Interpretable nonlinear model | Balanced class weight, max depth 12, min leaf size 5 |
| Random Forest | Ensemble nonlinear model | 200 trees, balanced subsampling, parallel training |
| **Tuned Random Forest** | Final model | `RandomizedSearchCV`, 3-fold stratified CV, **recall** as primary metric |

Tuned parameters: `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`.

## Results

Evaluated on the held-out test set (15,200 rows).

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|-------|---------:|----------:|-------:|---------:|--------:|-------:|
| Random Forest | 99.18% | 93.14% | 59.27% | 72.44% | 99.10% | 88.39% |
| **Tuned Random Forest** | 97.47% | 41.14% | **92.00%** | 56.85% | 99.00% | 76.06% |
| Logistic Regression | 97.41% | 40.48% | **92.00%** | 56.22% | 97.51% | 78.22% |
| Decision Tree | 97.24% | 38.47% | 88.00% | 53.54% | 88.73% | 47.23% |

**Confusion matrix: Tuned Random Forest**

|  | Predicted: No Failure | Predicted: Failure |
|--|:--:|:--:|
| **Actual: No Failure** | 14,563 (TN) | 362 (FP) |
| **Actual: Failure** | 22 (FN) | 253 (TP) |

The tuned model detected 253 of 275 actual APS failures. The 362 false positives mean some healthy vehicles get flagged for inspection, which is an acceptable trade-off in this setting.

## Project Structure

> Adjust this section to match your actual repository.

```
├── data/                  # APS Failure at Scania Trucks dataset files
├── notebooks/             # Jupyter notebook(s): EDA, training, evaluation
├── models/                # Saved model (Joblib)
├── figures/               # Plots used in the report
├── report/                # Project report (PDF)
├── requirements.txt
└── README.md
```

## Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

**2. Install dependencies**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

**3. Add the dataset**

Download the APS Failure at Scania Trucks dataset from the link above and place the files in the `data/` folder.

**4. Run the notebook**

```bash
jupyter notebook
```

Open the notebook in `notebooks/` and run all cells.

**5. Load the saved model for prediction** *(optional)*

```python
import joblib
import pandas as pd

model = joblib.load("models/tuned_random_forest.joblib")
X_new = pd.read_csv("new_sensor_data.csv", na_values="na")

pred = model.predict(X_new)               # 0 = non-APS failure, 1 = APS failure
prob = model.predict_proba(X_new)[:, 1]   # failure probability
```

## Key Takeaways

- Accuracy alone is misleading on this dataset: a model that always predicts "negative" would already score about 98%.
- Class weighting and recall-focused tuning raised recall from **59% to 92%**.
- Higher recall comes at the cost of precision, so the model suits **inspection prioritization**, not fully automated decisions.
- The system should support, not replace, professional maintenance judgement.

## Future Work

- Explore cost-sensitive learning and gradient-boosting models (XGBoost, LightGBM)
- Tune the decision threshold to balance precision and recall
- Add explainability (for example SHAP) to identify influential sensors
- Validate on other fleets and datasets, such as the SCANIA Component X dataset

## Team

| Name |
|------|
| Cherukumalli Sai Hima Sekhar |
| Akhil Chowdary |

**Guide:** Nirmalajyothi Narisetty, Associate Professor, Department of CSE, KL University

## Acknowledgements

- Scania CV AB for the APS Failure dataset
- Scikit-learn documentation
- Literature surveyed in the project report, including studies on cost-sensitive predictive maintenance, Scania Component X data and class-imbalance handling in automotive data

---

*This project was developed for academic purposes.*
