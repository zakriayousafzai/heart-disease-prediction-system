# Model Development Guide

This directory contains the scripts, datasets, visuals, and model artifacts used to build and evaluate the heart disease prediction models.

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Getting Started](#getting-started)
- [Pipeline Workflow](#pipeline-workflow)
- [Scripts Documentation](#scripts-documentation)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Data Visualization](#2-data-visualization)
  - [3. ML Model Training](#3-ml-model-training)
  - [4. ANN Model Training](#4-ann-model-training)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Dataset Reference](#dataset-reference)
- [Model Outputs](#model-outputs)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

The model development module implements a complete ML pipeline for heart disease prediction:

1. Data preprocessing from raw CSV to cleaned numeric data
2. Exploratory visualization for feature understanding
3. Traditional ML training (Logistic Regression and Random Forest)
4. ANN training for multi-class risk prediction
5. Optional local dashboard testing with Streamlit

---

## Directory Structure

```text
model_development/
|-- README.md
|-- dashboard.py
|-- requirements.txt
|
|-- Datasets/
|   |-- heart.csv
|   `-- heart_cleaned.csv
|
|-- Scripts/
|   |-- preprocessing.py
|   |-- visualization.py
|   |-- ml-model-training.py
|   `-- ann-model-training.py
|
|-- Models/
|   |-- ann_model.pth
|   |-- scaler.pkl
|   |-- ann_accuracy.pkl
|   |-- log_reg_model.pkl
|   |-- lr_accuracy.pkl
|   |-- rfc_model.pkl
|   |-- rf_accuracy.pkl
|   `-- ann_shap_samples.pkl
|
`-- Visuals/
    |-- correlations-with-heart-disease.png
    |-- heartDisease-distribution.png
    |-- histogram-*.png
    |-- sunburstChart-*.png
    `-- violinPlot-*.png
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

1. Navigate to this directory:

```bash
cd model_development
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Key Dependencies

| Package | Version |
|---|---|
| pandas | 3.0.0 |
| numpy | 2.4.1 |
| scikit-learn | 1.8.0 |
| shap | 0.46.0 |
| torch | 2.10.0 |
| plotly | 6.5.0 |
| streamlit | 1.52.2 |

Note: `requirements.txt` currently also includes some backend packages (Flask, SQLAlchemy, CORS, dotenv, Gemini, psycopg2). They are not required for training scripts themselves but are present in this environment file.

---

## Pipeline Workflow

Run scripts in this order:

```text
Step 1: Scripts/preprocessing.py
Input:  Datasets/heart.csv
Output: Datasets/heart_cleaned.csv

Step 2 (optional): Scripts/visualization.py
Input:  Datasets/heart_cleaned.csv
Output: Visuals/*.png

Step 3A: Scripts/ml-model-training.py
Output: Models/log_reg_model.pkl, Models/rfc_model.pkl,
        Models/lr_accuracy.pkl, Models/rf_accuracy.pkl

Step 3B: Scripts/ann-model-training.py
Output: Models/ann_model.pth, Models/scaler.pkl, Models/ann_accuracy.pkl,
        Models/ann_shap_samples.pkl
        (and additional SHAP artifacts, see below)

Step 4 (optional): dashboard.py
Command: streamlit run dashboard.py
```

### Quick Start Commands

```bash
cd Scripts
python preprocessing.py
python visualization.py
python ml-model-training.py
python ann-model-training.py

cd ..
streamlit run dashboard.py
```

---

## Scripts Documentation

### 1. Data Preprocessing

**File:** `Scripts/preprocessing.py`

**Purpose:** Clean and encode raw data into model-ready numeric form.

What it does:

1. Basic EDA checks (`info`, `describe`, duplicates, nulls)
2. Encodes categorical columns to integer codes
3. Replaces biologically implausible zeros (`Cholesterol`, `RestingBP`) with NaN
4. Imputes missing values with `KNNImputer(n_neighbors=3)`
5. Casts all columns except `Oldpeak` to `int32`
6. Saves `Datasets/heart_cleaned.csv`

Input/Output:

| Type | File |
|---|---|
| Input | `../Datasets/heart.csv` |
| Output | `../Datasets/heart_cleaned.csv` |

---

### 2. Data Visualization

**File:** `Scripts/visualization.py`

**Purpose:** Generate exploratory plots from cleaned data.

Produces:

- Correlation line plot with `HeartDisease`
- Heart disease distribution pie chart
- Histograms for key categorical/numeric features
- Sunburst charts for hierarchical feature views
- Violin plots for `MaxHR` and `Oldpeak`

Input:

| Type | File |
|---|---|
| Input | `../Datasets/heart_cleaned.csv` |

Note: This script builds Plotly figures directly. If you want image files consistently, run it in an environment/workflow that explicitly saves figures.

---

### 3. ML Model Training

**File:** `Scripts/ml-model-training.py`

**Purpose:** Train binary classifiers for `HeartDisease`.

Models:

- Logistic Regression
  - Tests solvers: `lbfgs`, `liblinear`, `newton-cg`, `newton-cholesky`, `sag`, `saga`
  - Picks best test-set solver, retrains final model
- Random Forest
  - `GridSearchCV` over:
    - `n_estimators`: `[50, 100, 150, 500]`
    - `max_depth`: `[3, 6, 9, 19]`
    - `max_features`: `['sqrt', 'log2', None]`
    - `max_leaf_nodes`: `[3, 6, 9]`

Data split:

- `train_test_split(test_size=0.2, random_state=42, stratify=HeartDisease)`

Saved outputs:

- `../Models/log_reg_model.pkl`
- `../Models/rfc_model.pkl`
- `../Models/lr_accuracy.pkl`
- `../Models/rf_accuracy.pkl`

---

### 4. ANN Model Training

**File:** `Scripts/ann-model-training.py`

**Purpose:** Train a 3-class ANN (Low/Medium/High risk) and SHAP artifacts.

Risk labeling:

- Low (0): `HeartDisease == 0`
- High (2): `HeartDisease == 1` and (`Oldpeak > 2.0` or `ST_Slope == 2` or `MaxHR < 120`)
- Medium (1): all other disease-positive cases

Architecture:

- Input: 11 features
- `Linear(11,64)` + `BatchNorm1d` + `ReLU`
- `Dropout(0.3)`
- `Linear(64,32)` + `BatchNorm1d` + `ReLU`
- `Linear(32,3)`

Training setup:

- Optimizer: Adam (`lr=0.001`)
- Loss: `CrossEntropyLoss` with class weights (`compute_class_weight`)
- 500 epochs, keep best weights by test accuracy

Scaling:

- `StandardScaler` fit on training split, reused for test/inference

Saved outputs (script paths):

- `../Models/ann_model.pth`
- `../Models/scaler.pkl`
- `../Models/ann_accuracy.pkl`
- `../Models/ann_shap_values.pkl`
- `../Models/ann_shap_samples.pkl`
- `../Models/ann_shap_feature_names.pkl`
- `../Models/ann_shap_importance.csv`

---

## Streamlit Dashboard

**File:** `dashboard.py`

**Purpose:** Local UI for testing model predictions and viewing stored accuracies.

Tabs:

1. Predict
   - Collects patient inputs
   - Runs Logistic Regression + Random Forest + ANN
   - Displays ANN class probabilities
2. Model Information
   - Displays LR/RF/ANN accuracy bar chart

Run:

```bash
cd model_development
streamlit run dashboard.py
```

Access: `http://localhost:8501`

Model path note:

- `dashboard.py` and all training scripts now use `model_development/Models/`.
- Retrain models once after this change so all artifacts exist in `Models/`.

---

## Dataset Reference

### Raw Dataset

- File: `Datasets/heart.csv`
- Source: Kaggle Heart Failure Prediction Dataset
- Records: 918

### Features

`Age`, `Sex`, `ChestPainType`, `RestingBP`, `Cholesterol`, `FastingBS`, `RestingECG`, `MaxHR`, `ExerciseAngina`, `Oldpeak`, `ST_Slope`, `HeartDisease`

### Cleaned Dataset

- File: `Datasets/heart_cleaned.csv`
- Categorical features encoded to numeric
- Missing values for zero cholesterol/resting BP imputed
- Integer dtypes restored except continuous `Oldpeak`

---

## Model Outputs

Model artifacts are written to a single location:

1. `Models/` (inside `model_development`) for both ML and ANN outputs

Typical artifacts:

- `ann_model.pth`
- `scaler.pkl`
- `ann_accuracy.pkl`
- `log_reg_model.pkl`
- `rfc_model.pkl`
- `lr_accuracy.pkl`
- `rf_accuracy.pkl`
- `ann_shap_samples.pkl`

---

## Best Practices

- Run preprocessing before any training.
- Use consistent feature encoding between training and inference.
- Keep scaler and ANN weights together.
- Keep all model artifacts in `Models/`.
- Recompute SHAP artifacts after ANN retraining.

---

## Troubleshooting

### 1. Module import errors

```bash
pip install -r requirements.txt
```

### 2. Dataset not found

Run scripts from `model_development/Scripts` so relative paths resolve.

### 3. Torch installation issues

Install CPU or CUDA build from official PyTorch index if needed.

### 4. Streamlit port in use

```bash
streamlit run dashboard.py --server.port 8502
```

### 5. Model file path mismatch

If files are missing in `Models/`, rerun the training scripts from `model_development/Scripts`.

---

Happy model development.
