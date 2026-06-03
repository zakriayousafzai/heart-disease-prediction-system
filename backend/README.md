# Model Development Guide

This directory contains all the scripts, datasets, and resources for developing, training, and evaluating the heart disease prediction machine learning models.

## 📋 Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Getting Started](#getting-started)
- [Pipeline Workflow](#pipeline-workflow)
- [Scripts Documentation](#scripts-documentation)
  - [Data Preprocessing](#1-data-preprocessing)
  - [Data Visualization](#2-data-visualization)
  - [ML Model Training](#3-ml-model-training)
  - [ANN Model Training](#4-ann-model-training)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Dataset Reference](#dataset-reference)
- [Model Outputs](#model-outputs)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

The model development module implements a complete machine learning pipeline for heart disease risk prediction:

1. **Data Preprocessing** - Clean and prepare raw data for model training
2. **Exploratory Data Analysis** - Visualize patterns and correlations
3. **Traditional ML Training** - Train Logistic Regression and Random Forest models
4. **Deep Learning Training** - Train an Artificial Neural Network for multi-class risk prediction
5. **Interactive Dashboard** - Test predictions with a Streamlit web interface

---

## Directory Structure

```
model_development/
├── README.md                    # This documentation file
├── dashboard.py                 # Streamlit interactive prediction app
├── requirements.txt             # Python dependencies
│
├── Datasets/
│   ├── heart.csv                # Original raw dataset (918 records)
│   └── heart_cleaned.csv        # Preprocessed dataset (output of preprocessing.py)
│
├── Scripts/
│   ├── preprocessing.py         # Data cleaning and preparation
│   ├── visualization.py         # EDA and data visualization
│   ├── ml-model-training.py     # Logistic Regression & Random Forest training
│   └── ann-model-training.py    # PyTorch Neural Network training
│
└── Visuals/                     # Generated visualization images
    ├── correlations-with-heart-disease.png
    ├── heartDisease-distribution.png
    ├── histogram-*.png          # Distribution histograms
    ├── sunburstChart-*.png      # Hierarchical sunburst charts
    └── violinPlot-*.png         # Violin distribution plots
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Installation

1. **Navigate to the model_development directory:**
   ```bash
   cd model_development
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate (Windows)
   venv\Scripts\activate
   
   # Activate (Linux/macOS)
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | 2.3.3 | Data manipulation |
| numpy | 2.4.0 | Numerical computing |
| scikit-learn | 1.8.0 | ML algorithms & preprocessing |
| plotly | 6.5.0 | Interactive visualizations |
| matplotlib | 3.10.8 | Static visualizations |
| seaborn | 0.13.2 | Statistical visualizations |
| streamlit | 1.52.2 | Dashboard web interface |
| torch | - | Neural network (installed separately) |

---

## Pipeline Workflow

Execute the scripts in this order for a complete model development cycle:

```
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL DEVELOPMENT PIPELINE                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Data Preprocessing                                      │
│  Script: Scripts/preprocessing.py                                │
│  Input:  Datasets/heart.csv                                      │
│  Output: Datasets/heart_cleaned.csv                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Exploratory Data Analysis (Optional)                    │
│  Script: Scripts/visualization.py                                │
│  Input:  Datasets/heart_cleaned.csv                              │
│  Output: Visuals/*.png                                           │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────────┐ ┌─────────────────────────────────┐
│  STEP 3A: ML Model Training │ │  STEP 3B: ANN Model Training    │
│  Script: ml-model-training  │ │  Script: ann-model-training.py  │
│  Output:                    │ │  Output:                        │
│  - log_reg_model.pkl        │ │  - ann_model.pth                │
│  - rfc_model.pkl            │ │  - scaler.pkl                   │
│  - lr_accuracy.pkl          │ │  - ann_accuracy.pkl             │
│  - rf_accuracy.pkl          │ │                                 │
└─────────────────────────────┘ └─────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Test with Dashboard                                     │
│  Script: dashboard.py                                            │
│  Command: streamlit run dashboard.py                             │
└─────────────────────────────────────────────────────────────────┘
```

### Quick Start Commands

```bash
# Step 1: Preprocess data
cd Scripts
python preprocessing.py

# Step 2: Generate visualizations (optional)
python visualization.py

# Step 3: Train models
python ml-model-training.py    # Train Logistic Regression & Random Forest
python ann-model-training.py   # Train Neural Network

# Step 4: Launch dashboard
cd ..
streamlit run dashboard.py
```

---

## Scripts Documentation

### 1. Data Preprocessing

**File:** [`Scripts/preprocessing.py`](Scripts/preprocessing.py)

**Purpose:** Clean and prepare the raw heart disease dataset for model training.

#### What it does:

1. **Exploratory Data Analysis**
   - Displays dataset structure and statistics
   - Checks for missing values and duplicates
   - Analyzes feature cardinality

2. **Categorical Encoding**
   - Converts string categorical variables to numeric codes:
   
   | Feature | Original Values | Encoded Values |
   |---------|-----------------|----------------|
   | Sex | M, F | 0, 1 |
   | ChestPainType | ATA, NAP, ASY, TA | 0, 1, 2, 3 |
   | RestingECG | Normal, ST, LVH | 0, 1, 2 |
   | ExerciseAngina | N, Y | 0, 1 |
   | ST_Slope | Up, Flat, Down | 0, 1, 2 |

3. **Missing Value Imputation**
   - Replaces biologically impossible values (0 for Cholesterol and RestingBP) with NaN
   - Uses **KNN Imputation** (k=3) to estimate missing values based on similar patients

4. **Data Type Optimization**
   - Converts discrete variables to `int32` for memory efficiency
   - Keeps `Oldpeak` as float (continuous variable)

#### Usage:

```bash
cd Scripts
python preprocessing.py
```

#### Input/Output:

| Type | File | Description |
|------|------|-------------|
| Input | `../Datasets/heart.csv` | Raw dataset with 918 records |
| Output | `../Datasets/heart_cleaned.csv` | Cleaned dataset ready for training |

#### Key Code Sections:

```python
# KNN Imputation for Cholesterol
heart_df['Cholesterol'].replace(0, np.nan, inplace=True)
imputer = KNNImputer(n_neighbors=3)
after_impute = imputer.fit_transform(heart_df)
```

---

### 2. Data Visualization

**File:** [`Scripts/visualization.py`](Scripts/visualization.py)

**Purpose:** Generate exploratory visualizations to understand feature relationships and patterns.

#### Visualizations Generated:

| Visualization Type | Features Analyzed | Output File |
|-------------------|-------------------|-------------|
| Pie Chart | HeartDisease distribution | `heartDisease-distribution.png` |
| Line Plot | Feature correlations | `correlations-with-heart-disease.png` |
| Histograms | Age, Sex, ChestPainType, FastingBS, ST_Slope, ExerciseAngina | `histogram-*.png` |
| Sunburst Charts | Age, MaxHR, RestingBP by HeartDisease | `sunburstChart-*.png` |
| Violin Plots | MaxHR, Oldpeak distributions | `violinPlot-*.png` |

#### Usage:

```bash
cd Scripts
python visualization.py
```

#### Key Insights from Visualizations:

1. **Correlation Analysis:**
   - **Positive predictors:** Oldpeak, ST_Slope, ExerciseAngina, ChestPainType
   - **Negative predictors:** MaxHR (lower heart rate = higher risk)

2. **Feature Importance:**
   - Asymptomatic chest pain (ASY) strongly correlates with heart disease
   - Downsloping ST_Slope indicates high disease probability
   - Exercise-induced angina is a strong positive predictor

---

### 3. ML Model Training

**File:** [`Scripts/ml-model-training.py`](Scripts/ml-model-training.py)

**Purpose:** Train traditional machine learning models for binary heart disease classification.

#### Models Trained:

##### Logistic Regression

A linear probabilistic classifier that uses the sigmoid function to estimate disease probability.

**Optimization Strategy:**
- Tests 6 different optimization solvers
- Selects the solver with highest test accuracy

```python
# Solvers tested
solver = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
```

##### Random Forest Classifier

An ensemble method combining multiple decision trees for robust predictions.

**Hyperparameter Tuning via GridSearchCV:**

| Parameter | Search Space | Description |
|-----------|--------------|-------------|
| `n_estimators` | [50, 100, 150, 500] | Number of trees |
| `max_depth` | [3, 6, 9, 19] | Maximum tree depth |
| `max_features` | ['sqrt', 'log2', None] | Features per split |
| `max_leaf_nodes` | [3, 6, 9] | Maximum leaf nodes |

#### Usage:

```bash
cd Scripts
python ml-model-training.py
```

#### Output Files:

| File | Location | Description |
|------|----------|-------------|
| `log_reg_model.pkl` | `../../saved_models/` | Trained Logistic Regression model |
| `rfc_model.pkl` | `../../saved_models/` | Trained Random Forest model |
| `lr_accuracy.pkl` | `../../saved_models/` | Logistic Regression accuracy (%) |
| `rf_accuracy.pkl` | `../../saved_models/` | Random Forest accuracy (%) |

#### Expected Console Output:

```
Best Logistic Regression Solver: liblinear
Logistic Regression Accuracy: 0.8587
Saved LR Accuracy: 85.87%
Random Forest Accuracy: 0.8804
Saved RF Accuracy: 88.04%
```

---

### 4. ANN Model Training

**File:** [`Scripts/ann-model-training.py`](Scripts/ann-model-training.py)

**Purpose:** Train a PyTorch neural network for multi-class heart disease risk prediction.

#### Risk Classification Logic:

The ANN performs **3-class classification** instead of binary:

| Risk Level | Code | Criteria |
|------------|------|----------|
| **Low Risk** | 0 | HeartDisease == 0 |
| **Medium Risk** | 1 | HeartDisease == 1 AND moderate indicators |
| **High Risk** | 2 | HeartDisease == 1 AND (Oldpeak > 2.0 OR ST_Slope == 2 OR MaxHR < 120) |

```python
def assign_risk(row):
    if row['HeartDisease'] == 0:
        return 0  # Low Risk
    elif (row['Oldpeak'] > 2.0) or (row['ST_Slope'] == 2) or (row['MaxHR'] < 120):
        return 2  # High Risk
    else:
        return 1  # Medium Risk
```

#### Neural Network Architecture:

```
┌─────────────────────────────────────┐
│  Input Layer (11 features)          │
└─────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Linear Layer (64 neurons)          │
│  BatchNorm1d(64)                    │
│  ReLU Activation                    │
└─────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Dropout (30%)                      │
└─────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Linear Layer (32 neurons)          │
│  BatchNorm1d(32)                    │
│  ReLU Activation                    │
└─────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Output Layer (3 classes)           │
│  Low Risk / Medium Risk / High Risk │
└─────────────────────────────────────┘
```

#### Training Configuration:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Optimizer | Adam | Adaptive learning rate optimization |
| Learning Rate | 0.001 | Step size for weight updates |
| Loss Function | CrossEntropyLoss | Multi-class classification loss |
| Class Weights | Computed | Balances imbalanced dataset |
| Epochs | 500 | Maximum training iterations |
| Dropout | 0.3 (30%) | Regularization to prevent overfitting |
| Early Stopping | Yes | Saves best model based on validation accuracy |

#### Feature Scaling:

```python
# StandardScaler normalizes features to mean=0, std=1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit only on training data
X_test = scaler.transform(X_test)        # Apply same transformation to test
```

**Important:** The scaler is saved and must be used in production to ensure consistent predictions.

#### Usage:

```bash
cd Scripts
python ann-model-training.py
```

#### Output Files:

| File | Location | Description |
|------|----------|-------------|
| `ann_model.pth` | `../../saved_models/` | PyTorch model weights |
| `scaler.pkl` | `../../saved_models/` | StandardScaler for feature normalization |
| `ann_accuracy.pkl` | `../../saved_models/` | Best validation accuracy (%) |

#### Expected Console Output:

```
Best ANN Accuracy Achieved: 89.13%
```

---

## Streamlit Dashboard

**File:** [`dashboard.py`](dashboard.py)

**Purpose:** Interactive web application for testing model predictions.

### Features:

1. **Predict Tab:**
   - Input patient clinical data via form fields
   - Get predictions from all three models
   - Visualize risk probability distribution

2. **Model Information Tab:**
   - Compare model accuracies with bar chart
   - View performance metrics

### Running the Dashboard:

```bash
cd model_development
streamlit run dashboard.py
```

Access at: `http://localhost:8501`

### Input Fields:

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| Age | Number | 1-120 | Patient age in years |
| Sex | Select | Male/Female | Biological sex |
| Chest Pain Type | Select | 4 options | Type of chest pain experienced |
| Resting BP | Number | 0-300 | Resting blood pressure (mm Hg) |
| Cholesterol | Number | 0+ | Serum cholesterol (mg/dl) |
| Fasting BS | Select | 2 options | Fasting blood sugar category |
| Resting ECG | Select | 3 options | ECG results at rest |
| Max HR | Number | 60-202 | Maximum heart rate during exercise |
| Exercise Angina | Select | Yes/No | Pain during exercise |
| Oldpeak | Number | 0-10 | ST depression value |
| ST Slope | Select | 3 options | ST segment slope |

### Screenshot Workflow:

```
┌────────────────────────────────────────────────────────┐
│                    Predict Tab                          │
├────────────────────────────────────────────────────────┤
│  [Age: 55]                                              │
│  [Sex: Male ▼]                                          │
│  [Chest Pain: Typical Angina ▼]                         │
│  ...                                                    │
│  [Submit Button]                                        │
├────────────────────────────────────────────────────────┤
│  Results:                                               │
│  ─────────────────────────────                          │
│  Logistic Regression: Heart Disease Detected            │
│  Random Forest: Heart Disease Detected                  │
│  ─────────────────────────────                          │
│  ANN Prediction: High Risk                              │
│  [Bar Chart: Risk Probability Breakdown]                │
└────────────────────────────────────────────────────────┘
```

---

## Dataset Reference

### Raw Dataset: `Datasets/heart.csv`

**Source:** [Kaggle Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

**Records:** 918 patients

### Feature Descriptions:

| Feature | Type | Description |
|---------|------|-------------|
| Age | Numeric | Patient age in years |
| Sex | Categorical | M = Male, F = Female |
| ChestPainType | Categorical | TA, ATA, NAP, ASY |
| RestingBP | Numeric | Resting blood pressure (mm Hg) |
| Cholesterol | Numeric | Serum cholesterol (mg/dl) |
| FastingBS | Binary | 1 if fasting blood sugar > 120 mg/dl |
| RestingECG | Categorical | Normal, ST, LVH |
| MaxHR | Numeric | Maximum heart rate achieved |
| ExerciseAngina | Categorical | Y = Yes, N = No |
| Oldpeak | Numeric | ST depression induced by exercise |
| ST_Slope | Categorical | Up, Flat, Down |
| HeartDisease | Binary | 1 = Disease, 0 = No Disease |

### Cleaned Dataset: `Datasets/heart_cleaned.csv`

After preprocessing:
- All categorical variables encoded as integers
- Missing Cholesterol values imputed (172 records had 0)
- Missing RestingBP values imputed (1 record had 0)
- Data types optimized

---

## Model Outputs

All trained models are saved to `../saved_models/` (project root):

| File | Format | Size | Description |
|------|--------|------|-------------|
| `ann_model.pth` | PyTorch state dict | ~15 KB | Neural network weights |
| `scaler.pkl` | Pickle | ~2 KB | StandardScaler parameters |
| `log_reg_model.pkl` | Pickle | ~3 KB | Logistic Regression model |
| `rfc_model.pkl` | Pickle | ~500 KB | Random Forest model |
| `ann_accuracy.pkl` | Pickle | <1 KB | ANN test accuracy |
| `lr_accuracy.pkl` | Pickle | <1 KB | Logistic Regression accuracy |
| `rf_accuracy.pkl` | Pickle | <1 KB | Random Forest accuracy |

### Loading Models in Production:

```python
import pickle
import torch

# Load traditional ML models
log_reg = pickle.load(open('saved_models/log_reg_model.pkl', 'rb'))
rfc = pickle.load(open('saved_models/rfc_model.pkl', 'rb'))

# Load scaler (required for ANN)
scaler = pickle.load(open('saved_models/scaler.pkl', 'rb'))

# Load PyTorch ANN
from model import HeartANN  # Define architecture first
model = HeartANN(11)
model.load_state_dict(torch.load('saved_models/ann_model.pth'))
model.eval()
```

---

## Best Practices

### 1. Data Preprocessing

- ✅ Always run preprocessing before training
- ✅ Check for new missing value patterns in updated datasets
- ✅ Verify encoding consistency with production API

### 2. Model Training

- ✅ Use stratified train-test splits to maintain class balance
- ✅ Save the scaler along with the ANN model
- ✅ Track and compare multiple training runs

### 3. Hyperparameter Tuning

- ✅ Use cross-validation for reliable performance estimates
- ✅ Document best hyperparameters for reproducibility
- ✅ Consider computation time vs. accuracy trade-offs

### 4. Version Control

- ✅ Never commit large model files to Git (use `.gitignore`)
- ✅ Track model versions with clear naming conventions
- ✅ Document changes between model versions

---

## Troubleshooting

### Common Issues:

#### 1. Import Error: `ModuleNotFoundError`

```bash
# Ensure virtual environment is activated
source venv/bin/activate  # or venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements.txt
```

#### 2. File Not Found: Dataset

```bash
# Ensure you're running from the correct directory
cd Scripts
python preprocessing.py  # Creates heart_cleaned.csv
```

#### 3. PyTorch Not Installed

```bash
# Install PyTorch separately (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Or GPU version (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 4. Streamlit Port Already in Use

```bash
# Run on different port
streamlit run dashboard.py --server.port 8502
```

#### 5. Model Path Issues

The scripts use relative paths. Ensure you run them from the correct directory:

```bash
# From Scripts directory
cd model_development/Scripts
python ann-model-training.py

# Models saved to: ../../saved_models/
```

---

## Contributing

When modifying the model development pipeline:

1. Document any changes to the preprocessing steps
2. Update encoding mappings if dataset categories change
3. Re-run all downstream scripts after preprocessing changes
4. Update accuracy metrics in documentation after retraining

---

<p align="center">
  <strong>Happy Model Development! 🧠</strong>
</p>
