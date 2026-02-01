# Heart Disease Prediction System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0-red.svg)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-3.1.2-green.svg)](https://flask.palletsprojects.com)
[![Next.js](https://img.shields.io/badge/Next.js-16.1.6-black.svg)](https://nextjs.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive **Deep Learning-based Heart Disease Prediction System** that analyzes patient clinical data and predicts cardiovascular disease risk using multiple machine learning models. The system provides multi-class risk classification (Low, Medium, High Risk) through an Artificial Neural Network (ANN) and binary disease detection via Random Forest and Logistic Regression models.

![Heart Disease Prediction](model_development/Visuals/heartDisease-distribution.png)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Architecture](#project-architecture)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
  - [Model Development Setup](#model-development-setup)
- [Usage](#usage)
  - [Running the Backend API](#running-the-backend-api)
  - [Running the Frontend](#running-the-frontend)
  - [Running the Streamlit Dashboard](#running-the-streamlit-dashboard)
- [API Documentation](#api-documentation)
- [Machine Learning Models](#machine-learning-models)
  - [Artificial Neural Network (ANN)](#artificial-neural-network-ann)
  - [Random Forest Classifier](#random-forest-classifier)
  - [Logistic Regression](#logistic-regression)
- [Dataset](#dataset)
- [Data Insights & Visualizations](#data-insights--visualizations)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

Heart disease is one of the leading causes of death worldwide. Early and accurate detection can save millions of lives by enabling timely treatment. This project implements a **multi-model prediction system** that:

1. **Analyzes patient health records** including age, gender, blood pressure, cholesterol level, blood sugar, ECG results, and other clinical features
2. **Applies multiple ML/DL models** for comprehensive risk assessment
3. **Provides multi-class risk classification** (Low, Medium, High Risk) using an Artificial Neural Network
4. **Offers binary disease detection** using Random Forest and Logistic Regression models
5. **Stores prediction history** in a PostgreSQL database for tracking and analysis
6. **Visualizes health risk trends** with interactive charts and graphs

---

## Features

### Core Functionality
- ✅ **Multi-Model Predictions**: Get predictions from 3 different models for comprehensive assessment
- ✅ **Risk Classification**: Multi-class output (Low Risk, Medium Risk, High Risk) with probability scores
- ✅ **Binary Detection**: Heart disease detected/not detected from traditional ML models
- ✅ **Patient History**: Store and retrieve past predictions from the database
- ✅ **Model Metrics**: View accuracy comparisons across all models

### Technical Features
- ✅ **RESTful API**: Flask-based backend with CORS support
- ✅ **Modern Frontend**: Next.js 16 with React 19 and Tailwind CSS
- ✅ **Interactive Dashboard**: Streamlit-based visualization tool
- ✅ **Data Preprocessing**: KNN imputation for missing values, categorical encoding
- ✅ **Feature Scaling**: StandardScaler for neural network optimization
- ✅ **Regularization**: Dropout and batch normalization to prevent overfitting
- ✅ **Class Balancing**: Weighted loss function for imbalanced dataset handling

---

## Project Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CLIENT LAYER                                  │
├────────────────────────────┬────────────────────────────────────────────┤
│     Next.js Frontend       │         Streamlit Dashboard                │
│   (React 19 + Tailwind)    │    (Interactive Visualization)            │
└────────────────────────────┴────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            API LAYER                                    │
│                    Flask REST API (Port 5000)                          │
│         ┌──────────┬──────────────┬────────────┬─────────────┐         │
│         │  /       │  /predict    │  /history  │  /metrics   │         │
│         │ (Health) │ (Inference)  │  (Records) │ (Accuracy)  │         │
│         └──────────┴──────────────┴────────────┴─────────────┘         │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          MODEL LAYER                                    │
├───────────────────┬─────────────────────┬───────────────────────────────┤
│   HeartANN        │  Random Forest      │   Logistic Regression         │
│   (PyTorch)       │  (scikit-learn)     │   (scikit-learn)             │
│   Multi-class     │  Binary             │   Binary                      │
│   3 risk levels   │  Detection          │   Detection                   │
└───────────────────┴─────────────────────┴───────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA LAYER                                     │
│                    PostgreSQL Database (heart_db)                       │
│                    Patient Records + Predictions                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

### Backend
| Component | Technology | Version |
|-----------|------------|---------|
| Web Framework | Flask | 3.1.2 |
| Database ORM | Flask-SQLAlchemy | 3.1.1 |
| Database | PostgreSQL | - |
| Deep Learning | PyTorch | 2.10.0 |
| Machine Learning | scikit-learn | 1.8.0 |
| Data Processing | pandas, NumPy | 3.0.0, 2.4.1 |
| CORS | flask-cors | 6.0.2 |

### Frontend
| Component | Technology | Version |
|-----------|------------|---------|
| Framework | Next.js | 16.1.6 |
| UI Library | React | 19.2.3 |
| Styling | Tailwind CSS | 4.x |
| Language | TypeScript | 5.x |

### Model Development
| Component | Technology | Purpose |
|-----------|------------|---------|
| Visualization | Plotly, Matplotlib, Seaborn | Data exploration |
| Dashboard | Streamlit | Interactive UI |
| Imputation | scikit-learn KNNImputer | Missing value handling |

---

## Installation

### Prerequisites

Ensure you have the following installed:

- **Python** 3.10 or higher
- **Node.js** 18.x or higher
- **PostgreSQL** 13 or higher
- **pip** (Python package manager)
- **npm** (Node package manager)

### Backend Setup

1. **Navigate to the backend directory:**
   ```bash
   cd backend
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/macOS
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure PostgreSQL database:**
   
   Create a database named `heart_db` and update the connection string in [`backend/app.py`](backend/app.py:31) if needed:
   ```python
   app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:your_password@localhost:5432/heart_db'
   ```

5. **Verify model files exist in [`backend/Models/`](backend/Models/):**
   - `scaler.pkl` - Feature normalization scaler
   - `ann_model.pth` - PyTorch ANN model weights
   - `log_reg_model.pkl` - Logistic Regression model
   - `rfc_model.pkl` - Random Forest model
   - `ann_accuracy.pkl`, `lr_accuracy.pkl`, `rf_accuracy.pkl` - Accuracy metrics

### Frontend Setup

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

### Model Development Setup

1. **Navigate to the model development directory:**
   ```bash
   cd model_development
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Running the Backend API

1. **Start the Flask server:**
   ```bash
   cd backend
   python app.py
   ```

2. **The API will be available at:** `http://localhost:5000`

3. **Verify the server is running:**
   ```bash
   curl http://localhost:5000
   # Response: Heart Disease Risk Prediction API is Running.
   ```

### Running the Frontend

1. **Start the Next.js development server:**
   ```bash
   cd frontend
   npm run dev
   ```

2. **Access the application at:** `http://localhost:3000`

3. **For production build:**
   ```bash
   npm run build
   npm run start
   ```

### Running the Streamlit Dashboard

1. **Start the Streamlit application:**
   ```bash
   cd model_development
   streamlit run dashboard.py
   ```

2. **Access the dashboard at:** `http://localhost:8501`

---

## API Documentation

Complete API documentation is available in [`backend/API.md`](backend/API.md).

### Quick Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check - verify API is running |
| `/predict` | POST | Submit patient data for risk prediction |
| `/history` | GET | Retrieve recent prediction records (max 50) |
| `/metrics` | GET | Get model accuracy metrics |

### Example Prediction Request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55,
    "sex": "Male",
    "chest_pain_type": "Typical Angina",
    "resting_bp": 140,
    "cholesterol": 250,
    "fasting_bs": "> 120 mg/dl",
    "resting_ecg": "Normal",
    "max_hr": 150,
    "exercise_angina": "Yes",
    "oldpeak": 1.5,
    "st_slope": "Flat"
  }'
```

### Example Response

```json
{
  "ann_prediction": {
    "result": "High Risk",
    "probability": 85.42
  },
  "rf_prediction": "Heart Disease Detected",
  "lr_prediction": "Heart Disease Detected",
  "id": 123
}
```

---

## Machine Learning Models

### Artificial Neural Network (ANN)

The ANN model performs **multi-class risk classification** with the following architecture:

```
Input Layer (11 features)
    │
    ▼
┌─────────────────────────────────┐
│  Linear Layer (64 neurons)      │
│  BatchNorm1d + ReLU            │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Dropout (30%)                  │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Linear Layer (32 neurons)      │
│  BatchNorm1d + ReLU            │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Output Layer (3 classes)       │
│  Low Risk / Medium Risk / High  │
└─────────────────────────────────┘
```

**Key Features:**
- **Batch Normalization**: Stabilizes training and accelerates convergence
- **Dropout (30%)**: Prevents overfitting by randomly deactivating neurons
- **Class Weighting**: Handles imbalanced dataset using weighted cross-entropy loss
- **Early Stopping**: Saves best model weights based on validation accuracy

**Risk Classification Logic:**
- **Low Risk (0)**: No heart disease detected
- **Medium Risk (1)**: Heart disease with moderate indicators
- **High Risk (2)**: Heart disease with severe indicators (Oldpeak > 2.0, Downsloping ST, MaxHR < 120)

### Random Forest Classifier

An ensemble learning method using multiple decision trees for **binary classification**:

- **Hyperparameter Optimization**: GridSearchCV for optimal configuration
- **Parameters Tuned**:
  - `n_estimators`: [50, 100, 150, 500]
  - `max_depth`: [3, 6, 9, 19]
  - `max_features`: ['sqrt', 'log2', None]
  - `max_leaf_nodes`: [3, 6, 9]

### Logistic Regression

A linear probabilistic classifier for **binary classification**:

- **Solver Optimization**: Tests multiple solvers (lbfgs, liblinear, newton-cg, newton-cholesky, sag, saga)
- **Automatic Selection**: Chooses best-performing solver for the dataset

---

## Dataset

The project uses the **Kaggle Heart Failure Prediction Dataset** containing 918 patient records.

**Source:** [https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

### Features

| Feature | Type | Description | Values |
|---------|------|-------------|--------|
| `Age` | Numeric | Patient age in years | 28-77 |
| `Sex` | Categorical | Patient biological sex | M (Male), F (Female) |
| `ChestPainType` | Categorical | Type of chest pain | TA, ATA, NAP, ASY |
| `RestingBP` | Numeric | Resting blood pressure (mm Hg) | 0-200 |
| `Cholesterol` | Numeric | Serum cholesterol (mg/dl) | 0-603 |
| `FastingBS` | Binary | Fasting blood sugar > 120 mg/dl | 0, 1 |
| `RestingECG` | Categorical | Resting ECG results | Normal, ST, LVH |
| `MaxHR` | Numeric | Maximum heart rate achieved | 60-202 |
| `ExerciseAngina` | Binary | Exercise-induced angina | Y (Yes), N (No) |
| `Oldpeak` | Numeric | ST depression induced by exercise | -2.6 to 6.2 |
| `ST_Slope` | Categorical | Peak exercise ST segment slope | Up, Flat, Down |
| `HeartDisease` | Binary | Target variable | 0 (No), 1 (Yes) |

### Chest Pain Type Descriptions

| Code | Type | Description |
|------|------|-------------|
| TA | Typical Angina | Classic angina symptoms |
| ATA | Atypical Angina | Partial angina symptoms |
| NAP | Non-Anginal Pain | Pain not related to angina |
| ASY | Asymptomatic | No chest pain symptoms |

### Data Preprocessing

The preprocessing pipeline ([`model_development/Scripts/preprocessing.py`](model_development/Scripts/preprocessing.py)) includes:

1. **Exploratory Data Analysis**: Statistical summaries and data quality checks
2. **Categorical Encoding**: Label encoding for string variables
3. **Missing Value Imputation**: KNN imputation (k=3) for cholesterol and resting BP values of 0
4. **Data Type Optimization**: Convert appropriate columns to int32 for memory efficiency

---

## Data Insights & Visualizations

The project includes comprehensive exploratory data analysis with visualizations generated using Plotly. All visualizations are available in the [`model_development/Visuals/`](model_development/Visuals/) directory.

### Dataset Distribution

Understanding the class balance in the dataset is crucial for model training:

![Heart Disease Distribution](model_development/Visuals/heartDisease-distribution.png)

*The pie chart shows the distribution of heart disease cases in the dataset. An imbalanced dataset requires special handling techniques like class weighting, which is implemented in the ANN training.*

---

### Feature Correlations with Heart Disease

![Correlations with Heart Disease](model_development/Visuals/correlations-with-heart-disease.png)

*This visualization shows Pearson correlation coefficients between each feature and the target variable (HeartDisease). Key insights:*
- **Positive correlations** (higher values increase disease risk): Oldpeak, ST_Slope, ExerciseAngina, ChestPainType
- **Negative correlations** (higher values decrease disease risk): MaxHR, Sex

---

### Demographic Analysis

#### Age Distribution by Heart Disease Status

![Age vs Heart Disease](model_development/Visuals/histogram-age-heartDisease.png)

*Histogram showing age distribution segmented by heart disease status. The visualization reveals that heart disease prevalence increases with age, with patients aged 50-65 showing the highest risk.*

---

#### Sex Distribution

![Sex vs Heart Disease](model_development/Visuals/histogram-sex-heartDisease.png)

*Distribution of heart disease by biological sex (0=Male, 1=Female). Male patients in this dataset show a higher prevalence of heart disease.*

---

### Clinical Feature Analysis

#### Chest Pain Type Distribution

![Chest Pain Type vs Heart Disease](model_development/Visuals/histogram-chestPainType-heartDisease.png)

*Distribution of chest pain types by disease status:*
- **0 (ATA)**: Atypical Angina - Lower disease correlation
- **1 (NAP)**: Non-Anginal Pain - Moderate disease correlation
- **2 (ASY)**: Asymptomatic - **Highest disease correlation** (patients with no symptoms often have undetected heart disease)
- **3 (TA)**: Typical Angina - Variable correlation

---

#### Exercise-Induced Angina

![Exercise Angina vs Heart Disease](model_development/Visuals/histogram-excersiseAngina-heartDisease.png)

*Exercise-induced angina (0=No, 1=Yes) is a strong predictor of heart disease. Patients experiencing chest pain during exercise have significantly higher disease rates.*

---

#### ST Slope Analysis

![ST Slope vs Heart Disease](model_development/Visuals/histogram-ST_Slope-heartDisease.png)

*The slope of the ST segment during peak exercise is a critical indicator:*
- **0 (Upsloping)**: Generally indicates good cardiac health
- **1 (Flat)**: Moderate risk indicator
- **2 (Downsloping)**: **Strong indicator of heart disease** - used in high-risk classification

---

#### Fasting Blood Sugar

![Fasting Blood Sugar vs Heart Disease](model_development/Visuals/histogram-fastingBS-heartDisease.png)

*Fasting blood sugar above 120 mg/dl (1) shows correlation with heart disease, linking diabetes risk to cardiovascular health.*

---

### Continuous Variable Analysis

#### Maximum Heart Rate Distribution

![MaxHR Violin Plot](model_development/Visuals/violinPlot-maxHR-heartDisease.png)

*Violin plot showing MaxHR distribution by disease status. Key insight: Patients with heart disease tend to have **lower maximum heart rates** during exercise testing, indicating reduced cardiovascular capacity.*

---

#### ST Depression (Oldpeak)

![Oldpeak Violin Plot](model_development/Visuals/violinPlot-oldpeak-heartDisease.png)

*Violin plot of Oldpeak (ST depression induced by exercise relative to rest):*
- Patients **without** heart disease cluster around 0
- Patients **with** heart disease show higher Oldpeak values
- Values > 2.0 are used as a criterion for **High Risk** classification in the ANN model

---

### Hierarchical Visualizations

#### Age Groups Sunburst Chart

![Age Sunburst](model_development/Visuals/sunburstChart-age-heartDisease.png)

*Hierarchical sunburst chart showing the relationship between heart disease status and age distribution. The inner ring represents disease status (0/1), and outer segments show age distributions within each group.*

---

#### Maximum Heart Rate Sunburst

![MaxHR Sunburst](model_development/Visuals/sunburstChart-maxHR-heartDisease.png)

*Sunburst visualization of maximum heart rate distributions segmented by disease status, providing a hierarchical view of how exercise capacity varies between healthy and diseased patients.*

---

#### Resting Blood Pressure Sunburst

![Resting BP Sunburst](model_development/Visuals/sunburstChart-restingBP-heartDisease.png)

*Hierarchical view of resting blood pressure distributions by heart disease status. While resting BP shows weaker correlation with heart disease compared to exercise metrics, extreme values still indicate increased risk.*

---

## Project Structure

```
FYP/
├── README.md                          # This documentation file
├── project_details.md                 # Project requirements and specifications
├── .gitignore                         # Git ignore rules
│
├── backend/                           # Flask REST API
│   ├── app.py                         # Main API application
│   ├── API.md                         # API documentation
│   ├── requirements.txt               # Python dependencies
│   └── Models/                        # Trained model files
│       ├── ann_model.pth              # PyTorch ANN weights
│       ├── ann_accuracy.pkl           # ANN accuracy metric
│       ├── log_reg_model.pkl          # Logistic Regression model
│       ├── lr_accuracy.pkl            # LR accuracy metric
│       ├── rfc_model.pkl              # Random Forest model
│       ├── rf_accuracy.pkl            # RF accuracy metric
│       └── scaler.pkl                 # StandardScaler for features
│
├── frontend/                          # Next.js web application
│   ├── app/                           # Next.js App Router
│   │   ├── page.tsx                   # Home page component
│   │   ├── layout.tsx                 # Root layout
│   │   ├── globals.css                # Global styles
│   │   └── favicon.ico                # Site favicon
│   ├── public/                        # Static assets
│   ├── package.json                   # Node.js dependencies
│   ├── tsconfig.json                  # TypeScript configuration
│   ├── next.config.ts                 # Next.js configuration
│   ├── postcss.config.mjs             # PostCSS configuration
│   └── eslint.config.mjs              # ESLint configuration
│
├── model_development/                 # ML model training & analysis
│   ├── dashboard.py                   # Streamlit visualization app
│   ├── requirements.txt               # Python dependencies
│   ├── Datasets/
│   │   ├── heart.csv                  # Original dataset (918 records)
│   │   └── heart_cleaned.csv          # Preprocessed dataset
│   ├── Scripts/
│   │   ├── preprocessing.py           # Data cleaning & preparation
│   │   ├── ann-model-training.py      # ANN model training
│   │   ├── ml-model-training.py       # RF & LR training
│   │   └── visualization.py           # EDA visualizations
│   └── Visuals/                       # Generated charts & plots
│       ├── correlations-with-heart-disease.png
│       ├── heartDisease-distribution.png
│       ├── histogram-age-heartDisease.png
│       └── ... (additional visualizations)
│
└── saved_models/                      # Backup of trained models
    ├── ann_model.pth
    ├── ann_accuracy.pkl
    ├── log_reg_model.pkl
    ├── lr_accuracy.pkl
    ├── rfc_model.pkl
    ├── rf_accuracy.pkl
    └── scaler.pkl
```

---

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes and commit:**
   ```bash
   git commit -m "Add: your feature description"
   ```
4. **Push to your branch:**
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Open a Pull Request**

### Code Style Guidelines

- Follow PEP 8 for Python code
- Use TypeScript for frontend development
- Include docstrings for all functions and classes
- Write meaningful commit messages

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Dataset Source**: [Kaggle Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Frameworks**: PyTorch, Flask, Next.js, Streamlit
- **Libraries**: scikit-learn, pandas, NumPy, Plotly

---

## Contact

For questions, issues, or contributions, please open an issue on GitHub or contact the project maintainer.

---

<p align="center">
  <strong>Early detection saves lives. ❤️</strong>
</p>
