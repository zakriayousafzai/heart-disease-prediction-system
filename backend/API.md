# Heart Disease Risk Prediction API Documentation

## Overview

The Heart Disease Risk Prediction API is a Flask-based REST API that provides heart disease risk predictions using multiple machine learning models. The API analyzes patient clinical data and returns predictions from three different models for comprehensive risk assessment.

### Base URL

```
http://localhost:5000
```

### Models Used

| Model | Type | Output |
|-------|------|--------|
| **Artificial Neural Network (ANN)** | PyTorch Neural Network | Multi-class risk classification (Low/Medium/High Risk) |
| **Random Forest Classifier** | scikit-learn | Binary disease detection |
| **Logistic Regression** | scikit-learn | Binary disease detection |

---

## Technology Stack

- **Framework**: Flask 3.1.2
- **Database**: PostgreSQL (via Flask-SQLAlchemy)
- **ML Libraries**: PyTorch 2.10.0, scikit-learn 1.8.0
- **Data Processing**: pandas 3.0.0, NumPy 2.4.1
- **CORS**: flask-cors 6.0.2

---

## API Endpoints

### 1. Health Check

Verify that the API is running and accessible.

```http
GET /
```

#### Response

```
200 OK
```

```text
Heart Disease Risk Prediction API is Running.
```

---

### 2. Predict Heart Disease Risk

Submit patient clinical data to receive heart disease risk predictions from all three models.

```http
POST /predict
```

#### Request Headers

| Header | Value |
|--------|-------|
| `Content-Type` | `application/json` |

#### Request Body

| Field | Type | Required | Description | Valid Values |
|-------|------|----------|-------------|--------------|
| `age` | integer | Yes | Patient's age in years | 1-120 |
| `sex` | string | Yes | Patient's biological sex | `"Male"`, `"Female"` |
| `chest_pain_type` | string | Yes | Type of chest pain experienced | `"Typical Angina"`, `"Atypical Angina"`, `"Non-anginal Pain"`, `"Asymptomatic"` |
| `resting_bp` | integer | Yes | Resting blood pressure (mm Hg) | 0-300 |
| `cholesterol` | integer | Yes | Serum cholesterol (mg/dl) | 0-600 |
| `fasting_bs` | string | Yes | Fasting blood sugar level | `"> 120 mg/dl"`, `"<= 120 mg/dl"` |
| `resting_ecg` | string | Yes | Resting ECG results | `"Normal"`, `"ST-T wave Abnormality"`, `"Left Ventricular Hypertrophy"` |
| `max_hr` | integer | Yes | Maximum heart rate achieved | 60-220 |
| `exercise_angina` | string | Yes | Exercise-induced angina | `"Yes"`, `"No"` |
| `oldpeak` | float | Yes | ST depression induced by exercise | -5.0 to 10.0 |
| `st_slope` | string | Yes | Slope of peak exercise ST segment | `"Upsloping"`, `"Flat"`, `"Downsloping"` |

#### Example Request

```json
{
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
}
```

#### Success Response

```
200 OK
```

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

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `ann_prediction.result` | string | ANN model risk classification: `"Low Risk"`, `"Medium Risk"`, or `"High Risk"` |
| `ann_prediction.probability` | float | Confidence percentage (0-100) for the predicted class |
| `rf_prediction` | string | Random Forest prediction: `"Heart Disease Detected"` or `"No Heart Disease Detected"` |
| `lr_prediction` | string | Logistic Regression prediction: `"Heart Disease Detected"` or `"No Heart Disease Detected"` |
| `id` | integer | Database ID of the saved patient record |

#### Error Response

```
500 Internal Server Error
```

```json
{
  "error": "Error message describing what went wrong"
}
```

---

### 3. Get Prediction History

Retrieve the most recent prediction records from the database.

```http
GET /history
```

#### Response

```
200 OK
```

```json
[
  {
    "id": 123,
    "age": 55,
    "sex": "Male",
    "chest_pain": "Typical Angina",
    "resting_bp": 140,
    "cholesterol": 250,
    "fasting_bs": "> 120 mg/dl",
    "resting_ecg": "Normal",
    "max_hr": 150,
    "exercise_angina": "Yes",
    "oldpeak": 1.5,
    "st_slope": "Flat",
    "ann_prediction": "High Risk",
    "ann_probability": 85.42,
    "lr_prediction": "Heart Disease Detected",
    "rf_prediction": "Heart Disease Detected",
    "timestamp": "2026-02-01T10:30:00"
  }
]
```

#### Notes

- Returns up to **50** most recent records
- Results are ordered by timestamp (newest first)
- Each record includes all input features and predictions from all models

---

### 4. Get Model Metrics

Retrieve pre-computed accuracy metrics for all machine learning models.

```http
GET /metrics
```

#### Response

```
200 OK
```

```json
{
  "ann": 0.89,
  "rf": 0.87,
  "lr": 0.84
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `ann` | float | ANN model accuracy (0-1 scale) |
| `rf` | float | Random Forest model accuracy (0-1 scale) |
| `lr` | float | Logistic Regression model accuracy (0-1 scale) |

---

## Data Models

### Patient Schema

The database stores patient records with the following schema:

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `id` | Integer | No | Primary key (auto-generated) |
| `age` | Integer | No | Patient's age |
| `sex` | String(10) | No | Patient's sex |
| `chest_pain` | String(50) | No | Chest pain type |
| `resting_bp` | Integer | No | Resting blood pressure |
| `cholesterol` | Integer | No | Serum cholesterol |
| `fasting_bs` | String(20) | No | Fasting blood sugar category |
| `resting_ecg` | String(50) | No | Resting ECG results |
| `max_hr` | Integer | No | Maximum heart rate |
| `exercise_angina` | String(10) | No | Exercise-induced angina |
| `oldpeak` | Float | No | ST depression value |
| `st_slope` | String(20) | No | ST slope category |
| `ann_prediction` | String(20) | No | ANN model prediction |
| `ann_probability` | Float | No | ANN confidence percentage |
| `lr_prediction` | String(50) | Yes | Logistic Regression prediction |
| `rf_prediction` | String(50) | Yes | Random Forest prediction |
| `timestamp` | DateTime | No | Record creation time (UTC) |

---

## Neural Network Architecture

The ANN model ([`HeartANN`](app.py:90)) uses the following architecture:

```
Input Layer (11 features)
    ↓
Linear Layer (64 neurons) + BatchNorm + ReLU
    ↓
Dropout (30%)
    ↓
Linear Layer (32 neurons) + BatchNorm + ReLU
    ↓
Output Layer (3 classes)
```

### Input Features (11 total)

1. Age
2. Sex (encoded)
3. Chest Pain Type (encoded)
4. Resting Blood Pressure
5. Cholesterol
6. Fasting Blood Sugar (encoded)
7. Resting ECG (encoded)
8. Maximum Heart Rate
9. Exercise Angina (encoded)
10. Oldpeak
11. ST Slope (encoded)

---

## Setup & Configuration

### Database Configuration

The API connects to PostgreSQL using the following configuration in [`app.py`](app.py:31):

```python
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:7861@localhost:5432/heart_db'
```

### Required Model Files

The following files must be present in the [`Models/`](Models/) directory:

| File | Description |
|------|-------------|
| `scaler.pkl` | StandardScaler for feature normalization |
| `ann_model.pth` | PyTorch ANN model weights |
| `log_reg_model.pkl` | Trained Logistic Regression model |
| `rfc_model.pkl` | Trained Random Forest model |
| `ann_accuracy.pkl` | ANN accuracy metric |
| `rf_accuracy.pkl` | Random Forest accuracy metric |
| `lr_accuracy.pkl` | Logistic Regression accuracy metric |

### Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure PostgreSQL is running with the database `heart_db`

3. Start the server:

```bash
python app.py
```

The API will be available at `http://localhost:5000`

---

## Error Handling

| Status Code | Description |
|-------------|-------------|
| `200` | Successful request |
| `500` | Internal server error (check error message in response) |

### Common Errors

- **Model Loading Errors**: Ensure all `.pkl` and `.pth` files are present in the `Models/` directory
- **Database Connection Errors**: Verify PostgreSQL is running and credentials are correct
- **Invalid Input**: Ensure all required fields are provided with valid values

---

## CORS Configuration

The API has CORS enabled via [`flask-cors`](app.py:14), allowing cross-origin requests from any domain. This enables frontend applications to communicate with the API from different origins.

---

## Example Usage

### cURL

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

### Python (requests)

```python
import requests

url = "http://localhost:5000/predict"
data = {
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
}

response = requests.post(url, json=data)
print(response.json())
```

### JavaScript (fetch)

```javascript
const response = await fetch('http://localhost:5000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    age: 55,
    sex: 'Male',
    chest_pain_type: 'Typical Angina',
    resting_bp: 140,
    cholesterol: 250,
    fasting_bs: '> 120 mg/dl',
    resting_ecg: 'Normal',
    max_hr: 150,
    exercise_angina: 'Yes',
    oldpeak: 1.5,
    st_slope: 'Flat'
  })
});

const result = await response.json();
console.log(result);
```

---

## Version

- **API Version**: 1.0
- **Last Updated**: February 2026
