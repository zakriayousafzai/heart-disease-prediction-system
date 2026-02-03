"""
Heart Disease Risk Prediction API

A Flask-based REST API that provides heart disease risk predictions using multiple machine learning models:
- Artificial Neural Network (ANN) for multi-class risk classification
- Random Forest Classifier for binary disease detection
- Logistic Regression for binary disease detection

The API stores patient data and predictions in a PostgreSQL database.
"""

from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Initialize Flask application with CORS support for cross-origin requests
app = Flask(__name__)
CORS(app)

# Configure PostgreSQL database connection
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:9832954@localhost:5432/heart_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class Patient(db.Model):
    """
    SQLAlchemy ORM model representing patient records and prediction results.
    
    Stores patient medical data including vital signs, ECG readings, and predictions
    from multiple machine learning models. Each record represents a single prediction request.
    """
    __tablename__ = 'patients'
    
    # Primary key
    id = db.Column(db.Integer, primary_key=True)
    
    # Patient clinical features
    age = db.Column(db.Integer, nullable=False)
    sex = db.Column(db.String(10), nullable=False)
    chest_pain = db.Column(db.String(50), nullable=False)
    resting_bp = db.Column(db.Integer, nullable=False)
    cholesterol = db.Column(db.Integer, nullable=False)
    fasting_bs = db.Column(db.String(20), nullable=False)
    resting_ecg = db.Column(db.String(50), nullable=False)
    max_hr = db.Column(db.Integer, nullable=False)
    exercise_angina = db.Column(db.String(10), nullable=False)
    oldpeak = db.Column(db.Float, nullable=False)
    st_slope = db.Column(db.String(20), nullable=False)
    
    # Model predictions and confidence scores
    ann_prediction = db.Column(db.String(20), nullable=False)
    ann_probability = db.Column(db.Float, nullable=False)
    lr_prediction = db.Column(db.String(50))
    rf_prediction = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        """Convert patient record to dictionary format for JSON serialization."""
        return {
            'id': self.id,
            'age': self.age,
            'sex': self.sex,
            'chest_pain': self.chest_pain,
            'resting_bp': self.resting_bp,
            'cholesterol': self.cholesterol,
            'fasting_bs': self.fasting_bs,
            'resting_ecg': self.resting_ecg,
            'max_hr': self.max_hr,
            'exercise_angina': self.exercise_angina,
            'oldpeak': self.oldpeak,
            'st_slope': self.st_slope,
            'ann_prediction': self.ann_prediction,
            'ann_probability': self.ann_probability,
            'lr_prediction': self.lr_prediction,
            'rf_prediction': self.rf_prediction,
            'timestamp': self.timestamp.isoformat()
        }

class HeartANN(nn.Module):
    """
    PyTorch Artificial Neural Network for heart disease risk classification.
    
    Architecture:
    - Input layer: 11 features (patient clinical data)
    - Hidden layer 1: 64 neurons with batch normalization and ReLU activation
    - Dropout layer: 30% dropout rate for regularization
    - Hidden layer 2: 32 neurons with batch normalization and ReLU activation
    - Output layer: 3 classes (Low Risk, Medium Risk, High Risk)
    """
    def __init__(self, input_dim):
        super(HeartANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(32, 3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """Forward pass through the network with batch normalization and dropout."""
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.output(x)
        return x

# Load all pre-trained machine learning models and preprocessing utilities
try:
    # Load StandardScaler for feature normalization (required for ANN input)
    scaler = pickle.load(open('./Models/scaler.pkl', 'rb'))
    
    # Load PyTorch ANN model and set to evaluation mode
    ann_model = HeartANN(11)
    ann_model.load_state_dict(torch.load('./Models/ann_model.pth'))
    ann_model.eval()
    
    # Load scikit-learn models (trained on encoded but unscaled data)
    log_reg_model = pickle.load(open('./Models/log_reg_model.pkl', 'rb'))
    rfc_model = pickle.load(open('./Models/rfc_model.pkl', 'rb'))
    
    # Load pre-computed model accuracy metrics for comparison
    metrics_data = {
        "ann": pickle.load(open('./Models/ann_accuracy.pkl', 'rb')),
        "rf": pickle.load(open('./Models/rf_accuracy.pkl', 'rb')),
        "lr": pickle.load(open('./Models/lr_accuracy.pkl', 'rb'))
    }
    print("All models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    metrics_data = {"ann": 0, "rf": 0, "lr": 0}

# Initialize database tables if they don't exist
with app.app_context():
    db.create_all()

@app.route('/')
def home():
    """Health check endpoint to verify API is running."""
    return "Heart Disease Risk Prediction API is Running."

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict heart disease risk based on patient clinical data.
    
    Expected JSON payload contains patient features including age, sex, chest pain type,
    blood pressure, cholesterol, ECG readings, and exercise test results.
    
    Returns predictions from three models:
    - ANN: Multi-class risk level (Low/Medium/High) with probability
    - Random Forest: Binary disease detection
    - Logistic Regression: Binary disease detection
    
    Also saves the patient record and predictions to the database.
    """
    data = request.json
    
    try:
        # Encode categorical features to numerical values for model input
        sex_enc = 0 if data['sex'] == "Male" else 1
        
        cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
        cp_enc = cp_map.get(data['chest_pain_type'], 3)
        
        bs_enc = 1 if data['fasting_bs'] == "> 120 mg/dl" else 0
        
        ecg_map = {"Normal": 0, "ST-T wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
        ecg_enc = ecg_map.get(data['resting_ecg'], 0)
        
        ex_enc = 1 if data['exercise_angina'] == "Yes" else 0
        
        slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
        slope_enc = slope_map.get(data['st_slope'], 1)

        # Prepare feature vectors in DataFrame format for model compatibility
        raw_features_df = pd.DataFrame([{
            'Age': int(data['age']),
            'Sex': sex_enc,
            'ChestPainType': cp_enc,
            'RestingBP': int(data['resting_bp']),
            'Cholesterol': int(data['cholesterol']),
            'FastingBS': bs_enc,
            'RestingECG': ecg_enc,
            'MaxHR': int(data['max_hr']),
            'ExerciseAngina': ex_enc,
            'Oldpeak': float(data['oldpeak']),
            'ST_Slope': slope_enc
        }])
        
        # Apply StandardScaler transformation for ANN input (ANN requires normalized features)
        features_scaled = scaler.transform(raw_features_df.values)
        
        # Generate predictions from ANN model using PyTorch
        with torch.no_grad():
            tensor_in = torch.FloatTensor(features_scaled)
            output = ann_model(tensor_in)
            probs = F.softmax(output, dim=1).numpy()[0]
            
        categories = ["Low Risk", "Medium Risk", "High Risk"]
        pred_idx = np.argmax(probs)
        ann_result = categories[pred_idx]
        ann_prob = float(probs[pred_idx] * 100)
        
        # Generate predictions from Random Forest model (binary classification)
        rf_pred_binary = rfc_model.predict(raw_features_df)[0]
        rf_result = "Heart Disease Detected" if rf_pred_binary == 1 else "No Heart Disease Detected"
        
        # Generate predictions from Logistic Regression model (binary classification)
        lr_pred_binary = log_reg_model.predict(raw_features_df)[0]
        lr_result = "Heart Disease Detected" if lr_pred_binary == 1 else "No Heart Disease Detected"

        # Create new patient record with input data and all model predictions
        new_patient = Patient(
            age=data['age'], sex=data['sex'], chest_pain=data['chest_pain_type'],
            resting_bp=data['resting_bp'], cholesterol=data['cholesterol'],
            fasting_bs=data['fasting_bs'], resting_ecg=data['resting_ecg'],
            max_hr=data['max_hr'], exercise_angina=data['exercise_angina'],
            oldpeak=data['oldpeak'], st_slope=data['st_slope'],
            ann_prediction=ann_result, ann_probability=ann_prob,
            lr_prediction=lr_result, rf_prediction=rf_result
        )
        
        db.session.add(new_patient)
        db.session.commit()
        
        # Return predictions from all three models with confidence scores
        return jsonify({
            "ann_prediction": {
                "result": ann_result,
                "probability": round(ann_prob, 2)
            },
            "rf_prediction": rf_result,
            "lr_prediction": lr_result,
            "id": new_patient.id
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    """
    Retrieve prediction history from the database.
    
    Returns the 50 most recent patient records with their predictions,
    ordered by timestamp (newest first).
    """
    patients = Patient.query.order_by(Patient.timestamp.desc()).limit(50).all()
    return jsonify([p.to_dict() for p in patients])

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """
    Retrieve pre-computed accuracy metrics for all models.
    
    Returns accuracy scores for ANN, Random Forest, and Logistic Regression models.
    """
    return jsonify(metrics_data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)