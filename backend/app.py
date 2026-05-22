"""
Heart Disease Risk Prediction API

A Flask-based REST API that provides heart disease risk predictions using multiple machine learning models:
- Artificial Neural Network (ANN) for multi-class risk classification
- Random Forest Classifier for binary disease detection
- Logistic Regression for binary disease detection

The API stores patient data and predictions in a PostgreSQL database.
Uses SHAP for model explainability and Google Gemini AI for personalized recommendations.
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
import shap
import json
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore")

# Initialize Flask application with CORS support for cross-origin requests
app = Flask(__name__)
CORS(app)

# Configure PostgreSQL database connection
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DB_URI')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ========================= GEMINI AI CONFIGURATION =========================

GEMINI_API_KEY = os.getenv('API_KEY')

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    print("Gemini AI initialized successfully.")
else:
    gemini_model = None
    print("Warning: GEMINI_API_KEY not set. Recommendations will not be generated.")

# Gemini prompt template for generating personalized health recommendations
GEMINI_PROMPT_TEMPLATE = """You are a medical AI assistant specializing in cardiovascular health.
A patient has been assessed by our heart disease prediction model.

**Patient Data:**
- Age: {age} years, Sex: {sex}
- Chest Pain Type: {chest_pain}
- Resting Blood Pressure: {resting_bp} mmHg
- Cholesterol: {cholesterol} mg/dl
- Fasting Blood Sugar: {fasting_bs}
- Resting ECG: {resting_ecg}
- Maximum Heart Rate: {max_hr} bpm
- Exercise-Induced Angina: {exercise_angina}
- ST Depression (Oldpeak): {oldpeak}
- ST Slope: {st_slope}

**Prediction Results:**
- ANN Model: {ann_result} (Confidence: {ann_prob}%)
- Random Forest: {rf_result}
- Logistic Regression: {lr_result}

**Top Risk Factors (by SHAP importance):**
{risk_factors_text}

Based on this analysis, provide 3-5 personalized health recommendations.

IMPORTANT: Respond ONLY with a valid JSON array. No markdown, no code blocks, no extra text.
Each recommendation must have these exact fields:
[
  {{
    "category": "dietary" or "medical" or "lifestyle",
    "icon": "an emoji that represents the recommendation",
    "title": "short recommendation title",
    "description": "detailed actionable advice in 2-3 sentences",
    "priority": "high" or "medium" or "low"
  }}
]

Guidelines:
- Be specific to this patient's values, not generic advice
- Prioritize based on which risk factors have highest SHAP impact
- Include a mix of dietary, medical, and lifestyle recommendations
- Use empathetic but professional medical tone
- Always recommend consulting a healthcare provider for critical findings
- Do NOT diagnose - only suggest actions
- Sort recommendations by priority (high first)"""


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

# Feature names mapping for explainability
FEATURE_NAMES = [
    'Age', 'Sex', 'Chest Pain Type', 'Resting Blood Pressure',
    'Cholesterol', 'Fasting Blood Sugar', 'Resting ECG',
    'Max Heart Rate', 'Exercise Angina', 'Oldpeak', 'ST Slope'
]

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
    
    # Initialize SHAP KernelExplainer for ANN model
    ann_shap_background = pickle.load(open('./Models/ann_shap_samples.pkl', 'rb'))

    def ann_predict_proba_fn(data):
        data_t = torch.FloatTensor(data)
        with torch.no_grad():
            logits = ann_model(data_t)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    ann_explainer = shap.KernelExplainer(ann_predict_proba_fn, ann_shap_background)
    
    # Load pre-computed model accuracy metrics for comparison
    metrics_data = {
        "ann": pickle.load(open('./Models/ann_accuracy.pkl', 'rb')),
        "rf": pickle.load(open('./Models/rf_accuracy.pkl', 'rb')),
        "lr": pickle.load(open('./Models/lr_accuracy.pkl', 'rb'))
    }
    print("All models loaded successfully.")
    print("SHAP KernelExplainer initialized for ANN model.")
except Exception as e:
    print(f"Error loading models: {e}")
    metrics_data = {"ann": 0, "rf": 0, "lr": 0}
    ann_explainer = None

# ========================= EXPLAINABILITY HELPER FUNCTIONS =========================

def format_feature_value(feature_idx, patient_data, original_input):
    """
    Format feature values for human-readable display.
    
    Args:
        feature_idx: Index of the feature (0-10)
        patient_data: DataFrame with encoded values
        original_input: Dictionary with original string values
        
    Returns:
        Formatted string representing the feature value
    """
    feature_map = {
        0: lambda: f"{int(patient_data.iloc[0, feature_idx])} years",
        1: lambda: original_input['sex'],
        2: lambda: original_input['chest_pain_type'],
        3: lambda: f"{int(patient_data.iloc[0, feature_idx])} mmHg",
        4: lambda: f"{int(patient_data.iloc[0, feature_idx])} mg/dl",
        5: lambda: original_input['fasting_bs'],
        6: lambda: original_input['resting_ecg'],
        7: lambda: f"{int(patient_data.iloc[0, feature_idx])} bpm",
        8: lambda: original_input['exercise_angina'],
        9: lambda: f"{float(patient_data.iloc[0, feature_idx]):.1f}",
        10: lambda: original_input['st_slope']
    }
    return feature_map.get(feature_idx, lambda: str(patient_data.iloc[0, feature_idx]))()

def get_feature_description(feature_idx, patient_data, original_input):
    """
    Generate descriptive text explaining the clinical significance of a feature value.
    
    Args:
        feature_idx: Index of the feature (0-10)
        patient_data: DataFrame with encoded values
        original_input: Dictionary with original string values
        
    Returns:
        Human-readable description of the feature
    """
    age = int(patient_data.iloc[0, 0])
    cholesterol = int(patient_data.iloc[0, 4])
    resting_bp = int(patient_data.iloc[0, 3])
    max_hr = int(patient_data.iloc[0, 7])
    oldpeak = float(patient_data.iloc[0, 9])
    fasting_bs = original_input['fasting_bs']
    chest_pain = original_input['chest_pain_type']
    exercise_angina = original_input['exercise_angina']
    st_slope = original_input['st_slope']
    
    descriptions = {
        0: f"Age of {age} years contributes to overall cardiovascular risk assessment",
        1: f"Biological sex affects heart disease risk patterns",
        2: f"{chest_pain} chest pain pattern is clinically significant",
        3: f"Resting blood pressure of {resting_bp} mmHg {'is elevated' if resting_bp > 140 else 'is within normal range' if resting_bp >= 120 else 'is low'}",
        4: f"Cholesterol level of {cholesterol} mg/dl {'is high' if cholesterol > 200 else 'is within normal range'}",
        5: f"Fasting blood sugar {fasting_bs} {'indicates potential diabetes risk' if '>' in fasting_bs else 'is within normal range'}",
        6: f"Resting ECG shows {original_input['resting_ecg']}",
        7: f"Maximum heart rate of {max_hr} bpm {'is lower than expected' if max_hr < 120 else 'is within normal range'}",
        8: f"Exercise-induced angina {'present' if exercise_angina == 'Yes' else 'not present'}",
        9: f"ST depression of {oldpeak} {'indicates significant ischemia' if oldpeak > 2.0 else 'suggests some cardiac stress' if oldpeak > 1.0 else 'is minimal'}",
        10: f"ST slope is {st_slope} {'(concerning pattern)' if st_slope in ['Flat', 'Downsloping'] else '(normal response)'}"
    }
    return descriptions.get(feature_idx, "Clinical parameter")

def categorize_impact(shap_value):
    """
    Categorize SHAP importance score into impact levels.
    
    Args:
        shap_value: Absolute SHAP value
        
    Returns:
        String: 'high', 'medium', or 'low'
    """
    if shap_value > 0.5:
        return 'high'
    elif shap_value > 0.2:
        return 'medium'
    else:
        return 'low'

# ========================= GEMINI AI RECOMMENDATIONS =========================

def generate_ai_recommendations(original_input, risk_factors, prediction_results):
    """
    Generate AI-powered personalized health recommendations using Google Gemini.
    
    Sends the patient's clinical data, model predictions, and SHAP-identified risk factors
    to Gemini AI to generate context-aware, personalized recommendations.
    
    Returns empty list if Gemini is unavailable or fails.
    
    Args:
        original_input: Dictionary with original patient input values
        risk_factors: List of SHAP-identified risk factors
        prediction_results: Dictionary with prediction results from all models
        
    Returns:
        List of recommendation dictionaries, or empty list if Gemini unavailable
    """
    if gemini_model is None:
        print("Gemini not available, no recommendations will be generated.")
        return []
    
    try:
        # Build risk factors text for the prompt
        if risk_factors:
            risk_factors_text = "\n".join([
                f"  {i+1}. {rf['feature']}: {rf['value']} - {rf['impact']} impact ({rf['direction']})"
                for i, rf in enumerate(risk_factors)
            ])
        else:
            risk_factors_text = "  No SHAP risk factors computed."
        
        # Fill the prompt template with patient data
        prompt = GEMINI_PROMPT_TEMPLATE.format(
            age=original_input['age'],
            sex=original_input['sex'],
            chest_pain=original_input['chest_pain_type'],
            resting_bp=original_input['resting_bp'],
            cholesterol=original_input['cholesterol'],
            fasting_bs=original_input['fasting_bs'],
            resting_ecg=original_input['resting_ecg'],
            max_hr=original_input['max_hr'],
            exercise_angina=original_input['exercise_angina'],
            oldpeak=original_input['oldpeak'],
            st_slope=original_input['st_slope'],
            ann_result=prediction_results['ann_result'],
            ann_prob=prediction_results['ann_prob'],
            rf_result=prediction_results['rf_result'],
            lr_result=prediction_results['lr_result'],
            risk_factors_text=risk_factors_text
        )
        
        # Call Gemini API
        response = gemini_model.generate_content(prompt)
        
        # Clean the response text - remove markdown code blocks if present
        response_text = response.text.strip()
        if response_text.startswith("```"):
            # Remove ```json and ``` markers
            lines = response_text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            response_text = "\n".join(lines).strip()
        
        # Parse JSON response
        recommendations = json.loads(response_text)
        
        # Validate the structure of each recommendation
        valid_categories = {'dietary', 'medical', 'lifestyle'}
        valid_priorities = {'high', 'medium', 'low'}
        validated_recs = []
        
        for rec in recommendations:
            if isinstance(rec, dict) and all(k in rec for k in ['category', 'icon', 'title', 'description', 'priority']):
                # Normalize values
                rec['category'] = rec['category'].lower() if rec['category'].lower() in valid_categories else 'lifestyle'
                rec['priority'] = rec['priority'].lower() if rec['priority'].lower() in valid_priorities else 'medium'
                validated_recs.append(rec)
        
        if validated_recs:
            print(f"Gemini generated {len(validated_recs)} recommendations successfully.")
            return validated_recs
        else:
            print("Gemini returned invalid format, no recommendations will be generated.")
            return []
        
    except Exception as e:
        print(f"Gemini API error: {e}")
        import traceback
        traceback.print_exc()
        return []

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
    
    Also returns SHAP-based risk factor analysis and Gemini AI-powered recommendations.
    Saves the patient record and predictions to the database.
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

        # ========================= GENERATE EXPLAINABILITY DATA =========================
        risk_factors = []
        recommendations = []
        
        # Compute SHAP values if ANN explainer is available
        if ann_explainer is not None:
            try:
                # Compute ANN SHAP values on scaled features
                shap_values = ann_explainer.shap_values(features_scaled, nsamples=100)

                # Use SHAP values for the predicted ANN class
                if isinstance(shap_values, list):
                    # Common multi-class format: list[n_classes] of (n_samples, n_features)
                    raw_shap = np.array(shap_values[pred_idx][0]).flatten()
                else:
                    shap_arr = np.array(shap_values)
                    if shap_arr.ndim == 3:
                        # Could be (n_samples, n_features, n_classes)
                        if shap_arr.shape[0] == features_scaled.shape[0]:
                            raw_shap = shap_arr[0, :, pred_idx].flatten()
                        else:
                            # Could be (n_classes, n_samples, n_features)
                            raw_shap = shap_arr[pred_idx, 0, :].flatten()
                    elif shap_arr.ndim == 2:
                        # Single-output format fallback
                        raw_shap = shap_arr[0].flatten()
                    else:
                        raw_shap = shap_arr.flatten()
                
                # Ensure we have a 1D array of length 11 (one per feature)
                shap_direction = raw_shap[:11]
                feature_importance = np.abs(shap_direction)
                
                # Get top 5 most important features (sorted by absolute SHAP value)
                top_indices = list(np.argsort(feature_importance)[::-1][:5])
                
                # Build risk_factors list with detailed information
                for idx in top_indices:
                    i = int(idx.item()) if hasattr(idx, 'item') else int(idx)
                    impact_score = float(feature_importance[i])
                    direction_value = float(shap_direction[i])
                    
                    risk_factors.append({
                        'feature': FEATURE_NAMES[i],
                        'value': format_feature_value(i, raw_features_df, data),
                        'impact': categorize_impact(impact_score),
                        'impact_score': round(impact_score, 4),
                        'direction': 'increases risk' if direction_value > 0 else 'decreases risk',
                        'description': get_feature_description(i, raw_features_df, data)
                    })
                
            except Exception as e:
                print(f"Error computing SHAP values: {e}")
                import traceback
                traceback.print_exc()
                risk_factors = []

        # ========================= GENERATE AI RECOMMENDATIONS =========================
        # Use Gemini AI for personalized recommendations (with rule-based fallback)
        prediction_results = {
            'ann_result': ann_result,
            'ann_prob': round(ann_prob, 2),
            'rf_result': rf_result,
            'lr_result': lr_result
        }
        recommendations = generate_ai_recommendations(data, risk_factors, prediction_results)

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
        
        # Return predictions from all three models with confidence scores and explainability data
        response_data = {
            "ann_prediction": {
                "result": ann_result,
                "probability": round(ann_prob, 2)
            },
            "rf_prediction": rf_result,
            "lr_prediction": lr_result,
            "id": new_patient.id
        }
        
        # Add explainability data if available
        if risk_factors:
            response_data["risk_factors"] = risk_factors
        if recommendations:
            response_data["recommendations"] = recommendations
            
        return jsonify(response_data)

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