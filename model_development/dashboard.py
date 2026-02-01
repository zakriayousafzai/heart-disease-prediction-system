# Import required libraries for web interface, data handling, and machine learning
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.express as px

# Neural Network Architecture Definition
# Multi-layer perceptron for multi-class heart disease risk classification
class HeartANN(nn.Module):
    """
    Artificial Neural Network for heart disease risk prediction.
    Architecture: Input -> FC(64) -> BN -> ReLU -> Dropout -> FC(32) -> BN -> ReLU -> FC(3)
    Output: 3 classes (Low Risk, Medium Risk, High Risk)
    """
    def __init__(self, input_dim):
        super(HeartANN, self).__init__()
        
        # First fully connected layer: input_dim -> 64 neurons
        self.fc1 = nn.Linear(input_dim, 64)
        # Batch normalization for first layer to stabilize training
        self.bn1 = nn.BatchNorm1d(64)
        
        # Second fully connected layer: 64 -> 32 neurons
        self.fc2 = nn.Linear(64, 32)
        # Batch normalization for second layer
        self.bn2 = nn.BatchNorm1d(32)
        
        # Dropout layer with 30% probability to prevent overfitting
        self.dropout = nn.Dropout(0.3)
        
        # Output layer: 32 -> 3 classes (Low/Medium/High Risk)
        self.output = nn.Linear(32, 3)
        # ReLU activation function for non-linearity
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        Returns:
            Output tensor of shape (batch_size, 3) with class logits
        """
        # Layer 1: Linear -> BatchNorm -> ReLU
        x = self.relu(self.bn1(self.fc1(x)))
        # Apply dropout for regularization
        x = self.dropout(x)
        # Layer 2: Linear -> BatchNorm -> ReLU
        x = self.relu(self.bn2(self.fc2(x)))
        # Output layer (no activation, logits returned)
        x = self.output(x)
        return x

# ========================= STREAMLIT APPLICATION =========================
st.title("Multi-Class Heart Disease Risk Prediction")

# Create two tabs for different sections of the application
tab1, tab2 = st.tabs(["Predict", "Model Information"])

# ========================= TAB 1: PREDICTION INTERFACE =========================
with tab1:
    # -------------------- User Input Collection --------------------
    # Collect patient demographic and clinical information
    age = st.number_input("Age", min_value=1, max_value=120)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
    cholesterol = st.number_input("Serum Cholesterol (mm/dl)", min_value=0)
    fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("(Oldpeak) ST Depression Induced by Exercise", min_value=0.0, max_value=10.0)
    st_slope = st.selectbox("Slope of the Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
    
    # -------------------- Data Preprocessing --------------------
    # Convert categorical string inputs to numerical values for model compatibility
    sex = 0 if sex == "Male" else 1
    chest_pain = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(chest_pain)
    fasting_bs = 1 if fasting_bs == "> 120 mg/dl" else 0
    resting_ecg = ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope)
    
    # Create DataFrame with proper column names matching training data
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    })
    
    # -------------------- Model Configuration --------------------
    # Define traditional ML models for binary classification
    algo_names = ["Logistic Regression", "Random Forest"]
    model_names = ["./Models/log_reg_model.pkl", "./Models/rfc_model.pkl"]
    
    # Storage for predictions from multiple models
    predictions = []
    
    def predict_heart_disease(data):
        """
        Load and run predictions using traditional ML models.
        Args:
            data: DataFrame containing patient features
        Returns:
            List of predictions from each model
        """
        for model_name in model_names:
            model = pickle.load(open(model_name, 'rb'))
            prediction = model.predict(data)
            predictions.append(prediction)
        return predictions
    
    # -------------------- Prediction Execution --------------------
    if st.button("Submit"):
        st.subheader('Results......')
        st.markdown('-----------------------------------')
        
        # Execute traditional ML model predictions
        result = predict_heart_disease(input_data)
        
        # Display binary classification results (Disease/No Disease)
        for i in range(len(predictions)):
            st.header(algo_names[i])
            if result[i][0] == 0:
                st.write("No Heart Disease Detected")
            else:
                st.write("Heart Disease Detected")
            st.markdown('--------------------------------')
            
        # -------------------- Neural Network Prediction --------------------
        st.header('Artificial Neural Network Prediction')
        
        # Step 1: Load and apply feature scaling (same scaler used during training)
        scaler = pickle.load(open('./Models/scaler.pkl', 'rb'))
        input_scaled = scaler.transform(input_data.values)
    
        # Step 2: Load trained PyTorch model and set to evaluation mode
        model = HeartANN(input_data.shape[1])
        model.load_state_dict(torch.load('./Models/ann_model.pth'))
        model.eval()
    
        # Step 3: Generate predictions with gradient computation disabled
        with torch.no_grad():
            output = model(torch.FloatTensor(input_scaled))
            # Convert logits to probability distribution using softmax
            probs = F.softmax(output, dim=1).numpy()[0]
    
        # Step 4: Determine predicted class and display result
        categories = ["Low Risk", "Medium Risk", "High Risk"]
        prediction = np.argmax(probs)
    
        st.subheader(f"Result: {categories[prediction]}")
    
        # Step 5: Visualize probability distribution across risk categories
        prob_df = pd.DataFrame({"Risk": categories, "Probability": probs * 100})
        fig = px.bar(prob_df, x="Risk", y="Probability", color="Risk", title="Risk Probability Breakdown")
        st.plotly_chart(fig)

# ========================= TAB 2: MODEL INFORMATION =========================
with tab2:
    import plotly.express as px
    
    # -------------------- Load Model Accuracies --------------------
    # Dynamically load accuracy metrics saved during model training
    # Use default value of 0.0 if accuracy file is missing
    
    try:
        lr_acc = pickle.load(open('./Models/lr_accuracy.pkl', 'rb'))
    except FileNotFoundError:
        lr_acc = 0.0
        
    try:
        rf_acc = pickle.load(open('./Models/rf_accuracy.pkl', 'rb'))
    except FileNotFoundError:
        rf_acc = 0.0
        
    try:
        ann_acc = pickle.load(open('./Models/ann_accuracy.pkl', 'rb'))
    except FileNotFoundError:
        ann_acc = 0.0

    # -------------------- Prepare Visualization Data --------------------
    # Create dictionary mapping model names to their accuracy scores
    data = {
        "Logistic Regression": lr_acc,
        "Random Forest": rf_acc,
        "ANN": ann_acc
    }
    
    # Extract model names and accuracy values for plotting
    Models = list(data.keys())
    Accuracies = list(data.values())
    
    # Create DataFrame for Plotly visualization
    df = pd.DataFrame(list(zip(Models, Accuracies)), columns=['Models', 'Accuracies'])
    
    # Round accuracy values to 2 decimal places for cleaner display
    df['Accuracies'] = df['Accuracies'].round(2)
    
    # -------------------- Generate Comparison Chart --------------------
    # Create bar chart comparing model performance
    fig = px.bar(df, y='Accuracies', x='Models', title="Model Accuracy Comparison")
    
    # Add percentage labels on top of bars for precise values
    fig.update_traces(texttemplate='%{y}%', textposition='outside')
    # Set y-axis range from 0-100% for standardized scale
    fig.update_layout(yaxis_range=[0, 100])
    
    # Display the interactive chart
    st.plotly_chart(fig)