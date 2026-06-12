"""
Artificial Neural Network Training Script for Multi-Class Heart Disease Risk Prediction

This script trains a PyTorch-based neural network to classify patients into three risk categories:
- Low Risk (0): No heart disease
- Medium Risk (1): Heart disease with moderate indicators
- High Risk (2): Heart disease with severe clinical indicators

Key Features:
- Class weight balancing for imbalanced dataset
- Batch normalization for training stability
- Dropout regularization to prevent overfitting
- Early stopping to save best model weights
- Feature scaling for improved convergence
"""

# Import required libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import pickle
import copy

# ========================= DATA LOADING AND PREPARATION =========================

# Load preprocessed heart disease dataset
heart_df = pd.read_csv('../Datasets/heart_cleaned.csv')

# -------------------- Multi-Class Risk Category Assignment --------------------
def assign_risk(row):
    """
    Assigns risk category based on heart disease status and clinical severity indicators.
    
    Risk Categories:
    - 0 (Low Risk): No heart disease detected
    - 1 (Medium Risk): Heart disease present with moderate symptoms
    - 2 (High Risk): Heart disease with severe indicators:
        * Oldpeak > 2.0 (significant ST depression)
        * ST_Slope == 2 (downsloping - poor prognosis)
        * MaxHR < 120 (reduced exercise capacity)
    
    Args:
        row: DataFrame row containing patient clinical data
        
    Returns:
        int: Risk category (0, 1, or 2)
    """
    if row['HeartDisease'] == 0:
        return 0
    elif (row['Oldpeak'] > 2.0) or (row['ST_Slope'] == 2) or (row['MaxHR'] < 120):
        return 2
    else:
        return 1

# Apply risk categorization to entire dataset
heart_df['RiskCategory'] = heart_df.apply(assign_risk, axis=1)

# -------------------- Feature and Target Extraction --------------------
# Separate features (X) and target labels (y)
# Remove both original binary label and new multi-class label from features
feature_names = heart_df.drop(['HeartDisease', 'RiskCategory'], axis=1).columns.tolist()
X = heart_df.drop(['HeartDisease', 'RiskCategory'], axis=1).values
y = heart_df['RiskCategory'].values

# Split data into training (80%) and testing (20%) sets
# Stratify ensures proportional class distribution in both sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# -------------------- Feature Scaling --------------------
# Standardize features to have mean=0 and std=1 for better neural network performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit on training data only
X_test = scaler.transform(X_test)        # Transform test data using training statistics
# Save scaler for use in production (app.py)
pickle.dump(scaler, open('../Models/scaler.pkl', 'wb'))

# ========================= HANDLING CLASS IMBALANCE =========================

# Compute class weights to address imbalanced dataset
# Assigns higher weights to minority classes, forcing model to pay more attention
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_t = torch.FloatTensor(class_weights)

# ========================= NEURAL NETWORK ARCHITECTURE =========================

class HeartANN(nn.Module):
    """
    Optimized Artificial Neural Network for multi-class heart disease risk prediction.
    
    Architecture:
        Input Layer: 11 features
        Hidden Layer 1: 64 neurons with BatchNorm + ReLU + Dropout
        Hidden Layer 2: 32 neurons with BatchNorm + ReLU
        Output Layer: 3 neurons (one per risk category)
    
    Techniques:
        - Batch Normalization: Stabilizes training and accelerates convergence
        - Dropout (30%): Reduces overfitting by randomly deactivating neurons
        - ReLU Activation: Introduces non-linearity for complex pattern learning
    """
    
    def __init__(self, input_dim):
        super(HeartANN, self).__init__()
        
        # First hidden layer: Input -> 64 neurons
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)  # Normalizes activations for stable training
        
        # Second hidden layer: 64 -> 32 neurons
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        
        # Dropout layer for regularization (prevents overfitting)
        self.dropout = nn.Dropout(0.3)
        
        # Output layer: 32 -> 3 classes
        self.output = nn.Linear(32, 3)
        
        # Activation function
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Forward propagation through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, 3) with raw logits
        """
        # Layer 1: Linear -> BatchNorm -> ReLU
        x = self.relu(self.bn1(self.fc1(x)))
        # Apply dropout after first layer
        x = self.dropout(x)
        # Layer 2: Linear -> BatchNorm -> ReLU
        x = self.relu(self.bn2(self.fc2(x)))
        # Output layer (no activation - raw logits for CrossEntropyLoss)
        x = self.output(x)
        return x

# ========================= MODEL INITIALIZATION =========================

# Initialize model with appropriate input dimensions
model = HeartANN(X_train.shape[1])

# Adam optimizer with standard learning rate
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Cross-entropy loss with class weights to handle imbalance
criterion = nn.CrossEntropyLoss(weight=class_weights_t)

# -------------------- Convert NumPy Arrays to PyTorch Tensors --------------------
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

# ========================= TRAINING WITH EARLY STOPPING =========================

# Track best model performance for early stopping
best_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())

# Training loop: 500 epochs with validation-based early stopping
for epoch in range(500):
    # -------------------- Training Phase --------------------
    model.train()  # Enable training mode (activates dropout, batch norm training)
    optimizer.zero_grad()  # Clear gradients from previous iteration
    outputs = model(X_train_t)  # Forward pass
    loss = criterion(outputs, y_train_t)  # Compute weighted cross-entropy loss
    loss.backward()  # Backpropagation to compute gradients
    optimizer.step()  # Update model parameters
    
    # -------------------- Validation Phase --------------------
    model.eval()  # Enable evaluation mode (disables dropout, uses running batch norm stats)
    with torch.no_grad():  # Disable gradient computation for efficiency
        test_out = model(X_test_t)  # Forward pass on test set
        _, predicted = torch.max(test_out, 1)  # Get predicted class (highest logit)
        correct = (predicted == y_test_t).sum().item()  # Count correct predictions
        acc = (correct / y_test_t.size(0)) * 100  # Calculate accuracy percentage
        
        # Early stopping: Save model only if it achieves new best accuracy
        if acc > best_acc:
            best_acc = acc
            best_model_wts = copy.deepcopy(model.state_dict())

# ========================= MODEL SAVING =========================

# Load the best model weights found during training
model.load_state_dict(best_model_wts)

# Display final training results
print(f"Best ANN Accuracy Achieved: {best_acc:.2f}%")

# Save model architecture weights for deployment
torch.save(model.state_dict(), '../Models/ann_model.pth')

# Save accuracy metric for display in web application
pickle.dump(best_acc, open('../Models/ann_accuracy.pkl', 'wb'))

# ========================= SHAP EXPLAINABILITY (KERNEL EXPLAINER) =========================

# KernelExplainer works with any black-box model, including PyTorch ANNs.
# It explains predictions by approximating Shapley values around a background dataset.
model.eval()

def predict_proba_fn(data):
    """
    Wrapper for SHAP that returns class probabilities from the trained ANN.

    Args:
        data: NumPy array of shape (n_samples, n_features)

    Returns:
        np.ndarray: Predicted probabilities of shape (n_samples, 3)
    """
    data_t = torch.FloatTensor(data)
    with torch.no_grad():
        logits = model(data_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs

# Use a representative subset as SHAP background to keep KernelExplainer tractable.
rng = np.random.default_rng(42)
background_size = min(100, X_train.shape[0])
background_idx = rng.choice(X_train.shape[0], size=background_size, replace=False)
background_data = X_train[background_idx]

# Explain a subset of test samples (can be increased if needed, but this is faster).
explain_size = min(50, X_test.shape[0])
explain_idx = rng.choice(X_test.shape[0], size=explain_size, replace=False)
X_explain = X_test[explain_idx]

explainer = shap.KernelExplainer(predict_proba_fn, background_data)
shap_values = explainer.shap_values(X_explain)

# Persist SHAP artifacts for later inspection/visualization in notebooks or app layers.
pickle.dump(shap_values, open('../Models/ann_shap_values.pkl', 'wb'))
pickle.dump(X_explain, open('../Models/ann_shap_samples.pkl', 'wb'))
pickle.dump(feature_names, open('../Models/ann_shap_feature_names.pkl', 'wb'))

# Build a simple global feature-importance file from mean absolute SHAP values.
if isinstance(shap_values, list):
    # Multi-class format in many SHAP versions: list[n_classes] of (n_samples, n_features)
    abs_vals = np.mean([np.abs(class_vals) for class_vals in shap_values], axis=0)
else:
    # Newer SHAP versions may return a single array with output dimension included.
    vals = np.array(shap_values)
    if vals.ndim == 3:
        abs_vals = np.mean(np.abs(vals), axis=2)
    else:
        abs_vals = np.abs(vals)

global_importance = np.mean(abs_vals, axis=0)
importance_df = pd.DataFrame({
    'feature': feature_names,
    'mean_abs_shap': global_importance
}).sort_values('mean_abs_shap', ascending=False)
importance_df.to_csv('../Models/ann_shap_importance.csv', index=False)

print("SHAP artifacts saved: ann_shap_values.pkl, ann_shap_samples.pkl, ann_shap_feature_names.pkl, ann_shap_importance.csv")
