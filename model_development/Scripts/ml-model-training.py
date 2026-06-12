"""
Traditional Machine Learning Model Training Script for Binary Heart Disease Classification

This script trains and optimizes two classical machine learning models:
1. Logistic Regression - Linear probabilistic classifier
2. Random Forest - Ensemble of decision trees

Both models perform binary classification (Heart Disease: Yes/No) and are optimized
through hyperparameter tuning to achieve maximum predictive accuracy.
"""

# Import required libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# ========================= DATA LOADING AND SPLITTING =========================

# Load preprocessed heart disease dataset
heart_df = pd.read_csv('../Datasets/heart_cleaned.csv')

# Split dataset into training (80%) and testing (20%) sets
# Features (X): All clinical variables except target
# Target (y): Binary heart disease indicator (0 = No Disease, 1 = Disease)
X_train, X_test, y_train, y_test = train_test_split(
    heart_df.drop('HeartDisease', axis=1),  # Feature matrix (all columns except target)
    heart_df['HeartDisease'],                # Target variable (binary classification)
    test_size=0.2,                           # Reserve 20% of data for testing
    random_state=42,                         # Seed for reproducible splits
    stratify=heart_df['HeartDisease']        # Maintain class distribution in both sets
)

# ========================= LOGISTIC REGRESSION TRAINING =========================

# Logistic Regression: Linear model that estimates probability of binary outcomes
# Uses sigmoid function to map linear combination of features to [0,1] probability range

# -------------------- Solver Optimization --------------------
# Different optimization algorithms have varying convergence properties
# Test all available solvers to find the most effective for this dataset
solver = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
best_solver = ''
test_score = np.zeros(6)

for i, n in enumerate(solver):
    # Initialize model with current solver
    log_reg = LogisticRegression(solver=n, max_iter=1000)
    
    # Train model on training data
    log_reg.fit(X_train, y_train)
    
    # Evaluate performance on held-out test set
    test_score[i] = log_reg.score(X_test, y_test)
    
    # Track solver with highest test accuracy
    if log_reg.score(X_test, y_test) == test_score.max():
        best_solver = n

# -------------------- Final Model Training --------------------
# Train final logistic regression model using optimal solver
log_reg = LogisticRegression(solver=best_solver)
log_reg.fit(X_train, y_train)

# Generate predictions on test set
log_reg_pred = log_reg.predict(X_test)

# Calculate and convert accuracy to percentage
lr_acc = accuracy_score(y_test, log_reg_pred) * 100

# -------------------- Model Persistence --------------------
# Save trained model and accuracy for deployment
pickle.dump(lr_acc, open('../Models/lr_accuracy.pkl', 'wb'))
pickle.dump(log_reg, open('../Models/log_reg_model.pkl', 'wb'))

# Display training results
print(f"Best Logistic Regression Solver: {best_solver}")
print(f'Logistic Regression Accuracy: {accuracy_score(y_test, log_reg_pred):.4f}')
print(f"Saved LR Accuracy: {lr_acc:.2f}%")

# ========================= RANDOM FOREST TRAINING =========================

# Random Forest: Ensemble learning method using multiple decision trees
# Combines predictions from many trees to reduce overfitting and improve accuracy
# Each tree is trained on random subset of data and features (bagging + feature randomness)

# -------------------- Hyperparameter Grid Search --------------------
# Define hyperparameter search space for optimization
param_grid = {
    'n_estimators': [50, 100, 150, 500],      # Number of trees in the forest
    'max_depth': [3, 6, 9, 19],               # Maximum depth of each tree
    'max_features': ['sqrt', 'log2', None],   # Number of features considered per split
    'max_leaf_nodes': [3, 6, 9]               # Maximum number of leaf nodes per tree
}

# Initialize base Random Forest classifier
rfc = RandomForestClassifier()

# GridSearchCV: Exhaustive search over parameter combinations with cross-validation
# Automatically finds optimal hyperparameter configuration
grid_search = GridSearchCV(rfc, param_grid)
grid_search.fit(X_train, y_train)

# -------------------- Final Model Training --------------------
# Train final model using best hyperparameters discovered by grid search
rfctree = RandomForestClassifier(**grid_search.best_params_)
rfctree.fit(X_train, y_train)

# Generate predictions on test set
rfc_pred = rfctree.predict(X_test)

# Calculate and convert accuracy to percentage
rf_acc = accuracy_score(y_test, rfc_pred) * 100

# -------------------- Model Persistence --------------------
# Save trained model and accuracy for deployment
pickle.dump(rf_acc, open('../Models/rf_accuracy.pkl', 'wb'))
pickle.dump(rfctree, open('../Models/rfc_model.pkl', 'wb'))

# Display training results
print(f'Random Forest Accuracy: {accuracy_score(y_test, rfc_pred):.4f}')
print(f"Saved RF Accuracy: {rf_acc:.2f}%")
