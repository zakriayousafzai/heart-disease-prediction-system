"""
Exploratory Data Visualization Script for Heart Disease Dataset

This script generates comprehensive visualizations to understand:
- Feature distributions and their relationship with heart disease
- Correlation patterns between clinical variables and disease presence
- Class balance and demographic distributions
- Statistical relationships using multiple chart types

All visualizations use Plotly Express for interactive exploration.

Input: ../Datasets/heart_cleaned.csv (preprocessed dataset)
Output: Interactive visualizations displayed in browser/notebook
"""

# Import required libraries
import plotly.express as px
import pandas as pd

# ========================= DATA LOADING =========================

# Load preprocessed heart disease dataset
heart_df = pd.read_csv('../Datasets/heart_cleaned.csv')

# ========================= INITIAL DATA INSPECTION =========================

# Display random sample to verify data loading
heart_df.sample()

# ========================= CORRELATION ANALYSIS =========================

# Compute Pearson correlation matrix for all numerical features
heart_df.corr()

# Extract and sort correlations with target variable (HeartDisease)
# Excluding the target's self-correlation (last element)
heart_df.corr()["HeartDisease"][:-1].sort_values()

# Visualize feature correlations with heart disease using line plot
# Helps identify strongest positive and negative predictors
px.line(heart_df.corr()["HeartDisease"][:-1].sort_values())

# ========================= DEMOGRAPHIC VISUALIZATIONS =========================

# -------------------- Age Distribution --------------------
# Hierarchical sunburst chart: HeartDisease status -> Age groups
# Shows age distribution within disease-positive and disease-negative groups
px.sunburst(heart_df, path=["HeartDisease", "Age"])

# Histogram showing age distribution colored by heart disease status
# Reveals age-related risk patterns
px.histogram(heart_df, x="Age", color="HeartDisease")

# -------------------- Sex Distribution --------------------
# Histogram showing sex distribution (0=Male, 1=Female) colored by disease status
# Identifies gender-based disease prevalence
px.histogram(heart_df, x="Sex", color="HeartDisease")

# ========================= TARGET VARIABLE DISTRIBUTION =========================

# Pie chart showing class balance in dataset
# Critical for understanding potential class imbalance issues
px.pie(heart_df, names='HeartDisease', title='Heart Disease Distribution')

# ========================= CLINICAL FEATURE VISUALIZATIONS =========================

# -------------------- Chest Pain Type --------------------
# Histogram of chest pain categories (ATA=0, NAP=1, ASY=2, TA=3)
# Asymptomatic patients (ASY) often show high disease correlation
px.histogram(heart_df, x="ChestPainType", color="HeartDisease")

# -------------------- Resting Blood Pressure --------------------
# Hierarchical view of blood pressure distribution by disease status
px.sunburst(heart_df, path=["HeartDisease", "RestingBP"])

# -------------------- Fasting Blood Sugar --------------------
# Binary indicator: 0 = ≤120 mg/dl, 1 = >120 mg/dl
# Shows relationship between diabetes indicator and heart disease
px.histogram(heart_df, x="FastingBS", color="HeartDisease")

# -------------------- Maximum Heart Rate --------------------
# Hierarchical view: Shows how exercise capacity relates to disease
px.sunburst(heart_df, path=["HeartDisease", "MaxHR"])

# Violin plot showing MaxHR distribution and density by disease status
# Reveals central tendency, spread, and multimodal patterns
px.violin(heart_df, x='HeartDisease', y='MaxHR', color='HeartDisease')

# -------------------- ST Depression (Oldpeak) --------------------
# Violin plot of ST depression induced by exercise
# Higher values typically indicate more severe coronary issues
px.violin(heart_df, x='HeartDisease', y='Oldpeak', color='HeartDisease')

# -------------------- ST Slope --------------------
# Histogram of ST segment slope (Up=0, Flat=1, Down=2)
# Downsloping ST segments strongly indicate disease
px.histogram(heart_df, x="ST_Slope", color="HeartDisease")

# -------------------- Exercise-Induced Angina --------------------
# Binary indicator: 0 = No, 1 = Yes
# Strong predictor of coronary artery disease
px.histogram(heart_df, x="ExerciseAngina", color="HeartDisease")