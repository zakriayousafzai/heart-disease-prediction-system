"""
Data Preprocessing Script for Heart Disease Dataset

This script performs comprehensive data cleaning and preparation including:
- Exploratory data analysis (EDA)
- Categorical variable encoding
- Missing value imputation using K-Nearest Neighbors
- Data type optimization
- Export of cleaned dataset for downstream analysis

Input: ../Datasets/heart.csv (raw dataset)
Output: ../Datasets/heart_cleaned.csv (preprocessed dataset)
"""

# Import required libraries
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

# ========================= DATA LOADING =========================

# Load raw heart disease dataset
heart_df = pd.read_csv('../Datasets/heart.csv')

# ========================= EXPLORATORY DATA ANALYSIS =========================

# Display complete dataset
heart_df

# Display random sample of 5 rows for initial inspection
heart_df.sample(5)

# Display dataset structure: row count, column types, non-null counts, memory usage
heart_df.info()

# Generate statistical summary for numerical columns (mean, std, min, max, quartiles)
heart_df.describe()

# Generate comprehensive summary including categorical columns
heart_df.describe(include='all')

# ========================= DATA QUALITY ASSESSMENT =========================

# Count missing (null) values per column
heart_df.isnull().sum()

# Identify duplicate records to prevent data redundancy
heart_df.duplicated().sum()

# Count unique values per column to understand cardinality
heart_df.nunique()

# ========================= CATEGORICAL ENCODING =========================

# Identify columns with object (string) data type
cat_col = heart_df.select_dtypes(include=['object']).columns
cat_col

# Convert categorical variables to numeric using label encoding
# Encoding scheme:
#   Sex: M=0, F=1
#   ChestPainType: ATA=0, NAP=1, ASY=2, TA=3
#   RestingECG: Normal=0, ST=1, LVH=2
#   ExerciseAngina: N=0, Y=1
#   ST_Slope: Up=0, Flat=1, Down=2
for col in cat_col:
    print(col)
    # Display unique values and their assigned numeric codes
    print((heart_df[col].unique()), list(range(heart_df[col].nunique())))
    
    # Replace categorical values with sequential integers
    heart_df[col].replace(heart_df[col].unique(), list(range(heart_df[col].nunique())), inplace=True)
    print('*' * 90)
    print()

# Display transformed dataset
heart_df

# ========================= MISSING VALUE IMPUTATION =========================

# -------------------- Cholesterol Imputation --------------------
# Identify biologically impossible cholesterol values (0 mg/dl)
heart_df['Cholesterol'].value_counts()

# Replace zero values with NaN to mark as missing
heart_df['Cholesterol'].replace(0, np.nan, inplace=True)

# Initialize K-Nearest Neighbors imputer (k=3)
# KNN imputation estimates missing values based on 3 most similar patients
imputer = KNNImputer(n_neighbors=3)

# Apply imputation and reconstruct DataFrame
after_impute = imputer.fit_transform(heart_df)
heart_df = pd.DataFrame(after_impute, columns=heart_df.columns)

# Verify successful imputation (should return 0)
heart_df['Cholesterol'].isna().sum()

# -------------------- Resting Blood Pressure Imputation --------------------
# Identify biologically impossible resting BP values (0 mm Hg)
heart_df['RestingBP'][heart_df['RestingBP'] == 0]

# Replace zero values with NaN
heart_df['RestingBP'].replace(0, np.nan, inplace=True)

# Initialize second KNN imputer for blood pressure
imputer2 = KNNImputer(n_neighbors=3)

# Apply imputation and reconstruct DataFrame
after_impute2 = imputer2.fit_transform(heart_df)
heart_df = pd.DataFrame(after_impute2, columns=heart_df.columns)

# Verify successful imputation
heart_df['RestingBP'].isnull().sum()

# ========================= DATA TYPE OPTIMIZATION =========================

# After KNN imputation, all columns become float64
# Convert discrete variables back to integers for memory efficiency and semantic correctness

# Get all column names
without_oldPeak = heart_df.columns

# Exclude 'Oldpeak' column as it requires continuous decimal values
without_oldPeak = without_oldPeak.drop('Oldpeak')

# Convert all columns except Oldpeak to 32-bit integers
heart_df[without_oldPeak] = heart_df[without_oldPeak].astype('int32')

# Display final dataset structure
heart_df.info()

# ========================= EXPORT CLEANED DATA =========================

# Save preprocessed dataset for model training and analysis
heart_df.to_csv('../Datasets/heart_cleaned.csv', index=False)