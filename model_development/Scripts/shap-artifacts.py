"""
SHAP Artifact Generation Script for the ANN Heart Disease Risk Model

Regenerates the explainability artifacts consumed by the backend API **without
retraining**. The already-trained ANN weights and the fitted StandardScaler are
loaded from disk, so model predictions and the published accuracy figures stay
exactly as they are.

Shapley values are only interpretable relative to the distribution they are
explained against, so the background used to derive the percentiles below must
be the same background the API explains against at request time. This script
therefore writes that background to its own file rather than overloading the
explained-samples file.

Artifacts written to ../Models/:
    ann_shap_background.pkl   - Scaled training rows used as the SHAP background
                                (the reference distribution for all attributions)
    ann_shap_thresholds.json  - Feature names plus per-feature, per-class percentiles
                                of |SHAP| used to band impact as high/medium/low
    ann_shap_values.pkl       - Raw SHAP values for the explained samples
    ann_shap_samples.pkl      - The scaled rows those SHAP values were computed on
    ann_shap_feature_names.pkl - Feature name list, kept for backwards compatibility
    ann_shap_importance.csv   - Global mean |SHAP| per feature, for offline plots

Usage (from model_development/Scripts):
    python shap-artifacts.py
"""

# Import required libraries
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import shap
from sklearn.model_selection import train_test_split

# Paths for the trained model, the fitted scaler, and the source dataset
MODEL_WEIGHTS_PATH = '../Models/ann_model.pth'
SCALER_PATH = '../Models/scaler.pkl'
DATASET_PATH = '../Datasets/heart_cleaned.csv'

# Size of the SHAP background and of the explained sample set
BACKGROUND_SIZE = 100
EXPLAIN_SIZE = 100
N_CLASSES = 3

# Percentiles of |SHAP| persisted per feature and used to band impact at request
# time. p90 marks a genuinely dominant factor for that feature, p50 a typical one.
IMPACT_PERCENTILES = {'high': 90.0, 'medium': 50.0}


# ========================= MODEL DEFINITION =========================
# Must stay structurally identical to HeartANN in ann-model-training.py and in
# backend/app.py, otherwise the saved state dict will not load.

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


# ========================= LOAD TRAINED MODEL =========================

print("Loading trained ANN and fitted scaler...")
model = HeartANN(11)
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))
model.eval()
scaler = pickle.load(open(SCALER_PATH, 'rb'))


def predict_proba_fn(data):
    """
    Black-box adapter handed to SHAP: converts the model's logits into the
    class probabilities that get explained.

    Args:
        data: NumPy array of shape (n_samples, n_features)

    Returns:
        np.ndarray: Predicted probabilities of shape (n_samples, 3)
    """
    with torch.no_grad():
        logits = model(torch.FloatTensor(data))
        return F.softmax(logits, dim=1).cpu().numpy()


# ========================= REBUILD THE TRAINING SPLIT =========================
# The background must be drawn from the same population the scaler was fitted on,
# so the split performed during training is replayed here deterministically.

heart_df = pd.read_csv(DATASET_PATH)


def assign_risk(row):
    """
    Assigns risk category based on heart disease status and clinical severity.

    Risk Categories:
        - 0 (Low Risk): No heart disease detected
        - 1 (Medium Risk): Heart disease present with moderate symptoms
        - 2 (High Risk): Heart disease with severe indicators
    """
    if row['HeartDisease'] == 0:
        return 0
    elif (row['Oldpeak'] > 2.0) or (row['ST_Slope'] == 2) or (row['MaxHR'] < 120):
        return 2
    else:
        return 1


heart_df['RiskCategory'] = heart_df.apply(assign_risk, axis=1)

feature_names = heart_df.drop(['HeartDisease', 'RiskCategory'], axis=1).columns.tolist()
X = heart_df[feature_names].values
y = heart_df['RiskCategory'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -------------------- Verify the reconstruction matches the scaler --------------------
# StandardScaler fitted on the training split reproduces its own mean_/scale_, so this
# is a cheap guard against the dataset or split having changed since training. Without
# it a silently mismatched background would corrupt every attribution.
if not (np.allclose(scaler.mean_, X_train.mean(axis=0), rtol=1e-6, atol=1e-8)
        and np.allclose(scaler.scale_, X_train.std(axis=0), rtol=1e-6, atol=1e-8)):
    raise SystemExit(
        'Reconstructed training split does not match the fitted scaler. The dataset or '
        'the split in ann-model-training.py has changed since the model was trained. '
        'Retrain the ANN before regenerating SHAP artifacts.'
    )
print('Verified: reconstructed training split matches the fitted scaler.')

X_train_scaled = scaler.transform(X_train)

# ========================= SHAP BACKGROUND =========================
# Drawn from the scaled training rows. Only the first draw is taken from the
# generator so the selected indices match ann-model-training.py.
rng = np.random.default_rng(42)
background_size = min(BACKGROUND_SIZE, X_train_scaled.shape[0])
background_idx = rng.choice(X_train_scaled.shape[0], size=background_size, replace=False)
background_data = X_train_scaled[background_idx]

# ========================= COMPUTE SHAP VALUES =========================
# Explained on scaled training rows: the percentiles derived from them are a
# property of the model over its own training distribution, which is exactly the
# reference the API's attributions are measured against.
explainer = shap.KernelExplainer(predict_proba_fn, background_data)
explain_size = min(EXPLAIN_SIZE, X_train_scaled.shape[0])
explain_idx = rng.choice(X_train_scaled.shape[0], size=explain_size, replace=False)
X_explain = X_train_scaled[explain_idx]

shap_values = explainer.shap_values(X_explain)


def as_class_arrays(values, n_samples, n_features, n_classes):
    """
    Normalise SHAP output into one (n_samples, n_features) array per class.

    SHAP's layout for a multi-output model changed between releases: older
    versions return a list of per-class arrays, newer versions a single array of
    shape (n_samples, n_features, n_classes). Both are accepted.

    Args:
        values: Raw output of KernelExplainer.shap_values
        n_samples: Number of explained rows
        n_features: Number of input features
        n_classes: Number of model outputs

    Returns:
        list: n_classes arrays each of shape (n_samples, n_features)
    """
    arr = np.asarray(values, dtype=float)

    if arr.ndim == 3:
        if arr.shape == (n_samples, n_features, n_classes):
            return [arr[:, :, c] for c in range(n_classes)]
        if arr.shape[0] == n_classes:
            return [arr[c] for c in range(n_classes)]

    if arr.ndim == 2 and arr.shape == (n_samples, n_features):
        return [arr]

    raise ValueError(f'Unexpected SHAP output shape: {arr.shape}')


per_class = as_class_arrays(shap_values, X_explain.shape[0], len(feature_names), N_CLASSES)

# Mean model output per class over the background. This is the reference point
# that SHAP attributions are measured from.
expected_value = predict_proba_fn(background_data).mean(axis=0)

# ========================= BUILD CALIBRATION THRESHOLDS =========================
# Per feature and per class percentiles of |SHAP|. At request time a factor is
# banded by comparing its |SHAP| against the percentiles for that same feature and
# class, so the bands stay meaningful across retrainings instead of relying on
# hardcoded absolute cut-offs.

thresholds = {
    'feature_names': feature_names,
    'n_classes': N_CLASSES,
    'percentiles': IMPACT_PERCENTILES,
    'n_background': int(background_data.shape[0]),
    'n_explained': int(X_explain.shape[0]),
    'shap_version': shap.__version__,
    'expected_value': {str(c): round(float(expected_value[c]), 6) for c in range(N_CLASSES)},
    'per_feature': {},
}

abs_values = np.stack([np.abs(v) for v in per_class], axis=0)  # (n_classes, n_samples, n_features)

for j, name in enumerate(feature_names):
    thresholds['per_feature'][name] = {
        str(c): {
            'p50': round(float(np.percentile(abs_values[c, :, j], 50)), 6),
            'p75': round(float(np.percentile(abs_values[c, :, j], 75)), 6),
            'p90': round(float(np.percentile(abs_values[c, :, j], 90)), 6),
        }
        for c in range(N_CLASSES)
    }

# ========================= PERSIST ARTIFACTS =========================

with open('../Models/ann_shap_background.pkl', 'wb') as f:
    pickle.dump(background_data, f)

with open('../Models/ann_shap_thresholds.json', 'w') as f:
    json.dump(thresholds, f, indent=2)

pickle.dump(shap_values, open('../Models/ann_shap_values.pkl', 'wb'))
pickle.dump(X_explain, open('../Models/ann_shap_samples.pkl', 'wb'))
pickle.dump(feature_names, open('../Models/ann_shap_feature_names.pkl', 'wb'))

# Global feature importance from mean absolute SHAP values, averaged over classes.
global_importance = abs_values.mean(axis=(0, 1))
importance_df = pd.DataFrame({
    'feature': feature_names,
    'mean_abs_shap': global_importance
}).sort_values('mean_abs_shap', ascending=False)
importance_df.to_csv('../Models/ann_shap_importance.csv', index=False)

print(f'Background: {background_data.shape[0]} scaled training rows')
print(f'Explained:  {X_explain.shape[0]} scaled training rows')
print(f'Expected model output per class over background: {np.round(expected_value, 4)}')
print('\nGlobal mean |SHAP| per feature:')
for _, row in importance_df.iterrows():
    print(f"  {row['feature']:<16} {row['mean_abs_shap']:.5f}")
print('\nSHAP artifacts saved: ann_shap_background.pkl, ann_shap_thresholds.json, '
      'ann_shap_values.pkl, ann_shap_samples.pkl, ann_shap_feature_names.pkl, '
      'ann_shap_importance.csv')
