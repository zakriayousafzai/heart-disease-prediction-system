"""
Invariant checks for the SHAP explainability output of /predict

The explainability fields are derived presentation values, and a subtle mistake in
them produces output that looks plausible but is misleading: an importance band that
inverts against the factor ordering, or a direction label that disagrees with the
underlying value. These checks assert the properties that must hold for every
response, over the whole held-out split, so such mistakes fail loudly instead of
reaching a patient.

Run from the backend directory:
    python test_shap_explanations.py

No test framework is required. The database is replaced with an in-memory SQLite
database and Gemini is disabled, so the script needs no external services. It exits
non-zero if any invariant fails.
"""

import os
import sys

# Must be set before importing app.py; load_dotenv() does not override these.
os.environ['API_KEY'] = ''
os.environ['DB_URI'] = 'sqlite:///:memory:'

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(
    os.path.dirname(BACKEND_DIR), 'model_development', 'Datasets', 'heart_cleaned.csv'
)

sys.path.insert(0, BACKEND_DIR)
os.chdir(BACKEND_DIR)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import app as api

# Severity ordering, so bands can be checked for monotonicity down a list.
BAND_ORDER = {'high': 2, 'medium': 1, 'low': 0}

# Additivity is approximate: KernelSHAP solves a least-squares problem from a finite
# number of permutations, so a small residual is expected.
PP_TOLERANCE = 0.5

failures = []


def check(condition, message):
    if not condition:
        failures.append(message)


class _FakeSession:
    """Stands in for the SQLAlchemy session so the route runs without PostgreSQL."""

    def add(self, obj):
        obj.id = 1

    def commit(self):
        pass

    def remove(self):
        pass


def load_holdout():
    """Rebuilds the held-out split the ANN was evaluated on."""
    df = pd.read_csv(DATASET_PATH)

    def assign_risk(row):
        if row['HeartDisease'] == 0:
            return 0
        if (row['Oldpeak'] > 2.0) or (row['ST_Slope'] == 2) or (row['MaxHR'] < 120):
            return 2
        return 1

    df['RiskCategory'] = df.apply(assign_risk, axis=1)
    features = df.drop(['HeartDisease', 'RiskCategory'], axis=1)
    _, X_test, _, _ = train_test_split(
        features.values, df['RiskCategory'].values,
        test_size=0.2, stratify=df['RiskCategory'].values, random_state=42
    )
    return X_test


def to_payload(row):
    """Maps an encoded dataset row back into the API's input schema."""
    return {
        'age': int(row[0]),
        'sex': 'Male' if row[1] == 0 else 'Female',
        'chest_pain_type': ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain',
                            'Asymptomatic'][int(row[2])],
        'resting_bp': int(row[3]),
        'cholesterol': int(row[4]),
        'fasting_bs': '> 120 mg/dl' if row[5] == 1 else '<= 120 mg/dl',
        'resting_ecg': ['Normal', 'ST-T wave Abnormality',
                        'Left Ventricular Hypertrophy'][int(row[6])],
        'max_hr': int(row[7]),
        'exercise_angina': 'Yes' if row[8] == 1 else 'No',
        'oldpeak': float(row[9]),
        'st_slope': ['Upsloping', 'Flat', 'Downsloping'][int(row[10])],
    }


def main():
    assert api.ann_explainer is not None, 'SHAP explainer failed to initialise'
    assert api.SHAP_EXPECTED_VALUE is not None, 'baseline metadata artifact not loaded'

    api.db.session = _FakeSession()
    client = api.app.test_client()

    X_test = load_holdout()
    band_counts = {'high': 0, 'medium': 0, 'low': 0}
    worst_residual = 0.0
    n_factors = 0

    for row in X_test:
        body = client.post('/predict', json=to_payload(row)).get_json()
        factors = body.get('risk_factors') or []
        if not factors:
            failures.append('no risk_factors returned')
            continue
        n_factors += len(factors)
        probability = body['ann_prediction']['probability']

        # 1. Bands must be non-increasing down the list. Because factors are ordered by
        #    |SHAP| and the band is a monotonic function of |SHAP| relative to the
        #    strongest factor, any increase means the banding scale is inconsistent.
        bands = [BAND_ORDER[f['impact']] for f in factors]
        if bands != sorted(bands, reverse=True):
            failures.append(
                'bands not monotonic: %s' % [f['impact'] for f in factors]
            )

        # 2. The strongest factor is the reference point for the relative scale.
        check(abs(factors[0]['relative_importance'] - 1.0) < 1e-6,
              'top factor relative_importance != 1.0')

        for i, f in enumerate(factors):
            band_counts[f['impact']] += 1

            # 3. relative_importance must decrease down the list.
            if i and f['relative_importance'] > factors[i - 1]['relative_importance'] + 1e-9:
                failures.append('relative_importance increased down the list')

            # 4. The signed contribution must agree with the stated direction.
            if f['contribution_pp'] > 0 and f['direction'] != 'increases risk':
                failures.append('positive contribution labelled %r' % f['direction'])
            if f['contribution_pp'] < 0 and f['direction'] != 'decreases risk':
                failures.append('negative contribution labelled %r' % f['direction'])

            # 5. The label must match its magnitude band and direction.
            expected = api.IMPACT_LABELS[(f['impact'], f['direction'])]
            if f['impact_label'] != expected:
                failures.append('impact_label %r != %r' % (f['impact_label'], expected))

        # 6. SHAP is additive, so baseline + the shown factors + the residual must
        #    reconcile with the reported probability.
        if 'baseline_probability' in body and 'other_factors_pp' in body:
            total = (body['baseline_probability']
                     + sum(f['contribution_pp'] for f in factors)
                     + body['other_factors_pp'])
            residual = abs(total - probability)
            worst_residual = max(worst_residual, residual)
            check(residual <= PP_TOLERANCE,
                  'additivity residual %.2f pp exceeds tolerance' % residual)

    print('Checked %d patients, %d risk factors' % (len(X_test), n_factors))
    print('Band distribution: high %d, medium %d, low %d'
          % (band_counts['high'], band_counts['medium'], band_counts['low']))
    print('Worst additivity residual: %.2f pp (tolerance %.1f)'
          % (worst_residual, PP_TOLERANCE))

    if failures:
        print('\nFAILED (%d):' % len(failures))
        for f in failures[:20]:
            print('  - %s' % f)
        return 1

    print('\nAll invariants hold.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
