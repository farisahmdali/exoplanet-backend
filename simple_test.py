"""
Simple test to check if the model works correctly
"""
import joblib
import pandas as pd
import numpy as np
import os

# Load the model
model_path = os.path.join('..', 'exoplannet', 'exo.joblib')

print("=" * 60)
print("SIMPLE MODEL TEST")
print("=" * 60)

if not os.path.exists(model_path):
    print(f"❌ Model not found: {model_path}")
    exit(1)

print(f"✓ Loading model from: {model_path}")
model = joblib.load(model_path)
print(f"✓ Model type: {type(model).__name__}")
print(f"✓ Has predict_proba: {hasattr(model, 'predict_proba')}")

# Feature order (18 features)
feature_order = [
    'koi_fpflag_ss', 'koi_fpflag_ec', 'koi_period', 'koi_time0bk',
    'koi_impact', 'koi_duration', 'koi_depth', 'koi_prad',
    'koi_teq', 'koi_insol', 'koi_model_snr', 'koi_tce_plnt_num',
    'koi_steff', 'koi_slogg', 'koi_srad', 'ra', 'dec', 'koi_kepmag'
]

print(f"\n{'=' * 60}")
print("TEST 1: CONFIRMED Example (from training data)")
print("=" * 60)

# Real confirmed exoplanet from training data (row 0 from the notebook)
confirmed = {
    'koi_fpflag_ss': 0,
    'koi_fpflag_ec': 0,
    'koi_period': 9.488036,
    'koi_time0bk': 170.538750,
    'koi_impact': 0.146,
    'koi_duration': 2.95750,
    'koi_depth': 615.8,
    'koi_prad': 2.26,
    'koi_teq': 793,
    'koi_insol': 93.59,
    'koi_model_snr': 35.8,
    'koi_tce_plnt_num': 1,
    'koi_steff': 5455,
    'koi_slogg': 4.467,
    'koi_srad': 0.927,
    'ra': 291.93423,
    'dec': 48.141651,
    'koi_kepmag': 15.347
}

X_confirmed = pd.DataFrame([confirmed], columns=feature_order)
pred = model.predict(X_confirmed)[0]
prob = model.predict_proba(X_confirmed)[0]

print(f"Prediction: {pred}")
print(f"Probabilities: {prob}")
print(f"Result: {('✓ CONFIRMED (0)' if pred == 0 else '✗ FALSE POSITIVE (1)')}")
print(f"Confidence: {prob[int(pred)] * 100:.2f}%")

print(f"\n{'=' * 60}")
print("TEST 2: FALSE POSITIVE Example (from training data)")
print("=" * 60)

# Real false positive from training data (row 3 from the notebook)
false_positive = {
    'koi_fpflag_ss': 1,  # Red flag!
    'koi_fpflag_ec': 0,
    'koi_period': 1.736952,
    'koi_time0bk': 169.046370,
    'koi_impact': 0.000,
    'koi_duration': 1.85750,
    'koi_depth': 1423.0,
    'koi_prad': 3.67,
    'koi_teq': 1707,
    'koi_insol': 1160.73,
    'koi_model_snr': 10.6,
    'koi_tce_plnt_num': 1,
    'koi_steff': 5853,
    'koi_slogg': 4.544,
    'koi_srad': 0.868,
    'ra': 297.00482,
    'dec': 48.134129,
    'koi_kepmag': 15.436
}

X_false = pd.DataFrame([false_positive], columns=feature_order)
pred = model.predict(X_false)[0]
prob = model.predict_proba(X_false)[0]

print(f"Prediction: {pred}")
print(f"Probabilities: {prob}")
print(f"Result: {('✗ CONFIRMED (0)' if pred == 0 else '✓ FALSE POSITIVE (1)')}")
print(f"Confidence: {prob[int(pred)] * 100:.2f}%")

print(f"\n{'=' * 60}")
print("TEST 3: Your Frontend Default Values")
print("=" * 60)

default = {
    'koi_fpflag_ss': 0,
    'koi_fpflag_ec': 0,
    'koi_period': 4.2,
    'koi_time0bk': 132.5,
    'koi_impact': 0.5,
    'koi_duration': 2.3,
    'koi_depth': 100,
    'koi_prad': 1.5,
    'koi_teq': 300,
    'koi_insol': 1.0,
    'koi_model_snr': 15,
    'koi_tce_plnt_num': 1,
    'koi_steff': 5500,
    'koi_slogg': 4.5,
    'koi_srad': 1.0,
    'ra': 285.0,
    'dec': 45.0,
    'koi_kepmag': 14.0
}

X_default = pd.DataFrame([default], columns=feature_order)
pred = model.predict(X_default)[0]
prob = model.predict_proba(X_default)[0]

print(f"Prediction: {pred}")
print(f"Probabilities: {prob}")
print(f"Result: {('CONFIRMED (0)' if pred == 0 else 'FALSE POSITIVE (1)')}")
print(f"Confidence: {prob[int(pred)] * 100:.2f}%")

print(f"\n{'=' * 60}")
print("SUMMARY")
print("=" * 60)

if pred == 0:
    print("⚠️  DEFAULT VALUES predict CONFIRMED")
    print("    This is why you're seeing FALSE POSITIVE for everything -")
    print("    the default values might need adjustment, or there's a mismatch")
    print("    between training data distribution and input values.")
else:
    print("✓  DEFAULT VALUES predict FALSE POSITIVE")
    print("    This is expected for generic/default values.")

print(f"\n{'=' * 60}")
print("RECOMMENDATIONS")
print("=" * 60)
print("1. Use TEST 1 and TEST 2 values in your frontend")
print("2. These are from actual training data and should work correctly")
print("3. Try other real exoplanet parameters from NASA archives")

