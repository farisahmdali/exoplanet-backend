"""
Debug script to check the model type and test predictions
"""
import joblib
import pandas as pd
import numpy as np
import os

# Load the model
model_path = os.path.join('..', 'exoplannet', 'exo.joblib')

print("=" * 60)
print("Model Debug Information")
print("=" * 60)

if os.path.exists(model_path):
    print(f"✓ Model file found: {model_path}")
    model = joblib.load(model_path)
    print(f"✓ Model loaded successfully")
    print(f"\nModel Type: {type(model).__name__}")
    print(f"Model Class: {type(model)}")
    
    # Check if it has predict_proba
    if hasattr(model, 'predict_proba'):
        print("✓ Model has predict_proba method")
    
    # Test with sample data (CONFIRMED case)
    print("\n" + "=" * 60)
    print("Test Case 1: Should be CONFIRMED")
    print("=" * 60)
    
    confirmed_params = {
        'koi_fpflag_ss': 0,
        'koi_fpflag_ec': 0,
        'koi_period': 289.9,
        'koi_time0bk': 131.51,
        'koi_impact': 0.72,
        'koi_duration': 5.6,
        'koi_depth': 492,
        'koi_prad': 2.4,
        'koi_teq': 262,
        'koi_insol': 1.11,
        'koi_model_snr': 47.3,
        'koi_tce_plnt_num': 1,
        'koi_steff': 5518,
        'koi_slogg': 4.45,
        'koi_srad': 0.97,
        'ra': 290.4,
        'dec': 47.9,
        'koi_kepmag': 11.7
    }
    
    feature_order = [
        'koi_fpflag_ss', 'koi_fpflag_ec', 'koi_period', 'koi_time0bk',
        'koi_impact', 'koi_duration', 'koi_depth', 'koi_prad',
        'koi_teq', 'koi_insol', 'koi_model_snr', 'koi_tce_plnt_num',
        'koi_steff', 'koi_slogg', 'koi_srad', 'ra', 'dec', 'koi_kepmag'
    ]
    
    X_confirmed = pd.DataFrame([confirmed_params], columns=feature_order)
    
    pred_confirmed = model.predict(X_confirmed)[0]
    if hasattr(model, 'predict_proba'):
        prob_confirmed = model.predict_proba(X_confirmed)[0]
        print(f"Prediction: {pred_confirmed} (0=CONFIRMED, 1=FALSE POSITIVE)")
        print(f"Probabilities: [CONFIRMED: {prob_confirmed[0]:.4f}, FALSE POSITIVE: {prob_confirmed[1]:.4f}]")
        print(f"Result: {'✓ CONFIRMED' if pred_confirmed == 0 else '✗ FALSE POSITIVE'}")
    
    # Test with sample data (FALSE POSITIVE case)
    print("\n" + "=" * 60)
    print("Test Case 2: Should be FALSE POSITIVE")
    print("=" * 60)
    
    false_positive_params = {
        'koi_fpflag_ss': 1,  # Red flag!
        'koi_fpflag_ec': 0,
        'koi_period': 0.85,
        'koi_time0bk': 120.33,
        'koi_impact': 0.95,
        'koi_duration': 1.2,
        'koi_depth': 18500,
        'koi_prad': 15.2,
        'koi_teq': 2100,
        'koi_insol': 856.4,
        'koi_model_snr': 8.3,
        'koi_tce_plnt_num': 1,
        'koi_steff': 4200,
        'koi_slogg': 4.65,
        'koi_srad': 0.82,
        'ra': 195.3,
        'dec': 38.2,
        'koi_kepmag': 15.8
    }
    
    X_false = pd.DataFrame([false_positive_params], columns=feature_order)
    
    pred_false = model.predict(X_false)[0]
    if hasattr(model, 'predict_proba'):
        prob_false = model.predict_proba(X_false)[0]
        print(f"Prediction: {pred_false} (0=CONFIRMED, 1=FALSE POSITIVE)")
        print(f"Probabilities: [CONFIRMED: {prob_false[0]:.4f}, FALSE POSITIVE: {prob_false[1]:.4f}]")
        print(f"Result: {'✗ CONFIRMED' if pred_false == 0 else '✓ FALSE POSITIVE'}")
    
    # Test with default parameters from frontend
    print("\n" + "=" * 60)
    print("Test Case 3: Default Frontend Values")
    print("=" * 60)
    
    default_params = {
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
    
    X_default = pd.DataFrame([default_params], columns=feature_order)
    
    pred_default = model.predict(X_default)[0]
    if hasattr(model, 'predict_proba'):
        prob_default = model.predict_proba(X_default)[0]
        print(f"Prediction: {pred_default} (0=CONFIRMED, 1=FALSE POSITIVE)")
        print(f"Probabilities: [CONFIRMED: {prob_default[0]:.4f}, FALSE POSITIVE: {prob_default[1]:.4f}]")
        print(f"Result: {'CONFIRMED' if pred_default == 0 else 'FALSE POSITIVE'}")
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Model is working: {'✓ Yes' if pred_confirmed == 0 and pred_false == 1 else '✗ No - Check model'}")
    print(f"\nNote: If RandomForest or LogisticRegression, features may need scaling")
    print(f"      LightGBM models don't require scaling")
    
else:
    print(f"✗ Model file not found: {model_path}")
    print("Make sure exo.joblib is in the ../exoplannet/ directory")

