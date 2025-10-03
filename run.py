#!/usr/bin/env python
"""
Quick start script for the Exoplanet Detection API
"""
import os
import sys

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import flask
        import flask_cors
        import pandas
        import numpy
        import sklearn
        import joblib
        import lightgbm
        print("✓ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nPlease install dependencies with:")
        print("  pip install -r requirements.txt")
        return False

def check_model():
    """Check if the model file exists"""
    model_path = os.path.join('..', 'exoplannet', 'exo.joblib')
    if os.path.exists(model_path):
        print(f"✓ Model file found at: {model_path}")
        return True
    else:
        print(f"✗ Model file not found at: {model_path}")
        print("\nPlease ensure 'exo.joblib' is in the '../exoplannet/' directory")
        return False

def main():
    print("=" * 60)
    print("Exoplanet Detection API - Startup Check")
    print("=" * 60)
    print()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check model
    if not check_model():
        print("\nWarning: API will start but predictions may not work without the model.")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    print()
    print("=" * 60)
    print("Starting Flask Server...")
    print("=" * 60)
    print()
    print("API will be available at: http://localhost:5000")
    print("API Documentation: http://localhost:5000/api/health")
    print()
    print("Press Ctrl+C to stop the server")
    print()
    
    # Import and run the app
    from app import app
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    main()

