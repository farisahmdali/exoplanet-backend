# Installation Troubleshooting Guide

## Error: pip subprocess failed / build dependencies error

This error typically occurs on Windows when trying to install packages with C extensions like `lightgbm`.

## Solutions (Try in order)

### Solution 1: Install without LightGBM (Recommended for Quick Start)

```bash
cd exoplanet-backend
pip install -r requirements-minimal.txt
```

This installs everything except LightGBM. Your model should still work if it's a scikit-learn model.

### Solution 2: Use Pre-built Wheels

```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install packages one by one
pip install Flask==3.0.0
pip install Flask-CORS==4.0.0
pip install pandas==2.1.3
pip install numpy==1.26.2
pip install scikit-learn==1.3.2
pip install joblib==1.3.2

# Try LightGBM from conda-forge or skip it
pip install lightgbm --prefer-binary
```

### Solution 3: Install Visual C++ Build Tools (Windows)

LightGBM requires Microsoft Visual C++ 14.0 or greater.

1. Download **Build Tools for Visual Studio**: https://visualstudio.microsoft.com/downloads/
2. Select "C++ build tools" during installation
3. Restart your terminal
4. Try installing again:
   ```bash
   pip install -r requirements.txt
   ```

### Solution 4: Use Conda (Alternative)

```bash
# Install Miniconda from: https://docs.conda.io/en/latest/miniconda.html

# Create conda environment
conda create -n exoplanet python=3.10
conda activate exoplanet

# Install packages
conda install -c conda-forge flask flask-cors pandas numpy scikit-learn joblib lightgbm
pip install werkzeug python-dotenv
```

### Solution 5: Use Python 3.10 (If on 3.12+)

Some packages don't have wheels for Python 3.12 yet.

```bash
# Check your Python version
python --version

# If Python 3.12+, install Python 3.10:
# Download from: https://www.python.org/downloads/

# Then create venv with Python 3.10
py -3.10 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Test Without LightGBM

If your model (`exo.joblib`) was trained with scikit-learn RandomForest, you don't need LightGBM:

```bash
# Install minimal requirements
pip install -r requirements-minimal.txt

# Start the server
python app.py
```

## Check What Model You Have

```python
import joblib

# Load your model
model = joblib.load('../exoplannet/exo.joblib')

# Check model type
print(type(model))
# If it shows sklearn, you don't need lightgbm
# If it shows lightgbm, you need to install it
```

## Still Having Issues?

### Option A: Skip the problematic package
Comment out `lightgbm==4.1.0` in `requirements.txt` and install the rest.

### Option B: Use Docker (Advanced)
```bash
# Use Python slim image that has build tools
docker run -it python:3.10-slim bash
```

### Option C: Use WSL (Windows Subsystem for Linux)
Install Ubuntu from Microsoft Store, then use Linux commands.

## Verify Installation

```bash
python -c "import flask, pandas, numpy, sklearn, joblib; print('All core packages installed!')"
```

## Common Error Messages

### "Microsoft Visual C++ 14.0 or greater is required"
→ Install Visual C++ Build Tools (Solution 3)

### "Could not build wheels for lightgbm"
→ Use requirements-minimal.txt (Solution 1)

### "No module named 'lightgbm'"
→ Model uses LightGBM, install it via conda or build tools

### "ERROR: Failed building wheel for numpy"
→ Upgrade pip: `python -m pip install --upgrade pip`

## Need More Help?

1. Check your Python version: `python --version`
2. Check your pip version: `pip --version`
3. Check installed packages: `pip list`
4. Try installing packages individually to isolate the problem

## Recommended Setup for Windows

```bash
# 1. Use Python 3.10 (most compatible)
# 2. Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# 3. Install minimal requirements
pip install -r requirements-minimal.txt

# 4. Start server
python app.py
```

This should get you running! The backend will work with or without LightGBM depending on your model type.

