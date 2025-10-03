# 🔧 Quick Fix for Installation Error

## The Problem
You're getting a build error when trying to install packages. This is common on Windows.

## ✅ Quick Solution (Recommended)

### Step 1: Install Minimal Requirements
```bash
cd exoplanet-backend

# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install minimal requirements (no LightGBM)
pip install -r requirements-minimal.txt
```

### Step 2: Check Your Model Type
```bash
python
```

Then in Python:
```python
import joblib
model = joblib.load('../exoplannet/exo.joblib')
print(type(model))
```

**If you see `sklearn.ensemble` → You're good! Don't need LightGBM**
**If you see `lightgbm` → You need to install LightGBM (see below)**

### Step 3: Start the Server
```bash
python app.py
```

Server will start at http://localhost:5000

## 🎯 If You Need LightGBM

### Option A: Use Conda (Easiest)
```bash
# Install Miniconda first from: https://docs.conda.io/en/latest/miniconda.html

conda create -n exoplanet python=3.10
conda activate exoplanet
conda install -c conda-forge lightgbm flask flask-cors pandas numpy scikit-learn joblib
pip install werkzeug python-dotenv
```

### Option B: Pre-built Wheel
```bash
pip install --upgrade pip
pip install lightgbm --prefer-binary
```

### Option C: Install Build Tools
Download and install: https://visualstudio.microsoft.com/downloads/
- Choose "Build Tools for Visual Studio"
- Select "Desktop development with C++"
- Restart terminal and try again

## 🚀 Automated Installation

Run the install script:
```bash
install.bat
```

This will handle everything automatically.

## ✅ Verify Installation

```bash
python -c "import flask, pandas, numpy, sklearn; print('✓ Ready!')"
```

## 🆘 Still Having Issues?

1. **Use Python 3.10** (not 3.12)
   - Download from: https://www.python.org/downloads/release/python-31011/
   
2. **Try each package individually:**
   ```bash
   pip install Flask
   pip install Flask-CORS
   pip install pandas
   pip install numpy
   pip install scikit-learn
   pip install joblib
   ```

3. **Check your Python version:**
   ```bash
   python --version
   ```
   Should be 3.8 to 3.11 (avoid 3.12 for now)

## 💡 Pro Tip

The backend will work fine without LightGBM if your model is a scikit-learn model (RandomForest, etc.). Just use `requirements-minimal.txt`!

---

**Need more help?** See INSTALL_GUIDE.md for detailed solutions.

