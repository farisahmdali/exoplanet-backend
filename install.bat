@echo off
echo ============================================
echo Exoplanet Backend - Easy Installation
echo ============================================
echo.

echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    echo Make sure Python is installed and in PATH
    pause
    exit /b 1
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo ============================================
echo Installing packages (minimal version)...
echo This skips LightGBM which often causes issues
echo ============================================
echo.

pip install -r requirements-minimal.txt

if errorlevel 1 (
    echo.
    echo Installation had some errors.
    echo Trying to install packages individually...
    echo.
    pip install Flask==3.0.0
    pip install Flask-CORS==4.0.0
    pip install pandas==2.1.3
    pip install numpy==1.26.2
    pip install scikit-learn==1.3.2
    pip install joblib==1.3.2
    pip install werkzeug==3.0.1
    pip install python-dotenv==1.0.0
)

echo.
echo ============================================
echo Installation Complete!
echo ============================================
echo.
echo To start the server, run:
echo   start.bat
echo.
echo Or manually:
echo   venv\Scripts\activate
echo   python app.py
echo.
pause

