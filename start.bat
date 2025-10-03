@echo off
echo ============================================
echo Starting Exoplanet Detection Backend
echo ============================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing/updating dependencies...
pip install -q -r requirements.txt

echo.
echo ============================================
echo Starting Flask Server on http://localhost:5000
echo ============================================
echo.

python app.py

