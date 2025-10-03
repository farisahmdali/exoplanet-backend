#!/bin/bash

echo "============================================"
echo "Starting Exoplanet Detection Backend"
echo "============================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing/updating dependencies..."
pip install -q -r requirements.txt

echo ""
echo "============================================"
echo "Starting Flask Server on http://localhost:5000"
echo "============================================"
echo ""

python app.py

