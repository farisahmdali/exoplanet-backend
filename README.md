# Exoplanet Detection Backend API

Flask-based REST API for exoplanet detection using machine learning models.

## Features

- 🚀 RESTful API for exoplanet predictions
- 🧠 Support for pretrained ML models (LightGBM, Random Forest)
- 📊 CSV data file upload and processing
- 🔧 Custom model training with hyperparameter tuning
- 🔐 CORS enabled for frontend integration
- 📝 Comprehensive logging and error handling

## Installation

### 1. Create Virtual Environment

```bash
cd exoplanet-backend
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

```bash
cp .env.example .env
```

Edit `.env` to configure your settings.

### 4. Place Model File

Ensure the trained model file `exo.joblib` is in the parent directory at:
```
../exoplannet/exo.joblib
```

Or update the `MODEL_PATH` in `.env` to point to your model location.

## Running the Server

### Development Mode

```bash
python app.py
```

The API will be available at `http://localhost:5000`

### Production Mode

```bash
# Using Gunicorn (recommended)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## API Endpoints

### Health Check
```http
GET /api/health
```

Returns server health status and information.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00",
  "models_loaded": 1
}
```

### Get Available Models
```http
GET /api/models
```

Returns list of available pretrained models.

**Response:**
```json
{
  "models": [
    {
      "id": "kepler-v2",
      "name": "Kepler Transit Model v2.0",
      "accuracy": "96.8%",
      "loaded": true
    }
  ]
}
```

### Make Predictions
```http
POST /api/predict
Content-Type: multipart/form-data
```

Upload a CSV file and get exoplanet predictions.

**Parameters:**
- `file`: CSV file containing light curve data
- `model_id`: (optional) Model ID to use for prediction

**Response:**
```json
{
  "confidence": {
    "overall": 94.7,
    "exoplanetDetected": true,
    "numberOfCandidates": 2,
    "processingTime": "1.2s"
  },
  "predictions": [
    {
      "id": 1,
      "transitTime": "2.456 days",
      "probability": 0.947,
      "planetRadius": "1.23 Earth radii",
      "orbitalPeriod": "4.2 days",
      "confidence": "High"
    }
  ]
}
```

### Train New Model
```http
POST /api/train
Content-Type: multipart/form-data
```

Train a new model with custom hyperparameters.

**Parameters:**
- `training_file`: CSV file with labeled training data
- `validation_file`: (optional) CSV file for validation
- `learning_rate`: Learning rate (default: 0.001)
- `batch_size`: Batch size (default: 32)
- `epochs`: Number of epochs (default: 100)
- `hidden_layers`: Number of hidden layers (default: 3)
- `neurons_per_layer`: Neurons per layer (default: 128)
- `dropout`: Dropout rate (default: 0.2)
- `optimizer`: Optimizer type (default: 'adam')
- `activation`: Activation function (default: 'relu')

**Response:**
```json
{
  "success": true,
  "accuracy": 96.8,
  "valAccuracy": 94.2,
  "trainingTime": "2h 34m",
  "modelSize": "45.2 MB"
}
```

## File Format

### Input CSV Format

The CSV file should contain Kepler exoplanet features:

```csv
koi_period,koi_time0bk,koi_impact,koi_duration,koi_depth,koi_prad,koi_teq,koi_insol,koi_model_snr,koi_tce_plnt_num,koi_steff,koi_slogg,koi_srad,ra,dec,koi_kepmag
```

Required columns may vary depending on your model.

## Project Structure

```
exoplanet-backend/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── .gitignore            # Git ignore file
├── README.md             # This file
├── uploads/              # Uploaded files (created automatically)
└── models/               # Saved models (created automatically)
```

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid input)
- `413`: File Too Large
- `500`: Internal Server Error

All errors include a JSON response with an `error` field describing the issue.

## CORS Configuration

CORS is enabled for all origins by default. For production, update the CORS configuration in `app.py`:

```python
CORS(app, resources={r"/api/*": {"origins": "https://yourdomain.com"}})
```

## Logging

Logs are output to console with INFO level. Configure logging in `app.py` for production use.

## Security Considerations

For production deployment:

1. Set `FLASK_DEBUG=False` in `.env`
2. Configure proper CORS origins
3. Add authentication/authorization
4. Use HTTPS
5. Implement rate limiting
6. Add input validation and sanitization
7. Use environment variables for sensitive data

## Contributing

1. Follow PEP 8 style guidelines
2. Add docstrings to functions
3. Include error handling
4. Update tests when adding features

## License

MIT License

