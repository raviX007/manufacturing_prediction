# Manufacturing Predictive Maintenance API

## Overview

This API predicts machine failures in manufacturing operations using machine learning. It provides endpoints for uploading training data, training a predictive model, and making predictions about potential machine failures.

## Dataset

The project uses the "Machine Predictive Maintenance Classification" dataset from Kaggle:

- **Source**: [Machine Predictive Maintenance Classification Dataset](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)
- **Size**: 10,000 data points
- **License**: CC0: Public Domain

### Dataset Features

- `UDI`: Unique identifier ranging from 1 to 10000
- `Product ID`: Product identifier (L/M/H + number) indicating quality variants
- `Type`: Machine type categorized as L (Low), M (Medium), and H (High)
- `Air temperature [K]`: Measured in Kelvin
- `Process temperature [K]`: Measured in Kelvin
- `Rotational speed [rpm]`: Rotational speed
- `Torque [Nm]`: Torque in Nm
- `Tool wear [min]`: Minutes of tool wear
- `Target`: Binary indicator of failure (0: No failure, 1: Failure)
- `Failure Type`: Categories of failure (No Failure, Heat Dissipation Failure, Power Failure, etc.)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup Steps

1. Clone the repository

```bash
git clone <repository-url>
cd manufacturing_predict_api
```

2. Create and activate virtual environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment

# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Run the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

The Local API will be available at `http://0.0.0.0:8001`

## API Endpoints

### 1. Upload Data (POST /upload)

Upload training data in CSV format.

```bash
curl -X POST -F "file=@data/predictive_maintenance.csv" http://0.0.0.0:8001/upload
```

### 2. Train Model (POST /train)

#### Note: Training Takes Time (Approx 3 minutes).

Train the model using the uploaded dataset.

```bash
curl -X POST http://0.0.0.0:8001/train
```

Response:

```json
{
  "accuracy": 0.984,
  "f1_score": 0.6862745098039216,
  "training_time": 136.29223227500916
}
```

### 3. Predict (POST /predict)

Make predictions for potential machine failures.

#### Example :

```json
{
  "air_temperature": 298.5,
  "process_temperature": 308.7,
  "rotational_speed": 1500,
  "torque": 40.0,
  "tool_wear": 20,
  "type": "M"
}
```

Response:

```json
{
  "failure_predicted": false,
  "failure_type": "No Failure",
  "confidence": 0.95
}
```

## Testing

1. Access the Swagger UI documentation at `http://0.0.0.0:8001/docs`
2. Use the interactive API documentation to test endpoints
3. Alternative documentation available at `http://0.0.0.0:8001/redoc`

## Technologies Used

- FastAPI: Web framework for building APIs
- scikit-learn: Machine learning library
- pandas: Data manipulation and analysis
- numpy: Numerical computing
- uvicorn: ASGI server implementation

## Model Details

- Algorithm: Random Forest Classifier
- Features: Temperature, Rotational Speed, Torque, Tool Wear, Machine Type
- Target: Binary classification (Failure/No Failure)
- Metrics: Accuracy and F1-Score

## Code Quality

This project uses the following tools for code quality:

- `black`: Code formatter
- `ruff`: Style guide enforcement (linting)
- `isort`: Import sorting

Format code:

```bash
# Run Black formatter
black .

# Sort imports
isort .

# Run ruff for style guide enforcement(linting)
ruff check app/ tests/ --fix
```

## Error Handling

The API includes comprehensive error handling for:

- Invalid file formats
- Missing or incorrect data
- Model training failures
- Invalid prediction inputs
- Server errors

