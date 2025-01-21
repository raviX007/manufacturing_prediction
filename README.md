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
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`

## API Endpoints

### 1. Upload Data (POST /upload)
Upload training data in CSV format.

```bash
curl -X POST -F "file=@data/predictive_maintenance.csv" http://localhost:8000/upload
```

### 2. Train Model (POST /train)
Train the model using the uploaded dataset.

```bash
curl -X POST http://localhost:8000/train
```

Response:
```json
{
    "accuracy": 0.983,
    "f1_score": 0.679,
    "training_time": 0.716
}
```

### 3. Predict (POST /predict)
Make predictions for potential machine failures.

#### Example 1: Normal Operation
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

#### Example 2: High Risk Operation
```json
{
    "air_temperature": 308.5,
    "process_temperature": 312.7,
    "rotational_speed": 2200,
    "torque": 70.0,
    "tool_wear": 180,
    "type": "M"
}
```

Response:
```json
{
    "failure_predicted": true,
    "failure_type": "Failure Detected",
    "confidence": 0.87
}
```

## Testing
1. Access the Swagger UI documentation at `http://127.0.0.1:8000/docs`
2. Use the interactive API documentation to test endpoints
3. Alternative documentation available at `http://127.0.0.1:8000/redoc`


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

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License.




