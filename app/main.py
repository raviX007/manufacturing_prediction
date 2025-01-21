from fastapi import FastAPI, File, HTTPException, UploadFile

from app.models.ml_model import MachineLearningModel
from app.schemas.request_models import PredictionInput
from app.schemas.response_models import PredictionOutput, TrainingOutput
from app.utils.data_processor import DataProcessor

app = FastAPI(
    title="Manufacturing Prediction API",
    description="API for predicting machine downtime in manufacturing operations",
    version="1.0.0",
)

ml_model = MachineLearningModel()
data_processor = DataProcessor()


@app.post("/upload", response_model=dict)
async def upload_data(file: UploadFile = File(...)):
    """
    Upload training data in CSV format
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    try:
        contents = await file.read()
        data_processor.save_training_data(contents)
        return {"message": "Data uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/train", response_model=TrainingOutput)
async def train_model():
    """
    Train the model using uploaded data
    """
    try:
        metrics = ml_model.train(data_processor.load_training_data())
        return TrainingOutput(**metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Make predictions using trained model
    """
    try:
        prediction, failure_type, confidence = ml_model.predict(input_data.dict())
        return PredictionOutput(
            failure_predicted=prediction,
            failure_type=failure_type,
            confidence=confidence,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def check_health():
    """
    Health check endpoint
    """
    return {"message": "Manufacturing Prediction API is running"}
