from fastapi import FastAPI, File, HTTPException, UploadFile
from app.models.ml_model import MachineLearningModel
from app.schemas.request_models import PredictionInput
from app.schemas.response_models import PredictionOutput, TrainingOutput
from app.utils.data_processor import DataProcessor
from app.utils.logger import setup_logger
from fastapi.middleware.cors import CORSMiddleware
import time

# Initialize logger
logger = setup_logger("manufacturing_api")

app = FastAPI(
    title="Manufacturing Prediction API",
    description="API for predicting machine downtime in manufacturing operations",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ml_model = MachineLearningModel()
data_processor = DataProcessor()

@app.post("/upload", response_model=dict)
async def upload_data(file: UploadFile = File(...)):
    """
    Upload training data in CSV format
    """
    logger.info(f"Received file upload request: {file.filename}")
    
    if not file.filename.endswith(".csv"):
        logger.warning(f"Invalid file format attempted: {file.filename}")
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    try:
        contents = await file.read()
        data_processor.save_training_data(contents)
        logger.info(f"Successfully processed and saved training data: {file.filename}")
        return {"message": "Data uploaded successfully"}
    except Exception as e:
        logger.error(f"Error processing upload file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/train", response_model=TrainingOutput)
async def train_model():
    """
    Train the model using uploaded data
    """
    logger.info("Starting model training")
    start_time = time.time()
    
    try:
        metrics = ml_model.train(data_processor.load_training_data())
        training_time = time.time() - start_time
        logger.info(f"Model training completed successfully in {training_time:.2f} seconds. Metrics: {metrics}")
        return TrainingOutput(**metrics)
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Make predictions using trained model
    """
    logger.info("Received prediction request")
    
    try:
        prediction, failure_type, confidence = ml_model.predict(input_data.dict())
        logger.info(f"Prediction made: failure_predicted={prediction}, type={failure_type}, confidence={confidence}")
        return PredictionOutput(
            failure_predicted=prediction,
            failure_type=failure_type,
            confidence=confidence,
        )
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def check_health():
    """
    Health check endpoint
    """
    logger.debug("Health check requested")
    return {"message": "Manufacturing Prediction API is running"}