from pydantic import BaseModel, Field


class PredictionOutput(BaseModel):
    failure_predicted: bool = Field(
        ..., description="Predicted machine failure (True/False)"
    )
    failure_type: str = Field(..., description="Type of failure predicted")
    confidence: float = Field(..., description="Prediction confidence score")


class TrainingOutput(BaseModel):
    accuracy: float = Field(..., description="Model accuracy score")
    f1_score: float = Field(..., description="F1 score")
    training_time: float = Field(..., description="Training time in seconds")
