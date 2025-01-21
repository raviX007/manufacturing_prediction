from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    air_temperature: float = Field(..., description="Air temperature in Kelvin")
    process_temperature: float = Field(..., description="Process temperature in Kelvin")
    rotational_speed: int = Field(..., description="Rotational speed in rpm")
    torque: float = Field(..., description="Torque in Nm")
    tool_wear: int = Field(..., description="Tool wear in minutes")
    type: str = Field(..., description="Type of product (L, M, or H)")

    class Config:
        json_schema_extra = {
            "example": {
                "air_temperature": 298.5,
                "process_temperature": 308.7,
                "rotational_speed": 1500,
                "torque": 40.0,
                "tool_wear": 20,
                "type": "M",
            }
        }
