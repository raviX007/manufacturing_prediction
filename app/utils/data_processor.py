import io
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from app.utils.logger import setup_logger

class DataProcessor:
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.logger.info("Initializing DataProcessor")
        
        self.data_path = Path("data/predictive_maintenance.csv")
        self.required_columns = {
            "UDI": int,
            "Product ID": str,
            "Type": str,
            "Air temperature [K]": float,
            "Process temperature [K]": float,
            "Rotational speed [rpm]": int,
            "Torque [Nm]": float,
            "Tool wear [min]": int,
            "Target": int,
            "Failure Type": str,
        }
        self.label_encoders = {}
        self.logger.debug(f"Set data path to: {self.data_path}")
        self.logger.debug(f"Initialized with {len(self.required_columns)} required columns")

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate the dataframe structure and data types
        """
        self.logger.info("Starting data validation")
        self.logger.debug(f"Input DataFrame shape: {df.shape}")
        
        missing_cols = set(self.required_columns.keys()) - set(df.columns)
        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")

        self.logger.debug("Validating data types for each column")
        for col, dtype in self.required_columns.items():
            try:
                if dtype in [int, float]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                else:
                    df[col] = df[col].astype(str)
                self.logger.debug(f"Successfully converted column {col} to {dtype}")
            except Exception as e:
                self.logger.error(f"Data type validation failed for column {col}", exc_info=True)
                raise ValueError(
                    f"Invalid data type for column {col}. Expected {dtype}: {str(e)}"
                )

        self.logger.info("Data validation completed successfully")
        return True

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data by handling categorical variables and missing values
        """
        self.logger.info("Starting data preprocessing")
        self.logger.debug(f"Input DataFrame shape: {df.shape}")
        
        processed_df = df.copy()

        if "Type" in processed_df.columns:
            self.logger.debug("Processing 'Type' column")
            if "Type" not in self.label_encoders:
                self.logger.debug("Creating new LabelEncoder for 'Type' column")
                self.label_encoders["Type"] = LabelEncoder()
                processed_df["Type_encoded"] = self.label_encoders[
                    "Type"
                ].fit_transform(processed_df["Type"])
            else:
                self.logger.debug("Using existing LabelEncoder for 'Type' column")
                processed_df["Type_encoded"] = self.label_encoders["Type"].transform(
                    processed_df["Type"]
                )

        if "Target" in processed_df.columns:
            self.logger.debug("Converting 'Target' column to boolean")
            processed_df["Target"] = processed_df["Target"].astype(bool)

        numeric_columns = [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ]
        self.logger.debug(f"Handling missing values for numeric columns: {numeric_columns}")
        processed_df[numeric_columns] = processed_df[numeric_columns].fillna(
            processed_df[numeric_columns].mean()
        )

        self.logger.info("Data preprocessing completed")
        self.logger.debug(f"Output DataFrame shape: {processed_df.shape}")
        return processed_df

    def save_training_data(self, contents: Union[bytes, str]) -> None:
        """
        Save uploaded CSV data to file system
        """
        self.logger.info("Starting to save training data")
        try:
            if isinstance(contents, bytes):
                self.logger.debug("Reading bytes content")
                df = pd.read_csv(io.BytesIO(contents))
            else:
                self.logger.debug("Reading string content")
                df = pd.read_csv(io.StringIO(contents))

            self.logger.debug(f"Loaded DataFrame with shape: {df.shape}")
            self.validate_data(df)

            self.data_path.parent.mkdir(exist_ok=True)
            self.logger.debug(f"Saving data to: {self.data_path}")
            df.to_csv(self.data_path, index=False)
            self.logger.info("Training data saved successfully")

        except Exception as e:
            self.logger.error(f"Error saving training data: {str(e)}", exc_info=True)
            raise ValueError(f"Error processing CSV file: {str(e)}")

    def load_training_data(self) -> pd.DataFrame:
        """
        Load and preprocess training data from file system
        """
        self.logger.info("Starting to load training data")
        
        if not self.data_path.exists():
            self.logger.error(f"Training data not found at: {self.data_path}")
            raise FileNotFoundError("No training data found. Please upload data first.")

        try:
            self.logger.debug(f"Reading data from: {self.data_path}")
            df = pd.read_csv(self.data_path)
            self.logger.debug(f"Loaded DataFrame with shape: {df.shape}")

            self.validate_data(df)
            processed_df = self.preprocess_data(df)

            self.logger.info("Training data loaded and processed successfully")
            return processed_df

        except Exception as e:
            self.logger.error(f"Error loading training data: {str(e)}", exc_info=True)
            raise ValueError(f"Error loading training data: {str(e)}")

    def prepare_prediction_input(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare input data for prediction
        """
        self.logger.info("Starting to prepare prediction input")
        self.logger.debug(f"Input data: {input_data}")
        
        try:
            input_df = pd.DataFrame([input_data])
            processed_input = self.preprocess_data(input_df)
            
            self.logger.info("Prediction input prepared successfully")
            return processed_input

        except Exception as e:
            self.logger.error(f"Error preparing prediction input: {str(e)}", exc_info=True)
            raise ValueError(f"Error preparing prediction input: {str(e)}")

    def get_feature_names(self) -> list:
        """
        Get list of feature names used for training
        """
        features = [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
            "Type_encoded",
        ]
        self.logger.debug(f"Returning feature names: {features}")
        return features

    def inverse_transform_predictions(self, predictions: np.ndarray) -> list:
        """
        Transform encoded predictions back to original labels
        """
        self.logger.info("Starting inverse transform of predictions")
        try:
            if "type" in self.label_encoders:
                self.logger.debug("Using type label encoder for inverse transform")
                result = self.label_encoders["type"].inverse_transform(predictions)
                self.logger.info("Inverse transform completed successfully")
                return result
            self.logger.debug("No type encoder found, returning original predictions")
            return predictions
        except Exception as e:
            self.logger.error(f"Error in inverse transform: {str(e)}", exc_info=True)
            raise ValueError(f"Error inverse transforming predictions: {str(e)}")