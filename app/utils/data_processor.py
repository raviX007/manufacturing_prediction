import io
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DataProcessor:
    def __init__(self):
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

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate the dataframe structure and data types
        """
        
        missing_cols = set(self.required_columns.keys()) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        
        for col, dtype in self.required_columns.items():
            try:
                if dtype in [int, float]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                else:
                    df[col] = df[col].astype(str)
            except Exception as e:
                raise ValueError(
                    f"Invalid data type for column {col}. Expected {dtype}: {str(e)}"
                )

        return True

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data by handling categorical variables and missing values
        """
       
        processed_df = df.copy()

        
        if "Type" in processed_df.columns:
            if "Type" not in self.label_encoders:
                self.label_encoders["Type"] = LabelEncoder()
                processed_df["Type_encoded"] = self.label_encoders[
                    "Type"
                ].fit_transform(processed_df["Type"])
            else:
                processed_df["Type_encoded"] = self.label_encoders["Type"].transform(
                    processed_df["Type"]
                )

        
        if "Target" in processed_df.columns:
            processed_df["Target"] = processed_df["Target"].astype(bool)

        
        numeric_columns = [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ]
        processed_df[numeric_columns] = processed_df[numeric_columns].fillna(
            processed_df[numeric_columns].mean()
        )

        return processed_df

    def save_training_data(self, contents: Union[bytes, str]) -> None:
        """
        Save uploaded CSV data to file system
        """
        try:
            if isinstance(contents, bytes):
                df = pd.read_csv(io.BytesIO(contents))
            else:
                df = pd.read_csv(io.StringIO(contents))

            
            self.validate_data(df)

            self.data_path.parent.mkdir(exist_ok=True)

            df.to_csv(self.data_path, index=False)

        except Exception as e:
            raise ValueError(f"Error processing CSV file: {str(e)}")

    def load_training_data(self) -> pd.DataFrame:
        """
        Load and preprocess training data from file system
        """
        if not self.data_path.exists():
            raise FileNotFoundError("No training data found. Please upload data first.")

        try:
            df = pd.read_csv(self.data_path)

            self.validate_data(df)
            processed_df = self.preprocess_data(df)

            return processed_df

        except Exception as e:
            raise ValueError(f"Error loading training data: {str(e)}")

    def prepare_prediction_input(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare input data for prediction
        """
        try:
            input_df = pd.DataFrame([input_data])

            processed_input = self.preprocess_data(input_df)

            return processed_input

        except Exception as e:
            raise ValueError(f"Error preparing prediction input: {str(e)}")

    def get_feature_names(self) -> list:
        """
        Get list of feature names used for training
        """
        return [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
            "Type_encoded",
        ]

    def inverse_transform_predictions(self, predictions: np.ndarray) -> list:
        """
        Transform encoded predictions back to original labels
        """
        try:
            if "type" in self.label_encoders:
                return self.label_encoders["type"].inverse_transform(predictions)
            return predictions
        except Exception as e:
            raise ValueError(f"Error inverse transforming predictions: {str(e)}")
