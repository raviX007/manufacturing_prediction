import time
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class MachineLearningModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_columns = [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ]
        self.input_mapping = {
            "air_temperature": "Air temperature [K]",
            "process_temperature": "Process temperature [K]",
            "rotational_speed": "Rotational speed [rpm]",
            "torque": "Torque [Nm]",
            "tool_wear": "Tool wear [min]",
            "type": "Type",
        }
        self.categorical_columns = ["Type"]
        self.target_column = "Target"
        self.failure_type_column = "Failure Type"
        self.model_path = "data/model.joblib"
        self.scaler_path = "data/scaler.joblib"
        self.label_encoder = LabelEncoder()

    def train(self, data: pd.DataFrame) -> Dict[str, float]:
        try:
            start_time = time.time()
            print("Available columns in data:", data.columns.tolist())
            print("Looking for features:", self.feature_columns)
            X = data[self.feature_columns].copy()
            if "Type" in data.columns:
                type_encoded = self.label_encoder.fit_transform(data["Type"])
                X["Type_encoded"] = type_encoded
            y = data[self.target_column]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            self.model.fit(X_train_scaled, y_train)
            accuracy = self.model.score(X_test_scaled, y_test)
            y_pred = self.model.predict(X_test_scaled)
            f1 = f1_score(y_test, y_pred)
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            training_time = time.time() - start_time
            return {
                "accuracy": float(accuracy),
                "f1_score": float(f1),
                "training_time": training_time,
            }
        except Exception as e:
            raise Exception(f"Training error: {str(e)}")

    def predict(self, input_data: Dict[str, Any]) -> Tuple[bool, str, float]:
        try:
            if not hasattr(self, "model") or self.model is None:
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
            mapped_input = {
                self.input_mapping[k]: v
                for k, v in input_data.items()
                if k in self.input_mapping
            }
            input_df = pd.DataFrame([mapped_input])
            features = input_df[self.feature_columns].copy()
            if "Type" in input_df.columns:
                features["Type_encoded"] = self.label_encoder.transform(
                    input_df["Type"]
                )
            features_scaled = self.scaler.transform(features)
            prediction = self.model.predict(features_scaled)[0]
            confidence = np.max(self.model.predict_proba(features_scaled)[0])
            failure_type = "No Failure" if not prediction else "Failure Detected"
            return bool(prediction), failure_type, float(confidence)
        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}. Input data: {input_data}")

    def save_model(self):
        try:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
        except Exception as e:
            raise Exception(f"Error saving model: {str(e)}")

    def load_model(self):
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
