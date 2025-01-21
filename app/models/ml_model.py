import time
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from app.utils.logger import setup_logger

class MachineLearningModel:
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.logger.info("Initializing MachineLearningModel")
        
        self.model = None
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
        
        self.param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [5, 8, 10],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4],
            'max_features': ['sqrt', 'log2'],
            'class_weight': ['balanced']
        }
        self.logger.debug(f"Initialized with parameter grid: {self.param_grid}")

    def analyze_feature_importance(self, feature_names=None) -> Dict[str, float]:
        """Analyze and return feature importance scores."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance = self.model.feature_importances_
        feature_names = feature_names or self.feature_columns
        importance_dict = dict(zip(feature_names, importance))
        
        sorted_importance = {
            k: v for k, v in sorted(importance_dict.items(), 
                                  key=lambda item: item[1], 
                                  reverse=True)
        }
        
        self.logger.info("Feature importance analysis:")
        for feat, imp in sorted_importance.items():
            self.logger.info(f"{feat}: {imp:.4f}")
            
        return sorted_importance

    def perform_cross_validation(self, X: pd.DataFrame, y: pd.Series, cv=5) -> Dict[str, float]:
        self.logger.info(f"Starting {cv}-fold cross-validation")
        
        scoring = {
            'accuracy': 'accuracy',
            'f1': make_scorer(f1_score),
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score)
        }
        
        cv_results = {}
        for metric_name, scorer in scoring.items():
            self.logger.debug(f"Computing {metric_name} scores")
            scores = cross_val_score(self.model, X, y, cv=cv, scoring=scorer)
            cv_results[f'cv_{metric_name}_mean'] = float(scores.mean())
            cv_results[f'cv_{metric_name}_std'] = float(scores.std())
            self.logger.info(f"CV {metric_name}: mean={scores.mean():.4f}, std={scores.std():.4f}")
        
        return cv_results

    def tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        self.logger.info("Starting hyperparameter tuning with GridSearchCV")
        self.logger.debug(f"Data shapes - X: {X.shape}, y: {y.shape}")
        
        base_model = RandomForestClassifier(
            random_state=42,
            bootstrap=True,
            oob_score=True,
            n_jobs=-1
        )
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=self.param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        
        self.logger.info("Fitting GridSearchCV...")
        grid_search.fit(X, y)
        
        self.model = grid_search.best_estimator_
        
        mean_train_score = grid_search.cv_results_['mean_train_score'][grid_search.best_index_]
        mean_test_score = grid_search.cv_results_['mean_test_score'][grid_search.best_index_]
        
        self.logger.info(f"Best parameters found: {grid_search.best_params_}")
        self.logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        self.logger.info(f"Training Score: {mean_train_score:.4f}")
        self.logger.info(f"CV Score: {mean_test_score:.4f}")
        self.logger.info(f"Out-of-bag score: {self.model.oob_score_:.4f}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': float(grid_search.best_score_),
            'train_score': float(mean_train_score),
            'cv_score': float(mean_test_score),
            'oob_score': float(self.model.oob_score_),
            'cv_results': grid_search.cv_results_
        }

    def train(self, data: pd.DataFrame) -> Dict[str, float]:
        try:
            self.logger.info("Starting model training process")
            start_time = time.time()

            self.logger.debug(f"Available columns in data: {data.columns.tolist()}")
            self.logger.debug(f"Using features: {self.feature_columns}")

            X = data[self.feature_columns].copy()

            if "Type" in data.columns:
                self.logger.debug("Encoding 'Type' column")
                type_encoded = self.label_encoder.fit_transform(data["Type"])
                X["Type_encoded"] = type_encoded

            y = data[self.target_column]
            self.logger.info(f"Dataset shape - X: {X.shape}, y: {y.shape}")

            class_distribution = y.value_counts(normalize=True)
            self.logger.info(f"Class distribution:\n{class_distribution}")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            self.logger.debug(f"Train-test split - Train: {X_train.shape}, Test: {X_test.shape}")

            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            self.logger.debug("Data scaling completed")

            self.logger.info("Starting hyperparameter tuning...")
            tuning_results = self.tune_hyperparameters(X_train_scaled, y_train)
            
            self.logger.info("Starting cross-validation...")
            cv_results = self.perform_cross_validation(X_train_scaled, y_train)

            y_pred = self.model.predict(X_test_scaled)
            accuracy = self.model.score(X_test_scaled, y_test)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            cm = confusion_matrix(y_test, y_pred)
            
            self.logger.info(f"Final test set metrics:")
            self.logger.info(f"Accuracy: {accuracy:.4f}")
            self.logger.info(f"F1 Score: {f1:.4f}")
            self.logger.info(f"Precision: {precision:.4f}")
            self.logger.info(f"Recall: {recall:.4f}")
            self.logger.info(f"Confusion Matrix:\n{cm}")

            feature_importance = self.analyze_feature_importance()

            self.save_model()
            self.logger.info("Model and scaler saved to disk")

            training_time = time.time() - start_time
            self.logger.info(f"Total training time: {training_time:.2f} seconds")

            return {
                "accuracy": float(accuracy),
                "f1_score": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "training_time": training_time,
                "best_parameters": tuning_results['best_params'],
                "best_cv_score": tuning_results['best_score'],
                "train_score": tuning_results['train_score'],
                "cv_score": tuning_results['cv_score'],
                "oob_score": tuning_results['oob_score'],
                "feature_importance": feature_importance,
                **cv_results
            }
        except Exception as e:
            self.logger.error(f"Training error: {str(e)}", exc_info=True)
            raise Exception(f"Training error: {str(e)}")

    def predict(self, input_data: Dict[str, Any]) -> Tuple[bool, str, float]:
        try:
            self.logger.info("Starting prediction process")
            self.logger.debug(f"Input data: {input_data}")

            if not hasattr(self, "model") or self.model is None:
                self.logger.info("Loading model and scaler from disk")
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)

            # Create DataFrame with the correct columns first
            features = pd.DataFrame(columns=self.feature_columns)
            
            # Map input data to the correct column names
            mapped_input = {
                self.input_mapping[k]: v
                for k, v in input_data.items()
                if k in self.input_mapping
            }
            
            # Add the data as a new row
            features = pd.concat([features, pd.DataFrame([mapped_input])], ignore_index=True)
            
            # Ensure we only use the specified feature columns in the correct order
            features = features[self.feature_columns]

            if "type" in input_data:
                # Handle type encoding separately
                type_encoded = self.label_encoder.transform([input_data["type"]])
                features["Type_encoded"] = type_encoded

            features_scaled = self.scaler.transform(features)
            prediction = self.model.predict(features_scaled)[0]
            
            # Get probability estimates
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = float(np.max(probabilities))  # Get the highest probability
            
            # Add entropy calculation for prediction uncertainty
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            uncertainty = entropy / np.log2(len(probabilities))  # Normalized entropy
            
            failure_type = "No Failure" if not prediction else "Failure Detected"

            self.logger.info(f"Prediction: {failure_type}")
            self.logger.info(f"Confidence: {confidence:.4f}")
            self.logger.info(f"Uncertainty (normalized entropy): {uncertainty:.4f}")
            
            return bool(prediction), failure_type, confidence

        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}", exc_info=True)
            raise Exception(f"Prediction error: {str(e)}. Input data: {input_data}")

    def save_model(self):
        try:
            self.logger.info("Saving model and scaler to disk")
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            self.logger.info("Model and scaler saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}", exc_info=True)
            raise Exception(f"Error saving model: {str(e)}")

    def load_model(self):
        try:
            self.logger.info("Loading model and scaler from disk")
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.logger.info("Model and scaler loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}", exc_info=True)
            raise Exception(f"Error loading model: {str(e)}")