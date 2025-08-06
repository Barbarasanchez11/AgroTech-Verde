import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import pickle
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from src.config.config import get_data_path, get_model_path, ML_CONFIG
from src.services.firebase_service import FirebaseService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CropModelTrainer:
    def __init__(self):
        self.firebase_service = FirebaseService()
        self.label_encoder = None
        self.model = None
        self.is_trained = False
        self._initialize_preprocessor()

    def _initialize_preprocessor(self):
        numeric_columns = ["ph", "humedad", "temperatura", "precipitacion", "horas_de_sol"]
        categorical_columns = ["tipo_de_suelo", "temporada"]

        self.preprocessor = ColumnTransformer(transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            ("categorical", OneHotEncoder(), categorical_columns)
        ])

    def load_data(self) -> pd.DataFrame:
        try:
            data_path = get_data_path("agrotech_data.csv")
            if not data_path.exists():
                logger.error(f"Data file not found: {data_path}")
                return pd.DataFrame()

            df_original = pd.read_csv(data_path)
            logger.info(f"Loaded original data: {len(df_original)} records")

            if not self.firebase_service.is_initialized:
                self.firebase_service.initialize()

            if self.firebase_service.is_initialized:
                crops_data = self.firebase_service.get_all_crops()
                if crops_data:
                    df_new = pd.DataFrame(crops_data)
                    df_combined = pd.concat([df_original, df_new], ignore_index=True)
                    logger.info(f"Combined data: {len(df_combined)} records")
                    return df_combined

            return df_original

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()

    def validate_data(self, df: pd.DataFrame) -> bool:
        required_columns = ["tipo_de_cultivo", "ph", "humedad", "temperatura", "precipitacion", "horas_de_sol", "tipo_de_suelo", "temporada"]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        numeric_columns = ["ph", "humedad", "temperatura", "precipitacion", "horas_de_sol"]
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.error(f"Column {col} is not numeric")
                return False
        
        return True

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        if df.empty:
            logger.error("Empty dataframe provided")
            return pd.DataFrame(), pd.Series()

        if not self.validate_data(df):
            logger.error("Data validation failed")
            return pd.DataFrame(), pd.Series()

        try:
            self.label_encoder = LabelEncoder()
            df["crop_code"] = self.label_encoder.fit_transform(df["tipo_de_cultivo"])

            X = df.drop(columns=["tipo_de_cultivo", "crop_code"])
            y = df["crop_code"]

            return X, y

        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return pd.DataFrame(), pd.Series()

    def create_pipeline(self, model_type: str = "random_forest") -> Pipeline:
        if model_type == "random_forest":
            classifier = RandomForestClassifier(random_state=ML_CONFIG["random_state"])
        elif model_type == "svm":
            classifier = SVC(probability=True, random_state=ML_CONFIG["random_state"])
        else:
            logger.warning(f"Unknown model type '{model_type}', using Random Forest")
            classifier = RandomForestClassifier(random_state=ML_CONFIG["random_state"])

        pipeline = Pipeline([
            ("preprocessor", self.preprocessor),
            ("classifier", classifier)
        ])

        return pipeline

    def train_model(self, model_type: str = "random_forest") -> bool:
        try:
            df = self.load_data()
            if df.empty:
                logger.error("No data available for training")
                return False

            X, y = self.prepare_data(df)
            if X.empty or y.empty:
                logger.error("Failed to prepare data")
                return False

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=ML_CONFIG["test_size"], 
                random_state=ML_CONFIG["random_state"]
            )

            self.model = self.create_pipeline(model_type)
            self.model.fit(X_train, y_train)

            self.evaluate_model(X_test, y_test)
            self.save_model(model_type)

            self.is_trained = True
            logger.info("Model training completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        try:
            y_pred = self.model.predict(X_test)
            all_labels = list(self.label_encoder.transform(self.label_encoder.classes_))
            target_names = self.label_encoder.classes_

            report = classification_report(
                y_test, y_pred,
                labels=all_labels,
                target_names=target_names,
                zero_division=0,
                output_dict=True
            )

            logger.info(f"Model accuracy: {report['accuracy']:.4f}")
            
            self._create_confusion_matrix(y_test, y_pred, target_names)
            
            return report

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {}

    def _create_confusion_matrix(self, y_test: pd.Series, y_pred: np.ndarray, target_names: np.ndarray):
        try:
            plt.figure(figsize=(10, 8))
            ConfusionMatrixDisplay.from_predictions(
                y_test, y_pred,
                display_labels=target_names,
                cmap='Blues',
                normalize='true'
            )
            plt.title("Confusion Matrix")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            matrix_path = get_model_path("confusion_matrix.png")
            plt.savefig(matrix_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Confusion matrix saved to {matrix_path}")
            
        except Exception as e:
            logger.error(f"Error creating confusion matrix: {e}")

    def save_model(self, model_type: str = "random_forest") -> bool:
        try:
            model_path = get_model_path(f"modelo_{model_type}.pkl")
            encoder_path = get_model_path("label_encoder.pkl")

            with open(model_path, "wb") as f:
                pickle.dump(self.model, f)

            with open(encoder_path, "wb") as f:
                pickle.dump(self.label_encoder, f)

            logger.info(f"Model saved to {model_path}")
            logger.info(f"Encoder saved to {encoder_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    def validate_prediction_input(self, input_data: Dict[str, Any]) -> bool:
        required_fields = ["ph", "humedad", "temperatura", "precipitacion", "horas_de_sol", "tipo_de_suelo", "temporada"]
        
        missing_fields = [field for field in required_fields if field not in input_data]
        if missing_fields:
            logger.error(f"Missing required fields for prediction: {missing_fields}")
            return False
        
        numeric_fields = ["ph", "humedad", "temperatura", "precipitacion", "horas_de_sol"]
        for field in numeric_fields:
            if not isinstance(input_data[field], (int, float)):
                logger.error(f"Field {field} must be numeric")
                return False
        
        return True

    def predict(self, input_data: Dict[str, Any]) -> Optional[str]:
        if not self.is_trained or self.model is None:
            logger.error("Model not trained")
            return None

        if not self.validate_prediction_input(input_data):
            logger.error("Invalid input data for prediction")
            return None

        try:
            df = pd.DataFrame([input_data])
            prediction = self.model.predict(df)
            predicted_crop = self.label_encoder.inverse_transform(prediction)[0]
            return predicted_crop

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None

def main():
    trainer = CropModelTrainer()
    success = trainer.train_model("random_forest")
    
    if success:
        logger.info("Training completed successfully")
    else:
        logger.error("Training failed")

if __name__ == "__main__":
    main() 