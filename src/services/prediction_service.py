import os
import pickle
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_model_path(model_name: str):
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    if model_name == "random_forest":
        return models_dir / "modelo_random_forest.pkl"
    elif model_name == "label_encoder":
        return models_dir / "label_encoder.pkl"
    elif model_name == "preprocessor":
        return models_dir / "preprocessor.pkl"
    return None


class PredictionService:
    def __init__(self):
        self.model = None
        self.encoder = None
        self.preprocessor = None

    def load_models(self):
        try:
            model_path = get_model_path("random_forest")
            encoder_path = get_model_path("label_encoder")
            preprocessor_path = get_model_path("preprocessor")

            if not model_path.exists() or not encoder_path.exists() or not preprocessor_path.exists():
                logger.warning("Uno o más archivos de modelo no existen")
                return False

            with open(model_path, "rb") as f:
                self.model = pickle.load(f)

            with open(encoder_path, "rb") as f:
                self.encoder = pickle.load(f)

            with open(preprocessor_path, "rb") as f:
                self.preprocessor = pickle.load(f)

            logger.info("Modelos cargados correctamente")
            return True
        except Exception as e:
            logger.error(f"Error al cargar los modelos: {e}")
            return False

    def save_models(self):
        try:
            model_path = get_model_path("random_forest")
            encoder_path = get_model_path("label_encoder")
            preprocessor_path = get_model_path("preprocessor")

            with open(model_path, "wb") as f:
                pickle.dump(self.model, f)

            with open(encoder_path, "wb") as f:
                pickle.dump(self.encoder, f)

            with open(preprocessor_path, "wb") as f:
                pickle.dump(self.preprocessor, f)

            logger.info("Modelos guardados correctamente")
            return True
        except Exception as e:
            logger.error(f"Error al guardar los modelos: {e}")
            return False

    def train_initial_model(self):
        try:
            data_path = Path(__file__).parent.parent / "data" / "agroTech_data.csv"

            if not data_path.exists():
                logger.error("El archivo agroTech_data.csv no existe")
                return False

            df = pd.read_csv(data_path)

            features = ["ph", "humidity", "temperature"]
            target = "label"

            X = df[features]
            y = df[target]

            self.encoder = LabelEncoder()
            y_encoded = self.encoder.fit_transform(y)

            self.preprocessor = ColumnTransformer(
                transformers=[("num", StandardScaler(), features)]
            )

            X_preprocessed = self.preprocessor.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_preprocessed, y_encoded, test_size=0.2, random_state=42
            )

            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)

            acc = self.model.score(X_test, y_test)
            logger.info(f"Modelo entrenado con precisión: {acc:.2f}")

            return self.save_models()
        except Exception as e:
            logger.error(f"Error al entrenar el modelo inicial: {e}")
            return False

    def predict(self, ph, humedad, temperatura):
        try:
            if self.model is None or self.encoder is None or self.preprocessor is None:
                if not self.load_models():
                    logger.warning("Intentando entrenar modelo inicial...")
                    if not self.train_initial_model():
                        return False, "Model not available", "No se pudo cargar ni entrenar el modelo"

            if self.preprocessor is None:
                logger.error("Preprocessor is not available")
                return False, "Preprocessor error", "Preprocessor is not available"

            input_data = pd.DataFrame(
                [[ph, humedad, temperatura]], columns=["ph", "humidity", "temperature"]
            )
            input_preprocessed = self.preprocessor.transform(input_data)
            prediction = self.model.predict(input_preprocessed)
            predicted_label = self.encoder.inverse_transform(prediction)

            return True, predicted_label[0], "OK"
        except Exception as e:
            logger.error(f"Error during model prediction: {e}")
            return False, "Prediction error", str(e)

    def get_model_info(self):
        if self.model is None:
            self.load_models()
        return {
            "model_name": type(self.model).__name__ if self.model else None,
            "version": "1.0"
        }
