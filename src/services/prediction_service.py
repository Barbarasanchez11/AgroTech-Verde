import pickle
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path
from contextlib import suppress

from src.config.config import get_model_path, ML_CONFIG
from src.utils.validators import DataValidator, ModelValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.is_loaded = False

    def load_models(self) -> bool:
        try:
            model_path = get_model_path("random_forest")
            encoder_path = get_model_path("label_encoder")
            
            if model_path is None or not Path(model_path).exists():
                logger.error("Model path is invalid or does not exist")
                return False
                
            if encoder_path is None or not Path(encoder_path).exists():
                logger.error("Encoder path is invalid or does not exist")
                return False
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            self.is_loaded = True
            logger.info("Models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False

    def _validate_model_consistency(self) -> bool:
        try:
            if not self.is_loaded or not self.model or not self.label_encoder:
                return False
            
            if not hasattr(self.label_encoder, "classes_"):
                return False
            
            expected_crops = ['arroz', 'lentejas', 'maiz', 'naranjas', 'soja', 'trigo', 'uva', 'zanahoria']
            actual_crops = list(self.label_encoder.classes_)
            
            if set(actual_crops) != set(expected_crops):
                logger.warning("Model inconsistency detected, retraining automatically")
                self._auto_retrain_model()
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating model consistency: {e}")
            return False

    def _auto_retrain_model(self) -> bool:
        try:
            from src.models.fix_model import train_clean_model
            
            logger.info("Starting automatic model retraining...")
            pipeline, label_encoder, accuracy = train_clean_model()
            
            self.model = pipeline
            self.label_encoder = label_encoder
            self.is_loaded = True
            
            logger.info(f"Model automatically retrained with accuracy: {accuracy}")
            return True
            
        except Exception as e:
            logger.error(f"Error in automatic retraining: {e}")
            return False

    def predict_crop(self, terrain_params: Dict[str, Any]) -> Tuple[bool, str, Optional[str]]:
        try:
            is_valid, errors = DataValidator.validate_terrain_params(terrain_params)
            if not is_valid:
                return False, "Parameter error", errors[0] if errors else None

            if not self.is_loaded:
                if not self.load_models():
                    return False, "Model loading error", None

            if not self._validate_model_consistency():
                if not self.is_loaded:
                    return False, "Model consistency error", "Model validation failed"

            df = pd.DataFrame([terrain_params])
            prediction = self.model.predict(df)
            
            is_valid_pred, model_errors = ModelValidator.validate_model_prediction(self.model, df)
            if not is_valid_pred:
                return False, "Invalid model", model_errors[0] if model_errors else None
            
            predicted_crop = self.label_encoder.inverse_transform(prediction)[0]
            
            logger.info(f"Prediction successful: {predicted_crop}")
            return True, predicted_crop, None
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return False, "Prediction error", str(e)

    def get_model_info(self) -> Dict[str, Any]:
        if not self.is_loaded:
            return {"status": "not_loaded"}
        
        try:
            model_path = get_model_path("random_forest")
            encoder_path = get_model_path("label_encoder")
            
            return {
                "status": "loaded",
                "model_type": type(self.model).__name__,
                "available_crops": getattr(self.label_encoder, "classes_", []),
                "model_path": str(model_path) if model_path else "unknown",
                "encoder_path": str(encoder_path) if encoder_path else "unknown"
            }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"status": "error", "message": str(e)} 