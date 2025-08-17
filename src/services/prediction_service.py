import os
import pickle
import logging
import warnings
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import sklearn
from pathlib import Path

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.is_loaded = False
        self.load_models()
    
    def load_models(self) -> bool:
        try:
            from config.config import get_model_path
            
            model_path = get_model_path("random_forest")
            encoder_path = get_model_path("label_encoder")
            
            if not model_path or not encoder_path:
                logger.error("Model or encoder path not found")
                return False
            
            if not model_path.exists():
                logger.error(f"Model path does not exist: {model_path}")
                return False
                
            if not encoder_path.exists():
                logger.error(f"Encoder path does not exist: {encoder_path}")
                return False
            
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                    
                if hasattr(self.model, 'named_steps'):
                    classifier = self.model.named_steps.get('classifier')
                    if classifier and hasattr(classifier, 'monotonic_cst'):
                        logger.warning("Model has monotonic_cst attribute - potential version mismatch")
                        
            except Exception as e:
                logger.error(f"Error loading model with pickle: {e}")
                return False
            
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            self.is_loaded = True
            logger.info("Models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.is_loaded = False
            return False

    def predict_crop(self, terrain_params: Dict[str, Any]) -> Tuple[bool, str, Optional[str]]:
        try:
            from utils.validators import DataValidator
            
            is_valid, errors = DataValidator.validate_terrain_params(terrain_params)
            if not is_valid:
                error_msg = errors[0] if errors else "Invalid parameters"
                logger.error(f"Parameter validation failed: {error_msg}")
                return False, "Parameter error", error_msg

            if not self.is_loaded:
                if not self.load_models():
                    return False, "Model loading error", "Failed to load ML models"

            if self.model is None:
                logger.error("Model is None after loading")
                return False, "Model error", "Model is not available"
                
            if self.label_encoder is None:
                logger.error("Label encoder is None after loading")
                return False, "Encoder error", "Label encoder is not available"

            df = pd.DataFrame([terrain_params])
            logger.info(f"Prediction input: {terrain_params}")
            
            try:
                prediction = self.model.predict(df)
                logger.info(f"Raw prediction: {prediction}")
                
                try:
                    probabilities = self.model.predict_proba(df)
                    confidence = max(probabilities[0]) * 100
                    logger.info(f"Prediction confidence: {confidence:.2f}%")
                except:
                    confidence = None
                    
            except Exception as pred_error:
                logger.error(f"Error during model prediction: {pred_error}")
                return False, "Prediction error", str(pred_error)
            
            try:
                predicted_crop = self.label_encoder.inverse_transform(prediction)[0]
                logger.info(f"Prediction successful: {predicted_crop}")
                return True, predicted_crop, None
                
            except Exception as decode_error:
                logger.error(f"Error decoding prediction: {decode_error}")
                return False, "Decoding error", str(decode_error)
            
        except Exception as e:
            logger.error(f"Unexpected error during prediction: {e}")
            return False, "Unexpected error", str(e)

    def get_model_info(self) -> Dict[str, Any]:
        if not self.is_loaded:
            return {"status": "not_loaded"}
        
        try:
            from config.config import get_model_path
            
            model_path = get_model_path("random_forest")
            encoder_path = get_model_path("label_encoder")
            
            info = {
                "status": "loaded",
                "sklearn_version": sklearn.__version__,
                "model_type": type(self.model).__name__ if self.model else "Unknown",
                "model_path": str(model_path) if model_path else "unknown",
                "encoder_path": str(encoder_path) if encoder_path else "unknown"
            }
            
            if self.label_encoder and hasattr(self.label_encoder, 'classes_'):
                info["available_crops"] = self.label_encoder.classes_.tolist()
            else:
                info["available_crops"] = []
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"status": "error", "error": str(e)} 

    def reload_models(self) -> bool:
        logger.info("Reloading models...")
        self.model = None
        self.label_encoder = None
        self.is_loaded = False
        return self.load_models()

prediction_service = PredictionService() 