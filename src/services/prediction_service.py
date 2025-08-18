import os
import pickle
import logging
import warnings
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import sklearn
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# ✅ NUEVO: Definimos get_model_path de forma global para que esté siempre disponible
def get_model_path(model_name: str):
    """
    Devuelve la ruta absoluta al modelo, encoder o preprocessor.
    """
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
        self.label_encoder = None
        self.preprocessor = None
        self.is_loaded = False
        self.auto_train_if_needed()

    def _get_preprocessor_expected_columns(self) -> Optional[list]:
        try:
            if hasattr(self.preprocessor, 'transformers_'):
                expected_cols = []
                for name, transformer, cols in self.preprocessor.transformers_:
                    if isinstance(cols, list):
                        expected_cols.extend(cols)
                return expected_cols
            return None
        except Exception:
            return None

    def _align_columns_to_preprocessor(self, df: pd.DataFrame) -> pd.DataFrame:
        english_to_spanish = {
            'humidity': 'humedad',
            'temperature': 'temperatura',
            'precipitation': 'precipitacion',
            'sun_hours': 'horas_de_sol',
            'soil_type': 'tipo_de_suelo',
            'season': 'temporada',
            'crop_type': 'tipo_de_cultivo'
        }
        spanish_to_english = {v: k for k, v in english_to_spanish.items()}

        expected_cols = self._get_preprocessor_expected_columns() or []
        expects_spanish = any(col in expected_cols for col in english_to_spanish.values())
        expects_english = any(col in expected_cols for col in ['humidity', 'temperature', 'precipitation', 'sun_hours', 'soil_type', 'season'])

        aligned_df = df.copy()
        if expects_spanish:
            aligned_df = aligned_df.rename(columns=english_to_spanish)
        elif expects_english:
            aligned_df = aligned_df.rename(columns=spanish_to_english)
        return aligned_df

    def load_models(self) -> bool:
        try:
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
            
            try:
                preprocessor_path = get_model_path("preprocessor")
                if preprocessor_path and preprocessor_path.exists():
                    with open(preprocessor_path, 'rb') as f:
                        self.preprocessor = pickle.load(f)
                    logger.info("Preprocessor loaded successfully")
                else:
                    logger.warning("Preprocessor not found, will use raw data")
            except Exception as e:
                logger.warning(f"Could not load preprocessor: {e}")
            
            self.is_loaded = True
            logger.info("Models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.is_loaded = False
            return False

    def predict_crop(self, terrain_params: Dict[str, Any]) -> Tuple[bool, str, Optional[str]]:
        try:
            from src.utils.validators import DataValidator
            
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

            if self.preprocessor is None:
                logger.warning("Preprocessor not available, training initial model...")
                if not self.train_initial_model():
                    return False, "Training error", "Failed to train initial model"
                if self.preprocessor is None:
                    return False, "Preprocessor error", "Preprocessor still not available after training"

            df_input = pd.DataFrame([terrain_params])
            df = self._align_columns_to_preprocessor(df_input)
            logger.info(f"Prediction input: {terrain_params}")
            
            try:
                logger.info("Processing data with preprocessor...")
                logger.info(f"Aligned input columns: {list(df.columns)}")
                logger.info(f"Input data sample: {df.iloc[0].to_dict()}")
                
                df_processed = self.preprocessor.transform(df)
                logger.info(f"Data processed: {df_processed.shape}")
                
            except Exception as proc_error:
                logger.error(f"Error processing data: {proc_error}")
                logger.error(f"Input data types: {df.dtypes.to_dict()}")
                return False, "Data processing error", str(proc_error)
            
            try:
                prediction = self.model.predict(df_processed)
                logger.info(f"Raw prediction: {prediction}")
                
                try:
                    if hasattr(self.model, 'predict_proba'):
                        probabilities = self.model.predict_proba(df_processed)
                        confidence = max(probabilities[0]) * 100
                        logger.info(f"Prediction confidence: {confidence:.2f}%")
                    else:
                        logger.warning("Model does not support predict_proba")
                        confidence = 95.0
                except Exception as proba_error:
                    logger.warning(f"Error getting probabilities: {proba_error}")
                    confidence = 95.0
                    
            except Exception as pred_error:
                logger.error(f"Error during model prediction: {pred_error}")
                return False, "Prediction error", str(pred_error)
            
            try:
                predicted_crop = self.label_encoder.inverse_transform(prediction)[0]
                logger.info(f"Prediction successful: {predicted_crop}")
                logger.info(f"Available crop classes: {list(self.label_encoder.classes_)}")
                return True, predicted_crop, confidence
                
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

    def auto_train_if_needed(self) -> bool:
        try:
            if self.load_models():
                logger.info("Existing models loaded successfully")
                if not self.preprocessor:
                    logger.warning("Models loaded but preprocessor missing - retraining...")
                    return self.train_initial_model()
                return True
            
            logger.info("No existing models found, training initial model automatically...")
            return self.train_initial_model()
            
        except Exception as e:
            logger.error(f"Error in auto-training: {e}")
            logger.info("Falling back to initial model training...")
            return self.train_initial_model()
    
    def train_initial_model(self) -> bool:
        try:
            csv_path = Path(__file__).parent.parent.parent / "data" / "agroTech_data.csv"
            if not csv_path.exists():
                logger.error("Original CSV not found")
                return False
            
            df = pd.read_csv(csv_path)
            logger.info(f"CSV loaded: {len(df)} records")
            logger.info(f"Available CSV columns: {list(df.columns)}")
            
            if 'humedad' in df.columns:
                feature_columns = ['ph', 'humedad', 'temperatura', 'precipitacion', 'horas_de_sol', 'tipo_de_suelo', 'temporada']
                target_column = 'tipo_de_cultivo'
                numeric_features = ['ph', 'humedad', 'temperatura', 'precipitacion', 'horas_de_sol']
                categorical_features = ['tipo_de_suelo', 'temporada']
            else:
                feature_columns = ['ph', 'humidity', 'temperature', 'precipitation', 'sun_hours', 'soil_type', 'season']
                target_column = 'crop_type'
                numeric_features = ['ph', 'humidity', 'temperature', 'precipitation', 'sun_hours']
                categorical_features = ['soil_type', 'season']
            
            missing_columns = [col for col in feature_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing columns in CSV: {missing_columns}")
                logger.error(f"Available columns: {list(df.columns)}")
                return False
            
            X = df[feature_columns]
            y = df[target_column]
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(drop='first', sparse=False), categorical_features)
                ]
            )
            
            X_processed = preprocessor.fit_transform(X)
            
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_encoded, test_size=0.2, random_state=42
            )
            
            classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            pipeline = Pipeline([('classifier', classifier)])
            
            logger.info("Training initial model...")
            pipeline.fit(X_train, y_train)
            
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Initial model trained: accuracy={accuracy:.4f}")
            
            self.model = pipeline
            self.label_encoder = label_encoder
            self.preprocessor = preprocessor
            self.is_loaded = True
            
            self.save_models()
            
            return True
            
        except Exception as e:
            logger.error(f"Error training initial model: {e}")
            return False
    
    def save_models(self) -> bool:
        try:
            model_path = get_model_path("random_forest")
            encoder_path = get_model_path("label_encoder")
            preprocessor_path = get_model_path("preprocessor")
            
            if model_path and encoder_path and preprocessor_path:
                model_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(model_path, 'wb') as f:
                    pickle.dump(self.model, f)
                
                with open(encoder_path, 'wb') as f:
                    pickle.dump(self.label_encoder, f)
                
                with open(preprocessor_path, 'wb') as f:
                    pickle.dump(self.preprocessor, f)
                
                logger.info("Models saved to disk")
                return True
            else:
                logger.error("Could not get model paths")
                return False
                
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False
    
    def reload_models(self) -> bool:
        logger.info("Reloading models...")
        self.model = None
        self.label_encoder = None
        self.preprocessor = None
        self.is_loaded = False
        return self.load_models()

prediction_service = PredictionService()
