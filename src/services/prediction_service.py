import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging
from src.config.config import STANDARD_COLUMNS, SPANISH_COLUMNS, COLUMN_MAPPING

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self):
        self.models_dir = Path("src/models")
        self.model = None
        self.label_encoder = None
        self.preprocessor = None
        self.is_loaded = False
        self._load_models()
    
    def _load_models(self) -> bool:
       
        try:
            
            model_path = self.models_dir / "modelo_random_forest.pkl"
            encoder_path = self.models_dir / "label_encoder.pkl"
            preprocessor_path = self.models_dir / "preprocessor.pkl"
            
            
            if not all([model_path.exists(), encoder_path.exists(), preprocessor_path.exists()]):
                missing_files = []
                if not model_path.exists():
                    missing_files.append(str(model_path))
                if not encoder_path.exists():
                    missing_files.append(str(encoder_path))
                if not preprocessor_path.exists():
                    missing_files.append(str(preprocessor_path))
                
                logger.warning(f"Archivos de modelo faltantes: {missing_files}")
                return False
            
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
           
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
          
            with open(preprocessor_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
            
            self.is_loaded = True
            logger.info(" Todos los modelos cargados exitosamente")
            logger.info(f"Clases disponibles: {list(self.label_encoder.classes_)}")
            
            return True
            
        except Exception as e:
            logger.error(f" Error cargando modelos: {e}")
            self.is_loaded = False
            return False
    
    def _normalize_input_data(self, terrain_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normaliza los datos de entrada usando el mapeo centralizado
        """
        # Usar el mapeo centralizado de config.py
        spanish_to_english = COLUMN_MAPPING
        
        # Crear copia normalizada
        normalized_params = {}
        
        for key, value in terrain_params.items():
            # Si la clave está en español, convertir a inglés
            if key in spanish_to_english:
                english_key = spanish_to_english[key]
                normalized_params[english_key] = value
                logger.debug(f"🔄 Mapped {key} -> {english_key}")
            else:
                # Mantener la clave original
                normalized_params[key] = value
        
        # Asegurar que todas las columnas requeridas estén presentes
        required_features = STANDARD_COLUMNS['all_features']
        
        missing_features = [feature for feature in required_features if feature not in normalized_params]
        if missing_features:
            logger.error(f"❌ Características faltantes: {missing_features}")
            raise ValueError(f"Faltan las siguientes características: {missing_features}")
        
        logger.info(f"✅ Datos normalizados correctamente: {list(normalized_params.keys())}")
        return normalized_params
    
    def predict_crop(self, terrain_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predice el mejor cultivo para los parámetros del terreno dados
        """
        try:
            # Verificar que los modelos estén cargados
            if not self.is_loaded:
                logger.warning("⚠️ Modelos no cargados, intentando recargar...")
                if not self._load_models():
                    return {
                        "success": False,
                        "error": "Modelos no disponibles",
                        "details": "No se pudieron cargar los modelos necesarios para hacer predicciones"
                    }
            
            # Detectar qué columnas espera el preprocessor
            expected_features = []
            if hasattr(self.preprocessor, 'transformers_'):
                for name, transformer, features in self.preprocessor.transformers_:
                    if isinstance(features, list):
                        expected_features.extend(features)
                    elif isinstance(features, str):
                        expected_features.append(features)
            
            logger.info(f"🔍 Columnas esperadas por el preprocessor: {expected_features}")
            
            # Mapear datos según las columnas esperadas
            if 'humedad' in expected_features:
                # El preprocessor espera columnas en español
                logger.info("🇪🇸 Preprocessor espera columnas en español")
                mapped_params = {
                    'ph': terrain_params.get('ph', 0),
                    'humedad': terrain_params.get('humidity', terrain_params.get('humedad', 0)),
                    'temperatura': terrain_params.get('temperature', terrain_params.get('temperatura', 0)),
                    'precipitacion': terrain_params.get('precipitation', terrain_params.get('precipitacion', 0)),
                    'horas_de_sol': terrain_params.get('sun_hours', terrain_params.get('horas_de_sol', 0)),
                    'tipo_de_suelo': terrain_params.get('soil_type', terrain_params.get('tipo_de_suelo', 'arcilloso')),
                    'temporada': terrain_params.get('season', terrain_params.get('temporada', 'verano'))
                }
                features_order = SPANISH_COLUMNS['all_features']
            else:
                # El preprocessor espera columnas en inglés
                logger.info("🇺🇸 Preprocessor espera columnas en inglés")
                mapped_params = {
                    'ph': terrain_params.get('ph', 0),
                    'humidity': terrain_params.get('humidity', 0),
                    'temperature': terrain_params.get('temperature', 0),
                    'precipitation': terrain_params.get('precipitation', 0),
                    'sun_hours': terrain_params.get('sun_hours', 0),
                    'soil_type': terrain_params.get('soil_type', 'clay'),
                    'season': terrain_params.get('season', 'summer')
                }
                features_order = STANDARD_COLUMNS['all_features']
            
            # Crear DataFrame con el orden correcto de columnas
            input_data = pd.DataFrame([mapped_params])[features_order]
            
            logger.info(f"📊 Datos de entrada preparados:")
            logger.info(f"   Columnas: {list(input_data.columns)}")
            logger.info(f"   Valores: {input_data.iloc[0].to_dict()}")
            
            # Verificar que no haya valores nulos
            if input_data.isnull().any().any():
                null_columns = input_data.columns[input_data.isnull().any()].tolist()
                return {
                    "success": False,
                    "error": "Valores nulos detectados",
                    "details": f"Columnas con valores nulos: {null_columns}"
                }
            
            # Aplicar el preprocessor
            try:
                X_processed = self.preprocessor.transform(input_data)
                logger.info(f"✅ Preprocessor aplicado exitosamente: {X_processed.shape}")
            except Exception as prep_error:
                logger.error(f"❌ Error en preprocessing: {prep_error}")
                return {
                    "success": False,
                    "error": "Error en preprocessing",
                    "details": str(prep_error)
                }
            
            # Hacer predicción
            try:
                prediction_encoded = self.model.predict(X_processed)[0]
                prediction_proba = self.model.predict_proba(X_processed)[0]
                
                # Decodificar predicción
                predicted_crop = self.label_encoder.inverse_transform([prediction_encoded])[0]
                confidence = float(np.max(prediction_proba))
                
                logger.info(f"🎯 Predicción exitosa:")
                logger.info(f"   Cultivo: {predicted_crop}")
                logger.info(f"   Confianza: {confidence:.4f}")
                
                # Obtener todas las probabilidades para crops alternativos
                all_probabilities = {}
                for i, crop_class in enumerate(self.label_encoder.classes_):
                    all_probabilities[crop_class] = float(prediction_proba[i])
                
                # Ordenar por probabilidad
                sorted_crops = sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True)
                
                return {
                    "success": True,
                    "predicted_crop": predicted_crop,
                    "confidence": confidence,
                    "all_probabilities": all_probabilities,
                    "top_3_recommendations": sorted_crops[:3],
                    "input_data": mapped_params
                }
                
            except Exception as pred_error:
                logger.error(f"❌ Error en predicción: {pred_error}")
                return {
                    "success": False,
                    "error": "Error en predicción",
                    "details": str(pred_error)
                }
                
        except Exception as e:
            logger.error(f"❌ Error general en predicción: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": "Error general en predicción",
                "details": str(e)
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtiene información sobre el estado de los modelos cargados
        """
        try:
            if not self.is_loaded:
                return {
                    "loaded": False,
                    "error": "Modelos no cargados"
                }
            
            # Información del modelo
            model_info = {
                "loaded": True,
                "model_type": type(self.model).__name__,
                "available_crops": list(self.label_encoder.classes_) if self.label_encoder else [],
                "num_crops": len(self.label_encoder.classes_) if self.label_encoder else 0
            }
            
            # Información del preprocessor
            if hasattr(self.preprocessor, 'transformers_'):
                transformers_info = []
                for name, transformer, features in self.preprocessor.transformers_:
                    transformers_info.append({
                        "name": name,
                        "transformer": type(transformer).__name__,
                        "features": features
                    })
                model_info["preprocessing_pipeline"] = transformers_info
            
            return model_info
            
        except Exception as e:
            logger.error(f"Error obteniendo información del modelo: {e}")
            return {
                "loaded": False,
                "error": str(e)
            }
    
    def reload_models(self) -> bool:
        """
        Recarga los modelos manualmente
        """
        logger.info("🔄 Recargando modelos...")
        self.model = None
        self.label_encoder = None
        self.preprocessor = None
        self.is_loaded = False
        
        return self._load_models()
    
    def validate_input_parameters(self, terrain_params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Valida que los parámetros de entrada sean correctos
        """
        errors = []
        
        # Parámetros requeridos usando el mapeo centralizado
        required_params_english = STANDARD_COLUMNS['all_features']
        required_params_spanish = SPANISH_COLUMNS['all_features']
        
        # Verificar si tiene los parámetros en inglés O en español
        has_english = all(param in terrain_params for param in required_params_english)
        has_spanish = all(param in terrain_params for param in required_params_spanish)
        
        if not has_english and not has_spanish:
            missing_english = [p for p in required_params_english if p not in terrain_params]
            missing_spanish = [p for p in required_params_spanish if p not in terrain_params]
            errors.append(f"Faltan parámetros. En inglés: {missing_english} o en español: {missing_spanish}")
        
        # Validar rangos numéricos
        numeric_validations = {
            'ph': (0, 14, 'pH debe estar entre 0 y 14'),
            'humidity': (0, 100, 'Humedad debe estar entre 0 y 100%'),
            'humedad': (0, 100, 'Humedad debe estar entre 0 y 100%'),
            'temperature': (-50, 60, 'Temperatura debe estar entre -50 y 60°C'),
            'temperatura': (-50, 60, 'Temperatura debe estar entre -50 y 60°C'),
            'precipitation': (0, 5000, 'Precipitación debe estar entre 0 y 5000mm'),
            'precipitacion': (0, 5000, 'Precipitación debe estar entre 0 y 5000mm'),
            'sun_hours': (0, 24, 'Horas de sol debe estar entre 0 y 24'),
            'horas_de_sol': (0, 24, 'Horas de sol debe estar entre 0 y 24')
        }
        
        for param, value in terrain_params.items():
            if param in numeric_validations:
                min_val, max_val, error_msg = numeric_validations[param]
                try:
                    num_value = float(value)
                    if not (min_val <= num_value <= max_val):
                        errors.append(f"{error_msg}. Valor recibido: {num_value}")
                except (ValueError, TypeError):
                    errors.append(f"{param} debe ser un número válido. Valor recibido: {value}")
        
        # Validar valores categóricos
        valid_soil_types = ['clay', 'sandy', 'loamy', 'arcilloso', 'arenoso', 'franco']
        valid_seasons = ['spring', 'summer', 'autumn', 'winter', 'primavera', 'verano', 'otoño', 'invierno']
        
        soil_param = terrain_params.get('soil_type') or terrain_params.get('tipo_de_suelo')
        season_param = terrain_params.get('season') or terrain_params.get('temporada')
        
        if soil_param and soil_param.lower() not in [s.lower() for s in valid_soil_types]:
            errors.append(f"Tipo de suelo inválido: {soil_param}. Valores válidos: {valid_soil_types}")
        
        if season_param and season_param.lower() not in [s.lower() for s in valid_seasons]:
            errors.append(f"Temporada inválida: {season_param}. Valores válidos: {valid_seasons}")
        
        return len(errors) == 0, errors


prediction_service = PredictionService()