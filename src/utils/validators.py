import re
import pandas as pd
from typing import Dict, Any, List, Tuple
from src.config.config import VALIDATION_CONFIG, TERRAIN_PARAMS, SOIL_TYPES, SEASONS

class DataValidator:
    @staticmethod
    def validate_terrain_params(params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        
        if not TERRAIN_PARAMS["ph"]["min"] <= params.get("ph", 0) <= TERRAIN_PARAMS["ph"]["max"]:
            errors.append(f"pH must be between {TERRAIN_PARAMS['ph']['min']} and {TERRAIN_PARAMS['ph']['max']}")
        
        if not TERRAIN_PARAMS["humidity"]["min"] <= params.get("humidity", -1) <= TERRAIN_PARAMS["humidity"]["max"]:
            errors.append(f"Humidity must be between {TERRAIN_PARAMS['humidity']['min']} and {TERRAIN_PARAMS['humidity']['max']}")
        
        if not TERRAIN_PARAMS["temperature"]["min"] <= params.get("temperature", -1) <= TERRAIN_PARAMS["temperature"]["max"]:
            errors.append(f"Temperature must be between {TERRAIN_PARAMS['temperature']['min']} and {TERRAIN_PARAMS['temperature']['max']}")
        
        if not TERRAIN_PARAMS["precipitation"]["min"] <= params.get("precipitation", -1) <= TERRAIN_PARAMS["precipitation"]["max"]:
            errors.append(f"Precipitation must be between {TERRAIN_PARAMS['precipitation']['min']} and {TERRAIN_PARAMS['precipitation']['max']}")
        
        if not TERRAIN_PARAMS["sun_hours"]["min"] <= params.get("sun_hours", -1) <= TERRAIN_PARAMS["sun_hours"]["max"]:
            errors.append(f"Sun hours must be between {TERRAIN_PARAMS['sun_hours']['min']} and {TERRAIN_PARAMS['sun_hours']['max']}")
        
        if params.get("soil_type") not in SOIL_TYPES:
            errors.append(f"Soil type must be one of: {', '.join(SOIL_TYPES)}")
        
        if params.get("season") not in SEASONS:
            errors.append(f"Season must be one of: {', '.join(SEASONS)}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_crop_name(crop_name: str) -> Tuple[bool, List[str]]:
        errors = []
        
        if not crop_name or not crop_name.strip():
            errors.append("Crop name cannot be empty")
        elif len(crop_name.strip()) < VALIDATION_CONFIG["min_crop_name_length"]:
            errors.append(f"Crop name must have at least {VALIDATION_CONFIG['min_crop_name_length']} characters")
        elif len(crop_name.strip()) > VALIDATION_CONFIG["max_crop_name_length"]:
            errors.append(f"Crop name cannot have more than {VALIDATION_CONFIG['max_crop_name_length']} characters")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        errors = []
        required_fields = VALIDATION_CONFIG["required_fields"]
        
        missing_fields = [field for field in required_fields if field not in df.columns]
        if missing_fields:
            errors.append(f"Missing fields: {', '.join(missing_fields)}")
        
        if df.empty:
            errors.append("DataFrame is empty")
        
        return len(errors) == 0, errors

class ModelValidator:
    @staticmethod
    def validate_model_prediction(model, input_data: pd.DataFrame) -> Tuple[bool, List[str]]:
        errors = []
        
        try:
            prediction = model.predict(input_data)
            if prediction is None or len(prediction) == 0:
                errors.append("Model could not generate a prediction")
        except Exception as e:
            errors.append(f"Error during prediction: {str(e)}")
        
        return len(errors) == 0, errors 