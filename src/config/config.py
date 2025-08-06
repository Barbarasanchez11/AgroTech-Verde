import os
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "src" / "models"
STYLE_FILE = PROJECT_ROOT / "src" / "style.css"

APP_CONFIG = {
    "page_title": "AgroTech Verde",
    "page_icon": "🌱",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

TERRAIN_PARAMS = {
    "ph": {"min": 4.5, "max": 8.5, "default": 6.5, "step": 0.1},
    "humedad": {"min": 0, "max": 100, "default": 50, "step": 1},
    "temperatura": {"min": 0, "max": 40, "default": 20, "step": 1},
    "precipitacion": {"min": 0, "max": 300, "default": 150, "step": 5},
    "horas_de_sol": {"min": 0, "max": 16, "default": 8, "step": 0.5}
}

SOIL_TYPES = ["arcilloso", "arenoso", "limoso", "rocoso"]
SEASONS = ['verano', 'otoño', 'invierno', 'primavera']

ML_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "model_files": {
        "random_forest": "modelo_random_forest.pkl",
        "label_encoder": "label_encoder.pkl"
    }
}

FIREBASE_CONFIG = {
    "collection_name": "cultivos",
    "credentials_key": "firebase"
}

VALIDATION_CONFIG = {
    "min_crop_name_length": 2,
    "max_crop_name_length": 50,
    "required_fields": ["tipo_de_cultivo", "ph", "humedad", "temperatura", 
                       "precipitacion", "horas_de_sol", "tipo_de_suelo", "temporada"]
}

def get_model_path(model_name: str) -> Path:
    if model_name in ML_CONFIG["model_files"]:
        return MODELS_DIR / ML_CONFIG["model_files"][model_name]
    else:
        return MODELS_DIR / f"{model_name}.pkl"

def get_data_path(filename: str) -> Path:
    return DATA_DIR / filename

def validate_environment() -> bool:
    required_dirs = [DATA_DIR, MODELS_DIR]
    required_files = [STYLE_FILE]
    
    for dir_path in required_dirs:
        if not dir_path.exists():
            print(f"Required directory not found: {dir_path}")
            return False
    
    for file_path in required_files:
        if not file_path.exists():
            print(f"Required file not found: {file_path}")
            return False
    
    return True 