import os
from pathlib import Path
from typing import Dict, Any, Optional

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
    "humidity": {"min": 0, "max": 100, "default": 50, "step": 1},
    "temperature": {"min": 0, "max": 40, "default": 20, "step": 1},
    "precipitation": {"min": 0, "max": 300, "default": 150, "step": 5},
    "sun_hours": {"min": 0.0, "max": 16.0, "default": 8.0, "step": 0.5}
}

SOIL_TYPES = ["arcilloso", "arenoso", "limoso", "rocoso"]
SEASONS = ['verano', 'otoño', 'invierno', 'primavera']


COLUMN_MAPPING = {
    'humidity': 'humedad',
    'temperature': 'temperatura',
    'precipitation': 'precipitacion',
    'sun_hours': 'horas_de_sol',
    'soil_type': 'tipo_de_suelo',
    'season': 'temporada',
    'crop_type': 'tipo_de_cultivo'
}


STANDARD_COLUMNS = {
    'numeric_features': ['ph', 'humidity', 'temperature', 'precipitation', 'sun_hours'],
    'categorical_features': ['soil_type', 'season'],
    'target_column': 'crop_type',
    'all_features': ['ph', 'humidity', 'temperature', 'precipitation', 'sun_hours', 'soil_type', 'season']
}


SPANISH_COLUMNS = {
    'numeric_features': ['ph', 'humedad', 'temperatura', 'precipitacion', 'horas_de_sol'],
    'categorical_features': ['tipo_de_suelo', 'temporada'],
    'target_column': 'tipo_de_cultivo',
    'all_features': ['ph', 'humedad', 'temperatura', 'precipitacion', 'horas_de_sol', 'tipo_de_suelo', 'temporada']
}

ML_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "model_files": {
        "random_forest": "modelo_random_forest.pkl",
        "svm": "svm.pkl",  
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

def get_model_path(model_name: str) -> Optional[Path]:
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        if model_name in ML_CONFIG["model_files"]:
            model_path = MODELS_DIR / ML_CONFIG["model_files"][model_name]
        else:
            filename = f"{model_name}.pkl" if not model_name.endswith('.pkl') else model_name
            model_path = MODELS_DIR / filename
        
        return model_path
        
    except Exception as e:
        print(f"Error getting model path for {model_name}: {e}")
        return None

def get_data_path(filename: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / filename

def validate_environment() -> bool:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        if not STYLE_FILE.exists():
            print(f"Warning: Style file not found: {STYLE_FILE}")
            try:
                STYLE_FILE.parent.mkdir(parents=True, exist_ok=True)
                with open(STYLE_FILE, 'w') as f:
                    f.write("/* AgroTech Verde Styles */\n")
                print(f"Created basic style file: {STYLE_FILE}")
            except Exception as e:
                print(f"Could not create style file: {e}")
        
        return True
        
    except Exception as e:
        print(f"Error validating environment: {e}")
        return False
    
def get_model_info() -> Dict[str, Any]:
    info = {
        "models_dir": str(MODELS_DIR),
        "available_models": {},
        "missing_models": []
    }
    
    for model_name, filename in ML_CONFIG["model_files"].items():
        model_path = get_model_path(model_name)
        if model_path and model_path.exists():
            info["available_models"][model_name] = {
                "path": str(model_path),
                "size": model_path.stat().st_size if model_path.exists() else 0,
                "exists": True
            }
        else:
            info["missing_models"].append(model_name)
            info["available_models"][model_name] = {
                "path": str(model_path) if model_path else "unknown",
                "exists": False
            }
    
    return info 