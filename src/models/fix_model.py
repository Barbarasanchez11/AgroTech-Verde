import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import unicodedata

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

from src.config.config import get_data_path, get_model_path, ML_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_text(text: str) -> str:
    
    if pd.isna(text):
        return text
    
   
    text = str(text).lower()
    
  
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    
    return text

def clean_dataset():
    
    data_path = get_data_path("agroTech_data.csv")
    df = pd.read_csv(data_path)
    
    logger.info(f"Dataset original: {df.shape}")
    logger.info(f"Cultivos únicos originales: {df['tipo_de_cultivo'].unique()}")
    

    df['tipo_de_cultivo'] = df['tipo_de_cultivo'].apply(normalize_text)
    
   
    unique_crops = df['tipo_de_cultivo'].unique()
    logger.info(f"Cultivos únicos después de normalización: {unique_crops}")
    
    
    valid_crops = ['arroz', 'lentejas', 'maiz', 'naranjas', 'soja', 'trigo', 'uva', 'zanahoria']
    df_clean = df[df['tipo_de_cultivo'].isin(valid_crops)].copy()
    
    logger.info(f"Dataset limpio: {df_clean.shape}")
    logger.info(f"Cultivos en dataset limpio: {df_clean['tipo_de_cultivo'].unique()}")
    
    return df_clean

def train_clean_model():
    
    df_clean = clean_dataset()
    
    X = df_clean.drop(columns=["tipo_de_cultivo"])
    y = df_clean["tipo_de_cultivo"]
    
    logger.info(f"Features: {X.shape}, Target: {y.shape}")
    
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    logger.info(f"LabelEncoder classes: {label_encoder.classes_}")
    
  
    numeric_columns = ["ph", "humedad", "temperatura", "precipitacion", "horas_de_sol"]
    categorical_columns = ["tipo_de_suelo", "temporada"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            ("categorical", OneHotEncoder(drop='first', sparse=False), categorical_columns)
        ],
        remainder='passthrough'
    )
    
 
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=ML_CONFIG["random_state"]))
    ])
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=ML_CONFIG["test_size"], 
        random_state=ML_CONFIG["random_state"]
    )
    
   
    pipeline.fit(X_train, y_train)
    
   
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Modelo entrenado con accuracy: {accuracy:.4f}")
    
   
    model_path = get_model_path("random_forest")
    encoder_path = get_model_path("label_encoder")
    
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
    
    logger.info(f"Modelo guardado en: {model_path}")
    logger.info(f"Encoder guardado en: {encoder_path}")
    
    return pipeline, label_encoder, accuracy

def main():
   
    try:
        logger.info("Iniciando corrección del modelo...")
        
        pipeline, encoder, accuracy = train_clean_model()
        
        logger.info("Modelo corregido exitosamente!")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f" Cultivos soportados: {list(encoder.classes_)}")
        
        return True
        
    except Exception as e:
        logger.error(f" Error corrigiendo modelo: {e}")
        return False

if __name__ == "__main__":
    main() 