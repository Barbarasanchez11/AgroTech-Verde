import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle
import logging
from pathlib import Path
import unicodedata
import re

from src.config.config import get_data_path, get_model_path, ML_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_text(text):
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text).lower()
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def clean_dataset():
    try:
        data_path = get_data_path("agroTech_data.csv")
        if not data_path or not Path(data_path).exists():
            logger.error(f"Data path not found: {data_path}")
            return None
        
        df = pd.read_csv(data_path)
        logger.info(f"Dataset loaded: {df.shape}")
        
        expected_crops = ['arroz', 'lentejas', 'maiz', 'naranjas', 'soja', 'trigo', 'uva', 'zanahoria']
        
        df['tipo_de_cultivo'] = df['tipo_de_cultivo'].apply(normalize_text)
        
        df_clean = df[df['tipo_de_cultivo'].isin(expected_crops)].copy()
        
        logger.info(f"Cleaned dataset: {df_clean.shape}")
        logger.info(f"Unique crops after cleaning: {df_clean['tipo_de_cultivo'].unique()}")
        
        return df_clean
        
    except Exception as e:
        logger.error(f"Error cleaning dataset: {e}")
        return None

def train_compatible_model():
    try:
        df_clean = clean_dataset()
        if df_clean is None or df_clean.empty:
            logger.error("No clean data available for training")
            return None, None, 0.0
        
        feature_columns = ['ph', 'humedad', 'temperatura', 'precipitacion', 'horas_de_sol']
        categorical_columns = ['tipo_de_suelo', 'temporada']
        target_column = 'tipo_de_cultivo'
        
        X = df_clean[feature_columns + categorical_columns]
        y = df_clean[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, feature_columns),
                ('cat', categorical_transformer, categorical_columns)
            ])
        
        # Usar RandomForestClassifier sin parámetros avanzados para compatibilidad
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        label_encoder = LabelEncoder()
        label_encoder.fit(y)
        
        logger.info(f"Compatible model trained successfully with accuracy: {accuracy:.4f}")
        logger.info(f"Label encoder classes: {list(label_encoder.classes_)}")
        
        return pipeline, label_encoder, accuracy
        
    except Exception as e:
        logger.error(f"Error training compatible model: {e}")
        return None, None, 0.0

def save_model(pipeline, label_encoder):
    try:
        model_path = get_model_path("random_forest")
        encoder_path = get_model_path("label_encoder")
        
        if model_path and encoder_path:
            with open(model_path, 'wb') as f:
                pickle.dump(pipeline, f)
            
            with open(encoder_path, 'wb') as f:
                pickle.dump(label_encoder, f)
            
            logger.info(f"Model and encoder saved successfully")
            return True
        else:
            logger.error("Model paths not configured")
            return False
            
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return False

if __name__ == "__main__":
    print("🔄 Reentrenando modelo para compatibilidad con Streamlit Cloud...")
    
    pipeline, label_encoder, accuracy = train_compatible_model()
    
    if pipeline and label_encoder:
        if save_model(pipeline, label_encoder):
            print(f"✅ Modelo reentrenado exitosamente")
            print(f"📊 Precisión: {accuracy:.4f}")
            print(f"🌱 Cultivos disponibles: {list(label_encoder.classes_)}")
            print("\n🚀 Ahora haz push del modelo actualizado:")
            print("git add src/models/")
            print("git commit -m 'feat: retrain model for Streamlit Cloud compatibility'")
            print("git push origin main")
        else:
            print("❌ Error al guardar el modelo")
    else:
        print("❌ Error al reentrenar el modelo") 