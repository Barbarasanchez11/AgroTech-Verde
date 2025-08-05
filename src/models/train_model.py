"""
Script de entrenamiento mejorado para AgroTech-Verde
"""
import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
from datetime import datetime

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from src.config.config import get_data_path, get_model_path, ML_CONFIG
from src.utils.validators import DataValidator

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Clase para entrenar modelos de ML"""
    
    def __init__(self):
        self.data_path = get_data_path("agrotech_data.csv")
        self.models_dir = Path(get_model_path("random_forest")).parent
        
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Carga y prepara los datos para entrenamiento
        
        Returns:
            Tuple con X (features) e y (target)
        """
        try:
            # Cargar datos
            df = pd.read_csv(self.data_path)
            logger.info(f"Datos cargados: {df.shape}")
            
            # Validar datos
            is_valid, errors = DataValidator.validate_dataframe(df)
            if not is_valid:
                raise ValueError(f"Datos inválidos: {'; '.join(errors)}")
            
            # Preparar features y target
            X = df.drop(columns=["tipo_de_cultivo"])
            y = df["tipo_de_cultivo"]
            
            logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
            return X, y
            
        except Exception as e:
            logger.error(f"Error al cargar datos: {str(e)}")
            raise
    
    def create_preprocessor(self) -> ColumnTransformer:
        """
        Crea el preprocesador para las features
        
        Returns:
            ColumnTransformer configurado
        """
        # Columnas numéricas y categóricas
        num_cols = ["ph", "humedad", "temperatura", "precipitacion", "horas_de_sol"]
        cat_cols = ["tipo_de_suelo", "temporada"]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(drop='first', sparse=False), cat_cols)
            ],
            remainder='passthrough'
        )
        
        return preprocessor
    
    def train_random_forest(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Pipeline, LabelEncoder, Dict[str, Any]]:
        """
        Entrena un modelo Random Forest
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Tuple con modelo, encoder y métricas
        """
        # Codificar target
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Crear pipeline
        preprocessor = self.create_preprocessor()
        
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=ML_CONFIG["random_state"]
            ))
        ])
        
        # Split de datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=ML_CONFIG["test_size"], 
            random_state=ML_CONFIG["random_state"],
            stratify=y_encoded
        )
        
        # Entrenar modelo
        logger.info("Entrenando Random Forest...")
        pipeline.fit(X_train, y_train)
        
        # Evaluar modelo
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Validación cruzada
        cv_scores = cross_val_score(pipeline, X, y_encoded, cv=5)
        
        # Métricas
        metrics = {
            "accuracy": accuracy,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "classification_report": classification_report(
                y_test, y_pred, 
                target_names=label_encoder.classes_,
                output_dict=True
            )
        }
        
        logger.info(f"Random Forest - Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return pipeline, label_encoder, metrics
    
    def train_svm(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Pipeline, LabelEncoder, Dict[str, Any]]:
        """
        Entrena un modelo SVM
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Tuple con modelo, encoder y métricas
        """
        # Codificar target
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Crear pipeline
        preprocessor = self.create_preprocessor()
        
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", SVC(
                kernel='rbf',
                probability=True,
                random_state=ML_CONFIG["random_state"]
            ))
        ])
        
        # Split de datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=ML_CONFIG["test_size"], 
            random_state=ML_CONFIG["random_state"],
            stratify=y_encoded
        )
        
        # Entrenar modelo
        logger.info("Entrenando SVM...")
        pipeline.fit(X_train, y_train)
        
        # Evaluar modelo
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Validación cruzada
        cv_scores = cross_val_score(pipeline, X, y_encoded, cv=5)
        
        # Métricas
        metrics = {
            "accuracy": accuracy,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "classification_report": classification_report(
                y_test, y_pred, 
                target_names=label_encoder.classes_,
                output_dict=True
            )
        }
        
        logger.info(f"SVM - Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return pipeline, label_encoder, metrics
    
    def save_model(self, model: Pipeline, encoder: LabelEncoder, model_name: str):
        """
        Guarda el modelo y encoder
        
        Args:
            model: Modelo entrenado
            encoder: Label encoder
            model_name: Nombre del modelo
        """
        try:
            # Guardar modelo
            model_path = self.models_dir / f"{model_name}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            
            # Guardar encoder
            encoder_path = self.models_dir / f"{model_name}_encoder.pkl"
            with open(encoder_path, "wb") as f:
                pickle.dump(encoder, f)
            
            logger.info(f"Modelo guardado: {model_path}")
            logger.info(f"Encoder guardado: {encoder_path}")
            
        except Exception as e:
            logger.error(f"Error al guardar modelo: {str(e)}")
            raise
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            classes: np.ndarray, title: str):
        """
        Plotea matriz de confusión
        
        Args:
            y_true: Valores reales
            y_pred: Predicciones
            classes: Nombres de las clases
            title: Título del gráfico
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes)
        plt.title(title)
        plt.ylabel('Valor Real')
        plt.xlabel('Predicción')
        plt.tight_layout()
        
        # Guardar gráfico
        plot_path = self.models_dir / f"{title.lower().replace(' ', '_')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Matriz de confusión guardada: {plot_path}")
    
    def train_all_models(self) -> Dict[str, Any]:
        """
        Entrena todos los modelos y retorna métricas
        
        Returns:
            Dict con métricas de todos los modelos
        """
        # Cargar datos
        X, y = self.load_and_prepare_data()
        
        results = {}
        
        # Entrenar Random Forest
        rf_model, rf_encoder, rf_metrics = self.train_random_forest(X, y)
        self.save_model(rf_model, rf_encoder, "modelo_rf")
        
        # Entrenar SVM
        svm_model, svm_encoder, svm_metrics = self.train_svm(X, y)
        self.save_model(svm_model, svm_encoder, "modelo_svm")
        
        # Comparar modelos
        results = {
            "random_forest": rf_metrics,
            "svm": svm_metrics,
            "best_model": "random_forest" if rf_metrics["accuracy"] > svm_metrics["accuracy"] else "svm"
        }
        
        # Guardar métricas
        metrics_path = self.models_dir / "training_metrics.json"
        import json
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Métricas guardadas: {metrics_path}")
        logger.info(f"Mejor modelo: {results['best_model']}")
        
        return results

def main():
    """Función principal"""
    trainer = ModelTrainer()
    
    try:
        logger.info("Iniciando entrenamiento de modelos...")
        results = trainer.train_all_models()
        
        logger.info("✅ Entrenamiento completado exitosamente")
        logger.info(f"Mejor modelo: {results['best_model']}")
        
    except Exception as e:
        logger.error(f"❌ Error durante el entrenamiento: {str(e)}")
        raise

if __name__ == "__main__":
    main() 