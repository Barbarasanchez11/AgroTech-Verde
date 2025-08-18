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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.data_path = get_data_path("agrotech_data.csv")
        self.models_dir = Path(get_model_path("random_forest")).parent
        
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        try:
            df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded: {df.shape}")
            
            is_valid, errors = DataValidator.validate_dataframe(df)
            if not is_valid:
                raise ValueError(f"Invalid data: {'; '.join(errors)}")
            
                X = df.drop(columns=["crop_type"])
    y = df["crop_type"]
            
            logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def create_preprocessor(self) -> ColumnTransformer:
            numeric_columns = ["ph", "humidity", "temperature", "precipitation", "sun_hours"]
    categorical_columns = ["soil_type", "season"]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", StandardScaler(), numeric_columns),
                ("categorical", OneHotEncoder(drop='first', sparse=False), categorical_columns)
            ],
            remainder='passthrough'
        )
        
        return preprocessor
    
    def train_random_forest(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Pipeline, LabelEncoder, Dict[str, Any]]:
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        preprocessor = self.create_preprocessor()
        
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
        
        metrics = {
            "accuracy": accuracy,
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }
        
        logger.info(f"Random Forest accuracy: {accuracy:.4f}")
        
        return pipeline, label_encoder, metrics
    
    def train_svm(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Pipeline, LabelEncoder, Dict[str, Any]]:
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        preprocessor = self.create_preprocessor()
        
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", SVC(probability=True, random_state=ML_CONFIG["random_state"]))
        ])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=ML_CONFIG["test_size"], 
            random_state=ML_CONFIG["random_state"]
        )
        
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        metrics = {
            "accuracy": accuracy,
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }
        
        logger.info(f"SVM accuracy: {accuracy:.4f}")
        
        return pipeline, label_encoder, metrics
    
    def save_model(self, model: Pipeline, encoder: LabelEncoder, model_name: str):
        try:
            model_path = get_model_path(f"{model_name}.pkl")
            encoder_path = get_model_path("label_encoder.pkl")
            
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            
            with open(encoder_path, "wb") as f:
                pickle.dump(encoder, f)
            
            logger.info(f"Model saved to {model_path}")
            logger.info(f"Encoder saved to {encoder_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            classes: np.ndarray, title: str):
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        matrix_path = get_model_path("confusion_matrix.png")
        plt.savefig(matrix_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {matrix_path}")
    
    def train_all_models(self) -> Dict[str, Any]:
        try:
            X, y = self.load_and_prepare_data()
            
            results = {}
            
            rf_model, rf_encoder, rf_metrics = self.train_random_forest(X, y)
            results["random_forest"] = {
                "model": rf_model,
                "encoder": rf_encoder,
                "metrics": rf_metrics
            }
            
            svm_model, svm_encoder, svm_metrics = self.train_svm(X, y)
            results["svm"] = {
                "model": svm_model,
                "encoder": svm_encoder,
                "metrics": svm_metrics
            }
            
            best_model = "random_forest" if rf_metrics["accuracy"] > svm_metrics["accuracy"] else "svm"
            best_result = results[best_model]
            
            self.save_model(best_result["model"], best_result["encoder"], best_model)
            
            logger.info(f"Best model: {best_model} with accuracy: {best_result['metrics']['accuracy']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            raise

def main():
    trainer = ModelTrainer()
    results = trainer.train_all_models()
    
    logger.info("Training completed successfully")
    return results

if __name__ == "__main__":
    main() 