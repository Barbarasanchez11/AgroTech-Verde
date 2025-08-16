import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from pathlib import Path
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

logger = logging.getLogger(__name__)

class SmartRetrainingService:
    def __init__(self, firebase_service):
        self.firebase_service = firebase_service
        self.dataset_path = Path("data/agroTech_data.csv")
        self.models_dir = Path("src/models")
        
    def collect_new_crop_data(self, min_examples: int = 5) -> Dict[str, List[Dict]]:
        
        try:
            from src.services.crop_normalizer import CropNormalizer
            
            all_crops = self.firebase_service.get_all_crops()
            if not all_crops:
                return {}
            
            
            normalizer = CropNormalizer()
            normalized_crops, normalization_log = normalizer.normalize_dataset(all_crops)
            
            if normalization_log:
                logger.info(f"Normalization applied: {normalization_log}")
            
           
            df = pd.DataFrame(normalized_crops)
            crop_groups = df.groupby('tipo_de_cultivo')
            
            collected_data = {}
            for crop_name, group in crop_groups:
                if len(group) >= min_examples:
                    collected_data[crop_name] = group.to_dict('records')
                    logger.info(f"Collected {len(group)} examples for {crop_name}")
                else:
                    logger.warning(f"Insufficient data for {crop_name}: {len(group)} examples (need {min_examples})")
            
            return collected_data
            
        except Exception as e:
            logger.error(f"Error collecting crop data: {e}")
            return {}
    
    def merge_with_original_dataset(self, new_data: Dict[str, List[Dict]]) -> pd.DataFrame:
       
        try:
           
            if self.dataset_path.exists():
                original_df = pd.read_csv(self.dataset_path)
                logger.info(f"Loaded original dataset with {len(original_df)} records")
            else:
                logger.warning("Original dataset not found, creating new one")
                original_df = pd.DataFrame()
            
           
            new_records = []
            for crop_name, records in new_data.items():
                for record in records:
                    new_record = {
                        'ph': record['ph'],
                        'humedad': record['humedad'],
                        'temperatura': record['temperatura'],
                        'precipitacion': record['precipitacion'],
                        'horas_de_sol': record['horas_de_sol'],
                        'tipo_de_suelo': record['tipo_de_suelo'],
                        'temporada': record['temporada'],
                        'crop': crop_name
                    }
                    new_records.append(new_record)
            
            if new_records:
                new_df = pd.DataFrame(new_records)
                logger.info(f"Prepared {len(new_df)} new records")
                
                
                if not original_df.empty:
                    combined_df = pd.concat([original_df, new_df], ignore_index=True)
                    logger.info(f"Combined dataset: {len(original_df)} original + {len(new_df)} new = {len(combined_df)} total")
                else:
                    combined_df = new_df
                    logger.info(f"Using only new data: {len(combined_df)} records")
                
                return combined_df
            else:
                logger.warning("No new data to merge")
                return original_df
                
        except Exception as e:
            logger.error(f"Error merging datasets: {e}")
            return pd.DataFrame()
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara features para el modelo"""
        try:
           
            feature_columns = ['ph', 'humedad', 'temperatura', 'precipitacion', 'horas_de_sol', 'tipo_de_suelo', 'temporada']
            X = df[feature_columns]
            y = df['crop']
            
            
            numeric_features = ['ph', 'humedad', 'temperatura', 'precipitacion', 'horas_de_sol']
            categorical_features = ['tipo_de_suelo', 'temporada']
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(drop='first', sparse=False), categorical_features)
                ]
            )
            
          
            X_processed = preprocessor.fit_transform(X)
            
           
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            
            logger.info(f"Features prepared: {X_processed.shape}, {len(label_encoder.classes_)} crop classes")
            return X_processed, y_encoded, preprocessor, label_encoder
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None, None, None, None
    
    def train_model(self, X: np.ndarray, y: np.ndarray, preprocessor, label_encoder) -> Tuple[Pipeline, float]:
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
           
            classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', classifier)
            ])
            
            
            pipeline.fit(X_train, y_train)
            
           
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            
            cv_scores = cross_val_score(pipeline, X, y, cv=5)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            logger.info(f"Model trained successfully:")
            logger.info(f"  Test accuracy: {accuracy:.4f}")
            logger.info(f"  CV accuracy: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
            logger.info(f"  Available crops: {list(label_encoder.classes_)}")
            
            return pipeline, accuracy, cv_mean
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return None, 0.0, 0.0
    
    def save_updated_model(self, pipeline, label_encoder, accuracy: float) -> bool:
       
        try:
            
            self.models_dir.mkdir(parents=True, exist_ok=True)
            
            
            model_path = self.models_dir / "modelo_random_forest.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(pipeline, f)
            
           
            encoder_path = self.models_dir / "label_encoder.pkl"
            with open(encoder_path, 'wb') as f:
                pickle.dump(label_encoder, f)
            
           
            updated_dataset_path = self.dataset_path
            updated_dataset_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Updated model saved successfully:")
            logger.info(f"  Model: {model_path}")
            logger.info(f"  Encoder: {encoder_path}")
            logger.info(f"  Final accuracy: {accuracy:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving updated model: {e}")
            return False
    
    def retrain_with_new_data(self, min_examples: int = 5) -> Dict[str, Any]:
       
        try:
            logger.info("Starting smart retraining process...")
            
           
            new_data = self.collect_new_crop_data(min_examples)
            if not new_data:
                return {"success": False, "error": "No data available for retraining"}
            
           
            combined_df = self.merge_with_original_dataset(new_data)
            if combined_df.empty:
                return {"success": False, "error": "Failed to merge datasets"}
            
     
            result = self.prepare_features(combined_df)
            if result[0] is None:
                return {"success": False, "error": "Failed to prepare features"}
            
            X, y, preprocessor, label_encoder = result
            
           
            result = self.train_model(X, y, preprocessor, label_encoder)
            if result[0] is None:
                return {"success": False, "error": "Failed to train model"}
            
            pipeline, accuracy, cv_accuracy = result
            
           
            if not self.save_updated_model(pipeline, label_encoder, accuracy):
                return {"success": False, "error": "Failed to save updated model"}
            
           
            combined_df.to_csv(self.dataset_path, index=False)
            
            return {
                "success": True,
                "accuracy": accuracy,
                "cv_accuracy": cv_accuracy,
                "total_records": len(combined_df),
                "crop_classes": list(label_encoder.classes_),
                "new_crops_added": len(new_data),
                "message": f"Model successfully retrained with {len(combined_df)} total records and {len(label_encoder.classes_)} crop classes"
            }
            
        except Exception as e:
            logger.error(f"Error in smart retraining: {e}")
            return {"success": False, "error": str(e)}
    
    def get_retraining_status(self) -> Dict[str, Any]:
        
        try:
           
            new_data = self.collect_new_crop_data(min_examples=1)
            total_examples = sum(len(records) for records in new_data.values())
            
       
            model_path = self.models_dir / "modelo_random_forest.pkl"
            encoder_path = self.models_dir / "label_encoder.pkl"
            
            status = {
                "available_crops": list(new_data.keys()),
                "total_examples": total_examples,
                "model_exists": model_path.exists(),
                "encoder_exists": encoder_path.exists(),
                "can_retrain": total_examples >= 5,
                "recommended_examples": 5
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting retraining status: {e}")
            return {"error": str(e)} 