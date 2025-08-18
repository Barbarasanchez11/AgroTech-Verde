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
    def __init__(self, database_service):
        self.database_service = database_service
        self.dataset_path = Path("data/agroTech_data.csv")
        self.models_dir = Path("src/models")
        
    def collect_new_crop_data(self, min_examples: int = 5) -> Dict[str, List[Dict]]:
        
        try:
            from src.services.crop_normalizer import CropNormalizer
            
            all_crops = self.database_service.get_all_crops()
            if not all_crops:
                return {}
            
            
            normalizer = CropNormalizer()
            normalized_crops, normalization_log = normalizer.normalize_dataset(all_crops)
            
            if normalization_log:
                logger.info(f"Normalization applied: {normalization_log}")
            
           
            df = pd.DataFrame(normalized_crops)
            crop_groups = df.groupby('crop_type')
            
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
                        'humidity': record['humidity'],
                        'temperature': record['temperature'],
                        'precipitation': record['precipitation'],
                        'sun_hours': record['sun_hours'],
                        'soil_type': record['soil_type'],
                        'season': record['season'],
                        'crop_type': crop_name
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
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer, LabelEncoder]:

        try:
            
            required_columns = ['ph', 'humidity', 'temperature', 'precipitation', 'sun_hours', 'soil_type', 'season', 'crop_type']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing columns: {missing_columns}")
                logger.error(f"Available columns: {list(df.columns)}")
                return None, None, None, None
            
            feature_columns = ['ph', 'humidity', 'temperature', 'precipitation', 'sun_hours', 'soil_type', 'season']
            X = df[feature_columns]
            y = df['crop_type']
            
        
            if X.isnull().any().any() or y.isnull().any():
                logger.error("Datos con valores nulos detectados")
                return None, None, None, None
            
           
            numeric_features = ['ph', 'humidity', 'temperature', 'precipitation', 'sun_hours']
            categorical_features = ['soil_type', 'season']
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(drop='first', sparse=False), categorical_features)
                ]
            )
            
       
            label_encoder = LabelEncoder()
            
            logger.info(f"Features preparados: X={X.shape}, y={len(y)} clases")
            return X, y, preprocessor, label_encoder
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None, None, None, None
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, preprocessor, label_encoder) -> Tuple[Pipeline, float]:
       
        try:
            
            logger.info("Procesando features...")
            X_processed = preprocessor.fit_transform(X)
            
            logger.info(" Codificando target...")
            y_encoded = label_encoder.fit_transform(y)
            
            
            unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
            logger.info(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
            
            
            min_class_count = np.min(class_counts)
            can_stratify = min_class_count >= 2
            
            if can_stratify:
                logger.info(" Usando stratified split (todas las clases tienen ≥2 ejemplos)")
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                )
            else:
                logger.warning(f"Cannot use stratified split (minimum class: {min_class_count})")
                logger.info("Using simple random split")
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y_encoded, test_size=0.2, random_state=42, stratify=None
                )
            
            
            classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            
          
            pipeline = Pipeline([
                ('classifier', classifier)
            ])
            
           
            logger.info(" Entrenando RandomForest...")
            pipeline.fit(X_train, y_train)
            
            
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
           
            cv_accuracy = 0.0
            if can_stratify and len(unique_classes) >= 3:
                try:
                    cv_scores = cross_val_score(pipeline, X_processed, y_encoded, cv=min(3, min_class_count), n_jobs=-1)
                    cv_accuracy = cv_scores.mean()
                    logger.info(f"Cross-validation: {cv_accuracy:.4f}")
                except Exception as cv_error:
                    logger.warning(f"Cross-validation failed: {cv_error}")
            else:
                logger.info("Skipping cross-validation (insufficient examples)")
            
            logger.info(f" Modelo entrenado exitosamente:")
            logger.info(f"  Test accuracy: {accuracy:.4f}")
            logger.info(f"  CV accuracy: {cv_accuracy:.4f}")
            logger.info(f"  Cultivos disponibles: {list(label_encoder.classes_)}")
            
            return pipeline, accuracy, cv_accuracy
            
        except Exception as e:
            logger.error(f" Error entrenando modelo: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None, 0.0, 0.0
    
    def save_updated_model(self, pipeline, label_encoder, preprocessor, accuracy: float) -> bool:
       
        try:
            self.models_dir.mkdir(parents=True, exist_ok=True)
            
           
            model_path = self.models_dir / "modelo_random_forest.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(pipeline, f)
            
            
            encoder_path = self.models_dir / "label_encoder.pkl"
            with open(encoder_path, 'wb') as f:
                pickle.dump(label_encoder, f)
            
            
            preprocessor_path = self.models_dir / "preprocessor.pkl"
            with open(preprocessor_path, 'wb') as f:
                pickle.dump(preprocessor, f)
            
            logger.info(f" Modelo actualizado guardado exitosamente:")
            logger.info(f"  Modelo: {model_path}")
            logger.info(f"  Encoder: {encoder_path}")
            logger.info(f"  Preprocessor: {preprocessor_path}")
            logger.info(f"  Final accuracy: {accuracy:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f" Error guardando modelo actualizado: {e}")
            return False
    
    def retrain_with_new_data(self, min_examples: int = 5) -> Dict[str, Any]:
        
        try:
            logger.info(" Iniciando proceso de reentrenamiento inteligente...")
            
           
            logger.info("Cargando dataset original")
            if not self.dataset_path.exists():
                return {"success": False, "error": "Dataset original no encontrado"}
            
            original_df = pd.read_csv(self.dataset_path)
            logger.info(f" Dataset original cargado: {len(original_df)} registros")
            
            
            logger.info(" Obteniendo nuevos datos de Supabase...")
            new_crops = self.database_service.get_all_crops()
            if not new_crops:
                return {"success": False, "error": "No hay nuevos datos en Supabase"}
            
            logger.info(f" Nuevos datos obtenidos: {len(new_crops)} registros")
            
            
            new_records = []
            for crop in new_crops:
               
                if not crop.get('is_prediction', False):
                    new_record = {
                        'ph': float(crop['ph']),
                        'humidity': float(crop['humidity']),
                        'temperature': float(crop['temperature']),
                        'precipitation': float(crop['precipitation']),
                        'sun_hours': float(crop['sun_hours']),
                        'soil_type': str(crop['soil_type']),
                        'season': str(crop['season']),
                        'crop_type': str(crop['crop_type'])
                    }
                    new_records.append(new_record)
            
            if not new_records:
                return {"success": False, "error": "No hay datos válidos para combinar"}
            
            new_df = pd.DataFrame(new_records)
            logger.info(f" Nuevos datos preparados: {len(new_df)} registros")
            
           
            logger.info("Combinando datasets...")
            combined_df = pd.concat([original_df, new_df], ignore_index=True)
            logger.info(f" Datasets combinados: {len(original_df)} original + {len(new_df)} nuevos = {len(combined_df)} total")
            
         
            logger.info(" Preparando features...")
            result = self.prepare_features(combined_df)
            if result[0] is None:
                return {"success": False, "error": "Error preparando features"}
            
            X, y, preprocessor, label_encoder = result
            logger.info(f"Features preparados: X={X.shape}, y={len(y)} clases")
            
           
            logger.info(" Entrenando modelo...")
            result = self.train_model(X, y, preprocessor, label_encoder)
            if result[0] is None:
                return {"success": False, "error": "Error entrenando modelo"}
            
            pipeline, accuracy, cv_accuracy = result
            logger.info(f"Model trained: accuracy={accuracy:.4f}, CV={cv_accuracy:.4f}")
            
           
            logger.info(" Guardando modelo...")
            if not self.save_updated_model(pipeline, label_encoder, preprocessor, accuracy):
                return {"success": False, "error": "Error guardando modelo"}
            
            
            combined_df.to_csv(self.dataset_path, index=False)
            logger.info(" Dataset combinado guardado")
            
            return {
                "success": True,
                "accuracy": accuracy,
                "cv_accuracy": cv_accuracy,
                "total_records": len(combined_df),
                "crop_classes": list(label_encoder.classes_),
                "new_crops_added": len(new_df),
                "message": f" Modelo reentrenado exitosamente con {len(combined_df)} registros totales y {len(label_encoder.classes_)} tipos de cultivo"
            }
            
        except Exception as e:
            logger.error(f" Error en reentrenamiento inteligente: {e}")
            import traceback
            logger.error(f"Traceback completo: {traceback.format_exc()}")
            return {"success": False, "error": f"Error detallado: {str(e)}"}
    
    def get_retraining_status(self) -> Dict[str, Any]:
       
        try:
            
            new_crops = self.database_service.get_all_crops()
            if not new_crops:
                return {"error": "No hay datos disponibles"}
            
           
            unique_crops = set()
            total_examples = 0
            
            for crop in new_crops:
                if not crop.get('is_prediction', False):
                    unique_crops.add(crop['crop_type'].lower().strip())
                    total_examples += 1
            
           
            model_path = self.models_dir / "modelo_random_forest.pkl"
            encoder_path = self.models_dir / "label_encoder.pkl"
            
            status = {
                "available_crops": list(unique_crops),
                "total_examples": total_examples,
                "model_exists": model_path.exists(),
                "encoder_exists": encoder_path.exists(),
                "can_retrain": total_examples >= 5,
                "recommended_examples": 5
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error obteniendo estado del sistema: {e}")
            return {"error": str(e)} 