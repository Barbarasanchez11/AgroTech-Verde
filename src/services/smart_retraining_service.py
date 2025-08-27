import os
import pickle
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from src.config.config import STANDARD_COLUMNS, SPANISH_COLUMNS, COLUMN_MAPPING

logger = logging.getLogger(__name__)

class SmartRetrainingService:
    def __init__(self, database_service):
        self.database_service = database_service
        self.dataset_path = Path("data/agroTech_data.csv")
        self.models_dir = Path("src/models")
        
       
        self.STANDARD_COLUMNS = STANDARD_COLUMNS
        self.SPANISH_COLUMNS = SPANISH_COLUMNS
        self.SPANISH_TO_ENGLISH = COLUMN_MAPPING
    
    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
       
        try:
            logger.info(f" Columnas originales: {list(df.columns)}")
            
          
            normalized_df = df.copy()
            
           
            columns_to_rename = {}
            for col in normalized_df.columns:
                if col in self.SPANISH_TO_ENGLISH:
                    columns_to_rename[col] = self.SPANISH_TO_ENGLISH[col]
            
            if columns_to_rename:
                normalized_df = normalized_df.rename(columns=columns_to_rename)
                logger.info(f" Columnas renombradas: {columns_to_rename}")
            
            logger.info(f"Columnas normalizadas: {list(normalized_df.columns)}")
            return normalized_df
            
        except Exception as e:
            logger.error(f" Error normalizando nombres de columnas: {e}")
            return df
    
    def collect_new_crop_data(self, min_examples: int = 5) -> Dict[str, List[Dict]]:
       
        try:
          
            all_crops = self.database_service.get_all_crops()
            if not all_crops:
                logger.warning(" No se encontraron cultivos en la base de datos")
                return {}
            
            logger.info(f" Obtenidos {len(all_crops)} registros de la base de datos")
            
          
            real_crops = [crop for crop in all_crops if not crop.get('is_prediction', False)]
            logger.info(f"Datos reales (no predicciones): {len(real_crops)}")
            
            if not real_crops:
                logger.warning("⚠️ No hay datos reales para entrenar")
                return {}
            
           
            df = pd.DataFrame(real_crops)
            df = self._normalize_column_names(df)
            
      
            if 'crop_type' not in df.columns:
                logger.error(" Columna 'crop_type' no encontrada después de normalización")
                return {}
            
            crop_groups = df.groupby('crop_type')
            
            collected_data = {}
            for crop_name, group in crop_groups:
                if len(group) >= min_examples:
                    collected_data[crop_name] = group.to_dict('records')
                    logger.info(f" Recolectados {len(group)} ejemplos para {crop_name}")
                else:
                    logger.warning(f" Datos insuficientes para {crop_name}: {len(group)} ejemplos (necesarios {min_examples})")
            
            logger.info(f" Total de cultivos con datos suficientes: {len(collected_data)}")
            return collected_data
            
        except Exception as e:
            logger.error(f"Error recolectando datos de cultivos: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {}
    
    def merge_with_original_dataset(self, new_data: Dict[str, List[Dict]]) -> pd.DataFrame:
        
        try:
            
            if self.dataset_path.exists():
                original_df = pd.read_csv(self.dataset_path)
                original_df = self._normalize_column_names(original_df)
                logger.info(f" Dataset original cargado: {len(original_df)} registros")
            else:
                logger.warning(" Dataset original no encontrado, creando nuevo")
                original_df = pd.DataFrame()
            
           
            new_records = []
            for crop_name, records in new_data.items():
                for record in records:
                    
                    standard_record = {}
                    
                
                    field_mappings = {
                        'ph': ['ph'],
                        'humidity': ['humidity', 'humedad'],
                        'temperature': ['temperature', 'temperatura'],
                        'precipitation': ['precipitation', 'precipitacion'],
                        'sun_hours': ['sun_hours', 'horas_de_sol'],
                        'soil_type': ['soil_type', 'tipo_de_suelo'],
                        'season': ['season', 'temporada'],
                        'crop_type': ['crop_type', 'tipo_de_cultivo']
                    }
                    
                    for standard_field, possible_names in field_mappings.items():
                        value = None
                        for name in possible_names:
                            if name in record and record[name] is not None:
                                value = record[name]
                                break
                        
                        if value is not None:
                            standard_record[standard_field] = value
                        elif standard_field == 'crop_type':
                            
                            standard_record[standard_field] = crop_name
                    
                   
                    required_fields = self.STANDARD_COLUMNS['numeric_features'] + self.STANDARD_COLUMNS['categorical_features'] + [self.STANDARD_COLUMNS['target_column']]
                    
                    if all(field in standard_record for field in required_fields):
                        new_records.append(standard_record)
                    else:
                        missing = [field for field in required_fields if field not in standard_record]
                        logger.warning(f"Registro incompleto, faltan: {missing}")
            
            if new_records:
                new_df = pd.DataFrame(new_records)
                logger.info(f" Nuevos registros preparados: {len(new_df)}")
                
             
                if not original_df.empty:
                    combined_df = pd.concat([original_df, new_df], ignore_index=True)
                    logger.info(f" Datasets combinados: {len(original_df)} original + {len(new_df)} nuevos = {len(combined_df)} total")
                else:
                    combined_df = new_df
                    logger.info(f" Usando solo datos nuevos: {len(combined_df)} registros")
                
                return combined_df
            else:
                logger.warning(" No hay nuevos datos válidos para combinar")
                return original_df if not original_df.empty else pd.DataFrame()
                
        except Exception as e:
            logger.error(f" Error combinando datasets: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[ColumnTransformer], Optional[LabelEncoder]]:
      
        try:
           
            df = self._normalize_column_names(df)
            
            logger.info(f" Columnas disponibles: {list(df.columns)}")
            
           
            numeric_features = self.STANDARD_COLUMNS['numeric_features']
            categorical_features = self.STANDARD_COLUMNS['categorical_features']
            target_column = self.STANDARD_COLUMNS['target_column']
            feature_columns = numeric_features + categorical_features
            
          
            required_columns = feature_columns + [target_column]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f" Columnas faltantes: {missing_columns}")
                logger.error(f" Columnas disponibles: {list(df.columns)}")
                return None, None, None, None
            
          
            X = df[feature_columns].copy()
            y = df[target_column].copy()
            
         
            if X.isnull().any().any():
                logger.error(" Valores nulos detectados en características")
                logger.error(f"Valores nulos en X:\n{X.isnull().sum()}")
                return None, None, None, None
            
            if y.isnull().any():
                logger.error(" Valores nulos detectados en objetivo")
                logger.error(f"Valores nulos en y: {y.isnull().sum()}")
                return None, None, None, None
            
         
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
                ],
                remainder='drop'
            )
            
            
            label_encoder = LabelEncoder()
            
            logger.info(f" Características preparadas:")
            logger.info(f"   X shape: {X.shape}")
            logger.info(f"   y length: {len(y)}")
            logger.info(f"   Clases únicas: {sorted(y.unique())}")
            logger.info(f"   Características numéricas: {numeric_features}")
            logger.info(f"   Características categóricas: {categorical_features}")
            
            return X, y, preprocessor, label_encoder
            
        except Exception as e:
            logger.error(f" Error preparando características: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None, None, None, None
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, preprocessor, label_encoder) -> Tuple[Optional[Pipeline], float, float]:
       
        try:
            logger.info(" Iniciando entrenamiento del modelo...")
            
          
            logger.info(" Aplicando preprocessing...")
            X_processed = preprocessor.fit_transform(X)
            logger.info(f" Preprocessing aplicado: {X_processed.shape}")
            
           
            logger.info(" Codificando variable objetivo...")
            y_encoded = label_encoder.fit_transform(y)
            
           
            unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
            class_distribution = dict(zip(unique_classes, class_counts))
            logger.info(f" Distribución de clases: {class_distribution}")
            
           
            min_class_count = np.min(class_counts)
            can_stratify = min_class_count >= 2
            
        
            if can_stratify:
                logger.info(" Usando división estratificada (todas las clases tienen ≥2 ejemplos)")
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                )
            else:
                logger.warning(f" No se puede usar división estratificada (clase mínima: {min_class_count})")
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y_encoded, test_size=0.2, random_state=42
                )
            
            logger.info(f" División de datos: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
            
           
            classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1
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
                    logger.info(f" Cross-validation: {cv_accuracy:.4f}")
                except Exception as cv_error:
                    logger.warning(f"⚠️ Cross-validation falló: {cv_error}")
            else:
                logger.info(" Saltando cross-validation (ejemplos insuficientes)")
            
            logger.info(f" Modelo entrenado exitosamente:")
            logger.info(f"   Test accuracy: {accuracy:.4f}")
            logger.info(f"   CV accuracy: {cv_accuracy:.4f}")
            logger.info(f"   Cultivos disponibles: {list(label_encoder.classes_)}")
            
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
            logger.info(f"   Modelo: {model_path}")
            logger.info(f"   Encoder: {encoder_path}")
            logger.info(f"   Preprocessor: {preprocessor_path}")
            logger.info(f"   Accuracy final: {accuracy:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f" Error guardando modelo actualizado: {e}")
            return False
    
    def retrain_with_new_data(self, min_examples: int = 5) -> Dict[str, Any]:
       
        try:
            logger.info("Iniciando proceso de reentrenamiento inteligente...")
            
           
            logger.info("📁 Cargando dataset original...")
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
            
           
            logger.info(" Combinando datasets...")
            combined_df = pd.concat([original_df, new_df], ignore_index=True)
            logger.info(f" Datasets combinados: {len(original_df)} original + {len(new_df)} nuevos = {len(combined_df)} total")
            
          
            logger.info(" Preparando características...")
            result = self.prepare_features(combined_df)
            if result[0] is None:
                return {"success": False, "error": "Error preparando características"}
            
            X, y, preprocessor, label_encoder = result
            logger.info(f" Características preparadas: X={X.shape}, y={len(y)} clases")
            
           
            logger.info(" Entrenando modelo...")
            result = self.train_model(X, y, preprocessor, label_encoder)
            if result[0] is None:
                return {"success": False, "error": "Error entrenando modelo"}
            
            pipeline, accuracy, cv_accuracy = result
            logger.info(f" Modelo entrenado: accuracy={accuracy:.4f}, CV={cv_accuracy:.4f}")
            
           
            logger.info(" Guardando modelo...")
            if not self.save_updated_model(pipeline, label_encoder, preprocessor, accuracy):
                return {"success": False, "error": "Error guardando modelo"}
            
          
            combined_df.to_csv(self.dataset_path, index=False)
            logger.info("💾 Dataset combinado guardado")
            
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
                    crop_type = crop.get('crop_type', '').lower().strip()
                    if crop_type:
                        unique_crops.add(crop_type)
                        total_examples += 1
            
            return {
                "total_examples": total_examples,
                "available_crops": sorted(list(unique_crops)),
                "num_unique_crops": len(unique_crops)
            }
            
        except Exception as e:
            logger.error(f" Error obteniendo estado: {e}")
            return {"error": str(e)}