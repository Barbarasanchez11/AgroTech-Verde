import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class SupabaseService:
    def __init__(self):
        self.supabase = None
        self._initialized = False
        self._initialization_error = None
        self._initialize()
    
  
    COLUMN_MAPPING = {
        'humidity': 'humedad',
        'temperature': 'temperatura', 
        'precipitation': 'precipitacion',
        'sun_hours': 'horas_de_sol',
        'soil_type': 'tipo_de_suelo',
        'season': 'temporada',
        'crop_type': 'tipo_de_cultivo'
    }
    
    def _map_to_db_columns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        
        mapped_data = {}
        for key, value in data.items():
            if key in self.COLUMN_MAPPING:
                mapped_data[self.COLUMN_MAPPING[key]] = value
            else:
                mapped_data[key] = value
        return mapped_data
    
    def _map_from_db_columns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        
        reverse_mapping = {v: k for k, v in self.COLUMN_MAPPING.items()}
        mapped_data = {}
        for key, value in data.items():
            if key in reverse_mapping:
                mapped_data[reverse_mapping[key]] = value
            else:
                mapped_data[key] = value
        return mapped_data
    
    def _initialize(self):
        try:
            from supabase import create_client, Client
            
            url = None
            key = None
            
            try:
                import streamlit as st
                if hasattr(st, 'secrets') and 'supabase' in st.secrets:
                    url = st.secrets.supabase.get('SUPABASE_URL')
                    key = st.secrets.supabase.get('SUPABASE_ANON_KEY')
                elif hasattr(st, 'secrets'):
                    url = st.secrets.get('SUPABASE_URL')
                    key = st.secrets.get('SUPABASE_ANON_KEY')
            except Exception as e:
                logger.warning(f"No se pudieron obtener secrets de Streamlit: {e}")
            
            if not url or not key:
                try:
                    from dotenv import load_dotenv
                    load_dotenv()
                    url = os.getenv('SUPABASE_URL')
                    key = os.getenv('SUPABASE_ANON_KEY')
                except:
                    pass
            
            if not url or not key:
                self._initialization_error = "SUPABASE_URL y SUPABASE_ANON_KEY no encontrados en secrets o variables de entorno"
                logger.error(self._initialization_error)
                return False
            
            self.supabase = create_client(url, key)
            
            result = self.supabase.table('crops').select('id').limit(1).execute()
            
            self._initialized = True
            self._initialization_error = None
            logger.info(" Supabase inicializado correctamente")
            return True
            
        except ImportError as e:
            self._initialization_error = f"Biblioteca supabase no encontrada: {e}"
            logger.error(self._initialization_error)
            return False
        except Exception as e:
            self._initialization_error = f"Error inicializando Supabase: {str(e)}"
            logger.error(self._initialization_error)
            return False
    
    def is_initialized(self) -> bool:
        return self._initialized and self.supabase is not None
    
    def get_initialization_error(self) -> Optional[str]:
        return self._initialization_error
    
    def _ensure_initialized(self) -> bool:
        if not self.is_initialized():
            return self._initialize()
        return True
    
    def save_crop_data(self, crop_data: Dict[str, Any]) -> bool:
        try:
            if not self._ensure_initialized():
                logger.error(f"No se pudo inicializar Supabase: {self._initialization_error}")
                return False
            
            crop_data_with_timestamp = {
                **crop_data,
                "created_at": datetime.now().isoformat(),
                "is_prediction": False
            }
            
            
            db_crop_data = self._map_to_db_columns(crop_data_with_timestamp)
            
            result = self.supabase.table('crops').insert(db_crop_data).execute()
            
            if result.data and len(result.data) > 0:
                logger.info(f"Crop saved successfully: {result.data[0].get('id')}")
                
                try:
                    self._auto_retrain_model()
                except Exception as retrain_error:
                    logger.warning(f"Auto-retraining failed: {retrain_error}")
                
                return True
            else:
                logger.error("No data received after insert")
                return False
                
        except Exception as e:
            logger.error(f" Error guardando cultivo: {e}")
            return False
    
    def save_prediction(self, terrain_params: Dict[str, Any], crop: str, confidence: float) -> Dict[str, Any]:
        try:
            if not self._ensure_initialized():
                error_msg = f"Supabase no inicializado: {self._initialization_error}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
            
            prediction_data = {
                **terrain_params,
                "crop_type": crop,
                "confidence": confidence,
                "created_at": datetime.now().isoformat(),
                "is_prediction": True
            }
            
          
            db_prediction_data = self._map_to_db_columns(prediction_data)
            
            result = self.supabase.table('crops').insert(db_prediction_data).execute()
            
            if result.data and len(result.data) > 0:
                logger.info(f"Prediction saved successfully: {result.data[0].get('id')}")
                return {
                    "success": True, 
                    "data": result.data[0],
                    "message": "Predicción guardada exitosamente"
                }
            else:
                logger.error("No data received after inserting prediction")
                return {"success": False, "error": "No se pudo insertar en la base de datos"}
                
        except Exception as e:
            error_msg = f"Error guardando predicción: {str(e)}"
            logger.error(f" {error_msg}")
            return {"success": False, "error": error_msg}
    
    def get_all_crops(self) -> List[Dict[str, Any]]:
        try:
            if not self._ensure_initialized():
                logger.error(f"No se pudo inicializar Supabase: {self._initialization_error}")
                return []
            
           
            result = self.supabase.table('crops').select('id, ph, humedad, temperatura, precipitacion, horas_de_sol, tipo_de_suelo, temporada, tipo_de_cultivo, created_at, confidence').order('created_at', desc=True).execute()
            
            if result.data:
                logger.info(f" Obtenidos {len(result.data)} cultivos de Supabase")
                
                mapped_data = [self._map_from_db_columns(crop) for crop in result.data]
                return mapped_data
            else:
                logger.info(" No se encontraron cultivos en la base de datos")
                return []
                
        except Exception as e:
            logger.error(f" Error obteniendo cultivos: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        try:
            if not self._ensure_initialized():
                return {"total_records": 0, "unique_crops": 0, "error": self._initialization_error}
            
            crops = self.get_all_crops()
            total_records = len(crops)
            
            unique_crops = len(set(
                crop.get('crop_type', '').lower().strip() 
                for crop in crops 
                if crop.get('crop_type')
            ))
            
            predictions_count = len([crop for crop in crops if crop.get('is_prediction', False)])
            manual_entries = total_records - predictions_count
            
            return {
                "total_records": total_records,
                "unique_crops": unique_crops,
                "predictions": predictions_count,
                "manual_entries": manual_entries
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"total_records": 0, "unique_crops": 0, "error": str(e)}
    
    def delete_all_crops(self) -> bool:
        try:
            if not self._ensure_initialized():
                logger.error(f"No se pudo inicializar Supabase: {self._initialization_error}")
                return False
            
            result = self.supabase.table('crops').select('id').execute()
            if not result.data:
                logger.info("No hay registros que eliminar")
                return True
            
            delete_result = self.supabase.table('crops').delete().gte('id', 0).execute()
            
            logger.info(f" Eliminados {len(result.data)} registros")
            return True
            
        except Exception as e:
            logger.error(f" Error eliminando cultivos: {e}")
            return False
    
    def test_connection(self) -> Dict[str, Any]:
        if not self._ensure_initialized():
            return {
                "success": False, 
                "error": self._initialization_error,
                "message": "No se pudo inicializar conexión"
            }
        
        try:
            result = self.supabase.table('crops').select('id').limit(1).execute()
            
            return {
                "success": True,
                "message": "Conexión exitosa a Supabase",
                "records_sample": len(result.data) if result.data else 0
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Error probando conexión"
            } 
    
    def load_initial_data_from_csv(self) -> Dict[str, Any]:
       
        try:
            if not self._ensure_initialized():
                return {"success": False, "error": "No se pudo inicializar Supabase"}
            
            
            existing_data = self.supabase.table('crops').select('id').limit(1).execute()
            if existing_data.data and len(existing_data.data) > 0:
                logger.info(" Base de datos ya tiene datos, no se cargan datos iniciales")
                return {"success": True, "message": "Base de datos ya tiene datos", "loaded": 0}
            
            
            import pandas as pd
            import os
            from pathlib import Path
            
            
            csv_path = Path(__file__).parent.parent.parent / "data" / "agroTech_data.csv"
            
            if not csv_path.exists():
                return {"success": False, "error": f"Archivo CSV no encontrado en {csv_path}"}
            
            # Read CSV
            df = pd.read_csv(csv_path)
            logger.info(f" CSV cargado: {len(df)} registros")
            
           
            crops_to_insert = []
            for _, row in df.iterrows():
                crop_data = {
                    "ph": float(row['ph']),
                    "humidity": float(row['humidity']),
                    "temperature": float(row['temperature']),
                    "precipitation": float(row['precipitation']),
                    "sun_hours": float(row['sun_hours']),
                    "soil_type": str(row['soil_type']),
                    "season": str(row['season']),
                    "crop_type": str(row['crop_type']),
                    "created_at": datetime.now().isoformat(),
                    "is_prediction": False,
                    "confidence": None
                }
                
                
                db_crop_data = self._map_to_db_columns(crop_data)
                crops_to_insert.append(db_crop_data)
            batch_size = 100
            total_inserted = 0
            
            for i in range(0, len(crops_to_insert), batch_size):
                batch = crops_to_insert[i:i + batch_size]
                result = self.supabase.table('crops').insert(batch).execute()
                if result.data:
                    total_inserted += len(result.data)
                    logger.info(f"Lote insertado: {len(result.data)} registros")
                else:
                    logger.warning(f"Batch {i//batch_size + 1} was not inserted correctly")
            
            logger.info(f"Datos iniciales cargados: {total_inserted} registros")
            return {
                "success": True, 
                "message": f"Datos iniciales cargados exitosamente",
                "loaded": total_inserted
            }
            
        except Exception as e:
            error_msg = f"Error cargando datos iniciales: {str(e)}"
            logger.error(f"{error_msg}")
            return {"success": False, "error": error_msg}
    
    def _auto_retrain_model(self) -> bool:
        try:
            logger.info("Starting auto-retraining...")
            
            from src.services.smart_retraining_service import SmartRetrainingService
            
            smart_service = SmartRetrainingService(self)
            
            result = smart_service.retrain_with_new_data(min_examples=1)
            
            if result.get("success"):
                logger.info(f"Auto-retraining successful: {result.get('message', '')}")
                return True
            else:
                logger.warning(f"Auto-retraining failed: {result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            logger.error(f"Error in auto-retraining: {e}")
            return False
    
    def ensure_initial_data(self) -> bool:
        try:
            existing_data = self.supabase.table('crops').select('id').limit(1).execute()
            if not existing_data.data or len(existing_data.data) == 0:
                logger.info("Database is empty, loading initial data...")
                result = self.load_initial_data_from_csv()
                if result.get("success"):
                    logger.info(f"Initial data loaded: {result.get('loaded', 0)} records")
                    return True
                else:
                    logger.error(f"Failed to load initial data: {result.get('error', 'Unknown error')}")
                    return False
            return True
        except Exception as e:
            logger.error(f"Error ensuring initial data: {e}")
            return False