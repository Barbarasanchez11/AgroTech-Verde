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
            logger.info("✅ Supabase inicializado correctamente")
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
            
            result = self.supabase.table('crops').insert(crop_data_with_timestamp).execute()
            
            if result.data and len(result.data) > 0:
                logger.info(f"✅ Cultivo guardado exitosamente: {result.data[0].get('id')}")
                return True
            else:
                logger.error("❌ No se recibieron datos después de insertar")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error guardando cultivo: {e}")
            return False
    
    def save_prediction(self, terrain_params: Dict[str, Any], crop: str, confidence: float) -> Dict[str, Any]:
        try:
            if not self._ensure_initialized():
                error_msg = f"Supabase no inicializado: {self._initialization_error}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
            
            prediction_data = {
                **terrain_params,
                "tipo_de_cultivo": crop,
                "confidence": confidence,
                "created_at": datetime.now().isoformat(),
                "is_prediction": True
            }
            
            result = self.supabase.table('crops').insert(prediction_data).execute()
            
            if result.data and len(result.data) > 0:
                logger.info(f"✅ Predicción guardada exitosamente: {result.data[0].get('id')}")
                return {
                    "success": True, 
                    "data": result.data[0],
                    "message": "Predicción guardada exitosamente"
                }
            else:
                logger.error("❌ No se recibieron datos después de insertar predicción")
                return {"success": False, "error": "No se pudo insertar en la base de datos"}
                
        except Exception as e:
            error_msg = f"Error guardando predicción: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {"success": False, "error": error_msg}
    
    def get_all_crops(self) -> List[Dict[str, Any]]:
        try:
            if not self._ensure_initialized():
                logger.error(f"No se pudo inicializar Supabase: {self._initialization_error}")
                return []
            
            result = self.supabase.table('crops').select('*').order('created_at', desc=True).execute()
            
            if result.data:
                logger.info(f"✅ Obtenidos {len(result.data)} cultivos de Supabase")
                return result.data
            else:
                logger.info("📭 No se encontraron cultivos en la base de datos")
                return []
                
        except Exception as e:
            logger.error(f"❌ Error obteniendo cultivos: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        try:
            if not self._ensure_initialized():
                return {"total_records": 0, "unique_crops": 0, "error": self._initialization_error}
            
            crops = self.get_all_crops()
            total_records = len(crops)
            
            unique_crops = len(set(
                crop.get('tipo_de_cultivo', '').lower().strip() 
                for crop in crops 
                if crop.get('tipo_de_cultivo')
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
            logger.error(f"❌ Error obteniendo estadísticas: {e}")
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
            
            logger.info(f"✅ Eliminados {len(result.data)} registros")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error eliminando cultivos: {e}")
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