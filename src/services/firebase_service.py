import firebase_admin
from firebase_admin import credentials, firestore
import streamlit as st
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from functools import wraps

from src.config.config import FIREBASE_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def require_initialization(func: Callable) -> Callable:
    def wrapper(self, *args, **kwargs):
        if not self.is_initialized:
            if not self.initialize():
                logger.error(f"Failed to initialize Firebase for {func.__name__}")
                return func.__default_return__ if hasattr(func, '__default_return__') else None
        return func(self, *args, **kwargs)
    return wrapper

class FirebaseDataAccess:
    def __init__(self, db: firestore.Client):
        self.db = db
        self.collection_name = FIREBASE_CONFIG["collection_name"]
    
    def add_document(self, data: Dict[str, Any]) -> Optional[str]:
        try:
            doc_ref = self.db.collection(self.collection_name).add(data)
            return doc_ref[1].id
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            return None
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        try:
            docs = self.db.collection(self.collection_name).stream()
            return [doc.to_dict() for doc in docs]
        except Exception as e:
            logger.error(f"Error getting all documents: {str(e)}")
            return []
    
    def get_documents_by_field(self, field: str, value: Any) -> List[Dict[str, Any]]:
        try:
            docs = self.db.collection(self.collection_name)\
                .where(field, "==", value)\
                .stream()
            return [doc.to_dict() for doc in docs]
        except Exception as e:
            logger.error(f"Error getting documents by field: {str(e)}")
            return []
    
    def get_documents_by_date_range(self, field: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        try:
            docs = self.db.collection(self.collection_name)\
                .where(field, ">=", start_date.isoformat())\
                .where(field, "<=", end_date.isoformat())\
                .stream()
            return [doc.to_dict() for doc in docs]
        except Exception as e:
            logger.error(f"Error getting documents by date range: {str(e)}")
            return []
    
    def delete_document(self, doc_id: str) -> bool:
        try:
            doc_ref = self.db.collection(self.collection_name).document(doc_id)
            doc = doc_ref.get()
            if not doc.exists:
                logger.warning(f"Document {doc_id} does not exist")
                return False
            
            doc_ref.delete()
            return True
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False

class FirebaseService:
    def __init__(self):
        self.db = None
        self.is_initialized = False
        self.data_access = None
    
    def initialize(self) -> bool:
        try:
            if not firebase_admin._apps:
                if "firebase" not in st.secrets:
                    logger.error("Firebase credentials not configured in st.secrets")
                    return False
                
                cred = credentials.Certificate(dict(st.secrets["firebase"]))
                firebase_admin.initialize_app(cred)
            
            self.db = firestore.client()
            self.data_access = FirebaseDataAccess(self.db)
            self.is_initialized = True
            logger.info("Firebase initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Firebase: {str(e)}")
            return False
    
    @require_initialization
    def save_crop_data(self, crop_data: Dict[str, Any]) -> bool:
        crop_data["timestamp"] = datetime.now().isoformat()
        
        doc_id = self.data_access.add_document(crop_data)
        if doc_id:
            logger.info(f"Data saved successfully with ID: {doc_id}")
            return True
        return False
    
    @require_initialization
    def get_all_crops(self) -> List[Dict[str, Any]]:
        try:
            docs = self.data_access.get_all_documents()
            crops_data = []
            
            for data in docs:
                if data:
                    relevant_data = {
                        "temporada": data.get("temporada"),
                        "temperatura": data.get("temperatura"),
                        "ph": data.get("ph"),
                        "tipo_de_cultivo": data.get("tipo_de_cultivo"),
                        "horas_de_sol": data.get("horas_de_sol"),
                        "precipitacion": data.get("precipitacion"),
                        "tipo_de_suelo": data.get("tipo_de_suelo"),
                        "humedad": data.get("humedad")
                    }
                    crops_data.append(relevant_data)
            
            return crops_data
            
        except Exception as e:
            logger.error(f"Error getting all crops: {e}")
            return []

    @require_initialization
    def clean_empty_records(self) -> Dict[str, int]:
        try:
            doc_refs = self.db.collection(self.data_access.collection_name).stream()
            deleted_count = 0
            total_count = 0
            
            for doc_ref in doc_refs:
                total_count += 1
                data = doc_ref.to_dict()
                
                relevant_fields = ["temporada", "temperatura", "ph", "tipo_de_cultivo", 
                                 "horas_de_sol", "precipitacion", "tipo_de_suelo", "humedad"]
                
                has_empty_field = False
                for field in relevant_fields:
                    value = data.get(field)
                    if value is None or value == "" or value == "None":
                        has_empty_field = True
                        break
                
                if has_empty_field:
                    self.data_access.delete_document(doc_ref.id)
                    deleted_count += 1
                    logger.info(f"Deleted record {doc_ref.id} with empty fields")
            
            logger.info(f"Cleanup completed: {deleted_count}/{total_count} records deleted")
            return {
                "total_records": total_count,
                "deleted_records": deleted_count,
                "remaining_records": total_count - deleted_count
            }
            
        except Exception as e:
            logger.error(f"Error cleaning empty records: {e}")
            return {"error": str(e)}

    @require_initialization
    def get_clean_data_for_training(self) -> List[Dict[str, Any]]:
        try:
            docs = self.data_access.get_all_documents()
            clean_data = []
            
            for data in docs:
                if data:
                    relevant_fields = ["temporada", "temperatura", "ph", "tipo_de_cultivo", 
                                     "horas_de_sol", "precipitacion", "tipo_de_suelo", "humedad"]
                    
                    has_all_fields = True
                    for field in relevant_fields:
                        value = data.get(field)
                        if value is None or value == "" or value == "None":
                            has_all_fields = False
                            break
                    
                    if has_all_fields:
                        clean_data.append(data)
            
            logger.info(f"Retrieved {len(clean_data)} clean records for training")
            return clean_data
            
        except Exception as e:
            logger.error(f"Error getting clean data for training: {e}")
            return []
    
    @require_initialization
    def get_crops_by_date_range(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        crops_data = self.data_access.get_documents_by_date_range("timestamp", start_date, end_date)
        logger.info(f"Retrieved {len(crops_data)} records in date range")
        return crops_data
    
    @require_initialization
    def get_crops_by_type(self, crop_type: str) -> List[Dict[str, Any]]:
        crops_data = self.data_access.get_documents_by_field("tipo_de_cultivo", crop_type)
        logger.info(f"Retrieved {len(crops_data)} records of type {crop_type}")
        return crops_data
    
    @require_initialization
    def delete_crop_record(self, doc_id: str) -> bool:
        success = self.data_access.delete_document(doc_id)
        if success:
            logger.info(f"Record deleted successfully: {doc_id}")
        return success
    
    @require_initialization
    def get_collection_stats(self) -> Dict[str, Any]:
        crops_data = self.data_access.get_all_documents()
        
        if not crops_data:
            return {"total_records": 0, "crop_types": {}, "latest_record": None, "unique_crops": 0}
        
        crop_types = {}
        for crop in crops_data:
            crop_type = crop.get("tipo_de_cultivo", "unknown")
            crop_types[crop_type] = crop_types.get(crop_type, 0) + 1
        
        latest_record = max(crops_data, key=lambda x: x.get("timestamp", ""))
        
        stats = {
            "total_records": len(crops_data),
            "crop_types": crop_types,
            "latest_record": latest_record.get("timestamp"),
            "unique_crops": len(crop_types)
        }
        
        logger.info(f"Statistics retrieved: {stats}")
        return stats 

    @require_initialization
    def save_prediction(self, terrain_params: Dict[str, Any], predicted_crop: str, confidence: float) -> Dict[str, Any]:
        try:
            prediction_data = {
                **terrain_params,
                "tipo_de_cultivo": predicted_crop,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
                "prediction_type": "ml_prediction"
            }
            
            doc_id = self.data_access.add_document(prediction_data)
            if doc_id:
                logger.info(f"Prediction saved successfully with ID: {doc_id}")
                return {"success": True, "doc_id": doc_id, "message": "Predicción guardada exitosamente"}
            else:
                logger.error("Failed to save prediction")
                return {"success": False, "error": "No se pudo guardar la predicción"}
                
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")
            return {"success": False, "error": str(e)} 