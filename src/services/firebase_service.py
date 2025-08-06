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
        crops_data = self.data_access.get_all_documents()
        logger.info(f"Retrieved {len(crops_data)} crop records")
        return crops_data
    
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