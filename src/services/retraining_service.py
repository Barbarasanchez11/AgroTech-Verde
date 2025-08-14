import logging
import pandas as pd
from typing import Dict, Any
from pathlib import Path

from src.services.firebase_service import FirebaseService
from src.models.model import CropModelTrainer
from src.config.config import MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrainingService:
    def __init__(self):
        self.firebase_service = FirebaseService()
        self.trainer = CropModelTrainer()
        
    def retrain_with_firebase_data(self) -> Dict[str, Any]:
        try:
            logger.info("Starting retraining with Firebase data...")
           
            firebase_data = self.firebase_service.get_clean_data_for_training()
            
            if not firebase_data:
                return {
                    "success": False,
                    "message": "No clean data available for training",
                    "records_used": 0
                }
            
            
            df = pd.DataFrame(firebase_data)
            logger.info(f"Training with {len(df)} records from Firebase")
            
            
            success = self.trainer.train_with_dataframe(df)
            
            if success:
                logger.info("Retraining completed successfully")
                return {
                    "success": True,
                    "message": "Model retrained successfully with Firebase data",
                    "records_used": len(df)
                }
            else:
                logger.error("Retraining failed")
                return {
                    "success": False,
                    "message": "Model retraining failed",
                    "records_used": len(df)
                }
                
        except Exception as e:
            logger.error(f"Error during retraining: {e}")
            return {
                "success": False,
                "message": f"Error during retraining: {str(e)}",
                "records_used": 0
            }
    
    def get_training_stats(self) -> Dict[str, Any]:
        try:
            firebase_data = self.firebase_service.get_clean_data_for_training()
            
            if not firebase_data:
                return {
                    "total_records": 0,
                    "clean_records": 0,
                    "crop_distribution": {}
                }
            
            df = pd.DataFrame(firebase_data)
            crop_distribution = df["tipo_de_cultivo"].value_counts().to_dict()
            
            return {
                "total_records": len(firebase_data),
                "clean_records": len(df),
                "crop_distribution": crop_distribution
            }
            
        except Exception as e:
            logger.error(f"Error getting training stats: {e}")
            return {
                "error": str(e)
            } 