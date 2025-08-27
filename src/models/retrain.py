import logging
from src.models.model import CropModelTrainer
from src.config.config import ML_CONFIG
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def retrain_model():
    trainer = CropModelTrainer()
    
    logger.info("Starting model retraining...")
    success = trainer.train_model("random_forest")
    
    if success:
        logger.info("Model retraining completed successfully")
        return True
    else:
        logger.error("Model retraining failed")
        return False

if __name__ == "__main__":
    retrain_model()
