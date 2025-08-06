import pytest
import pandas as pd
from src.utils.validators import DataValidator, ModelValidator
from src.config.config import SOIL_TYPES, SEASONS

class TestDataValidator:
    def test_validate_terrain_params_valid(self):
        valid_params = {
            "ph": 6.5,
            "humedad": 50,
            "temperatura": 20,
            "precipitacion": 150,
            "horas_de_sol": 8,
            "tipo_de_suelo": "arcilloso",
            "temporada": "verano"
        }
        
        is_valid, errors = DataValidator.validate_terrain_params(valid_params)
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_terrain_params_invalid_ph(self):
        invalid_params = {
            "ph": 10.0,
            "humedad": 50,
            "temperatura": 20,
            "precipitacion": 150,
            "horas_de_sol": 8,
            "tipo_de_suelo": "arcilloso",
            "temporada": "verano"
        }
        
        is_valid, errors = DataValidator.validate_terrain_params(invalid_params)
        assert not is_valid
        assert len(errors) > 0
        assert "pH must be between" in errors[0]
    
    def test_validate_terrain_params_invalid_soil_type(self):
        invalid_params = {
            "ph": 6.5,
            "humedad": 50,
            "temperatura": 20,
            "precipitacion": 150,
            "horas_de_sol": 8,
            "tipo_de_suelo": "invalido",
            "temporada": "verano"
        }
        
        is_valid, errors = DataValidator.validate_terrain_params(invalid_params)
        assert not is_valid
        assert len(errors) > 0
        assert "Soil type must be one of" in errors[0]
    
    def test_validate_crop_name_valid(self):
        valid_name = "maíz"
        is_valid, errors = DataValidator.validate_crop_name(valid_name)
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_crop_name_empty(self):
        empty_name = ""
        is_valid, errors = DataValidator.validate_crop_name(empty_name)
        assert not is_valid
        assert len(errors) > 0
        assert "cannot be empty" in errors[0]
    
    def test_validate_crop_name_too_short(self):
        short_name = "a"
        is_valid, errors = DataValidator.validate_crop_name(short_name)
        assert not is_valid
        assert len(errors) > 0
        assert "must have at least" in errors[0]
    
    def test_validate_dataframe_valid(self):
        valid_df = pd.DataFrame({
            "tipo_de_cultivo": ["maíz"],
            "ph": [6.5],
            "humedad": [50],
            "temperatura": [20],
            "precipitacion": [150],
            "horas_de_sol": [8],
            "tipo_de_suelo": ["arcilloso"],
            "temporada": ["verano"]
        })
        
        is_valid, errors = DataValidator.validate_dataframe(valid_df)
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_dataframe_missing_columns(self):
        invalid_df = pd.DataFrame({
            "tipo_de_cultivo": ["maíz"],
            "ph": [6.5]
        })
        
        is_valid, errors = DataValidator.validate_dataframe(invalid_df)
        assert not is_valid
        assert len(errors) > 0
        assert "Missing fields" in errors[0]
    
    def test_validate_dataframe_empty(self):
        empty_df = pd.DataFrame()
        is_valid, errors = DataValidator.validate_dataframe(empty_df)
        assert not is_valid
        assert len(errors) > 0
        assert "Missing fields" in errors[0]

class TestModelValidator:
    def test_validate_model_prediction_mock(self):
        class MockModel:
            def predict(self, data):
                return ["maíz"]
        
        mock_model = MockModel()
        test_data = pd.DataFrame({
            "ph": [6.5],
            "humedad": [50],
            "temperatura": [20],
            "precipitacion": [150],
            "horas_de_sol": [8],
            "tipo_de_suelo": ["arcilloso"],
            "temporada": ["verano"]
        })
        
        is_valid, errors = ModelValidator.validate_model_prediction(mock_model, test_data)
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_model_prediction_error(self):
        class MockModel:
            def predict(self, data):
                raise Exception("Error de predicción")
        
        mock_model = MockModel()
        test_data = pd.DataFrame({
            "ph": [6.5],
            "humedad": [50],
            "temperatura": [20],
            "precipitacion": [150],
            "horas_de_sol": [8],
            "tipo_de_suelo": ["arcilloso"],
            "temporada": ["verano"]
        })
        
        is_valid, errors = ModelValidator.validate_model_prediction(mock_model, test_data)
        assert not is_valid
        assert len(errors) > 0
        assert "Error during prediction" in errors[0] 