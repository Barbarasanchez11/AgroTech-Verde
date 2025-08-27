import streamlit as st
import pandas as pd
from typing import Dict, Any
import logging
import warnings
import os
import sys
from pathlib import Path

warnings.filterwarnings('ignore')

# Shim para permitir `streamlit run src/app.py` añadiendo el raíz del proyecto al sys.path
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
except Exception:
    pass

os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'true'
os.environ['STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION'] = 'true'
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
os.environ['STREAMLIT_SERVER_RUN_ON_SAVE'] = 'false'

from src.config.config import APP_CONFIG, TERRAIN_PARAMS, SOIL_TYPES, SEASONS, STYLE_FILE
from src.services.prediction_service import PredictionService
from src.services.supabase_service import SupabaseService
from src.utils.validators import DataValidator

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def init_services():
    try:
        prediction_service = PredictionService()
        database_service = SupabaseService()
        
        try:
            database_service.ensure_initial_data()
        except Exception as e:
            logger.warning(f"Could not ensure initial data: {e}")
        
        return prediction_service, database_service
    except Exception as e:
        logger.error(f"Error initializing services: {e}")
        return None, None

def load_css():
    try:
        with open(STYLE_FILE, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading CSS: {e}")

def render_sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/plant-under-rain.png", width=100)
        st.title("AgroTech Verde")
        st.markdown("---")
        st.markdown("### Clasificador Inteligente de Cultivos")
        st.markdown("Sistema de recomendación de cultivos basado en condiciones ambientales y del suelo.")
        
        try:
            prediction_service, database_service = init_services()
        except Exception as e:
            logger.error(f"Init services error: {e}")
            st.warning("Servicios no disponibles")
            return
        
        if not prediction_service:
            st.warning("Servicio de predicción no disponible")
            return
        
        try:
            model_info = prediction_service.get_model_info()
        except Exception as e:
            logger.error(f"get_model_info error: {e}")
            st.warning("Información del modelo no disponible")
            return
        
        if model_info.get("loaded"):
            available_crops = model_info.get("available_crops", [])
        else:
            st.warning("Modelo no cargado")

def render_terrain_params():
    st.markdown("### Parámetros del Terreno")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ph = st.slider("pH del suelo", 
                      min_value=TERRAIN_PARAMS["ph"]["min"], 
                      max_value=TERRAIN_PARAMS["ph"]["max"], 
                      value=TERRAIN_PARAMS["ph"]["default"], 
                      step=TERRAIN_PARAMS["ph"]["step"])
        
        humidity = st.slider("Humedad (%)", 
                           min_value=TERRAIN_PARAMS["humidity"]["min"], 
                           max_value=TERRAIN_PARAMS["humidity"]["max"], 
                           value=TERRAIN_PARAMS["humidity"]["default"], 
                           step=TERRAIN_PARAMS["humidity"]["step"])
        
        temperature = st.slider("Temperatura (°C)", 
                              min_value=TERRAIN_PARAMS["temperature"]["min"], 
                              max_value=TERRAIN_PARAMS["temperature"]["max"], 
                              value=TERRAIN_PARAMS["temperature"]["default"], 
                              step=TERRAIN_PARAMS["temperature"]["step"])
        
        precipitation = st.slider("Precipitación (mm)", 
                                 min_value=TERRAIN_PARAMS["precipitation"]["min"], 
                                 max_value=TERRAIN_PARAMS["precipitation"]["max"], 
                                 value=TERRAIN_PARAMS["precipitation"]["default"], 
                                 step=TERRAIN_PARAMS["precipitation"]["step"])
    
    with col2:
        sun_hours = st.slider("Horas de sol", 
                                min_value=TERRAIN_PARAMS["sun_hours"]["min"], 
                                max_value=TERRAIN_PARAMS["sun_hours"]["max"], 
                                value=TERRAIN_PARAMS["sun_hours"]["default"], 
                                step=TERRAIN_PARAMS["sun_hours"]["step"])
        
        soil_type = st.selectbox("Tipo de suelo", SOIL_TYPES)
        season = st.selectbox("Temporada", SEASONS)
    
    return {
        "ph": ph,
        "humidity": humidity,
        "temperature": temperature,
        "precipitation": precipitation,
        "sun_hours": sun_hours,
        "soil_type": soil_type,
        "season": season
    }

def handle_prediction(terrain_params: Dict[str, Any]):
    try:
        prediction_service, database_service = init_services()
        
        if not all([prediction_service, database_service]):
            st.error("No se pudieron inicializar los servicios. Por favor, recarga la página.")
            return
        
        is_valid, errors = DataValidator.validate_terrain_params(terrain_params)
        if not is_valid:
            st.error("Parámetros inválidos:")
            for error in errors:
                st.error(f"- {error}")
            return
        
        st.markdown("### Parámetros Ingresados")
        params_df = pd.DataFrame([terrain_params])
        st.dataframe(params_df, use_container_width=True, hide_index=True)
        
        prediction_result = prediction_service.predict_crop(terrain_params)
        
        if not prediction_result.get("success"):
            st.error(f"Error en la predicción: {prediction_result.get('error', 'Error desconocido')}")
            if prediction_result.get("details"):
                st.error(f"Detalles técnicos: {prediction_result.get('details')}")
            return
        
        crop = prediction_result.get("predicted_crop")
        confidence = prediction_result.get("confidence", 95.0)
        
        if crop: 
            
            st.markdown("### Resultado de la Predicción")
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                color: white;
                padding: 30px;
                border-radius: 15px;
                text-align: center;
                box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);
                margin: 20px 0;
            ">
                <h2 style="margin: 0; font-size: 2.5rem;"> {crop.upper()}</h2>
                <p style="font-size: 1.2rem; margin: 10px 0;">Cultivo Recomendado</p>
               
                   
                
            </div>
            """, unsafe_allow_html=True)
        
            try:
                save_result = database_service.save_prediction(terrain_params, crop, confidence)
                if not save_result.get("success"):
                    st.error(f"**Error al guardar:** {save_result.get('error', 'Error desconocido')}")
            except Exception as e:
                logger.error(f"Error saving to Supabase: {e}")
                st.error("**Error al guardar la predicción** - Intenta nuevamente")
            

            
        else:
            st.error(f"Error en la predicción: {crop_or_error}")
            if error_details:
                st.error(f"Detalles técnicos: {error_details}")
            
    except Exception as e:
        logger.error(f"Error in handle_prediction: {e}")
        st.error("Ha ocurrido un error inesperado durante la predicción")
        st.exception(e)

def render_new_crop_form():
    st.markdown("### Agregar Nuevo Cultivo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ph = st.slider("pH del suelo", 
                      min_value=TERRAIN_PARAMS["ph"]["min"], 
                      max_value=TERRAIN_PARAMS["ph"]["max"], 
                      value=TERRAIN_PARAMS["ph"]["default"], 
                      step=TERRAIN_PARAMS["ph"]["step"],
                      key="new_ph")
        
        humidity = st.slider("Humedad (%)", 
                           min_value=TERRAIN_PARAMS["humidity"]["min"], 
                           max_value=TERRAIN_PARAMS["humidity"]["max"], 
                           value=TERRAIN_PARAMS["humidity"]["default"], 
                           step=TERRAIN_PARAMS["humidity"]["step"],
                           key="new_humidity")
        
        temperature = st.slider("Temperatura (°C)", 
                              min_value=TERRAIN_PARAMS["temperature"]["min"], 
                              max_value=TERRAIN_PARAMS["temperature"]["max"], 
                              value=TERRAIN_PARAMS["temperature"]["default"], 
                              step=TERRAIN_PARAMS["temperature"]["step"],
                              key="new_temperature")
        
        precipitation = st.slider("Precipitación (mm)", 
                                 min_value=TERRAIN_PARAMS["precipitation"]["min"], 
                                 max_value=TERRAIN_PARAMS["precipitation"]["max"], 
                                 value=TERRAIN_PARAMS["precipitation"]["default"], 
                                 step=TERRAIN_PARAMS["precipitation"]["step"],
                                 key="new_precipitation")
    
    with col2:
        sun_hours = st.slider("Horas de sol", 
                                min_value=TERRAIN_PARAMS["sun_hours"]["min"], 
                                max_value=TERRAIN_PARAMS["sun_hours"]["max"], 
                                value=TERRAIN_PARAMS["sun_hours"]["default"], 
                                step=TERRAIN_PARAMS["sun_hours"]["step"],
                                key="new_sun_hours")
        
        soil_type = st.selectbox("Tipo de suelo", SOIL_TYPES, key="new_soil_type")
        season = st.selectbox("Temporada", SEASONS, key="new_season")
    
    crop_type = st.text_input("Tipo de cultivo", placeholder="Ej: maíz, trigo, soja...")
    
    if crop_type:
        try:
            from src.services.crop_normalizer import CropNormalizer
            normalizer = CropNormalizer()
            validation = normalizer.validate_crop_name(crop_type)
            
            if validation["is_valid"]:
                if validation["needs_normalization"]:
                    pass
            else:
                st.error(f"{validation['error']}")
                
        except Exception as e:
            st.warning("No se pudo validar el nombre del cultivo")
    
    if st.button("Guardar Cultivo"):
        if crop_type:
            try:
                from src.services.crop_normalizer import CropNormalizer
                normalizer = CropNormalizer()
                normalized_name = normalizer.normalize_crop_name(crop_type)
                
                crop_data = {
                    "ph": ph,
                    "humidity": humidity,
                    "temperature": temperature,
                    "precipitation": precipitation,
                    "sun_hours": sun_hours,
                    "soil_type": soil_type,
                    "season": season,
                    "crop_type": normalized_name
                }
                
                _, database_service = init_services()
                if database_service.save_crop_data(crop_data):
                    if normalized_name != crop_type:
                        st.success(f"Cultivo guardado correctamente como '{normalized_name}'")
                    else:
                        st.success("Cultivo guardado correctamente")
                else:
                    st.error("Error al guardar el cultivo")
            except Exception as e:
                st.error(f"Error al procesar el cultivo: {str(e)}")
        else:
            st.warning("Por favor ingresa el tipo de cultivo")

def render_crops_history():
    _, database_service = init_services()
    
    if not database_service:
        st.error("Servicio de base de datos no disponible")
        st.info("Verificar configuración de Supabase en los secretos de Streamlit")
        return
    
    crops_data = database_service.get_all_crops()
    
    if crops_data:
        df = pd.DataFrame(crops_data)
        
        st.markdown("### Historial de Cultivos")
        
        stats = database_service.get_collection_stats()
        if stats:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Registros", stats.get("total_records", 0))
            with col2:
                st.metric("Tipos de Cultivo", stats.get("unique_crops", 0))
            with col3:
                predictions = [crop for crop in crops_data if crop.get('is_prediction', False)]
               
        
        try:
            display_df = df.copy()
            rename_map = {
                'humidity': 'humedad',
                'temperature': 'temperatura',
                'precipitation': 'precipitacion',
                'sun_hours': 'horas de sol',
                'soil_type': 'tipo de suelo',
                'season': 'temporada',
                'crop_type': 'tipo de cultivo'
            }
            cols_to_drop = [c for c in ['created_at', 'confidence'] if c in display_df.columns]
            if cols_to_drop:
                display_df = display_df.drop(columns=cols_to_drop)
            display_df = display_df.rename(columns=rename_map)
            preferred_order = [
                'id', 'ph', 'humedad', 'temperatura', 'precipitacion',
                'horas de sol', 'tipo de suelo', 'temporada', 'tipo de cultivo'
            ]
            existing_order = [c for c in preferred_order if c in display_df.columns]
            remaining_cols = [c for c in display_df.columns if c not in existing_order]
            ordered_df = display_df[existing_order + remaining_cols] if existing_order else display_df
            st.dataframe(
                ordered_df.style.background_gradient(cmap='Greens'),
                use_container_width=True
            )
        except Exception:
            st.dataframe(
                df.style.background_gradient(cmap='Greens'),
                use_container_width=True
            )

        st.markdown("---")
        st.markdown("## Estado del Sistema Inteligente")
        st.markdown("---")
    
        try:
            from src.services.smart_retraining_service import SmartRetrainingService
            smart_service = SmartRetrainingService(database_service)
            status = smart_service.get_retraining_status()
    
            if not status.get("error"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Ejemplos Disponibles", status['total_examples'])
                with col2:
                    st.metric("Cultivos Únicos", len(status['available_crops']))
                
                if status['available_crops']:
                    try:
                        from src.services.crop_normalizer import CropNormalizer
                        normalizer = CropNormalizer()
                        
                        normalized_crops = sorted(status['available_crops'])
                        st.info(f"Cultivos disponibles ({len(normalized_crops)}): {', '.join(normalized_crops)}")
                        
                        all_crops = database_service.get_all_crops()
                        if all_crops:
                            original_names = [crop['crop_type'] for crop in all_crops]
                            unique_original = len(set(original_names))
                            unique_normalized = len(normalized_crops)
                            
                            if unique_original > unique_normalized:
                                st.success(f"Sistema de limpieza activo: {unique_original} nombres → {unique_normalized} únicos")
                    except Exception as e:
                        st.info(f"Cultivos disponibles: {', '.join(status['available_crops'])}")
                        st.error(f"Error en normalización: {str(e)}")
                else:
                    st.error(f"Error en servicio: {status.get('error')}")
                
        except Exception as e:
            st.error(f"Error obteniendo estado del sistema: {str(e)}")
            st.exception(e)
        
    else:
        st.info("No hay registros de cultivos disponibles.")

def main():
    try:
        st.set_page_config(
            page_title=APP_CONFIG["page_title"],
            page_icon=APP_CONFIG["page_icon"],
            layout=APP_CONFIG["layout"],
            initial_sidebar_state=APP_CONFIG["initial_sidebar_state"]
        )
        
        st.markdown("""
        <style>
        .stApp {
            max-width: 100%;
        }
        .stButton > button {
            width: 100%;
        }
        </style>
        """, unsafe_allow_html=True)
        
        load_css()
        
        render_sidebar()
        
        st.title("AgroTech Verde")
        st.markdown("Sistema inteligente de recomendación de cultivos")
        
        tab1, tab2, tab3 = st.tabs(["Predicción", "Nuevo Cultivo", "Historial"])
        
        with tab1:
            st.markdown("### Predicción de Cultivos")
            st.markdown("Ingresa los parámetros del terreno para obtener una recomendación de cultivo.")
            
            terrain_params = render_terrain_params()
            
            if st.button("Predecir Cultivo", key="predict_button"):
                with st.spinner("Procesando predicción..."):
                    handle_prediction(terrain_params)
        
        with tab2:
            render_new_crop_form()
        
        with tab3:
            render_crops_history()
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        st.error("Ha ocurrido un error inesperado. Por favor, recarga la página.")
        st.exception(e)

if __name__ == "__main__":
    main()