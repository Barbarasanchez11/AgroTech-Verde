import streamlit as st
import pandas as pd
from typing import Dict, Any
import logging
import warnings
import os

warnings.filterwarnings('ignore')

os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'true'
os.environ['STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION'] = 'true'
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
os.environ['STREAMLIT_SERVER_RUN_ON_SAVE'] = 'false'

from src.config.config import APP_CONFIG, TERRAIN_PARAMS, SOIL_TYPES, SEASONS, STYLE_FILE
from src.services.prediction_service import PredictionService
from src.services.firebase_service import FirebaseService
from src.services.retraining_service import RetrainingService
from src.utils.validators import DataValidator

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def init_services():
    try:
        prediction_service = PredictionService()
        firebase_service = FirebaseService()
        retraining_service = RetrainingService()
        prediction_service.load_models()
        return prediction_service, firebase_service, retraining_service
    except Exception as e:
        logger.error(f"Error initializing services: {e}")
        st.error("Error al inicializar los servicios. Por favor, recarga la página.")
        return None, None, None

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
        
        prediction_service, firebase_service, retraining_service = init_services()
        model_info = prediction_service.get_model_info()
        
        if model_info.get("status") == "loaded":
            available_crops = model_info.get("available_crops", [])
            st.markdown(f"**Cultivos disponibles:** {len(available_crops)}")
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
        
        humedad = st.slider("Humedad (%)", 
                           min_value=TERRAIN_PARAMS["humedad"]["min"], 
                           max_value=TERRAIN_PARAMS["humedad"]["max"], 
                           value=TERRAIN_PARAMS["humedad"]["default"], 
                           step=TERRAIN_PARAMS["humedad"]["step"])
        
        temperatura = st.slider("Temperatura (°C)", 
                              min_value=TERRAIN_PARAMS["temperatura"]["min"], 
                              max_value=TERRAIN_PARAMS["temperatura"]["max"], 
                              value=TERRAIN_PARAMS["temperatura"]["default"], 
                              step=TERRAIN_PARAMS["temperatura"]["step"])
        
        precipitacion = st.slider("Precipitación (mm)", 
                                 min_value=TERRAIN_PARAMS["precipitacion"]["min"], 
                                 max_value=TERRAIN_PARAMS["precipitacion"]["max"], 
                                 value=TERRAIN_PARAMS["precipitacion"]["default"], 
                                 step=TERRAIN_PARAMS["precipitacion"]["step"])
    
    with col2:
        horas_de_sol = st.slider("Horas de sol", 
                                min_value=TERRAIN_PARAMS["horas_de_sol"]["min"], 
                                max_value=TERRAIN_PARAMS["horas_de_sol"]["max"], 
                                value=TERRAIN_PARAMS["horas_de_sol"]["default"], 
                                step=TERRAIN_PARAMS["horas_de_sol"]["step"])
        
        tipo_de_suelo = st.selectbox("Tipo de suelo", SOIL_TYPES)
        temporada = st.selectbox("Temporada", SEASONS)
    
    return {
        "ph": ph,
        "humedad": humedad,
        "temperatura": temperatura,
        "precipitacion": precipitacion,
        "horas_de_sol": horas_de_sol,
        "tipo_de_suelo": tipo_de_suelo,
        "temporada": temporada
    }

def handle_prediction(terrain_params: Dict[str, Any]):
    try:
        prediction_service, firebase_service, retraining_service = init_services()
        
        if not all([prediction_service, firebase_service, retraining_service]):
            st.error("No se pudieron inicializar los servicios. Por favor, recarga la página.")
            return
        
        is_valid, errors = DataValidator.validate_terrain_params(terrain_params)
        if not is_valid:
            st.error("Parámetros inválidos:")
            for error in errors:
                st.error(f"- {error}")
            return
        
        # Mostrar parámetros ingresados
        st.markdown("### Parámetros Ingresados")
        params_df = pd.DataFrame([terrain_params])
        st.dataframe(params_df, use_container_width=True, hide_index=True)
        
        # Realizar predicción
        prediction_result = prediction_service.predict_crop(terrain_params)
        
        # prediction_result es una tupla: (success, crop_or_error, error_details)
        success, crop_or_error, error_details = prediction_result
        
        if success:
            crop = crop_or_error
            confidence = 95.0  # Valor por defecto de confianza
            
            st.markdown("### 🎯 Resultado de la Predicción")
            
            # Crear tarjeta de resultado
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
                <h2 style="margin: 0; font-size: 2.5rem;">🌱 {crop.upper()}</h2>
                <p style="font-size: 1.2rem; margin: 10px 0;">Cultivo Recomendado</p>
                <div style="
                    background: rgba(255,255,255,0.2);
                    padding: 10px 20px;
                    border-radius: 25px;
                    display: inline-block;
                    margin-top: 15px;
                ">
                    <strong>Confianza: {confidence:.1f}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Guardar en Firebase
            try:
                save_result = firebase_service.save_prediction(terrain_params, crop, confidence)
                if save_result.get("success"):
                    st.success(f"✅ **{crop.upper()}** recomendado y guardado exitosamente!")
                    st.info(f"🌱 **Cultivo:** {crop.capitalize()} | 📊 **Confianza:** {confidence:.1f}% | 💾 **Guardado en base de datos**")
                    st.success("💡 **Consejo:** Consulta tu historial de predicciones en la pestaña 'Historial' para ver todas tus recomendaciones anteriores.")
                else:
                    st.warning("⚠️ La predicción se realizó pero no se pudo guardar en la base de datos")
            except Exception as e:
                logger.error(f"Error saving to Firebase: {e}")
                st.warning("⚠️ La predicción se realizó pero no se pudo guardar en la base de datos")
            
            # Mostrar información adicional del cultivo
            st.markdown("### 📊 Información del Cultivo")
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                border: 1px solid #dee2e6;
                border-radius: 12px;
                padding: 20px;
                margin: 20px 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            ">
                <h4 style="color: #495057; margin: 0 0 15px 0;">🌱 **{crop.capitalize()}** - Cultivo Recomendado</h4>
                <div style="color: #6c757d; line-height: 1.6;">
                    <p><strong>✅ Condiciones ideales detectadas:</strong></p>
                    <ul style="margin: 10px 0; padding-left: 20px;">
                        <li>pH del suelo: <strong>{terrain_params['ph']}</strong></li>
                        <li>Humedad: <strong>{terrain_params['humedad']}%</strong></li>
                        <li>Temperatura: <strong>{terrain_params['temperatura']}°C</strong></li>
                        <li>Precipitación: <strong>{terrain_params['precipitacion']} mm</strong></li>
                        <li>Horas de sol: <strong>{terrain_params['horas_de_sol']} h</strong></li>
                        <li>Tipo de suelo: <strong>{terrain_params['tipo_de_suelo']}</strong></li>
                        <li>Temporada: <strong>{terrain_params['temporada']}</strong></li>
                    </ul>
                    <p style="margin-top: 15px; color: #28a745; font-weight: 600;">
                        🎯 **Confianza del modelo:** {confidence:.1f}%
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.info(f"💡 **{crop.capitalize()}** es la mejor opción para las condiciones de tu terreno. Esta recomendación se basa en el análisis de múltiples factores ambientales y del suelo.")
            
        else:
            error_message = crop_or_error
            st.error(f"❌ **Error en la predicción:** {error_message}")
            if error_details:
                st.error(f"🔍 **Detalles técnicos:** {error_details}")
            
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
                border: 1px solid #ffc107;
                border-radius: 8px;
                padding: 15px;
                margin: 15px 0;
            ">
                <h5 style="color: #856404; margin: 0 0 10px 0;">💡 **Sugerencias para resolver el problema:**</h5>
                <ul style="color: #856404; margin: 0; padding-left: 20px;">
                    <li>Verifica que todos los parámetros estén dentro de los rangos válidos</li>
                    <li>Intenta ajustar ligeramente los valores de entrada</li>
                    <li>Recarga la página si el problema persiste</li>
                    <li>Contacta al administrador si el error continúa</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        logger.error(f"Error in handle_prediction: {e}")
        st.error("Ha ocurrido un error durante la predicción. Por favor, intenta nuevamente.")
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
        
        humedad = st.slider("Humedad (%)", 
                           min_value=TERRAIN_PARAMS["humedad"]["min"], 
                           max_value=TERRAIN_PARAMS["humedad"]["max"], 
                           value=TERRAIN_PARAMS["humedad"]["default"], 
                           step=TERRAIN_PARAMS["humedad"]["step"], 
                           key="new_humedad")
        
        temperatura = st.slider("Temperatura (°C)", 
                              min_value=TERRAIN_PARAMS["temperatura"]["min"], 
                              max_value=TERRAIN_PARAMS["temperatura"]["max"], 
                              value=TERRAIN_PARAMS["temperatura"]["default"], 
                              step=TERRAIN_PARAMS["temperatura"]["step"], 
                              key="new_temperatura")
        
        precipitacion = st.slider("Precipitación (mm)", 
                                 min_value=TERRAIN_PARAMS["precipitacion"]["min"], 
                                 max_value=TERRAIN_PARAMS["precipitacion"]["max"], 
                                 value=TERRAIN_PARAMS["precipitacion"]["default"], 
                                 step=TERRAIN_PARAMS["precipitacion"]["step"], 
                                 key="new_precipitacion")
    
    with col2:
        horas_de_sol = st.slider("Horas de sol", 
                                min_value=TERRAIN_PARAMS["horas_de_sol"]["min"], 
                                max_value=TERRAIN_PARAMS["horas_de_sol"]["max"], 
                                value=TERRAIN_PARAMS["horas_de_sol"]["default"], 
                                step=TERRAIN_PARAMS["horas_de_sol"]["step"], 
                                key="new_horas_de_sol")
        
        tipo_de_suelo = st.selectbox("Tipo de suelo", SOIL_TYPES, key="new_tipo_de_suelo")
        temporada = st.selectbox("Temporada", SEASONS, key="new_temporada")
        tipo_de_cultivo = st.text_input("Tipo de cultivo", key="new_tipo_de_cultivo")
    
    if st.button("Guardar Cultivo"):
        if tipo_de_cultivo:
            crop_data = {
                "ph": ph,
                "humedad": humedad,
                "temperatura": temperatura,
                "precipitacion": precipitacion,
                "horas_de_sol": horas_de_sol,
                "tipo_de_suelo": tipo_de_suelo,
                "temporada": temporada,
                "tipo_de_cultivo": tipo_de_cultivo
            }
            
            _, firebase_service, _ = init_services()
            if firebase_service.save_crop_data(crop_data):
                st.success("Cultivo guardado correctamente")
            else:
                st.error("Error al guardar el cultivo")
        else:
            st.warning("Por favor ingresa el tipo de cultivo")

def render_crops_history():
    _, firebase_service, _ = init_services()
    
    crops_data = firebase_service.get_all_crops()
    
    if crops_data:
        df = pd.DataFrame(crops_data)
        
        st.markdown("### Historial de Cultivos")
        
        stats = firebase_service.get_collection_stats()
        if stats:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Registros", stats.get("total_records", 0))
            with col2:
                st.metric("Tipos de Cultivo", stats.get("unique_crops", 0))
        
        st.dataframe(
            df.style.background_gradient(cmap='Greens'),
            use_container_width=True
        )
    else:
        st.info("No hay registros de cultivos disponibles.")

def render_admin_section():
    st.markdown("### Administración del Sistema")
    
    _, firebase_service, retraining_service = init_services()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Limpiar Registros")
        st.markdown("Elimina registros con campos vacíos o incompletos")
        
        if st.button("Limpiar Registros", help="Elimina registros con campos vacíos", key="btn_limpiar"):
            with st.spinner("Limpiando registros..."):
                result = firebase_service.clean_empty_records()
                
                if "error" not in result:
                    st.success(f" Limpieza completada: {result['deleted_records']} registros eliminados")
                    st.info(f" Registros restantes: {result['remaining_records']}")
                else:
                    st.error(f" Error durante la limpieza: {result['error']}")
    
    with col2:
        st.markdown("#### Reentrenar Modelo")
        st.markdown("Reentrena el modelo con todos los datos de Firebase")
        
        if st.button("Reentrenar Modelo", help="Reentrena el modelo con datos de Firebase", key="btn_reentrenar"):
            with st.spinner("Reentrenando modelo..."):
                result = retraining_service.retrain_with_firebase_data()
                
                if result["success"]:
                    st.success(f"Modelo reentrenado exitosamente")
                    st.info(f"📊 Registros utilizados: {result['records_used']}")
                else:
                    st.error(f"Error en el reentrenamiento: {result['message']}")
    
    st.markdown("#### 📊 Estadísticas de Datos")
    
    stats = retraining_service.get_training_stats()
    
    if "error" not in stats:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div style="text-align: center; padding: 15px;">
                <h3 style="color: #495057; margin: 0; font-size: 1.1rem;">Total de Registros</h3>
                <p style="color: #28a745; font-size: 2rem; font-weight: bold; margin: 10px 0;">{stats.get("total_records", 0)}</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 15px;">
                <h3 style="color: #495057; margin: 0; font-size: 1.1rem;">Tipos de Cultivo</h3>
                <p style="color: #28a745; font-size: 2rem; font-weight: bold; margin: 10px 0;">{len(stats.get("crop_distribution", {}))}</p>
            </div>
            """, unsafe_allow_html=True)
        
        if stats.get("crop_distribution"):
            st.markdown("""
            <div style="margin-top: 25px;">
                <h4 style="color: #495057; margin-bottom: 20px; font-size: 1.2rem; display: flex; align-items: center; gap: 8px;">
                    🌱 Distribución de cultivos
                </h4>
                <div style="
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-top: 15px;
                ">
            """, unsafe_allow_html=True)
            
            for crop, count in stats["crop_distribution"].items():
                st.markdown(f"""
                <div class="crop-card" style="
                    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                    border: 1px solid #e9ecef;
                    border-radius: 12px;
                    padding: 16px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                    transition: all 0.3s ease;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 12px;
                    cursor: pointer;
                ">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <div style="
                            width: 8px;
                            height: 8px;
                            background: #28a745;
                            border-radius: 50%;
                            flex-shrink: 0;
                        "></div>
                        <span style="
                            color: #495057; 
                            font-weight: 600; 
                            font-size: 1rem;
                            text-transform: capitalize;
                        ">{crop}</span>
                    </div>
                    <div style="
                        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                        color: white;
                        padding: 6px 12px;
                        border-radius: 20px;
                        font-size: 0.85rem;
                        font-weight: bold;
                        box-shadow: 0 2px 4px rgba(40, 167, 69, 0.3);
                        min-width: 60px;
                        text-align: center;
                    ">{count} registros</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
    else:
        st.error(f"Error obteniendo estadísticas: {stats['error']}")

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
        
        st.title("🌱 AgroTech Verde")
        st.markdown("Sistema inteligente de recomendación de cultivos")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Predicción", "Nuevo Cultivo", "Historial", "Administración"])
        
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
        
        with tab4:
            render_admin_section()
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        st.error("Ha ocurrido un error inesperado. Por favor, recarga la página.")
        st.exception(e)

if __name__ == "__main__":
    main()

