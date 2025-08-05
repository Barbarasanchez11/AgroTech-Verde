"""
Aplicación principal de AgroTech-Verde
"""
import streamlit as st
import pandas as pd
from typing import Dict, Any
import logging

# Importar servicios y configuraciones
from src.config.config import APP_CONFIG, TERRAIN_PARAMS, SOIL_TYPES, SEASONS, STYLE_FILE
from src.services.prediction_service import PredictionService
from src.services.firebase_service import FirebaseService
from src.utils.validators import DataValidator

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar servicios
@st.cache_resource
def init_services():
    """Inicializa los servicios de la aplicación"""
    prediction_service = PredictionService()
    firebase_service = FirebaseService()
    return prediction_service, firebase_service

def load_css():
    """Carga los estilos CSS"""
    try:
        with open(STYLE_FILE, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("❌ Archivo de estilos no encontrado")
        logger.error(f"Archivo de estilos no encontrado: {STYLE_FILE}")

def render_sidebar():
    """Renderiza la barra lateral"""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/plant-under-rain.png", width=100)
        st.title("AgroTech Verde")
        st.markdown("---")
        st.markdown("### 🌱 Clasificador Inteligente de Cultivos")
        st.markdown("Sistema de recomendación de cultivos basado en condiciones ambientales y del suelo.")
        
        # Mostrar información del modelo
        prediction_service, _ = init_services()
        model_info = prediction_service.get_model_info()
        
        if model_info.get("loaded"):
            st.markdown("---")
            st.markdown("### 📊 Información del Modelo")
            st.markdown(f"**Tipo**: {model_info.get('model_type', 'N/A')}")
            st.markdown(f"**Cultivos disponibles**: {model_info.get('num_crops', 0)}")
        else:
            st.warning("⚠️ Modelo no cargado")

def render_terrain_params() -> Dict[str, Any]:
    """Renderiza los parámetros del terreno y retorna los valores"""
    st.markdown('<div class="param-section">', unsafe_allow_html=True)
    st.markdown("#### 📊 Parámetros del Terreno")
    
    # Crear columnas para mejor organización
    col1, col2 = st.columns(2)
    
    with col1:
        ph = st.slider(
            "pH del suelo", 
            TERRAIN_PARAMS["ph"]["min"], 
            TERRAIN_PARAMS["ph"]["max"], 
            TERRAIN_PARAMS["ph"]["default"],
            TERRAIN_PARAMS["ph"]["step"]
        )
        humedad = st.slider(
            "Humedad (%)", 
            TERRAIN_PARAMS["humedad"]["min"], 
            TERRAIN_PARAMS["humedad"]["max"], 
            TERRAIN_PARAMS["humedad"]["default"],
            TERRAIN_PARAMS["humedad"]["step"]
        )
        temperatura = st.slider(
            "Temperatura (°C)", 
            TERRAIN_PARAMS["temperatura"]["min"], 
            TERRAIN_PARAMS["temperatura"]["max"], 
            TERRAIN_PARAMS["temperatura"]["default"],
            TERRAIN_PARAMS["temperatura"]["step"]
        )
        precipitacion = st.slider(
            "Precipitación (mm)", 
            TERRAIN_PARAMS["precipitacion"]["min"], 
            TERRAIN_PARAMS["precipitacion"]["max"], 
            TERRAIN_PARAMS["precipitacion"]["default"],
            TERRAIN_PARAMS["precipitacion"]["step"]
        )
    
    with col2:
        horas_de_sol = st.slider(
            "Horas de sol", 
            TERRAIN_PARAMS["horas_de_sol"]["min"], 
            TERRAIN_PARAMS["horas_de_sol"]["max"], 
            TERRAIN_PARAMS["horas_de_sol"]["default"],
            TERRAIN_PARAMS["horas_de_sol"]["step"]
        )
        tipo_de_suelo = st.selectbox("Tipo de suelo", SOIL_TYPES)
        temporada = st.selectbox("Temporada", SEASONS)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
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
    """Maneja la predicción de cultivos"""
    prediction_service, firebase_service = init_services()
    
    # Validar parámetros
    is_valid, errors = DataValidator.validate_terrain_params(terrain_params)
    if not is_valid:
        st.error(f"❌ Parámetros inválidos: {'; '.join(errors)}")
        return
    
    # Hacer predicción
    success, message, prediction = prediction_service.predict_crop(terrain_params)
    
    if success and prediction:
        st.markdown(f'<div class="success-text">🌿 Cultivo Recomendado: {prediction}</div>', unsafe_allow_html=True)
        
        # Guardar en Firebase
        crop_data = terrain_params.copy()
        crop_data["tipo_de_cultivo"] = prediction
        
        if firebase_service.save_crop_data(crop_data):
            st.success("✅ Datos guardados correctamente")
        else:
            st.warning("⚠️ No se pudieron guardar los datos en la base de datos")
    else:
        st.error(f"❌ Error en la predicción: {message}")

def render_new_crop_form():
    """Renderiza el formulario para añadir nuevos cultivos"""
    with st.expander("➕ Añadir Nuevo Registro de Cultivo", expanded=False):
        with st.form("formulario_nuevo_cultivo"):
            col1, col2 = st.columns(2)
            
            with col1:
                nuevo_cultivo = st.text_input("Nombre del cultivo")
                ph_nuevo = st.slider(
                    "pH del suelo", 
                    TERRAIN_PARAMS["ph"]["min"], 
                    TERRAIN_PARAMS["ph"]["max"], 
                    TERRAIN_PARAMS["ph"]["default"], 
                    key="ph_nuevo"
                )
                humedad_nuevo = st.slider(
                    "Humedad (%)", 
                    TERRAIN_PARAMS["humedad"]["min"], 
                    TERRAIN_PARAMS["humedad"]["max"], 
                    TERRAIN_PARAMS["humedad"]["default"], 
                    key="humedad_nuevo"
                )
                temperatura_nuevo = st.slider(
                    "Temperatura (°C)", 
                    TERRAIN_PARAMS["temperatura"]["min"], 
                    TERRAIN_PARAMS["temperatura"]["max"], 
                    TERRAIN_PARAMS["temperatura"]["default"], 
                    key="temperatura_nuevo"
                )
            
            with col2:
                precipitacion_nuevo = st.slider(
                    "Precipitación (mm)", 
                    TERRAIN_PARAMS["precipitacion"]["min"], 
                    TERRAIN_PARAMS["precipitacion"]["max"], 
                    TERRAIN_PARAMS["precipitacion"]["default"], 
                    key="precipitacion_nuevo"
                )
                horas_de_sol_nuevo = st.slider(
                    "Horas de sol", 
                    TERRAIN_PARAMS["horas_de_sol"]["min"], 
                    TERRAIN_PARAMS["horas_de_sol"]["max"], 
                    TERRAIN_PARAMS["horas_de_sol"]["default"], 
                    key="horas_de_sol_nuevo"
                )
                tipo_de_suelo_nuevo = st.selectbox(
                    "Tipo de suelo", 
                    SOIL_TYPES, 
                    key="tipo_de_suelo_nuevo"
                )
                temporada_nuevo = st.selectbox(
                    "Temporada", 
                    SEASONS, 
                    key="temporada_nuevo"
                )
            
            submit = st.form_submit_button("Guardar Registro", use_container_width=True)
            
            if submit:
                # Validar nombre del cultivo
                is_valid, errors = DataValidator.validate_crop_name(nuevo_cultivo)
                if not is_valid:
                    st.error(f"⚠️ {errors[0]}")
                else:
                    nuevo_registro = {
                        "tipo_de_cultivo": nuevo_cultivo.strip(),
                        "ph": ph_nuevo,
                        "humedad": humedad_nuevo,
                        "temperatura": temperatura_nuevo,
                        "precipitacion": precipitacion_nuevo,
                        "horas_de_sol": horas_de_sol_nuevo,
                        "tipo_de_suelo": tipo_de_suelo_nuevo,
                        "temporada": temporada_nuevo
                    }
                    
                    _, firebase_service = init_services()
                    if firebase_service.save_crop_data(nuevo_registro):
                        st.success("✅ Registro guardado correctamente.")
                    else:
                        st.error("❌ Error al guardar el registro.")

def render_crops_history():
    """Renderiza el historial de cultivos"""
    _, firebase_service = init_services()
    
    # Obtener datos de Firebase
    crops_data = firebase_service.get_all_crops()
    
    if crops_data:
        # Convertir a DataFrame
        df = pd.DataFrame(crops_data)
        
        # Ordenar por timestamp si existe
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp", ascending=False)
        
        st.markdown("### 📊 Historial de Cultivos")
        
        # Mostrar estadísticas
        stats = firebase_service.get_collection_stats()
        if stats:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Registros", stats.get("total_records", 0))
            with col2:
                st.metric("Tipos de Cultivo", stats.get("unique_crops", 0))
            with col3:
                st.metric("Último Registro", "Disponible" if stats.get("latest_record") else "N/A")
        
        # Mostrar tabla
        st.dataframe(
            df.style.background_gradient(cmap='Greens'),
            use_container_width=True
        )
    else:
        st.info("📝 No hay registros de cultivos disponibles.")

def main():
    """Función principal de la aplicación"""
    # Configurar página
    st.set_page_config(**APP_CONFIG)
    
    # Cargar estilos
    load_css()
    
    # Renderizar sidebar
    render_sidebar()
    
    # Título principal
    st.title("🌱 AgroTech Verde")
    st.markdown("### Sistema de Recomendación de Cultivos")
    
    # Renderizar parámetros del terreno
    terrain_params = render_terrain_params()
    
    # Botón de predicción
    if st.button("Predecir Cultivo", key="btn-predict"):
        handle_prediction(terrain_params)
    
    st.markdown("---")
    
    # Formulario para nuevos cultivos
    render_new_crop_form()
    
    # Historial de cultivos
    render_crops_history()

if __name__ == "__main__":
    main()

