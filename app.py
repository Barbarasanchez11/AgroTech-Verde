import pandas as pd
import pickle
import streamlit as st
from firebase_utils import init_firebase, guardar_datos_cultivo

st.set_page_config(
    page_title="AgroTech Verde",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #388E3C;
        color: white;
        border: none;
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    .stSlider>div>div>div {
        background-color: #4CAF50;
    }
    .stSlider>div>div>div>div {
        background-color: #ff4b4b;
    }
    .success-text {
        font-size: 1.2rem;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/plant-under-rain.png", width=100)
    st.title("AgroTech Verde")
    st.markdown("---")
    st.markdown("### 🌱 Clasificador Inteligente de Cultivos")
    st.markdown("Sistema de recomendación de cultivos basado en condiciones ambientales y del suelo.")

st.title("🌱 AgroTech Verde")
st.markdown("### Sistema de Recomendación de Cultivos")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("#### 📊 Parámetros del Terreno")
    ph = st.slider("pH del suelo", 4.5, 8.5, 6.5)
    humedad = st.slider("Humedad (%)", 0, 100, 50)
    temperatura = st.slider("Temperatura (°C)", 0, 40, 20)
    precipitacion = st.slider("Precipitación (mm)", 0, 300, 150)
    horas_de_sol = st.slider("Horas de sol", 0, 16, 8)
    tipo_de_suelo = st.selectbox("Tipo de suelo", ["arcilloso", "arenoso", "limoso", "rocoso"])
    temporada = st.selectbox("Temporada", ['verano', 'otoño', 'invierno', 'primavera'])
    
    if st.button("Predecir Cultivo", use_container_width=True):
        with open("modelo_rf.pkl", "rb") as f:
            model = pickle.load(f)
        with open("label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
            
        input_data = pd.DataFrame([{
            "ph": ph,
            "humedad": humedad,
            "temperatura": temperatura,
            "precipitacion": precipitacion,
            "horas_de_sol": horas_de_sol,
            "tipo_de_suelo": tipo_de_suelo,
            "temporada": temporada
        }])
        
        pred = model.predict(input_data)
        cultivo = label_encoder.inverse_transform(pred)[0]
        
        st.markdown(f'<div class="success-text">🌿 Cultivo Recomendado: {cultivo}</div>', unsafe_allow_html=True)
        
        datos_cultivo = input_data.iloc[0].to_dict()
        datos_cultivo["tipo_de_cultivo"] = cultivo
        guardar_datos_cultivo(datos_cultivo)

with col2:
    pass

st.markdown("---")

with st.expander("➕ Añadir Nuevo Registro de Cultivo", expanded=False):
    with st.form("formulario_nuevo_cultivo"):
        col1, col2 = st.columns(2)
        with col1:
            nuevo_cultivo = st.text_input("Nombre del cultivo")
            ph_nuevo = st.slider("pH del suelo", 4.5, 8.5, 6.5, key="ph_nuevo")
            humedad_nuevo = st.slider("Humedad (%)", 0, 100, 50, key="humedad_nuevo")
            temperatura_nuevo = st.slider("Temperatura (°C)", 0, 40, 20, key="temperatura_nuevo")
        with col2:
            precipitacion_nuevo = st.slider("Precipitación (mm)", 0, 300, 150, key="precipitacion_nuevo")
            horas_de_sol_nuevo = st.slider("Horas de sol", 0, 16, 8, key="horas_de_sol_nuevo")
            tipo_de_suelo_nuevo = st.selectbox("Tipo de suelo", ["arcilloso", "arenoso", "limoso", "rocoso"], key="tipo_de_suelo_nuevo")
            temporada_nuevo = st.selectbox("Temporada", ['verano', 'otoño', 'invierno', 'primavera'], key="temporada_nuevo")
        
        submit = st.form_submit_button("💾 Guardar Registro", use_container_width=True)
        
        if submit:
            if nuevo_cultivo.strip() == "":
                st.error("⚠️ Por favor, introduce el nombre del cultivo antes de guardar.")
            else:
                nuevo_registro = {
                    "tipo_de_cultivo": nuevo_cultivo,
                    "ph": ph_nuevo,
                    "humedad": humedad_nuevo,
                    "temperatura": temperatura_nuevo,
                    "precipitacion": precipitacion_nuevo,
                    "horas_de_sol": horas_de_sol_nuevo,
                    "tipo_de_suelo": tipo_de_suelo_nuevo,
                    "temporada": temporada_nuevo
                }
                guardar_datos_cultivo(nuevo_registro)
                st.success("✅ Registro guardado correctamente.")

db = init_firebase()
docs = db.collection("cultivos").stream()
datos = [doc.to_dict() for doc in docs]
df = pd.DataFrame(datos)

if not df.empty:
    st.markdown("### 📊 Historial de Cultivos")
    st.dataframe(
        df,
        use_container_width=True
    )


