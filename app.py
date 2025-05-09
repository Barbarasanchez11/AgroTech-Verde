import streamlit as st
import pandas as pd
import pickle
import firebase_admin
from firebase_admin import credentials, db

# Inicializar Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase_credentials.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://agrotech-edfb5-default-rtdb.europe-west1.firebasedatabase.app/'
    })

# Cargar modelo y codificador
try:
    with open("modelo_rf.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
except FileNotFoundError:
    st.error("Archivos modelo_rf.pkl o label_encoder.pkl no encontrados. Ejecuta modelo.py primero.")
    st.stop()

# Título
st.title("🌱 Clasificador de Cultivos")
st.write("Predice el cultivo adecuado o ingresa datos reales para mejorar el modelo.")

# Pestañas
tab1, tab2 = st.tabs(["Predecir Cultivo", "Ingresar Datos Reales"])

# Pestaña 1: Predecir cultivo
with tab1:
    st.header("Predecir Cultivo")
    st.write("Introduce las condiciones del terreno para predecir el cultivo adecuado.")

    ph_pred = st.slider("pH del suelo", 4.5, 8.5, 6.5, key="ph_pred")
    humedad_pred = st.slider("Humedad (%)", 0, 100, 50, key="humedad_pred")
    temperatura_pred = st.slider("Temperatura (°C)", 0, 40, 20, key="temperatura_pred")
    precipitacion_pred = st.slider("Precipitación (mm)", 0, 300, 150, key="precipitacion_pred")
    horas_de_sol_pred = st.slider("Horas de sol", 0, 16, 8, key="horas_de_sol_pred")
    tipo_de_suelo_pred = st.selectbox("Tipo de suelo", ["arcilloso", "arenoso", "limoso", "rocoso"], key="tipo_de_suelo_pred")
    temporada_pred = st.selectbox("Temporada", ["verano", "otoño", "invierno", "primavera"], key="temporada_pred")

    input_data = pd.DataFrame([{
        "ph": ph_pred,
        "humedad": humedad_pred,
        "temperatura": temperatura_pred,
        "precipitacion": precipitacion_pred,
        "horas_de_sol": horas_de_sol_pred,
        "tipo_de_suelo": tipo_de_suelo_pred,
        "temporada": temporada_pred
    }])

    if st.button("🌾 Predecir Cultivo", key="predict_button"):
        pred = model.predict(input_data)
        cultivo = label_encoder.inverse_transform(pred)[0]
        st.success(f"El modelo recomienda cultivar: **{cultivo}**")

# Pestaña 2: Ingresar datos reales
with tab2:
    st.header("Ingresar Datos Reales")
    st.write("Ingresa un cultivo y las condiciones reales de tu parcela para mejorar el modelo.")

    tipo_de_cultivo = st.selectbox("Tipo de cultivo", ["arroz", "lentejas", "maiz", "naranjas", "soja", "trigo", "uva", "zanahoria"], key="tipo_de_cultivo")
    ph = st.slider("pH del suelo", 4.5, 8.5, 6.5, key="ph_input")
    humedad = st.slider("Humedad (%)", 0, 100, 50, key="humedad_input")
    temperatura = st.slider("Temperatura (°C)", 0, 40, 20, key="temperatura_input")
    precipitacion = st.slider("Precipitación (mm)", 0, 300, 150, key="precipitacion_input")
    horas_de_sol = st.slider("Horas de sol", 0, 16, 8, key="horas_de_sol_input")
    tipo_de_suelo = st.selectbox("Tipo de suelo", ["arcilloso", "arenoso", "limoso", "rocoso"], key="tipo_de_suelo_input")
    temporada = st.selectbox("Temporada", ["verano", "otoño", "invierno", "primavera"], key="temporada_input")

    if st.button("💾 Guardar Datos", key="save_button"):
        try:
            user_data = {
                "tipo_de_cultivo": tipo_de_cultivo,
                "ph": ph,
                "humedad": humedad,
                "temperatura": temperatura,
                "precipitacion": precipitacion,
                "horas_de_sol": horas_de_sol,
                "tipo_de_suelo": tipo_de_suelo,
                "temporada": temporada
            }
            ref = db.reference('/user_data')
            ref.push(user_data)
            st.success("Datos guardados en Firebase Realtime Database. ¡Gracias por contribuir!")
        except Exception as e:
            st.error(f"Error al guardar datos: {str(e)}")