import streamlit as st
import pandas as pd
import pickle
import firebase_admin
from firebase_admin import credentials, firestore
import os

# Inicializar Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase_config.json")  # Ruta a tu JSON de clave de servicio
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Cargar modelo y codificador
with open("modelo_rf.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Título
st.title("🌱 Clasificador de Cultivos")
st.write("Introduce las condiciones del terreno para predecir el tipo de cultivo adecuado.")

# Entradas del usuario
ph = st.slider("pH del suelo", 4.5, 8.5, 6.5)
humedad = st.slider("Humedad (%)", 0, 100, 50)
temperatura = st.slider("Temperatura (°C)", 0, 40, 20)
precipitacion = st.slider("Precipitación (mm)", 0, 300, 150)
horas_de_sol = st.slider("Horas de sol", 0, 16, 8)
tipo_de_suelo = st.selectbox("Tipo de suelo", ["arcilloso", "arenoso", "limoso", "rocoso"])
temporada = st.selectbox("Temporada", ['verano', 'otoño', 'invierno', 'primavera'])

# Crear DataFrame de entrada
input_data = pd.DataFrame([{
    "ph": ph,
    "humedad": humedad,
    "temperatura": temperatura,
    "precipitacion": precipitacion,
    "horas_de_sol": horas_de_sol,
    "tipo_de_suelo": tipo_de_suelo,
    "temporada": temporada
}])

# Predicción
if st.button("🌾 Predecir Cultivo"):
    pred = model.predict(input_data)
    cultivo = label_encoder.inverse_transform(pred)[0]
    st.success(f"El modelo recomienda cultivar: **{cultivo}**")

# Formulario para agregar nuevos cultivos
st.markdown("---")
st.subheader("➕ Agregar un nuevo cultivo al dataset")

nuevo_cultivo = st.text_input("Nombre del cultivo")
if st.button("📤 Guardar en Firebase"):
    if nuevo_cultivo:
        nuevo_doc = input_data.iloc[0].to_dict()
        nuevo_doc["tipo_de_cultivo"] = nuevo_cultivo.lower()
        db.collection("cultivos").add(nuevo_doc)
        st.success("✅ Cultivo guardado correctamente en Firebase")
    else:
        st.error("❗ Por favor ingresa el nombre del cultivo")
