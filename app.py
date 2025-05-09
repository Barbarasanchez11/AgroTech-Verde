import streamlit as st
import pandas as pd
import pickle
from firebase_utils import init_firebase

# Inicializar Firebase
db = init_firebase()

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

# Botón para predecir
if st.button("🌾 Predecir Cultivo"):
    pred = model.predict(input_data)
    cultivo = label_encoder.inverse_transform(pred)[0]
    st.success(f"El modelo recomienda cultivar: **{cultivo}**")

st.markdown("---")
st.subheader("📥 Añadir Nuevo Registro de Cultivo")

with st.form("formulario_nuevo_cultivo"):
    nuevo_cultivo = st.text_input("Nombre del cultivo")
    ph_nuevo = st.slider("pH del suelo", 4.5, 8.5, 6.5, key="ph_nuevo")
    humedad_nuevo = st.slider("Humedad (%)", 0, 100, 50, key="humedad_nuevo")
    temperatura_nuevo = st.slider("Temperatura (°C)", 0, 40, 20, key="temperatura_nuevo")
    precipitacion_nuevo = st.slider("Precipitación (mm)", 0, 300, 150, key="precipitacion_nuevo")
    horas_de_sol_nuevo = st.slider("Horas de sol", 0, 16, 8, key="horas_de_sol_nuevo")
    tipo_de_suelo_nuevo = st.selectbox("Tipo de suelo", ["arcilloso", "arenoso", "limoso", "rocoso"], key="tipo_de_suelo_nuevo")
    temporada_nuevo = st.selectbox("Temporada", ['verano', 'otoño', 'invierno', 'primavera'], key="temporada_nuevo")
    submit = st.form_submit_button("Guardar")

    if submit:
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
        db.collection("cultivos").add(nuevo_registro)
        st.success("✅ Registro guardado correctamente.")
