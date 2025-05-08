import streamlit as st
import pandas as pd
import pickle

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
humedad = st.slider("Humedad (%)", 20, 90, 50)
temperatura = st.slider("Temperatura (°C)", 10, 40, 25)
precipitacion = st.slider("Precipitación (mm)", 0, 300, 100)
horas_de_sol = st.slider("Horas de sol", 4, 12, 8)
tipo_de_suelo = st.selectbox("Tipo de suelo", ["arcilloso", "arenoso", "franco"])

# Crear DataFrame de entrada
input_data = pd.DataFrame([{
    "ph": ph,
    "humedad": humedad,
    "temperatura": temperatura,
    "precipitacion": precipitacion,
    "horas_de_sol": horas_de_sol,
    "tipo_de_suelo": tipo_de_suelo
}])

# Botón para predecir
if st.button("🌾 Predecir Cultivo"):
    pred = model.predict(input_data)
    cultivo = label_encoder.inverse_transform(pred)[0]
    st.success(f"El modelo recomienda cultivar: **{cultivo}**")
