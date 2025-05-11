import pandas as pd
import pickle
import streamlit as st
from firebase_utils import init_firebase, guardar_datos_cultivo

# Título
st.set_page_config(page_title="Clasificador de Cultivos", layout="centered")
st.title("🌱 Clasificador de Cultivos")
st.write("Introduce las condiciones del terreno para predecir el tipo de cultivo adecuado.")

# Cargar modelo y codificador
with open("modelo_rf.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

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

    datos_cultivo = input_data.iloc[0].to_dict()
    datos_cultivo["tipo_de_cultivo"] = cultivo

    guardar_datos_cultivo(datos_cultivo)
    st.success("¡Datos guardados exitosamente en Firebase!")

st.markdown("---")
st.subheader("Añadir Nuevo Registro de Cultivo")

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

# Firebase solo se inicializa aquí, cuando realmente se necesita
db = init_firebase()

# Mostrar datos existentes de Firebase
docs = db.collection("cultivos").stream()
datos = [doc.to_dict() for doc in docs]
df = pd.DataFrame(datos)

if not df.empty:
    st.subheader("📊 Datos de cultivos ingresados por usuarios")
    st.dataframe(df)


