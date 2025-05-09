import firebase_admin
from firebase_admin import credentials, firestore
import streamlit as st

# Inicializar Firebase solo una vez
@st.cache_resource
def init_firebase():
    cred = credentials.Certificate(dict(st.secrets["firebase"]))
    firebase_admin.initialize_app(cred)
    return firestore.client()

# Función para guardar datos
def guardar_datos_cultivo(datos):
    db = init_firebase()
    db.collection("cultivos").add(datos)

