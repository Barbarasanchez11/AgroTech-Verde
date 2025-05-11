import firebase_admin
from firebase_admin import credentials, firestore
import streamlit as st



def init_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(dict(st.secrets["firebase"]))
        firebase_admin.initialize_app(cred)
    return firestore.client()

# Función para guardar datos
def guardar_datos_cultivo(datos):
    db = init_firebase()
    db.collection("cultivos").add(datos)

