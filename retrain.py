import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from modelo import label_encoder, rf_pipeline  
import pickle

# Inicializar Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase_config.json")
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Leer Firestore
docs = db.collection("cultivos").stream()
firebase_data = [doc.to_dict() for doc in docs]
df_firebase = pd.DataFrame(firebase_data)

# Combinar con datos existentes
df_original = pd.read_csv("agrotech_data.csv")
df_combined = pd.concat([df_original, df_firebase], ignore_index=True)
df_combined.to_csv("agrotech_data_actualizada.csv", index=False)

# Reentrenar
from modelo import entrenar_y_guardar_modelo 

entrenar_y_guardar_modelo("agrotech_data_actualizada.csv")
print("✅ Modelo reentrenado con nuevos datos")
