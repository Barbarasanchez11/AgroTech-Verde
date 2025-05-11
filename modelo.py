import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import pickle
import streamlit as st
import numpy as np
from firebase_utils import init_firebase

# ========================
# 1. Cargar y unir datos
# ========================

df_original = pd.read_csv('agrotech_data.csv')
pd.set_option('display.float_format', '{:.2f}'.format)

# Cargar datos nuevos desde Firebase
db = init_firebase()
docs = db.collection("cultivos").stream()
datos = [doc.to_dict() for doc in docs]
df_nuevos = pd.DataFrame(datos)

# Unir datasets
df = pd.concat([df_original, df_nuevos], ignore_index=True)

# 2. Análisis Exploratorio

print(df.head())
print(df.info())
print(df.describe())

plt.figure(figsize=(14, 6))
sns.countplot(data=df, x='tipo_de_cultivo', hue='temporada', dodge=True)
plt.title("Distribución de Cultivos por Temporada")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 3. Codificar objetivo


label_encoder = LabelEncoder()
df["cultivo_cod"] = label_encoder.fit_transform(df["tipo_de_cultivo"])
print(df.head())


# 4. Preparar datos

X = df.drop(columns=["tipo_de_cultivo", "cultivo_cod"])
y = df["cultivo_cod"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_cols = ["ph", "humedad", "temperatura", "precipitacion", "horas_de_sol"]
cat_cols = ["tipo_de_suelo", "temporada"]

# Preprocesador
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(), cat_cols)
])

# 5. Pipelines de modelos

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

svm_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", SVC(probability=True, random_state=42))
])

# 6. Entrenamiento

rf_pipeline.fit(X_train, y_train)
svm_pipeline.fit(X_train, y_train)

# 7. Evaluación

all_labels = list(label_encoder.transform(label_encoder.classes_))
target_names = label_encoder.classes_

# ----- Random Forest -----
y_pred_rf = rf_pipeline.predict(X_test)
print("Random Forest - Reporte de clasificación:")
print(classification_report(
    y_test,
    y_pred_rf,
    labels=all_labels,
    target_names=target_names,
    zero_division=0
))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_rf,
    display_labels=target_names,
    labels=all_labels
)
plt.title("Matriz de confusión - Random Forest")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ----- SVM -----
y_pred_svm = svm_pipeline.predict(X_test)
print("SVM - Reporte de clasificación:")
print(classification_report(
    y_test,
    y_pred_svm,
    labels=all_labels,
    target_names=target_names,
    zero_division=0
))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_svm,
    display_labels=target_names,
    labels=all_labels
)
plt.title("Matriz de confusión - SVM")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 8. Guardar modelos

with open("modelo_rf.pkl", "wb") as f:
    pickle.dump(rf_pipeline, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)


def entrenar_y_guardar_modelo(csv_path):
    df = pd.read_csv(csv_path)
    label_encoder = LabelEncoder()
    df["cultivo_cod"] = label_encoder.fit_transform(df["tipo_de_cultivo"])
    X = df.drop(columns=["tipo_de_cultivo", "cultivo_cod"])
    y = df["cultivo_cod"]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(), cat_cols)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    pipeline.fit(X, y)
    with open("modelo_entrenado.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    with open("label_encoder_entrenado.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
