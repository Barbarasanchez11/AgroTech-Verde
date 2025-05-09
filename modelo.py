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


df=pd.read_csv('agrotech_data.csv')
pd.set_option('display.float_format', '{:.2f}'.format)

# Cargar datos nuevos desde Firebase
from firebase_utils import init_firebase
db = init_firebase()
docs = db.collection("cultivos").stream()
datos = [doc.to_dict() for doc in docs]
df_nuevos = pd.DataFrame(datos)

# Unir datasets
df = pd.concat([df_original, df_nuevos], ignore_index=True)


print(df.head())
print(df.info())
print(df.describe())

sns.countplot(data=df, x='tipo_de_cultivo', hue='temporada')
plt.title('Distribución de Cultivos por Temporada')
plt.show()

# 2. Codificar la variable objetivo (y)
label_encoder = LabelEncoder()
df["cultivo_cod"] = label_encoder.fit_transform(df["tipo_de_cultivo"])

print(df.head())

# 3. Separar X e y
X = df.drop(columns=["tipo_de_cultivo", "cultivo_cod"])
y = df["cultivo_cod"]

print(X.head())
print(y.head())

# 4. Separar en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train, X_test, y_train, y_test)


# 5. Identificar columnas numéricas y categóricas
num_cols = ["ph", "humedad", "temperatura", "precipitacion", "horas_de_sol"]
cat_cols = ["tipo_de_suelo", "temporada"]

print(num_cols)
print(cat_cols)

# 6. Crear transformador (escalar y codificar)
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(), cat_cols)
])

# Crear pipelines para Random Forest y SVM
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

svm_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", SVC(probability=True, random_state=42))
])

# Entrenar los modelos
rf_pipeline.fit(X_train, y_train)
svm_pipeline.fit(X_train, y_train)

# Evaluar Random Forest
y_pred_rf = rf_pipeline.predict(X_test)
print("Random Forest - Reporte de clasificación:")
print(classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf, display_labels=label_encoder.classes_)
plt.title("Matriz de confusión - Random Forest")
plt.xticks(rotation=45)
plt.show()

# Evaluar SVM
y_pred_svm = svm_pipeline.predict(X_test)
print("SVM - Reporte de clasificación:")
print(classification_report(y_test, y_pred_svm, target_names=label_encoder.classes_))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_svm, display_labels=label_encoder.classes_)
plt.title("Matriz de confusión - SVM")
plt.xticks(rotation=45)
plt.show()

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