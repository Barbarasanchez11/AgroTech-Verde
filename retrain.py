import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Cargar datos
df = pd.read_csv("agrotech_data.csv")

# Codificar objetivo
label_encoder = LabelEncoder()
df["cultivo_cod"] = label_encoder.fit_transform(df["tipo_de_cultivo"])
X = df.drop(columns=["tipo_de_cultivo", "cultivo_cod"])
y = df["cultivo_cod"]

# Columnas
num_cols = ["ph", "humedad", "temperatura", "precipitacion", "horas_de_sol"]
cat_cols = ["tipo_de_suelo", "temporada"]

# Preprocesador
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(), cat_cols)
])

# Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# Entrenar
pipeline.fit(X, y)

# Guardar modelo y label encoder
with open("modelo_rf.pkl", "wb") as f:
    pickle.dump(pipeline, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Modelos guardados correctamente.")
