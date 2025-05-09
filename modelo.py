import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pickle
import numpy as np
import firebase_admin
from firebase_admin import credentials, db

try:
    # Inicializar Firebase
    if not firebase_admin._apps:
        cred = credentials.Certificate("firebase_credentials.json")
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://agrotech-edfb5-default-rtdb.europe-west1.firebasedatabase.app/'
        })

    # Cargar datos sintéticos
    df_synthetic = pd.read_csv('agrotech_data.csv')
    print("Datos sintéticos cargados:")
    print(df_synthetic.head())

    # Cargar datos reales desde Firebase
    ref = db.reference('/user_data')
    user_data = ref.get()
    if user_data:
        df_user = pd.DataFrame([data for data in user_data.values()])
        print("Datos reales cargados desde Firebase:")
        print(df_user.head())
        df = pd.concat([df_synthetic, df_user], ignore_index=True)
    else:
        df = df_synthetic
        print("No se encontraron datos reales en Firebase. Usando solo datos sintéticos.")

    # Codificar la variable objetivo
    label_encoder = LabelEncoder()
    df["cultivo_cod"] = label_encoder.fit_transform(df["tipo_de_cultivo"])

    # Separar X e y
    X = df.drop(columns=["tipo_de_cultivo", "cultivo_cod"])
    y = df["cultivo_cod"]

    # Separar en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Tamaño de X_train:", X_train.shape)
    print("Tamaño de X_test:", X_test.shape)

    # Identificar columnas numéricas y categóricas
    num_cols = ["ph", "humedad", "temperatura", "precipitacion", "horas_de_sol"]
    cat_cols = ["tipo_de_suelo", "temporada"]

    # Crear transformador
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(), cat_cols)
    ])

    # Crear pipelines
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    svm_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", SVC(probability=True, random_state=42))
    ])

    # Entrenar modelos
    print("Entrenando Random Forest...")
    rf_pipeline.fit(X_train, y_train)
    print("Entrenando SVM...")
    svm_pipeline.fit(X_train, y_train)

    # Evaluar Random Forest
    y_pred_rf = rf_pipeline.predict(X_test)
    print("Random Forest - Reporte de clasificación:")
    print(classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_))

    # Evaluar SVM
    y_pred_svm = svm_pipeline.predict(X_test)
    print("SVM - Reporte de clasificación:")
    print(classification_report(y_test, y_pred_svm, target_names=label_encoder.classes_))

    # Guardar modelos y codificador
    print("Guardando archivos .pkl...")
    pickle.dump(rf_pipeline, open("modelo_rf.pkl", "wb"))
    pickle.dump(svm_pipeline, open("modelo_svm.pkl", "wb"))
    pickle.dump(label_encoder, open("label_encoder.pkl", "wb"))
    print("Archivos .pkl generados: modelo_rf.pkl, modelo_svm.pkl, label_encoder.pkl")

except Exception as e:
    print(f"Error durante la ejecución: {str(e)}")