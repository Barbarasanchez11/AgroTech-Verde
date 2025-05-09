import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, db

# === Configuración de entorno y Firebase ===
load_dotenv()
cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if not cred_path or not os.path.exists(cred_path):
    raise FileNotFoundError("El archivo de credenciales no fue encontrado. Verifica el .env y la ruta.")

cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://agrotech-edfb5-default-rtdb.europe-west1.firebasedatabase.app/'
})

# === Generar dataset simulado ===
np.random.seed(42)
n = 1000
data = {
    'tipo_de_cultivo': [],
    'ph': [],
    'tipo_de_suelo': [],
    'humedad': [],
    'temperatura': [],
    'precipitacion': [],
    'horas_de_sol': [],
    'temporada': []
}
cultivos = ['maiz', 'trigo', 'arroz', 'soja', 'naranjas', 'uva', 'lentejas', 'zanahoria']

for _ in range(n):
    cultivo = np.random.choice(cultivos)

    if cultivo == 'maiz':
        data['tipo_de_cultivo'].append(cultivo)
        data['ph'].append(np.round(np.random.uniform(6.0, 6.8), 2))
        data['tipo_de_suelo'].append(np.random.choice(['arcilloso', 'limoso', 'arenoso'], p=[0.6, 0.3, 0.1]))
        data['humedad'].append(np.round(np.random.uniform(50, 70), 2))
        data['temperatura'].append(np.round(np.random.uniform(22, 32)))
        data['precipitacion'].append(np.round(np.random.uniform(120, 220), 2))
        data['horas_de_sol'].append(np.round(np.random.uniform(8, 12), 1))
        data['temporada'].append(np.random.choice(['primavera', 'verano'], p=[0.4, 0.6]))
    elif cultivo == 'trigo':
        data['tipo_de_cultivo'].append(cultivo)
        data['ph'].append(np.round(np.random.uniform(6.4, 7.3), 2))
        data['tipo_de_suelo'].append(np.random.choice(['arcilloso', 'limoso', 'arenoso'], p=[0.6, 0.3, 0.1]))
        data['humedad'].append(np.round(np.random.uniform(40, 60), 2))
        data['temperatura'].append(np.round(np.random.uniform(12, 22)))
        data['precipitacion'].append(np.round(np.random.uniform(110, 190), 2))
        data['horas_de_sol'].append(np.round(np.random.uniform(6, 10), 1))
        data['temporada'].append(np.random.choice(['otoño', 'invierno'], p=[0.9, 0.1]))
    elif cultivo == 'arroz':
        data['tipo_de_cultivo'].append(cultivo)
        data['ph'].append(np.round(np.random.uniform(5.5, 6.5), 2))
        data['tipo_de_suelo'].append(np.random.choice(['arcilloso', 'limoso'], p=[0.8, 0.2]))
        data['humedad'].append(np.round(np.random.uniform(80, 100), 2))
        data['temperatura'].append(np.round(np.random.uniform(20, 35)))
        data['precipitacion'].append(np.round(np.random.uniform(200, 300), 2))
        data['horas_de_sol'].append(np.round(np.random.uniform(8, 12), 1))
        data['temporada'].append(np.random.choice(['verano', 'primavera'], p=[0.6, 0.4]))
    elif cultivo == 'soja':
        data['tipo_de_cultivo'].append(cultivo)
        data['ph'].append(np.round(np.random.uniform(6.2, 7.0), 2))
        data['tipo_de_suelo'].append(np.random.choice(['limoso', 'arcilloso', 'arenoso'], p=[0.6, 0.3, 0.1]))
        data['humedad'].append(np.round(np.random.uniform(50, 70), 2))
        data['temperatura'].append(np.round(np.random.uniform(20, 28)))
        data['precipitacion'].append(np.round(np.random.uniform(90, 170), 2))
        data['horas_de_sol'].append(np.round(np.random.uniform(8, 12), 1))
        data['temporada'].append(np.random.choice(['primavera', 'verano'], p=[0.7, 0.3]))
    elif cultivo == 'naranjas':
        data['tipo_de_cultivo'].append(cultivo)
        data['ph'].append(np.round(np.random.uniform(5.6, 6.3), 2))
        data['tipo_de_suelo'].append(np.random.choice(['arenoso', 'limoso', 'arcilloso'], p=[0.5, 0.4, 0.1]))
        data['humedad'].append(np.round(np.random.uniform(50, 70), 2))
        data['temperatura'].append(np.round(np.random.uniform(20, 30)))
        data['precipitacion'].append(np.round(np.random.uniform(100, 200), 2))
        data['horas_de_sol'].append(np.round(np.random.uniform(8, 12), 1))
        data['temporada'].append(np.random.choice(['primavera', 'verano'], p=[0.5, 0.5]))
    elif cultivo == 'uva':
        data['tipo_de_cultivo'].append(cultivo)
        data['ph'].append(np.round(np.random.uniform(6.0, 7.0), 2))
        data['tipo_de_suelo'].append(np.random.choice(['arenoso', 'rocoso', 'limoso'], p=[0.5, 0.3, 0.2]))
        data['humedad'].append(np.round(np.random.uniform(40, 60), 2))
        data['temperatura'].append(np.round(np.random.uniform(15, 30)))
        data['precipitacion'].append(np.round(np.random.uniform(80, 150), 2))
        data['horas_de_sol'].append(np.round(np.random.uniform(8, 12), 1))
        data['temporada'].append(np.random.choice(['primavera', 'verano'], p=[0.4, 0.6]))
    elif cultivo == 'lentejas':
        data['tipo_de_cultivo'].append(cultivo)
        data['ph'].append(np.round(np.random.uniform(6.0, 7.0), 2))
        data['tipo_de_suelo'].append(np.random.choice(['limoso', 'arcilloso', 'arenoso'], p=[0.7, 0.2, 0.1]))
        data['humedad'].append(np.round(np.random.uniform(40, 60), 2))
        data['temperatura'].append(np.round(np.random.uniform(13, 23)))
        data['precipitacion'].append(np.round(np.random.uniform(80, 150), 2))
        data['horas_de_sol'].append(np.round(np.random.uniform(6, 10), 1))
        data['temporada'].append(np.random.choice(['otoño', 'invierno'], p=[0.3, 0.7]))
    elif cultivo == 'zanahoria':
        data['tipo_de_cultivo'].append(cultivo)
        data['ph'].append(np.round(np.random.uniform(6.0, 6.8), 2))
        data['tipo_de_suelo'].append(np.random.choice(['arenoso', 'limoso', 'arcilloso'], p=[0.6, 0.3, 0.1]))
        data['humedad'].append(np.round(np.random.uniform(60, 80), 2))
        data['temperatura'].append(np.round(np.random.uniform(15, 25)))
        data['precipitacion'].append(np.round(np.random.uniform(80, 150), 2))
        data['horas_de_sol'].append(np.round(np.random.uniform(6, 10), 1))
        data['temporada'].append(np.random.choice(['otoño', 'invierno', 'primavera'], p=[0.4, 0.3, 0.3]))

df = pd.DataFrame(data)
df.to_csv('agrotech_data.csv', index=False)
print("Dataset generado: agrotech_data.csv")
print(df.head())

# === Interacción con el usuario ===
print("\nOpciones:")
print("1 - Elegir fila del dataset")
print("2 - Introducir datos manualmente")
opcion = input("Selecciona una opción (1 o 2): ")

if opcion == '1':
    index = int(input(f"Introduce un número entre 0 y {len(df) - 1}: "))
    if 0 <= index < len(df):
        entrada = df.iloc[index].to_dict()
    else:
        print("Índice fuera de rango.")
        exit()

elif opcion == '2':
    entrada = {
        'tipo_de_cultivo': input("Tipo de cultivo: "),
        'ph': float(input("pH del suelo: ")),
        'tipo_de_suelo': input("Tipo de suelo: "),
        'humedad': float(input("Humedad (%): ")),
        'temperatura': float(input("Temperatura (°C): ")),
        'precipitacion': float(input("Precipitación (mm): ")),
        'horas_de_sol': float(input("Horas de sol: ")),
        'temporada': input("Temporada (primavera/verano/otoño/invierno): ")
    }
else:
    print("Opción inválida.")
    exit()

# === Subida a Firebase ===
ref = db.reference('datos_agricolas')
ref.push(entrada)
print("\n✅ Datos subidos exitosamente a Firebase.")
