import pandas as pd
import numpy as np

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
        data['ph'].append(np.round(np.random.uniform(6.4, 7.3), 2))  # Más neutro
        data['tipo_de_suelo'].append(np.random.choice(['arcilloso', 'limoso', 'arenoso'], p=[0.6, 0.3, 0.1]))  # Más arcilloso
        data['humedad'].append(np.round(np.random.uniform(40, 60), 2))
        data['temperatura'].append(np.round(np.random.uniform(12, 22)))  # Más fresco
        data['precipitacion'].append(np.round(np.random.uniform(110, 190), 2))  # Más lluvia
        data['horas_de_sol'].append(np.round(np.random.uniform(6, 10), 1))
        data['temporada'].append(np.random.choice(['otoño', 'invierno'], p=[0.9, 0.1]))  # Casi solo otoño
    
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