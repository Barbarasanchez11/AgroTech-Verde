# 🌱 AgroTech-Verde

Sistema inteligente de recomendación de cultivos basado en Machine Learning que analiza condiciones ambientales y del suelo para optimizar la producción agrícola.

## 🚀 Características

- **Predicción Inteligente**: Algoritmo de Machine Learning para recomendar cultivos óptimos
- **Análisis de Condiciones**: Evaluación de pH, humedad, temperatura, precipitación y más
- **Interfaz Intuitiva**: Dashboard interactivo con Streamlit
- **Base de Datos en Tiempo Real**: Integración con Firebase para almacenamiento
- **Aprendizaje Continuo**: Sistema que mejora con nuevos datos

## 📊 Tecnologías Utilizadas

- **Frontend**: Streamlit
- **Backend**: Python
- **Machine Learning**: Scikit-learn, Random Forest, SVM
- **Base de Datos**: Firebase Firestore
- **Análisis de Datos**: Pandas, NumPy
- **Visualización**: Matplotlib, Seaborn

## 🛠️ Instalación

### Prerrequisitos

- Python 3.8+
- pip
- Cuenta de Firebase (opcional para funcionalidad completa)

### Instalación Local

1. **Clonar el repositorio**
```bash
git clone https://github.com/tu-usuario/AgroTech-Verde.git
cd AgroTech-Verde
```

2. **Crear entorno virtual**
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Configurar Firebase (opcional)**
```bash
# Crear archivo .streamlit/secrets.toml
mkdir .streamlit
touch .streamlit/secrets.toml
```

Añadir las credenciales de Firebase en `.streamlit/secrets.toml`:
```toml
[firebase]
type = "service_account"
project_id = "tu-proyecto-id"
private_key_id = "tu-private-key-id"
private_key = "tu-private-key"
client_email = "tu-client-email"
client_id = "tu-client-id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "tu-cert-url"
```

5. **Ejecutar la aplicación**
```bash
streamlit run src/app.py
```

## 🎯 Uso

1. **Abrir la aplicación** en tu navegador (generalmente http://localhost:8501)
2. **Configurar parámetros** del terreno usando los sliders y selectores
3. **Hacer predicción** haciendo clic en "Predecir Cultivo"
4. **Ver resultados** con la recomendación del cultivo óptimo
5. **Añadir nuevos registros** para mejorar el modelo

## 📁 Estructura del Proyecto

```
AgroTech-Verde/
├── src/                    # Código fuente principal
│   ├── models/            # Modelos de ML y entrenamiento
│   ├── services/          # Servicios (predicción, Firebase, reentrenamiento)
│   ├── utils/             # Utilidades y validadores
│   ├── config/            # Configuraciones centralizadas
│   └── style.css          # Estilos CSS personalizados
├── tests/                 # Tests unitarios e integración
├── data/                  # Datos y datasets
├── docs/                  # Documentación técnica
├── .streamlit/            # Configuración de Streamlit
├── requirements.txt       # Dependencias del proyecto
├── pyproject.toml         # Configuración del proyecto
├── pytest.ini            # Configuración de tests
├── .gitignore            # Archivos ignorados por Git
└── README.md             # Este archivo
```

## 🔧 Desarrollo

### Ejecutar Tests
```bash
python -m pytest tests/
```

### Entrenar Modelo
```bash
python src/models/train_model.py
```

### Generar Datos de Ejemplo
```bash
python src/utils/generate_data.py
```

## 📈 Métricas del Modelo

- **Precisión**: 95.2%
- **Recall**: 94.8%
- **F1-Score**: 95.0%
- **Cultivos Soportados**: 8 tipos diferentes

## 🤝 Contribuir

1. Fork el proyecto
2. Crear una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 👨‍💻 Autor

**Bárbara Sánchez Urbano** - [barbarasanchezurbano@gmail.com](barbarasanchezurbano@gmail.com)


⭐ Si este proyecto te ha sido útil, ¡dale una estrella al repositorio!

## 🎯 **Valor del Mensaje "✅ Predicción guardada en la base de datos"**

### 💡 **Para el Usuario:**

1. **Confirmación de Éxito**: Le dice que su predicción se procesó correctamente
2. **Persistencia de Datos**: Sabe que su información no se perdió
3. **Historial Disponible**: Puede consultar sus predicciones anteriores
4. **Confianza en el Sistema**: Ve que todo funciona como debe

### 🔍 **Para el Sistema:**

1. **Trazabilidad**: Cada predicción queda registrada con timestamp
2. **Mejora del Modelo**: Los datos se pueden usar para reentrenar
3. **Análisis**: Se pueden generar estadísticas de uso
4. **Debugging**: Si hay problemas, se puede revisar el historial

### 📊 **Datos que se Guardan:**

```python
prediction_data = {
    # Parámetros del terreno
    "ph": 6.5,
    "humedad": 50,
    "temperatura": 20,
    "precipitacion": 150,
    "horas_de_sol": 8.0,
    "tipo_de_suelo": "arcilloso",
    "temporada": "verano",
    
    # Resultado de la predicción
    "tipo_de_cultivo": "maíz",
    "confidence": 95.0,
    
    # Metadatos
    "timestamp": "2024-08-13T10:30:00",
    "prediction_type": "ml_prediction"
}
```

### 🚀 **Beneficios del Usuario:**

- **Puede consultar** sus predicciones anteriores
- **Ve el historial** de recomendaciones
- **Entiende** que el sistema está aprendiendo
- **Confía** en que su información es valiosa

### 💭 **¿Quieres Personalizar el Mensaje?**

Podríamos hacer el mensaje más informativo, por ejemplo:

```python
st.success(f"✅ Predicción guardada: {crop} recomendado para tu terreno")
# o
st.success("✅ Predicción guardada. Puedes consultar tu historial en la pestaña 'Historial'")
```

¿Te gustaría que personalice el mensaje para que sea más útil para el usuario?