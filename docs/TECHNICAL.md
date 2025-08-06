# Documentación Técnica

## Arquitectura del Proyecto

### Estructura de Directorios

```
AgroTech-Verde/
├── src/                    # Código fuente principal
│   ├── config/            # Configuración centralizada
│   ├── services/          # Servicios de negocio
│   ├── utils/             # Utilidades y validadores
│   ├── models/            # Modelos de ML
│   └── app.py             # Aplicación principal
├── tests/                 # Tests unitarios e integración
├── data/                  # Datos y datasets
├── docs/                  # Documentación
├── requirements.txt       # Dependencias
└── README.md             # Documentación principal
```

### Componentes Principales

#### 1. Configuración (`src/config/`)

- **config.py**: Configuración centralizada del proyecto
- Maneja rutas, parámetros, y configuraciones de ML
- Proporciona funciones utilitarias para acceso a archivos

#### 2. Servicios (`src/services/`)

- **prediction_service.py**: Lógica de predicción de ML
- **firebase_service.py**: Operaciones de base de datos
- Separación clara entre lógica de negocio y acceso a datos

#### 3. Utilidades (`src/utils/`)

- **validators.py**: Validación de datos de entrada
- Validación de parámetros del terreno
- Validación de modelos de ML

#### 4. Aplicación Principal (`src/app.py`)

- Interfaz de usuario con Streamlit
- Coordinación entre servicios
- Manejo de errores y validación

## Flujo de Datos

### 1. Entrada de Usuario

1. Usuario introduce parámetros del terreno
2. Validación de datos de entrada
3. Procesamiento por el servicio de predicción

### 2. Predicción

1. Carga de modelos de ML
2. Preprocesamiento de datos
3. Predicción del cultivo
4. Decodificación del resultado

### 3. Almacenamiento

1. Guardado en Firebase
2. Actualización de estadísticas
3. Registro de operaciones

## Tecnologías Utilizadas

### Frontend
- **Streamlit**: Framework para aplicaciones web
- **CSS personalizado**: Estilos específicos del proyecto

### Backend
- **Python 3.8+**: Lenguaje principal
- **Pandas**: Manipulación de datos
- **NumPy**: Operaciones numéricas

### Machine Learning
- **Scikit-learn**: Algoritmos de ML
- **Random Forest**: Modelo principal
- **SVM**: Modelo alternativo

### Base de Datos
- **Firebase Firestore**: Base de datos en la nube
- **Firebase Admin SDK**: Acceso programático

### Testing
- **pytest**: Framework de testing
- **pytest-cov**: Cobertura de código
- **Mocking**: Tests aislados

## Configuración de Desarrollo

### Variables de Entorno

```bash
# Firebase (opcional)
FIREBASE_PROJECT_ID=tu-proyecto-id
FIREBASE_PRIVATE_KEY=tu-private-key
FIREBASE_CLIENT_EMAIL=tu-client-email

# Streamlit
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
```

### Configuración de Firebase

1. Crear proyecto en Firebase Console
2. Generar credenciales de servicio
3. Configurar en `.streamlit/secrets.toml`

### Configuración de Streamlit

```toml
# .streamlit/secrets.toml
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

## Deployment

### Local

```bash
streamlit run src/app.py
```

### Producción

1. Configurar variables de entorno
2. Instalar dependencias
3. Ejecutar con gunicorn o similar

### Docker (Futuro)

```dockerfile
FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "src/app.py"]
```

## Monitoreo y Logging

### Logging

- Configurado con `logging` estándar de Python
- Nivel INFO para desarrollo
- Nivel WARNING para producción

### Métricas

- Cobertura de tests: >80%
- Tiempo de respuesta: <2s
- Disponibilidad: >99%

## Seguridad

### Validación de Datos

- Validación de entrada en todos los puntos
- Sanitización de datos
- Prevención de inyección

### Credenciales

- No hardcodear credenciales
- Usar variables de entorno
- Rotar claves regularmente

## Performance

### Optimizaciones

- Caching de modelos con `@st.cache_resource`
- Lazy loading de servicios
- Validación temprana de datos

### Escalabilidad

- Arquitectura modular
- Separación de responsabilidades
- Fácil añadir nuevos servicios

## Troubleshooting

### Problemas Comunes

1. **Modelo no encontrado**
   - Verificar que `modelo_rf.pkl` existe
   - Ejecutar script de entrenamiento

2. **Firebase no conecta**
   - Verificar credenciales en secrets
   - Comprobar conectividad de red

3. **Tests fallan**
   - Ejecutar `pip install -e ".[dev]"`
   - Verificar que todos los archivos existen

### Logs

```bash
# Ver logs de Streamlit
streamlit run src/app.py --logger.level=debug

# Ver logs de Python
python -u src/app.py
``` 