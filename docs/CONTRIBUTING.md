# Guía de Contribución

¡Gracias por tu interés en contribuir a AgroTech-Verde! Este documento te ayudará a comenzar.

## Cómo Contribuir

### 1. Fork y Clone

1. Haz fork del repositorio
2. Clona tu fork localmente:
```bash
git clone https://github.com/tu-usuario/AgroTech-Verde.git
cd AgroTech-Verde
```

### 2. Configurar el Entorno

1. Crear entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

### 3. Crear una Rama

```bash
git checkout -b feature/tu-nueva-funcionalidad
```

### 4. Desarrollar

- Escribe código limpio y bien documentado
- Añade tests para nuevas funcionalidades
- Sigue las convenciones del proyecto

### 5. Ejecutar Tests

```bash
# Tests unitarios
pytest

# Tests con coverage
pytest --cov=src

# Linting
flake8 src/

# Formateo
black src/
```

### 6. Commit y Push

```bash
git add .
git commit -m "feat: añadir nueva funcionalidad"
git push origin feature/tu-nueva-funcionalidad
```

### 7. Crear Pull Request

1. Ve a tu fork en GitHub
2. Crea un Pull Request
3. Describe tus cambios claramente

## Convenciones

### Commits

Usamos [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` Nueva funcionalidad
- `fix:` Corrección de bug
- `docs:` Documentación
- `style:` Formato
- `refactor:` Refactorización
- `test:` Tests
- `chore:` Mantenimiento

### Código

- Usa type hints
- Documenta funciones y clases
- Sigue PEP 8
- Máximo 88 caracteres por línea

### Tests

- Cobertura mínima: 80%
- Tests unitarios para nuevas funcionalidades
- Tests de integración para APIs

## Reportar Bugs

1. Busca si ya existe un issue similar
2. Crea un nuevo issue con:
   - Descripción clara del problema
   - Pasos para reproducir
   - Comportamiento esperado vs actual
   - Información del sistema

## Sugerir Mejoras

1. Crea un issue con la etiqueta `enhancement`
2. Describe la mejora propuesta
3. Explica por qué sería útil

## Recursos

- [Documentación de Streamlit](https://docs.streamlit.io/)
- [Guía de estilo de Python](https://www.python.org/dev/peps/pep-0008/)
- [Conventional Commits](https://www.conventionalcommits.org/)

## ¡Gracias!

Tu contribución hace que AgroTech-Verde sea mejor para todos. ¡Gracias por tu tiempo y esfuerzo! 