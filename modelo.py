import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

df=pd.read_csv('agrotech_data.csv')
pd.set_option('display.float_format', '{:.2f}'.format)



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
cat_cols = ["tipo_suelo"]

# 6. Crear transformador (escalar y codificar)
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(), cat_cols)
])
