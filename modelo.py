import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('agrotech_data.csv')
pd.set_option('display.float_format', '{:.2f}'.format)



print(df.head())
print(df.info())
print(df.describe())

sns.countplot(data=df, x='tipo_de_cultivo', hue='temporada')
plt.title('Distribución de Cultivos por Temporada')
plt.show()

