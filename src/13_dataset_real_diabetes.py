import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

dataset = load_diabetes(as_frame=True) # Cargamos el dataset de diabetes utilizando la función load_diabetes de sklearn.datasets, y establecemos el parámetro as_frame=True para obtener los datos en formato de DataFrame de pandas.

df = dataset.frame # Guardamos la tabla completa del dataset en un DataFrame de pandas llamado df.

print("Primeras filas del dataset:")
print(df.head()) # Imprimimos las primeras filas del DataFrame

print("\nInformación general:")
print(df.info()) # Imprimimos información general sobre el DataFrame, incluyendo el número de filas, columnas, tipos de datos y valores nulos.

print("\nEstadísticas básicas:")
print(df.describe()) # Imprimimos estadísticas básicas del DataFrame, como la media, desviación estándar, valores mínimos y máximos, y percentiles.

X = df.drop(columns=["target"]) # Guardamos en la variable X todas las columnas del DataFrame df excepto la columna "target", que es la variable objetivo que queremos predecir.
y = df["target"] # Guardamos en la variable y la columna "target" del DataFrame df, que es la variable objetivo que queremos predecir.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
) # Dividimos los datos en conjuntos de entrenamiento y prueba utilizando la función train_test_split de sklearn.model_selection. Establecemos el tamaño del conjunto de prueba en un 20% (test_size=0.2) y fijamos la semilla aleatoria en 42 (random_state=42) para garantizar la reproducibilidad de los resultados.

modelo = LinearRegression() # Creamos un objeto de la clase LinearRegression de sklearn.linear_model, que representa un modelo de regresión lineal.

modelo.fit(X_train, y_train) # Entrenamos el modelo de regresión lineal utilizando los datos de entrenamiento (X_train y y_train) mediante el método fit() del objeto modelo.

predicciones = modelo.predict(X_test) # Realizamos predicciones utilizando el modelo entrenado sobre los datos de prueba (X_test) mediante el método predict() del objeto modelo, y guardamos los resultados en la variable predicciones.

error_medio = mean_absolute_error(y_test, predicciones) # Calculamos el error medio absoluto (MAE) y el coeficiente de determinación (R2 score) utilizando las funciones mean_absolute_error y r2_score de sklearn.metrics, respectivamente. Estas métricas nos permiten evaluar la precisión del modelo en los datos de prueba.
r2 = r2_score(y_test, predicciones) # Calculamos el error medio absoluto (MAE) y el coeficiente de determinación (R2 score) utilizando las funciones mean_absolute_error y r2_score de sklearn.metrics, respectivamente. Estas métricas nos permiten evaluar la precisión del modelo en los datos de prueba.

resultados = pd.DataFrame({
    "valor_real": y_test,
    "valor_predicho": predicciones
}) # Creamos un DataFrame de pandas llamado resultados que contiene dos columnas: "valor_real" con los valores reales de la variable objetivo (y_test) y "valor_predicho" con los valores predichos por el modelo (predicciones). Esto nos permite comparar fácilmente los resultados del modelo con los valores reales.

print("\nResultados de prueba:")
print(resultados.head()) # Imprimimos las primeras filas del DataFrame resultados para mostrar una comparación entre los valores reales y los valores predichos por el modelo.

print("\nError medio absoluto:", error_medio) # Imprimimos el valor del error medio absoluto (MAE) calculado anteriormente, que nos indica la magnitud promedio de los errores de predicción del modelo.
print("R2 score:", r2) # Imprimimos el valor del coeficiente de determinación (R2 score) calculado anteriormente, que nos indica qué tan bien se ajusta el modelo a los datos de prueba. Un valor cercano a 1 indica un buen ajuste, mientras que un valor cercano a 0 indica un mal ajuste.

df.to_csv("data/diabetes_dataset.csv", index=False) # Guardamos el DataFrame df en un archivo CSV llamado "diabetes_dataset.csv" en la carpeta "data", sin incluir los índices de las filas (index=False).

joblib.dump(modelo, "models/modelo_diabetes.pkl") # Guardamos el modelo entrenado en un archivo llamado "modelo_diabetes.pkl" en la carpeta "models" utilizando la función dump() de joblib. Esto nos permite reutilizar el modelo en el futuro sin necesidad de volver a entrenarlo.

plt.scatter(y_test, predicciones) # Creamos un gráfico de dispersión utilizando la función scatter() de matplotlib.pyplot, donde el eje x representa los valores reales de la variable objetivo (y_test) y el eje y representa los valores predichos por el modelo (predicciones). Esto nos permite visualizar la relación entre los valores reales y los valores predichos.
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()]) # Agregamos una línea diagonal al gráfico que representa la relación ideal entre los valores reales y los valores predichos. Esta línea se dibuja desde el valor mínimo hasta el valor máximo de y_test, y nos permite comparar visualmente qué tan cerca están los puntos del gráfico de la línea ideal.

plt.title("Valor real vs valor predicho") # Agregamos un título al gráfico utilizando la función title() de matplotlib.pyplot, que describe la relación que se está visualizando en el gráfico.
plt.xlabel("Valor real") # Agregamos una etiqueta al eje x del gráfico utilizando la función xlabel() de matplotlib.pyplot, que indica que el eje x representa los valores reales de la variable objetivo.
plt.ylabel("Valor predicho") # Agregamos una etiqueta al eje y del gráfico utilizando la función ylabel() de matplotlib.pyplot, que indica que el eje y representa los valores predichos por el modelo.

plt.savefig("outputs/grafico_diabetes_real_vs_predicho.png") # Guardamos el gráfico generado en un archivo PNG llamado "grafico_diabetes_real_vs_predicho.png" en la carpeta "outputs" utilizando la función savefig() de matplotlib.pyplot. Esto nos permite conservar una copia del gráfico para su posterior análisis o presentación.

plt.show() # Mostramos el gráfico generado en una ventana emergente utilizando la función show() de matplotlib.pyplot, lo que nos permite visualizar la relación entre los valores reales y los valores predichos por el modelo.

print("\nDataset guardado en: data/diabetes_dataset.csv")
print("Modelo guardado en: models/modelo_diabetes.pkl")
print("Gráfico guardado en: outputs/grafico_diabetes_real_vs_predicho.png")