import pandas as pd # Importamos la biblioteca pandas, que nos permite trabajar con estructuras de datos como DataFrames y Series, facilitando la manipulación y análisis de datos en Python.
import matplotlib.pyplot as plt # Importamos la biblioteca matplotlib.pyplot, que nos permite crear gráficos y visualizaciones de datos en Python.
import joblib # Importamos la biblioteca joblib, que nos permite guardar y cargar objetos de Python, como modelos entrenados, de manera eficiente.

from sklearn.linear_model import LinearRegression # Importamos la clase LinearRegression de sklearn.linear_model, que nos permite crear y entrenar modelos de regresión lineal en Python.
from sklearn.tree import DecisionTreeRegressor # Importamos la clase DecisionTreeRegressor de sklearn.tree, que nos permite crear y entrenar modelos de regresión basados en árboles de decisión en Python.
from sklearn.ensemble import RandomForestRegressor # Importamos la clase RandomForestRegressor de sklearn.ensemble, que nos permite crear y entrenar modelos de regresión basados en bosques aleatorios (Random Forest) en Python.
from sklearn.model_selection import train_test_split # Importamos la función train_test_split de sklearn.model_selection, que nos permite dividir nuestros datos en conjuntos de entrenamiento y prueba de manera aleatoria y reproducible.
from sklearn.metrics import mean_absolute_error, r2_score # Importamos las funciones mean_absolute_error y r2_score de sklearn.metrics, que nos permiten calcular métricas de evaluación para modelos de regresión, como el error medio absoluto (MAE) y el coeficiente de determinación (R2 score).

df = pd.read_csv("data/diabetes_dataset.csv") # Cargamos el archivo CSV "diabetes_dataset.csv" que contiene el dataset de diabetes previamente guardado en la carpeta "data", y lo almacenamos en un DataFrame de pandas llamado df.

X = df.drop(columns=["target"]) # Guardamos en la variable X todas las columnas del DataFrame df excepto la columna "target", que es la variable objetivo que queremos predecir. Esto nos permite separar las características (features) de la variable objetivo (target) para entrenar y evaluar nuestros modelos de regresión.
y = df["target"] # Guardamos en la variable y la columna "target" del DataFrame df, que es la variable objetivo que queremos predecir. Esto nos permite separar las características (features) de la variable objetivo (target) para entrenar y evaluar nuestros modelos de regresión.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
) # Dividimos los datos en conjuntos de entrenamiento y prueba utilizando la función train_test_split de sklearn.model_selection. Establecemos el tamaño del conjunto de prueba en un 20% (test_size=0.2) y fijamos la semilla aleatoria en 42 (random_state=42) para garantizar la reproducibilidad de los resultados.

modelos = {
    "Regresion Lineal": LinearRegression(), 
    "Arbol de Decision": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
} # Creamos un diccionario llamado modelos que contiene tres modelos de regresión diferentes: Regresión Lineal, Árbol de Decisión y Random Forest. Cada modelo se instancia con sus respectivos parámetros, como la semilla aleatoria (random_state) para garantizar la reproducibilidad de los resultados. Esto nos permite comparar el rendimiento de diferentes algoritmos de regresión en el mismo conjunto de datos.

resultados = [] # Creamos una lista vacía llamada resultados que utilizaremos para almacenar los resultados de la evaluación de cada modelo, incluyendo el nombre del modelo, el error medio absoluto (MAE) y el coeficiente de determinación (R2 score). Esto nos permitirá comparar fácilmente el rendimiento de los diferentes modelos de regresión en el mismo conjunto de datos.

mejor_modelo = None # Creamos una variable llamada mejor_modelo y la inicializamos como None. Esta variable se utilizará para almacenar el modelo que tenga el mejor rendimiento (es decir, el menor error medio absoluto) durante la comparación de modelos. Esto nos permitirá identificar fácilmente cuál de los modelos evaluados es el más adecuado para predecir la variable objetivo en nuestro conjunto de datos de diabetes.
mejor_nombre = None # Creamos una variable llamada mejor_nombre y la inicializamos como None. Esta variable se utilizará para almacenar el nombre del modelo que tenga el mejor rendimiento (es decir, el menor error medio absoluto) durante la comparación de modelos. Esto nos permitirá identificar fácilmente cuál de los modelos evaluados es el más adecuado para predecir la variable objetivo en nuestro conjunto de datos de diabetes.
mejor_error = None # Creamos una variable llamada mejor_error y la inicializamos como None. Esta variable se utilizará para almacenar el valor del error medio absoluto (MAE) del modelo que tenga el mejor rendimiento durante la comparación de modelos. Esto nos permitirá identificar fácilmente cuál de los modelos evaluados es el más adecuado para predecir la variable objetivo en nuestro conjunto de datos de diabetes.

for nombre, modelo in modelos.items(): # Iteramos sobre los elementos del diccionario modelos utilizando un bucle for. En cada iteración, obtenemos el nombre del modelo (nombre) y la instancia del modelo (modelo). Esto nos permite entrenar y evaluar cada modelo de regresión en el mismo conjunto de datos de diabetes, y almacenar sus resultados para compararlos posteriormente.
    modelo.fit(X_train, y_train)

    predicciones = modelo.predict(X_test)

    error_medio = mean_absolute_error(y_test, predicciones)
    r2 = r2_score(y_test, predicciones)

    resultados.append({
        "modelo": nombre,
        "error_medio_absoluto": error_medio,
        "r2_score": r2
    })

    if mejor_error is None or error_medio < mejor_error:
        mejor_error = error_medio
        mejor_modelo = modelo
        mejor_nombre = nombre

df_resultados = pd.DataFrame(resultados)

print("Comparación de modelos:")
print(df_resultados)

print("\nMejor modelo:")
print(mejor_nombre)

print("\nError medio absoluto del mejor modelo:")
print(mejor_error)

df_resultados.to_csv("outputs/comparacion_modelos.csv", index=False)

joblib.dump(mejor_modelo, "models/mejor_modelo_diabetes.pkl")

plt.bar(df_resultados["modelo"], df_resultados["error_medio_absoluto"])

plt.title("Comparación de modelos por error medio absoluto")
plt.xlabel("Modelo")
plt.ylabel("Error medio absoluto")
plt.xticks(rotation=20)

plt.tight_layout()
plt.savefig("outputs/comparacion_modelos_mae.png")

plt.show()

print("\nResultados guardados en: outputs/comparacion_modelos.csv")
print("Mejor modelo guardado en: models/mejor_modelo_diabetes.pkl")
print("Gráfico guardado en: outputs/comparacion_modelos_mae.png")