import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("data/estudiantes_avanzado.csv")

columnas_entrada = [
    "horas_estudio",
    "asistencia",
    "horas_sueno",
    "trabajos_entregados"
] # Creamos una lista con los nombres de las columnas que usará el modelo para aprender.

X = df[columnas_entrada] # Creamos X que son los datos que el modelo usa para predecir.
y = df["nota"] # Creamos y que es la variable que queremos predecir.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
) # Utilizamos la función train_test_split() para dividir nuestro conjunto de datos en conjuntos de entrenamiento y prueba. La variable test_size=0.25 indica que el 25% de los datos se utilizarán para la prueba, mientras que el 75% restante se utilizará para el entrenamiento. La variable random_state=42 se utiliza para garantizar que la división sea reproducible, es decir, que obtengamos los mismos conjuntos de entrenamiento y prueba cada vez que ejecutemos el código.

modelo = LinearRegression() # Creamos un modelo vacío.

modelo.fit(X_train, y_train) # Aquí entrenamos el modelo.

predicciones = modelo.predict(X_test) # Hacemos predicciones utilizando el conjunto de prueba X_test y almacenamos los resultados en la variable predicciones.

resultados = X_test.copy() # Creamos una copia del conjunto de prueba X_test para almacenar los resultados.
resultados["nota_real"] = y_test # Agregamos una nueva columna "nota_real" al DataFrame resultados que contiene las notas reales del conjunto de prueba y_test.
resultados["nota_predicha"] = predicciones # Agregamos una nueva columna "nota_predicha" al DataFrame resultados que contiene las notas predichas por el modelo.

error_medio = mean_absolute_error(y_test, predicciones) # Calculamos el error medio absoluto utilizando la función mean_absolute_error() de scikit-learn, pasando las notas reales (y_test) y las notas predichas (predicciones) como argumentos. El error medio absoluto nos indica, en promedio, cuánto se desvían las predicciones del modelo de los valores reales. Un valor más bajo indica un mejor rendimiento del modelo.
r2 = r2_score(y_test, predicciones) # Calculamos el R2 score utilizando la función r2_score() de scikit-learn, pasando las notas reales (y_test) y las notas predichas (predicciones) como argumentos. El R2 score nos indica qué tan bien se ajusta el modelo a los datos. Un valor de R2 cercano a 1 indica un buen ajuste, mientras que un valor cercano a 0 indica un mal ajuste.

print("Datos de prueba y predicciones:")
print(resultados) # Imprimimos la tabla de resultados que contiene las horas de estudio, asistencia, horas de sueño, trabajos entregados, notas reales y notas predichas para cada estudiante en el conjunto de prueba.

print("\nError medio absoluto:", error_medio)
print("R2 score:", r2)

print("\nImportancia aproximada de cada variable:")
for columna, coeficiente in zip(columnas_entrada, modelo.coef_): 
    print(columna, ":", coeficiente)
    # "modelo.coef_" contiene los coeficinetes (indican la importancia) que ha aprendido el modelo para cada variable de entrada.
    # "zip(columnas_entrada, modelo.coef_)" Une cada columna con su coeficiente.
    # El for recorre cada pareja de columna y coeficiente.

nuevo_estudiante = pd.DataFrame({
    "horas_estudio": [4],
    "asistencia": [90],
    "horas_sueno": [7],
    "trabajos_entregados": [4]
}) # Crea un nuevo DataFrame con los datos de un estudiante para el cual queremos hacer una predicción de su nota.

prediccion_nueva = modelo.predict(nuevo_estudiante) # Hacemos una predicción utilizando el modelo entrenado y los datos del nuevo estudiante, y almacenamos el resultado en la variable "prediccion_nueva".

print("\nPredicción para un nuevo estudiante:")
print("Nota estimada:", prediccion_nueva[0])
