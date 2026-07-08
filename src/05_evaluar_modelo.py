import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split # Importamos la función train_test_split del módulo model_selection de la biblioteca scikit-learn para dividir nuestro conjunto de datos en conjuntos de entrenamiento y prueba.
from sklearn.metrics import mean_absolute_error, r2_score # Importamos las funciones mean_absolute_error y r2_score del módulo metrics de la biblioteca scikit-learn para evaluar el rendimiento del modelo de regresión lineal.

df = pd.read_csv("data/estudiantes_transformados.csv")

X = df[["horas_estudio"]]
y = df["nota"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
) # Utilizamos la función train_test_split() para dividir nuestro conjunto de datos en conjuntos de entrenamiento y prueba. La variable test_size=0.3 indica que el 30% de los datos se utilizarán para la prueba, mientras que el 70% restante se utilizará para el entrenamiento. La variable random_state=42 se utiliza para garantizar que la división sea reproducible, es decir, que obtengamos los mismos conjuntos de entrenamiento y prueba cada vez que ejecutemos el código.

modelo = LinearRegression() # Creamos un modelo vacío.

modelo.fit(X_train, y_train) # Aquí entrenamos el modelo.

predicciones = modelo.predict(X_test) # Hacemos predicciones utilizando el conjunto de prueba X_test y almacenamos los resultados en la variable predicciones.

resultados = pd.DataFrame({
    "horas_estudio": X_test["horas_estudio"],
    "nota_real": y_test,
    "nota_predicha": predicciones
}) # Creamos una tabla de resultados.

error_medio = mean_absolute_error(y_test, predicciones) # Calculamos el error medio absoluto utilizando la función mean_absolute_error() de scikit-learn, pasando las notas reales (y_test) y las notas predichas (predicciones) como argumentos. El error medio absoluto nos indica, en promedio, cuánto se desvían las predicciones del modelo de los valores reales. Un valor más bajo indica un mejor rendimiento del modelo. 
r2 = r2_score(y_test, predicciones) # Calculamos el R2 score utilizando la función r2_score() de scikit-learn, pasando las notas reales (y_test) y las notas predichas (predicciones) como argumentos. El R2 score nos indica qué tan bien se ajusta el modelo a los datos. Un valor de R2 cercano a 1 indica un buen ajuste, mientras que un valor cercano a 0 indica un mal ajuste.

print("Datos de prueba y predicciones:")
print(resultados)

print("\nError medio absoluto:", error_medio)
print("R2 score:", r2)