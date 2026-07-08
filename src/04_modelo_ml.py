import pandas as pd
from sklearn.linear_model import LinearRegression # Importamos la clase LinearRegression del módulo linear_model de la biblioteca scikit-learn para crear un modelo de regresión lineal.

df = pd.read_csv("data/estudiantes_transformados.csv") # Leemos el archivo CSV que creaste antes y lo cargamos en un DataFrame llamado df utilizando el método read_csv() de pandas.

X = df[["horas_estudio"]] # Creamos una variable X que contiene las horas de estudio de los estudiantes, que será nuestra variable independiente (entrada) para el modelo de regresión lineal. La doble corchete [[]] se utiliza para seleccionar una columna y mantenerla como un DataFrame en lugar de una Serie.
y = df["nota"] # Creamos una variable y que contiene las notas de los estudiantes, que será nuestra variable dependiente (salida) para el modelo de regresión lineal.

modelo = LinearRegression() # Aquí creamos el modelo. Todavía no ha aprendido nada.

modelo.fit(X, y) # Entrenamos el modelo utilizando el método fit() de la clase LinearRegression, pasando las variables X e y como argumentos. El modelo aprenderá la relación entre las horas de estudio y las notas de los estudiantes.

horas_nuevas = pd.DataFrame({
    "horas_estudio": [4]
}) # Creamos un nuevo DataFrame llamado horas_nuevas que contiene una sola fila con el valor 4 en la columna "horas_estudio". Este DataFrame se utilizará para hacer una predicción de la nota estimada para un estudiante que estudia 4 horas.

prediccion = modelo.predict(horas_nuevas) # Utilizamos el método predict() del modelo entrenado para hacer una predicción de la nota estimada para el estudiante que estudia 4 horas. La predicción se almacena en la variable prediccion.

print("Datos usados para entrenar:")            # Muestra los datos originales.
print(df[["nombre", "horas_estudio", "nota"]])  # Muestra los datos originales.

print("\nPendiente del modelo:", modelo.coef_[0]) # Muestra la pendiente del modelo de regresión lineal, que indica cuánto cambia la nota estimada por cada hora adicional de estudio. La pendiente se obtiene del atributo coef_ del modelo, que es un array que contiene los coeficientes de las variables independientes. En este caso, solo hay una variable independiente (horas_estudio), por lo que accedemos al primer elemento del array con [0].
print("Intercepto del modelo:", modelo.intercept_) # Muestra el intercepto del modelo de regresión lineal, que indica la nota estimada cuando las horas de estudio son cero. El intercepto se obtiene del atributo intercept_ del modelo.

print("\nPredicción:")
print("Si una persona estudia 4 horas, la nota estimada es:", prediccion[0]) # Muestra la predicción de la nota estimada para un estudiante que estudia 4 horas. La predicción se obtiene del array prediccion, que contiene las notas estimadas para cada fila del DataFrame horas_nuevas. En este caso, solo hay una fila, por lo que accedemos al primer elemento del array con [0].