import os # Importamos la librería os para trabajar con cosas del sistema operativo, como carpetas y rutas.
import pandas as pd
import joblib # Importamos joblib para guardar y cargar modelos de machine learning en un archivo.

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/estudiantes_avanzado.csv")

columnas_entrada = [
    "horas_estudio",
    "asistencia",
    "horas_sueno",
    "trabajos_entregados"
]

X = df[columnas_entrada]
y = df["nota"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

modelo = LinearRegression()

modelo.fit(X_train, y_train)

predicciones = modelo.predict(X_test)

error_medio = mean_absolute_error(y_test, predicciones)

os.makedirs("modelos", exist_ok=True) # Creamos la carpeta "modelos" si no existe, para guardar el modelo entrenado, si existe no da error.

joblib.dump(modelo, "modelos/modelo_notas.pkl") # Guardamos el modelo entrenado en un archivo llamado "modelo_notas.pkl" dentro de la carpeta "modelos".

print("Modelo entrenado correctamente.")
print("Error medio absoluto:", error_medio)
print("Modelo guardado en: modelos/modelo_notas.pkl")