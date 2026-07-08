import pandas as pd
import joblib

modelo = joblib.load("models/modelo_notas.pkl") # Utilizamos joblib.load para cargar el modelo previamente guardado en el archivo "modelo_notas.pkl" dentro de la carpeta "modelos".

nuevo_estudiante = pd.DataFrame({
    "horas_estudio": [4],
    "asistencia": [90],
    "horas_sueno": [7],
    "trabajos_entregados": [4]
})

prediccion = modelo.predict(nuevo_estudiante)

print("Datos del nuevo estudiante:")
print(nuevo_estudiante)

print("\nNota estimada:")
print(prediccion[0]) # Imprimimos la predicción de la nota estimada para el nuevo estudiante haciendo uso de prediccion[0], donde el índice 0 se utiliza para acceder al primer (y único) valor de la predicción, ya que el método predict devuelve un array de predicciones, incluso si solo hay una entrada.