import pandas as pd
import joblib

modelo = joblib.load("models/modelo_notas.pkl")

print("Predicción de nota de un estudiante")
print("-----------------------------------")

# Utilizamos la estrutcutura float(input("Horas de estudio: ").replace(",", ".")) de forma en la que: 
# Input es el que muestra esta pregunta en la terminal y espera a que responda.
# float() es para convertir el valor introducido de un texto a un número decimal.
# replace(",", ".") es para reemplazar la coma por un punto, ya que en algunos países se utiliza la coma como separador decimal y en otros el punto.

horas_estudio = float(input("Horas de estudio: ").replace(",", "."))
asistencia = float(input("Asistencia en porcentaje: ").replace(",", "."))
horas_sueno = float(input("Horas de sueño: ").replace(",", "."))
trabajos_entregados = int(float(input("Trabajos entregados: ").replace(",", ".")))

nuevo_estudiante = pd.DataFrame({
    "horas_estudio": [horas_estudio],
    "asistencia": [asistencia],
    "horas_sueno": [horas_sueno],
    "trabajos_entregados": [trabajos_entregados]
})

prediccion = modelo.predict(nuevo_estudiante)

print("\nDatos introducidos:")
print(nuevo_estudiante)

print("\nNota estimada:")
print(round(prediccion[0], 2)) # Usamos round(prediccion[0], 2) para redondear la predicción de la nota estimada a dos decimales, haciendo uso de prediccion[0], donde el índice 0 se utiliza para acceder al primer (y único) valor de la predicción, ya que el método predict devuelve un array de predicciones, incluso si solo hay una entrada.