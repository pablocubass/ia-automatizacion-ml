import pandas as pd
import joblib

def pedir_numero_en_rango(mensaje, minimo, maximo):
    while True:
        entrada = input(mensaje)

        try:
            numero = float(entrada.replace(",", "."))

            if numero < minimo or numero > maximo:
                print(f"Error: el valor debe estar entre {minimo} y {maximo}.")
            else:
                return numero

        except ValueError:
            print("Error: introduce un número válido.")

def pedir_entero_en_rango(mensaje, minimo, maximo):
    while True:
        entrada = input(mensaje)

        try:
            numero = int(float(entrada.replace(",", ".")))

            if numero < minimo or numero > maximo:
                print(f"Error: el valor debe estar entre {minimo} y {maximo}.")
            else:
                return numero

        except ValueError:
            print("Error: introduce un número entero válido.")

modelo = joblib.load("models/modelo_notas.pkl")

print("Predicción de nota de un estudiante")
print("-----------------------------------")

horas_estudio = pedir_numero_en_rango("Horas de estudio: ", 0, 24)
asistencia = pedir_numero_en_rango("Asistencia en porcentaje: ", 0, 100)
horas_sueno = pedir_numero_en_rango("Horas de sueño: ", 0, 24)
trabajos_entregados = pedir_entero_en_rango("Trabajos entregados: ", 0, 10)

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
print(round(prediccion[0], 2))