import pandas as pd
import joblib

# Creamos una función para pedir un número decimal al usuario, que no acaba hasta que el usuario introduce un valor válido. En caso de que el usuario introduzca un valor no válido, se le muestra un mensaje de error y se le vuelve a pedir el número.

def pedir_numero(mensaje):
    while True:
        entrada = input(mensaje)

        try:
            numero = float(entrada.replace(",", "."))
            return numero
        except ValueError:
            print("Error: introduce un número válido.")

# Creamos una función para pedir un número entero al usuario, que no acaba hasta que el usuario introduce un valor válido. En caso de que el usuario introduzca un valor no válido, se le muestra un mensaje de error y se le vuelve a pedir el número.


def pedir_entero(mensaje):
    while True:
        entrada = input(mensaje)

        try:
            numero = int(float(entrada.replace(",", ".")))
            return numero
        except ValueError:
            print("Error: introduce un número entero válido.")

modelo = joblib.load("models/modelo_notas.pkl")

print("Predicción de nota de un estudiante")
print("-----------------------------------")

horas_estudio = pedir_numero("Horas de estudio: ")
asistencia = pedir_numero("Asistencia en porcentaje: ")
horas_sueno = pedir_numero("Horas de sueño: ")
trabajos_entregados = pedir_entero("Trabajos entregados: ")

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