import pandas as pd #implementamos la librería pandas para trabajar con dataframes y abreviamos su nombre como pd

datos = {
    "nombre": ["Ana", "Luis", "Marta", "Pablo"],
    "horas_estudio": [2, 5, 3, 6],
    "nota": [6.5, 8.0, 7.0, 9.0]
} # Creamos nuestra estructura de datos con un diccionario que contiene los nombres, horas de estudio y notas de los estudiantes.

df = pd.DataFrame(datos) # Creamos un DataFrame a partir del diccionario de datos, lo que nos permite trabajar con los datos de manera tabular.

print("Tabla de estudiantes:")
print(df)

media_nota = df["nota"].mean() # Calculamos la media de las notas de los estudiantes utilizando el método mean() (que sirve para calcular la media aritmética) de pandas y la almacenamos en la variable media_nota.
media_horas = df["horas_estudio"].mean()

print("\nMedia de nota:", media_nota)
print("Media de horas de estudio:", media_horas)

mejor_estudiante = df[df["nota"] == df["nota"].max()] # Buscamos la nota más alta utilizando el método max(), luego filtramos el DataFrame para obtener las filas correspondientes a esa nota.

print("\nMejor estudiante:")
print(mejor_estudiante)