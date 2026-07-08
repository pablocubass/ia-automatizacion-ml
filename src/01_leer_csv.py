import pandas as pd

df = pd.read_csv("data/estudiantes.csv") # Cargamos los datos desde un archivo CSV llamado "estudiantes.csv" y los almacenamos en un DataFrame llamado df.

print("Primeras filas de la tabla:")
print(df.head()) # Muestra las primeras 5 filas y la información general del DataFrame.

print("\nInformación general:")
print(df.info()) # Muestra información general sobre el DataFrame, incluyendo el número de entradas, columnas, tipos de datos y valores nulos.

print("\nEstadísticas básicas:")
print(df.describe()) # Muestra estadísticas básicas de las columnas numéricas del DataFrame, como la media, desviación estándar, valores mínimo y máximo, y percentiles.

media_nota = df["nota"].mean() # Calculamos la media de las notas de los estudiantes utilizando el método mean() (que sirve para calcular la media aritmética) de pandas y la almacenamos en la variable media_nota.
nota_maxima = df["nota"].max() # Calculamos la nota máxima de los estudiantes utilizando el método max() de pandas y la almacenamos en la variable nota_maxima.
nota_minima = df["nota"].min() # Calculamos la nota mínima de los estudiantes utilizando el método min() de pandas y la almacenamos en la variable nota_minima.

print("\nMedia de notas:", media_nota)
print("Nota más alta:", nota_maxima)
print("Nota más baja:", nota_minima)

aprobados = df[df["nota"] >= 5] # Filtramos para obtener solo los estudiantes que han aprobado y almacenamos el resultado en la variable aprobados.

print("\nEstudiantes aprobados:")
print(aprobados)