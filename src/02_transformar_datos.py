import pandas as pd

df = pd.read_csv("data/estudiantes.csv")

df["aprobado"] = df["nota"] >= 5 # Creamos una nueva columna llamada "aprobado".

df["nota_redondeada"] = df["nota"].round(1) # Creamos una nueva columna llamada "nota_redondeada" que contiene las notas redondeadas a un decimal utilizando el método round() de pandas.

def clasificar_rendimiento(nota): # Creamos una función llamada "clasificar_rendimiento" utilizando def.
    if nota < 5:
        return "bajo"
    elif nota < 8:
        return "medio"
    else:
        return "alto"

df["rendimiento"] = df["nota"].apply(clasificar_rendimiento) # Implementamos la función "clasificar_rendimiento" en una nueva columna llamada "rendimiento" utilizando el método apply() de pandas, que aplica la función a cada elemento de la columna "nota".

print("Tabla con nuevas columnas:")
print(df)

df.to_csv("estudiantes_transformados.csv", index=False) # Guardamos el DataFrame modificado en un nuevo archivo CSV llamado "estudiantes_transformados.csv" utilizando el método to_csv() de pandas, y establecemos index=False para no incluir los índices en el archivo.

print("\nArchivo estudiantes_transformados.csv creado correctamente.")