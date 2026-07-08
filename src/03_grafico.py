import pandas as pd
import matplotlib.pyplot as plt # Importamos la librería matplotlib.pyplot para crear gráficos

df = pd.read_csv("data/estudiantes_transformados.csv")

print("Datos cargados:")
print(df)

plt.scatter(df["horas_estudio"], df["nota"]) # Creamos un gráfico de dispersión utilizando el método scatter() de matplotlib.pyplot, donde el eje x representa las horas de estudio y el eje y representa las notas.

plt.title("Relación entre horas de estudio y nota") # Agregamos un título al gráfico utilizando el método title() de matplotlib.pyplot.
plt.xlabel("Horas de estudio") # Agregamos una etiqueta al eje x utilizando el método xlabel() de matplotlib.pyplot.
plt.ylabel("Nota") # Agregamos una etiqueta al eje y utilizando el método ylabel() de matplotlib.pyplot.

plt.savefig("grafico_horas_nota.png") # Guardamos el gráfico en un archivo PNG llamado "grafico_horas_nota.png" utilizando el método savefig() de matplotlib.pyplot.

plt.show() # Mostramos el gráfico en pantalla utilizando el método show() de matplotlib.pyplot.