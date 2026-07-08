import pandas as pd
import matplotlib # Importamos la biblioteca matplotlib, que es una biblioteca de visualización de datos en Python. Esta biblioteca nos permite crear gráficos y visualizaciones de manera sencilla y flexible.
matplotlib.use("Agg") # Esta línea de código se utiliza para configurar el backend de Matplotlib en un entorno sin interfaz gráfica, como un servidor o un script que se ejecuta en segundo plano. Al establecer el backend en "Agg", Matplotlib puede generar gráficos y guardarlos como archivos de imagen (por ejemplo, PNG) sin necesidad de mostrar una ventana gráfica. Esto es útil cuando se desea crear gráficos automáticamente y guardarlos en disco sin interacción del usuario.
import matplotlib.pyplot as plt # Importamos la biblioteca matplotlib.pyplot y le damos el alias plt para poder utilizar sus funciones de manera más sencilla. Matplotlib es una biblioteca de visualización de datos en Python que permite crear gráficos y visualizaciones de manera fácil y flexible.
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

plt.scatter(y_test, predicciones) # Creamos un gráfico de dispersión utilizando el método scatter() de matplotlib.pyplot, donde el eje x representa las notas reales y el eje y representa las notas predichas por el modelo. Esto nos permite visualizar cómo se comparan las predicciones del modelo con los valores reales.

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()]) # Agregamos una línea diagonal al gráfico utilizando el método plot() de matplotlib.pyplot. Esta línea representa la relación ideal entre las notas reales y las notas predichas, donde todos los puntos se encontrarían si el modelo fuera perfecto. La línea se dibuja desde el valor mínimo hasta el valor máximo de las notas reales (y_test.min() y y_test.max()) en ambos ejes, creando una referencia visual para evaluar la precisión del modelo.

plt.title("Nota real vs nota predicha")
plt.xlabel("Nota real")
plt.ylabel("Nota predicha")

plt.savefig("grafico_real_vs_predicha.png")

plt.close()

print("Error medio absoluto:", error_medio)
print("Gráfico guardado como grafico_real_vs_predicha.png")