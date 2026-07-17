import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("data/diabetes_dataset.csv")

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

modelos = {
    "Regresion Lineal": LinearRegression(),
    "Arbol de Decision": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
}

resultados = []

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)

    predicciones_train = modelo.predict(X_train)
    predicciones_test = modelo.predict(X_test)

    mae_train = mean_absolute_error(y_train, predicciones_train)
    mae_test = mean_absolute_error(y_test, predicciones_test)

    r2_train = r2_score(y_train, predicciones_train)
    r2_test = r2_score(y_test, predicciones_test)

    diferencia_mae = mae_test - mae_train

    resultados.append({
        "modelo": nombre,
        "mae_train": mae_train,
        "mae_test": mae_test,
        "diferencia_mae": diferencia_mae,
        "r2_train": r2_train,
        "r2_test": r2_test
    })

df_resultados = pd.DataFrame(resultados)

print("Análisis de overfitting:")
print(df_resultados)

df_resultados.to_csv("outputs/analisis_overfitting_modelos.csv", index=False)

plt.bar(df_resultados["modelo"], df_resultados["diferencia_mae"])

plt.title("Diferencia entre error de test y entrenamiento")
plt.xlabel("Modelo")
plt.ylabel("MAE test - MAE train")
plt.xticks(rotation=20)

plt.tight_layout()
plt.savefig("outputs/analisis_overfitting_mae.png")

plt.show()

print("\nResultados guardados en: outputs/analisis_overfitting_modelos.csv")
print("Gráfico guardado en: outputs/analisis_overfitting_mae.png")