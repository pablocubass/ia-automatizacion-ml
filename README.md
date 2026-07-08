# IA y Automatización — Predicción de notas con Machine Learning

Este proyecto es una primera práctica de Inteligencia Artificial y automatización desarrollada en Python.

El objetivo principal es aprender el flujo básico de un proyecto de Machine Learning: cargar datos, analizarlos, transformarlos, entrenar un modelo, evaluarlo, guardarlo y usarlo posteriormente para hacer predicciones con nuevos datos.

## Objetivo del proyecto

El programa predice la nota estimada de un estudiante a partir de varias variables:

* horas de estudio;
* porcentaje de asistencia;
* horas de sueño;
* número de trabajos entregados.

Para ello se utiliza un modelo de regresión lineal entrenado con datos de ejemplo.

## Tecnologías utilizadas

* Python
* pandas
* NumPy
* matplotlib
* scikit-learn
* joblib

## Estructura del proyecto

```text
IA-Automatizacion/
├── data/
│   ├── estudiantes.csv
│   ├── estudiantes_transformados.csv
│   └── estudiantes_avanzado.csv
│
├── src/
│   ├── 00_primer_script.py
│   ├── 01_leer_csv.py
│   ├── 02_transformar_datos.py
│   ├── 03_grafico.py
│   ├── 04_modelo_ml.py
│   ├── 05_evaluar_modelo.py
│   ├── 06_modelo_varias_variables.py
│   ├── 07_grafico_predicciones.py
│   ├── 08_guardar_modelo.py
│   ├── 09_usar_modelo_guardado.py
│   ├── 10_prediccion_interactiva.py
│   ├── 11_prediccion_con_errores.py
│   └── 12_validar_rangos.py
│
├── models/
│   └── modelo_notas.pkl
│
├── outputs/
│   ├── grafico_horas_nota.png
│   └── grafico_real_vs_predicha.png
│
├── notebooks/
├── docs/
├── requirements.txt
├── .gitignore
└── README.md
```

## Flujo del proyecto

El proyecto sigue estos pasos:

1. Crear una tabla inicial de estudiantes.
2. Leer datos desde archivos CSV.
3. Transformar datos y crear nuevas columnas.
4. Visualizar relaciones entre variables.
5. Entrenar un primer modelo de Machine Learning.
6. Evaluar el modelo con datos de prueba.
7. Usar varias variables para mejorar la predicción.
8. Comparar nota real frente a nota predicha.
9. Guardar el modelo entrenado.
10. Cargar el modelo guardado para hacer nuevas predicciones.
11. Crear una predicción interactiva desde terminal.
12. Controlar errores de entrada.
13. Validar rangos para evitar datos absurdos.
14. Organizar el proyecto en carpetas profesionales.

## Cómo instalar el proyecto

Primero se recomienda crear un entorno virtual:

```bash
python -m venv .venv
```

En Windows, se activa con:

```bash
.venv\Scripts\activate
```

Después se instalan las dependencias:

```bash
pip install -r requirements.txt
```

## Cómo ejecutar el programa principal

El archivo más completo del proyecto es:

```text
src/12_validar_rangos.py
```

Para ejecutarlo desde la carpeta principal:

```bash
python src/12_validar_rangos.py
```

El programa pedirá los datos del estudiante:

```text
Horas de estudio:
Asistencia en porcentaje:
Horas de sueño:
Trabajos entregados:
```

Después mostrará una nota estimada usando el modelo entrenado.

## Ejemplo de uso

Entrada:

```text
Horas de estudio: 4
Asistencia en porcentaje: 90
Horas de sueño: 7
Trabajos entregados: 4
```

Salida aproximada:

```text
Nota estimada:
7.8
```

La predicción puede variar ligeramente según el modelo entrenado y los datos utilizados.

## Qué he aprendido con este proyecto

Con este proyecto he practicado conceptos básicos de Inteligencia Artificial y desarrollo en Python:

* uso de pandas para manejar datos;
* lectura y escritura de archivos CSV;
* creación de nuevas columnas;
* visualización de datos con matplotlib;
* entrenamiento de modelos con scikit-learn;
* separación entre datos de entrenamiento y prueba;
* evaluación de modelos con métricas;
* guardado y carga de modelos con joblib;
* creación de programas interactivos por terminal;
* control de errores con try/except;
* validación de rangos;
* organización profesional de carpetas;
* preparación básica para subir un proyecto a GitHub.

## Estado del proyecto

Este proyecto es una primera práctica de Machine Learning. No está pensado para hacer predicciones reales sobre estudiantes, ya que el dataset es pequeño y de ejemplo.

Su finalidad es educativa: aprender la estructura y el flujo básico de un proyecto de IA.

## Próximos pasos

Algunas mejoras posibles serían:

* usar un dataset real más grande;
* añadir más variables;
* probar otros modelos de Machine Learning;
* crear una API con FastAPI;
* crear una interfaz web sencilla;
* subir el proyecto a GitHub;
* desplegar una demo online.
