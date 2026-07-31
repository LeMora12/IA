# Proyecto final Inteligencia Artifical

### Objetivo del proyecto:

Sistema compuesto por un modelo predictivo (KNN), una API REST (FastAPI) y un agente inteligente (Gradio + LangChain + Qwen) que evalua solicitudes de credito y toma una decision automatizada.

## Arquitectura

```
IA/
├── data/
│   └── credit_score_raw.csv
├── model/                       # se genera al ejecutar train_model.ipynb
│   ├── knn_model.pkl
│   ├── scaler.pkl
│   └── features.json
├── train_model.ipynb
├── api.ipynb
├── agent_app.ipynb
├── requirements.txt
└── README.md
```

Flujo: `train_model.ipynb` genera el modelo -> `api.ipynb` lo expone -> `agent_app.ipynb` consume la API, aplica las reglas de negocio, pide a Qwen una justificacion y guarda el resultado en SQLite.

## Requisitos

- Python 3.11.4
- Jupyter Notebbok
- Ollama
- Descargar el modelo Qwen (gratuito, corre localmente con Ollama):

```bash
ollama pull qwen2.5:3b
python -m pip install notebook # Jupyter Notebook, para abrir y correr los .ipynb
python -m pip install ipykernel # kernel de Python que usa Jupyter para ejecutar las celdas
python -m pip install pandas # limpieza y manejo del dataset (DataFrames)
python -m pip install numpy # operaciones numericas usadas por pandas y scikit-learn
python -m pip install scikit-learn # StandardScaler, KNeighborsClassifier, metricas, train_test_split
python -m pip install joblib # guardar y cargar el modelo y el scaler entrenados (.pkl)
python -m pip install fastapi # framework de la API (POST /predict)
python -m pip install uvicorn # servidor que ejecuta la API de FastAPI
python -m pip install sqlalchemy # ORM para la base de datos SQLite (registro de evaluaciones)
python -m pip install gradio # interfaz de chat del agente
python -m pip install requests # el agente llama a la API por HTTP
python -m pip install langchain-ollama # conecta LangChain con Qwen corriendo en Ollama
```

## Levantar Jupyter

```bash
jupyter notebook
```

## Entreno del modelo ('train_model.ipynb')

1. Limpieza del CSV (`data/credit_score_raw.csv`): conversion de columnas numericas con texto/ruido, parseo de `Credit_History_Age`, e imputacion de valores faltantes con la **mediana** de cada columna (igual que se vio en clase).
2. Codificacion manual de la variable objetivo `Credit_Score` (Good/Standard/Poor) a 0/1/2 respetando el orden de riesgo (label encoding manual, como en clase).
3. Eliminacion de valores atipicos con el metodo del **Rango Intercuartilico (IQR)**: se descartan los registros fuera de `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`.
4. Seleccion de las **5 caracteristicas** con mayor correlacion (en valor absoluto) con `Credit_Score`: `Interest_Rate`, `Num_Credit_Inquiries`, `Outstanding_Debt`, `Delay_from_due_date`, `Num_Credit_Card`.
5. Escalado estandar (`StandardScaler`, ajustado solo con el set de entrenamiento) y entrenamiento de **KNN**, probando varios valores de `k` (igual que la busqueda del codo de K-Means vista en clase) y quedandose con el de mejor accuracy.
6. Evaluacion (accuracy, F1-score macro, reporte de clasificacion) y guardado del modelo en `model/`.

## Levantar API

Abrir **`api.ipynb`** y ejecutar todas las celdas. La ultima celda usa `await server.serve()` (en vez de `uvicorn.run(...)`) porque Jupyter ya corre su propio event loop; esta es la forma correcta y moderna de servir FastAPI desde un notebook. Al ejecutarla, la celda queda "corriendo" (icono `[*]`).

La API queda disponible en `http://127.0.0.1:8000`.

Respuesta del endpoint (solo categoria, sin score numerico):

```json
{ "risk_category": 2, "risk_level": "alto" }
```

## Levantar el agente (chat con Gradio)

Con `api.ipynb` corriendo, abrir `agent_app.ipynb` y ejecutar todas las celdas. La ultima celda (`demo.launch()`) abre un **chat**, tanto inline en el notebook como en `http://127.0.0.1:7860`.

Ahi el solicitante escribe en lenguaje natural, por ejemplo:

> "Tengo una tasa de interes de 15, he hecho 3 consultas de credito, debo 800 de deuda pendiente, llevo 10 dias de atraso en mis pagos y tengo 4 tarjetas de credito."

Tambien puede escribirlo por partes en varios mensajes; el chat recuerda lo ya dicho. El agente:

1. Usa **Qwen** para leer toda la conversacion y extraer las 5 variables necesarias (tasa de interes, consultas de credito, deuda pendiente, dias de atraso, numero de tarjetas de credito).
2. Si falta algun dato, se lo vuelve a pedir especificamente al solicitante.
3. Cuando ya tiene las 5, envia los datos a `POST /predict`.
4. Aplica la regla de negocio segun la categoria devuelta (alto/medio/bajo).
5. Le pide a Qwen una justificacion breve basada unicamente en la categoria y la regla aplicada.
6. Guarda el registro (las 5 variables extraidas, la categoria y la decision) en `credit_risk.db` (SQLite), creado en la carpeta del proyecto.

## Reglas de decision del agente

| Categoria | Riesgo | Decision                                                |
| --------- | ------ | ------------------------------------------------------- |
| 0         | Bajo   | Aprobar solicitud con condiciones estandar.             |
| 1         | Medio  | Solicitar documentacion adicional y evaluar nuevamente. |
| 2         | Alto   | Rechazar solicitud y recomendar educacion financiera.   |

## Manual de usuario

### Como usar el chat

Escribe tu situacion crediticia en una o varias oraciones, en espanol natural. No hace falta usar los nombres tecnicos de las variables ni un formato especial — el chat entiende frases como "tengo 3 tarjetas" o "no me he atrasado en mis pagos". Si falta algun dato, el chat te lo va a pedir especificamente antes de darte una respuesta.

### Ejemplos de conversacion

**Riesgo bajo:**

> "Mi tasa de interes es del 3%, he tenido 1 consulta de credito, debo 100 dolares, nunca me he atrasado en mis pagos y tengo 1 tarjeta de credito."

**Riesgo medio:**

> "Tengo una tasa de interes del 15%, he hecho 3 consultas de credito, debo 800 dolares, me he atrasado 10 dias en mis pagos y tengo 4 tarjetas de credito."

**Riesgo alto:**

> "Mi tasa de interes es del 28%, he hecho 10 consultas de credito, debo 2500 dolares, me he atrasado 25 dias en mis pagos y tengo 7 tarjetas de credito."

En cualquiera de los 3 casos, el chat responde con la categoria, la decision tomada y una justificacion breve generada por Qwen.

### Consultar el historial de evaluaciones guardadas

Todas las evaluaciones completas quedan guardadas en `credit_risk.db` (carpeta del proyecto). Para verlas, en una celda nueva de `agent_app.ipynb`:

```python
import pandas as pd

session = SessionLocal()
registros = session.query(Evaluation).all()
session.close()

pd.DataFrame([{
    "id": r.id, "interest_rate": r.interest_rate,
    "num_credit_inquiries": r.num_credit_inquiries,
    "outstanding_debt": r.outstanding_debt,
    "delay_from_due_date": r.delay_from_due_date,
    "num_credit_card": r.num_credit_card,
    "categoria": r.predicted_category, "decision": r.decision,
    "fecha": r.created_at,
} for r in registros])
```

Tambien se puede abrir `credit_risk.db` con un programa como [DB Browser for SQLite](https://sqlitebrowser.org/) (gratuito) para verlo con una interfaz grafica, sin necesidad de Jupyter.


## Notas de diseño

- El modelo solo devuelve la categoria de riesgo (0/1/2), sin score ni probabilidades, por requerimiento del proyecto.
- El registro en base de datos guarda unicamente las 5 variables de entrada, la categoria predicha y la decision tomada (no se guarda ninguna justificacion ni campo de "porque").
- `api.ipynb` y `agent_app.ipynb` deben ejecutarse en **kernels/pestañas separadas** de Jupyter, ya que ambos quedan "corriendo" de forma indefinida (uno sirviendo la API, el otro sirviendo la interfaz Gradio).
