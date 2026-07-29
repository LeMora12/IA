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
```

## Entreno del modelo ('train_model.ipynb')

1. Limpieza del CSV (`data/credit_score_raw.csv`): conversion de columnas numericas con texto/ruido, parseo de `Credit_History_Age`, e imputacion de valores faltantes con la **media** de cada columna (igual que se vio en clase).
2. Codificacion manual de la variable objetivo `Credit_Score` (Good/Standard/Poor) a 0/1/2 respetando el orden de riesgo (label encoding manual, como en clase).
3. Eliminacion de valores atipicos con el metodo del **Rango Intercuartilico (IQR)**: se descartan los registros fuera de `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`.
4. Seleccion de las **5 caracteristicas** con mayor correlacion (en valor absoluto) con `Credit_Score`: `Interest_Rate`, `Num_Credit_Inquiries`, `Outstanding_Debt`, `Delay_from_due_date`, `Num_Credit_Card`.
5. Escalado estandar (`StandardScaler`, ajustado solo con el set de entrenamiento) y entrenamiento de **KNN**, probando varios valores de `k` (igual que la busqueda del codo de K-Means vista en clase) y quedandose con el de mejor accuracy.
6. Evaluacion (accuracy, F1-score macro, reporte de clasificacion) y guardado del modelo en `model/`.

## Levantar API

Abrir **`api.ipynb`** y ejecutar todas las celdas. La ultima celda usa `await server.serve()` (en vez de `uvicorn.run(...)`) porque Jupyter ya corre su propio event loop; esta es la forma correcta y moderna de servir FastAPI desde un notebook. Al ejecutarla, la celda queda "corriendo" (icono `[*]`) mientras el servidor esta activo — eso es normal, no se congelo.

La API queda disponible en `http://127.0.0.1:8000`.

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
