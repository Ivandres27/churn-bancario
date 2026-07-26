# 📉 Customer Churn Prediction System

Solución de Machine Learning de punta a punta para predecir la fuga (churn) de clientes de telecomunicaciones. Incluye entrenamiento del modelo, dashboard interactivo, API REST y generación automática de reportes ejecutivos en PDF.

Dataset: [Telco Customer Churn (IBM)](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv) — 7,043 clientes, 26.5% de tasa de churn.

## 🚀 Características

* **Modelo predictivo:** Random Forest con umbral de decisión optimizado (no el 0.5 por defecto).
* **Dashboard interactivo:** Streamlit, con carga masiva de clientes vía CSV y visualización de factores de riesgo.
* **API REST:** FastAPI, con endpoint de predicción individual, lista para integrarse a otros sistemas.
* **Reportes ejecutivos:** generación automática de PDF con gráficos y recomendaciones de negocio por segmento de riesgo.
* **Holdout real:** `predict.py` corre sobre clientes que el modelo nunca vio durante el entrenamiento, no sobre datos reciclados.

## 📊 Resultados del modelo

| Métrica | Umbral 0.5 (default) | Umbral optimizado (~0.27–0.35) |
|---|---|---|
| Accuracy | ~79% | ~74–77% |
| F1-score (clase Churn) | ~0.54 | **~0.59–0.61** |
| Recall (clase Churn) | ~47% | **~67–71%** |
| AUC-ROC | 0.82 | 0.82 (no cambia con el umbral) |

**Por qué se movió el umbral:** con un dataset donde solo ~26% de los clientes hace churn, el umbral 0.5 por defecto de scikit-learn favorece la clase mayoritaria y deja escapar más de la mitad de los clientes que sí se van. Se buscó el umbral que maximiza F1 sobre la clase Churn usando la curva precision-recall, priorizando **recall** sobre precisión: en un problema de retención, el costo de negocio de no detectar a un cliente que se va es mayor que el costo de ofrecer una promoción de retención a alguien que en realidad se iba a quedar.

`class_weight='balanced'` se probó primero y **no mejoró el F1 de forma significativa** — el ajuste de umbral fue la palanca real. (Detalle documentado en los comentarios de `train_model.py`.)

## 📂 Estructura del Proyecto

```text
churn_project/
├── data/
│   ├── Telco-Customer-Churn.csv       # dataset completo (se descarga automático si no existe)
│   └── holdout_new_customers.csv      # 5% separado antes de entrenar, nunca visto por el modelo
├── models/                            # generados por train_model.py (no se versionan en git)
│   ├── churn_model.pkl
│   ├── model_columns.pkl
│   ├── feature_importance.pkl
│   └── decision_threshold.pkl
├── notebooks/                         # análisis exploratorio (Jupyter)
├── outputs/predictions/               # predicciones generadas por predict.py
├── src/
│   ├── train_model.py                 # entrenamiento + evaluación + umbral óptimo
│   ├── preprocessing.py               # limpieza y encoding, compartido por api.py/app.py/predict.py
│   ├── predict.py                     # inferencia batch sobre el holdout
│   ├── api.py                         # API REST (FastAPI)
│   ├── app.py                         # Dashboard (Streamlit)
│   ├── report_generator.py            # motor de reportes PDF (ReportLab)
│   └── fonts/                         # DejaVuSans*.ttf requeridas por ReportLab
└── requirements.txt
```

## ⚙️ Instalación y uso

```bash
# 1. Crear y activar entorno virtual
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Entrenar el modelo (descarga el dataset si no existe localmente)
python src/train_model.py

# 4. Generar predicciones sobre el holdout de "clientes nuevos"
python src/predict.py

# 5. Levantar el dashboard
streamlit run src/app.py

# 6. Levantar la API
uvicorn src.api:app --reload
```

## 🔍 Próximos pasos (documentados, no implementados aún)

* Tests automatizados (`pytest`) para `preprocessing.py` y el endpoint de la API.
* Calibración de probabilidades (`CalibratedClassifierCV`) en vez de solo ajuste de umbral.
* Comparación contra otros algoritmos (XGBoost, Logistic Regression) con el mismo pipeline de evaluación.
* Modo "Predicción Manual" en el dashboard (actualmente placeholder).