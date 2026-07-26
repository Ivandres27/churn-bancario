import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from pathlib import Path
import pandas as pd

from preprocessing import preprocess_customer_df, load_decision_threshold

# ==========================================
# 📍 CONFIGURACIÓN Y CARGA DE RECURSOS
# ==========================================
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent
MODELS_PATH = PROJECT_ROOT / "models"

logger = logging.getLogger("churn_api")
logging.basicConfig(level=logging.INFO)

try:
    MODEL = joblib.load(MODELS_PATH / "churn_model.pkl")
    MODEL_COLS = joblib.load(MODELS_PATH / "model_columns.pkl")
except FileNotFoundError:
    raise FileNotFoundError("Error: Los archivos del modelo (.pkl) no se encontraron. Ejecuta train_model.py.")

# FIX: antes el umbral 0.5 estaba hardcodeado en el código. Ahora se carga
# el umbral óptimo calculado por train_model.py vía precision-recall curve.
DECISION_THRESHOLD = load_decision_threshold(MODELS_PATH)
logger.info(f"Umbral de decisión cargado: {DECISION_THRESHOLD:.3f}")

app = FastAPI(
    title="Churn Prediction API",
    description="API REST para predecir la probabilidad de fuga de clientes (Churn).",
    version="1.1.0"
)


# ==========================================
# 📐 ESQUEMA DE DATOS DE ENTRADA (Pydantic)
# ==========================================
class CustomerData(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int  # 0 o 1
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    # Nota: no incluimos 'Churn' aquí porque es lo que vamos a predecir


# ==========================================
# 🛠️ FUNCIÓN DE PREDICCIÓN
# ==========================================
def preprocess_and_predict(data: CustomerData):
    """Procesa un único registro de cliente y genera la predicción."""
    df = pd.DataFrame([data.dict()])

    # FIX: preprocesamiento ahora vive en un solo lugar (preprocessing.py),
    # compartido con app.py y predict.py, en vez de estar copiado y pegado.
    df_final = preprocess_customer_df(df, MODEL_COLS)

    proba = MODEL.predict_proba(df_final)[:, 1][0]  # Probabilidad de Churn (Clase 1)

    # FIX: se usa el umbral óptimo guardado, no un 0.5 fijo sin justificación.
    riesgo = "Alto" if proba >= DECISION_THRESHOLD else "Bajo o Medio"

    return {
        "customerID": data.customerID,
        "probabilidad_churn": round(float(proba), 4),
        "umbral_decision": round(float(DECISION_THRESHOLD), 4),
        "riesgo": riesgo
    }


# ==========================================
# 🌐 ENDPOINTS DE LA API
# ==========================================
@app.get("/")
def read_root():
    return {"message": "Churn Prediction API v1.1.0 - Listo para recibir datos."}


@app.post("/predict")
def predict_churn(customer: CustomerData):
    """
    Recibe los datos de un cliente y devuelve la probabilidad de Churn.
    """
    try:
        return preprocess_and_predict(customer)
    except Exception as e:
        # FIX: antes se devolvía str(e) directo al cliente (fuga de info interna
        # de infraestructura/código). Ahora se loguea internamente y se responde
        # con un mensaje genérico.
        logger.exception(f"Error prediciendo churn para customerID={customer.customerID}")
        raise HTTPException(
            status_code=500,
            detail="Ocurrió un error interno al procesar la predicción. Intenta de nuevo más tarde."
        )