## Script para realizar predicciones de churn en nuevos datos de clientes ##

import pandas as pd
from pathlib import Path

from preprocessing import preprocess_customer_df, load_decision_threshold

# ==========================================
# 📍 BLOQUE DE CONFIGURACIÓN DINÁMICA
# ==========================================
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent

MODELS_PATH = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "data"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "predictions"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# FIX: antes este script "predecía" sobre el MISMO CSV público usado para
# entrenar (incluyendo las filas que el modelo ya vio en train_test_split).
# Eso no prueba nada sobre generalización a clientes nuevos, solo repite
# inferencia sobre datos conocidos. Aquí usamos un holdout separado, guardado
# por train_model.py, que el modelo nunca vio durante el entrenamiento.
HOLDOUT_FILE = DATA_PATH / "holdout_new_customers.csv"
# ==========================================


def make_predictions():
    model_file = MODELS_PATH / "churn_model.pkl"
    cols_file = MODELS_PATH / "model_columns.pkl"

    if not model_file.exists():
        print(f"❌ Error: No se encontró el modelo en {model_file}")
        return

    if not HOLDOUT_FILE.exists():
        print(f"❌ Error: No se encontró el holdout en {HOLDOUT_FILE}.")
        print("   Ejecuta train_model.py (versión con generación de holdout) primero.")
        return

    import joblib
    print("📥 Cargando modelo...")
    model = joblib.load(model_file)
    model_cols = joblib.load(cols_file)
    threshold = load_decision_threshold(MODELS_PATH)

    print(f"🔍 Analizando clientes nuevos desde: {HOLDOUT_FILE}")
    df_new = pd.read_csv(HOLDOUT_FILE)
    ids = df_new['customerID'] if 'customerID' in df_new.columns else df_new.index

    # FIX: preprocesamiento centralizado (antes copiado y pegado aquí también)
    X_final = preprocess_customer_df(df_new, model_cols)

    probs = model.predict_proba(X_final)[:, 1]
    # FIX: se usa el umbral óptimo guardado, no una regla de suavizado manual
    # que no resolvía ningún problema estadístico real.
    preds = (probs >= threshold).astype(int)

    results = pd.DataFrame({
        'CustomerID': ids,
        'Probabilidad_Fuga': probs,
        'Prediccion_Churn': preds
    })

    output_file = OUTPUT_PATH / "predicciones_churn.csv"
    results.to_csv(output_file, index=False)

    print(f"✅ Predicciones completadas sobre {len(results)} clientes nuevos (holdout).")
    print(f"📄 Archivo generado: {output_file}")


if __name__ == "__main__":
    make_predictions()