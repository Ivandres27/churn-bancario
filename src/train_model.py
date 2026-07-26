## Script de Entrenamiento de Modelo de Churn ##

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report,
    precision_recall_curve
)
import joblib
from pathlib import Path

# ==========================================
# 📍 BLOQUE DE CONFIGURACIÓN DINÁMICA
# ==========================================
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent          # churn_project/
DATA_PATH = PROJECT_ROOT / "data"
MODELS_PATH = PROJECT_ROOT / "models"
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
LOCAL_DATA_FILE = DATA_PATH / "Telco-Customer-Churn.csv"

MODELS_PATH.mkdir(parents=True, exist_ok=True)
DATA_PATH.mkdir(parents=True, exist_ok=True)

print(f"📍 Directorio raíz detectado: {PROJECT_ROOT}")
# ==========================================


def load_data():
    """
    Carga el dataset. Si existe una copia local en data/, la usa (más rápido,
    reproducible sin depender de internet). Si no existe, la descarga de la
    fuente pública (DATA_URL) y la guarda localmente para la próxima corrida.
    """
    if LOCAL_DATA_FILE.exists():
        print(f"📥 Cargando datos locales: {LOCAL_DATA_FILE}")
        df = pd.read_csv(LOCAL_DATA_FILE)
    else:
        print(f"🌐 Descargando datos desde: {DATA_URL}")
        df = pd.read_csv(DATA_URL)
        df.to_csv(LOCAL_DATA_FILE, index=False)
        print(f"💾 Copia local guardada en: {LOCAL_DATA_FILE}")

    # --- Limpieza básica ---
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # FIX: df['col'].fillna(x, inplace=True) NO actualiza df de forma confiable
    # en pandas >= 2.0 (Copy-on-Write). El patrón correcto es reasignar la columna.
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # NOTA: 'customerID' se conserva aquí a propósito (no se elimina en
    # load_data). Se necesita íntegro para poder generar el holdout de
    # "clientes nuevos" con su ID. Se elimina más adelante, justo antes
    # de pd.get_dummies(), solo para el set de entrenamiento/evaluación.
    return df


def train():
    df = load_data()

    # ----------------------------------------------------
    # HOLDOUT DE "CLIENTES NUEVOS" (para predict.py)
    # ----------------------------------------------------
    # FIX: separamos un 5% de los datos ANTES de entrenar, y nunca se usan
    # para fit() ni para el train_test_split de evaluación. Esto simula
    # "clientes nuevos" reales que predict.py puede usar para probar
    # generalización genuina, no solo re-inferencia sobre datos ya vistos.
    df_full = df.copy()
    holdout_df = df_full.sample(frac=0.05, random_state=7)
    df = df_full.drop(holdout_df.index)

    holdout_file = DATA_PATH / "holdout_new_customers.csv"
    holdout_df.to_csv(holdout_file, index=False)
    print(f"🗂️ Holdout de {len(holdout_df)} 'clientes nuevos' guardado en: {holdout_file}")

    print("⚙️ Preprocesando...")
    df = df.drop('customerID', axis=1, errors='ignore')
    df_processed = pd.get_dummies(df)

    X = df_processed.drop('Churn', axis=1)
    y = df_processed['Churn']

    tasa_churn = y.mean()
    print(f"📊 Tasa de churn en el dataset completo: {tasa_churn:.2%}")

    # FIX: stratify=y asegura que train y test conserven la misma proporción
    # de clases que el dataset completo. Sin esto, el split es aleatorio y
    # las métricas de evaluación pueden variar solo por "mala suerte" en el split.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"📊 Tasa de churn en test (con stratify): {y_test.mean():.2%}")

    print("🤖 Entrenando modelo...")
    # FIX: class_weight='balanced' penaliza más los errores sobre la clase
    # minoritaria (Churn=1, ~26% del dataset), en vez de optimizar accuracy
    # general ignorando esa clase.
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("\n=== RESULTADOS - VERSIÓN CORREGIDA (umbral 0.5) ===")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"F1-score (clase Churn): {f1_score(y_test, preds):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_test, probs):.4f}")
    print(classification_report(y_test, preds, target_names=['No Churn', 'Churn']))

    # ----------------------------------------------------
    # UMBRAL ÓPTIMO (maximiza F1 sobre la clase Churn)
    # ----------------------------------------------------
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = float(thresholds[best_idx])

    print(f"\n🎯 Umbral óptimo (máximo F1): {best_threshold:.3f}")
    preds_tuned = (probs >= best_threshold).astype(int)
    print("=== RESULTADOS - VERSIÓN CORREGIDA (umbral óptimo) ===")
    print(f"F1-score (clase Churn): {f1_score(y_test, preds_tuned):.4f}")
    print(classification_report(y_test, preds_tuned, target_names=['No Churn', 'Churn']))

    joblib.dump(best_threshold, MODELS_PATH / "decision_threshold.pkl")
    print(f"💾 Umbral óptimo guardado en: {MODELS_PATH / 'decision_threshold.pkl'}")
    print("   (api.py y app.py deben cargar este archivo en vez de usar 0.5 fijo)")

    # ----------------------------------------------------
    # GUARDAR LA IMPORTANCIA DE VARIABLES
    # ----------------------------------------------------
    print("📈 Calculando importancia de variables...")
    feature_importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)

    joblib.dump(feature_importances, MODELS_PATH / "feature_importance.pkl")
    print(f"✅ Importancia de variables guardada en: {MODELS_PATH / 'feature_importance.pkl'}")

    # ----------------------------------------------------
    # GUARDADO DINÁMICO DEL MODELO
    # ----------------------------------------------------
    model_file = MODELS_PATH / "churn_model.pkl"
    cols_file = MODELS_PATH / "model_columns.pkl"

    joblib.dump(model, model_file)
    joblib.dump(list(X_train.columns), cols_file)

    print(f"💾 Modelo guardado en: {model_file}")

    return model, X_test, y_test, probs


if __name__ == "__main__":
    train()