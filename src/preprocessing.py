"""
preprocessing.py
Lógica de preprocesamiento COMPARTIDA por api.py, app.py y predict.py.

Antes esta misma lógica estaba copiada y pegada en los 3 archivos.
Si mañana cambia el preprocesamiento (ej. nueva feature, otro manejo de
nulos), había que recordar tocar los 3 lugares. Ahora se toca uno solo.
"""

import pandas as pd
import numpy as np


def preprocess_customer_df(df_raw: pd.DataFrame, model_columns: list) -> pd.DataFrame:
    """
    Transforma un DataFrame de clientes "crudo" (una o varias filas) al
    formato exacto de columnas que el modelo espera.

    Args:
        df_raw: DataFrame con las columnas originales del cliente
                 (puede incluir customerID y/o Churn, se descartan si existen).
        model_columns: lista de columnas con las que se entrenó el modelo
                        (guardada en models/model_columns.pkl).

    Returns:
        DataFrame alineado exactamente a model_columns, listo para
        model.predict_proba().
    """
    df = df_raw.copy()

    # 1. Limpieza de TotalCharges (idéntico a train_model.py)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # NOTA: se reasigna la columna en vez de usar fillna(inplace=True),
    # que en pandas >= 2.0 con Copy-on-Write no garantiza persistir el cambio.
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # 2. Quitar columnas que no son features
    df = df.drop(['customerID', 'Churn'], axis=1, errors='ignore')

    # 3. One-Hot Encoding
    df_encoded = pd.get_dummies(df)

    # 4. Alineación de columnas con las del modelo entrenado
    df_final = pd.DataFrame(0, index=df_encoded.index, columns=model_columns)
    common_cols = list(set(df_encoded.columns) & set(model_columns))
    df_final[common_cols] = df_encoded[common_cols]

    return df_final


def load_decision_threshold(models_path, default: float = 0.5) -> float:
    """
    Carga el umbral de decisión óptimo guardado por train_model.py
    (models/decision_threshold.pkl). Si no existe (ej. modelo entrenado
    con una versión anterior del script), regresa `default` para no
    romper la app, mostrando un aviso.
    """
    threshold_file = models_path / "decision_threshold.pkl"
    if threshold_file.exists():
        import joblib
        return joblib.load(threshold_file)
    print(f"⚠️ No se encontró {threshold_file}. Usando umbral por defecto {default}. "
          f"Vuelve a correr train_model.py para generar el umbral óptimo.")
    return default