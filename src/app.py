import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import numpy as np
import plotly.express as px
from report_generator import generar_pdf
from preprocessing import preprocess_customer_df, load_decision_threshold

# ==========================================
# 📍 CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(
    page_title="Predicción de Retención",
    page_icon="📉",
    layout="wide"
)

FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent
MODELS_PATH = PROJECT_ROOT / "models"


# ==========================================
# 🧠 CARGA DEL MODELO Y RECURSOS
# ==========================================
@st.cache_resource
def load_model_resources():
    model_file = MODELS_PATH / "churn_model.pkl"
    cols_file = MODELS_PATH / "model_columns.pkl"
    importance_file = MODELS_PATH / "feature_importance.pkl"

    if not model_file.exists():
        return None, None, None, None

    model = joblib.load(model_file)
    model_cols = joblib.load(cols_file)
    feat_importance = joblib.load(importance_file) if importance_file.exists() else None

    # FIX: se carga el umbral óptimo (precision-recall) guardado por
    # train_model.py, en vez de dejar que model.predict() decida con 0.5.
    decision_threshold = load_decision_threshold(MODELS_PATH)

    return model, model_cols, feat_importance, decision_threshold


model, model_cols, feat_importance, decision_threshold = load_model_resources()

# ==========================================
# 🎨 INTERFAZ DE USUARIO
# ==========================================
st.title("📉 Predicción de Retención de Clientes (Churn)")
st.markdown("Esta aplicación permite predecir la probabilidad de que un cliente abandone el servicio.")
st.caption(f"Umbral de decisión del modelo (optimizado por F1): **{decision_threshold:.2f}** "
           f"— un cliente se clasifica como 'Churn' si su probabilidad supera este valor.")

st.sidebar.header("Panel de Control")
opcion = st.sidebar.radio("Modo de uso:", ["Carga Masiva (CSV)", "Predicción Manual"])

if model is None:
    st.error("❌ Error: No se encontró el modelo. Ejecuta train_model.py primero.")
    st.stop()

# --- MODO 1: CARGA MASIVA ---
if opcion == "Carga Masiva (CSV)":
    st.subheader("📂 Carga Masiva de Clientes")
    uploaded_file = st.file_uploader("Sube un archivo CSV", type="csv")

    if uploaded_file:
        df_raw = pd.read_csv(uploaded_file)
        st.write(f"✅ Archivo cargado: {len(df_raw)} clientes.")

        if st.button("⚡ Analizar Riesgo de Abandono", type="primary"):
            with st.spinner('El modelo está pensando...'):
                # Guardamos IDs antes de que preprocess_customer_df los descarte
                ids = df_raw['customerID'] if 'customerID' in df_raw.columns else df_raw.index

                # FIX: preprocesamiento centralizado (antes duplicado aquí, en
                # api.py y en predict.py).
                X_pred = preprocess_customer_df(df_raw, model_cols)

                probs = model.predict_proba(X_pred)[:, 1]

                # FIX: la clasificación binaria ahora usa el umbral óptimo
                # guardado, no el 0.5 implícito de model.predict().
                preds = (probs >= decision_threshold).astype(int)

                results = pd.DataFrame({
                    'CustomerID': ids.values,
                    'Probabilidad': probs,
                    'Predicción': preds
                })

                # Nota: estos 3 buckets (Bajo/Medio/Alto) son solo para
                # visualización y priorización interna del analista; son
                # independientes del umbral binario del modelo (que decide
                # Churn/No Churn para la métrica oficial reportada arriba).
                def clasificar_riesgo(p):
                    if p < 0.3:
                        return 'Bajo 🟢'
                    elif p < 0.7:
                        return 'Medio 🟡'
                    else:
                        return 'Alto 🔴'

                results['Nivel Riesgo'] = results['Probabilidad'].apply(clasificar_riesgo)

                # --- MOSTRAR RESULTADOS ---
                col1, col2, col3 = st.columns(3)
                total_alto = int(results['Predicción'].sum())
                porcentaje_fuga = (total_alto / len(results)) * 100

                col1.metric("Clientes Analizados", len(results))
                col2.metric("En Riesgo de Fuga (según umbral)", total_alto)
                col3.metric("Tasa de Churn Estimada", f"{porcentaje_fuga:.1f}%")

                if feat_importance is not None:
                    st.write("---")
                    st.subheader("🔍 ¿Qué está impulsando el Churn?")
                    top_n = feat_importance.head(10)
                    fig_imp = px.bar(
                        top_n,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title='Top 10 Factores de Riesgo (Feature Importance)',
                        labels={'importance': 'Impacto en la Predicción', 'feature': 'Variable'},
                        color='importance',
                        color_continuous_scale='Reds'
                    )
                    fig_imp.update_layout(yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig_imp, use_container_width=True)

                st.divider()

                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Distribución de Riesgo")
                    fig_pie = px.pie(results, names='Nivel Riesgo',
                                      color='Nivel Riesgo',
                                      color_discrete_map={'Bajo 🟢': '#2ecc71', 'Medio 🟡': '#f1c40f', 'Alto 🔴': '#e74c3c'})
                    st.plotly_chart(fig_pie, use_container_width=True)

                with c2:
                    st.subheader("Top Clientes en Riesgo")
                    top_risk = results.sort_values('Probabilidad', ascending=False).head(10)
                    st.dataframe(
                        top_risk[['CustomerID', 'Probabilidad', 'Nivel Riesgo']].style.format({'Probabilidad': '{:.4f}'}),
                        use_container_width=True
                    )

                st.divider()
                st.subheader("📑 Exportar Resultados")

                csv = results.to_csv(index=False).encode('utf-8')

                resumen = {
                    'total': len(results),
                    'riesgo_alto': total_alto,
                    'tasa': f"{porcentaje_fuga:.1f}"
                }
                pdf_bytes = generar_pdf(results, resumen)

                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    st.download_button(
                        "📥 Descargar CSV (Excel)",
                        data=csv,
                        file_name="predicciones_churn.csv",
                        mime="text/csv",
                        key='download-csv'
                    )
                with col_dl2:
                    st.download_button(
                        "📄 Descargar Reporte Ejecutivo (PDF)",
                        data=pdf_bytes,
                        file_name="reporte_ejecutivo_churn.pdf",
                        mime="application/pdf",
                        key='download-pdf'
                    )

# --- MODO 2: PREDICCIÓN MANUAL (Placeholder para futuro) ---
elif opcion == "Predicción Manual":
    st.info("🚧 Esta funcionalidad se desarrollará en el siguiente paso.")