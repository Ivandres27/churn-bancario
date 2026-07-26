# report_generator.py
# Generador de PDF ejecutivo (opción 1 - azul corporativo)
# Usa ReportLab (platypus) + matplotlib para gráficos embebidos.
# Devuelve bytes (lista de bytes) que Streamlit puede usar en st.download_button.

import io
import os
from datetime import datetime
from typing import Dict

import matplotlib.pyplot as plt
import joblib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)


# -----------------------------
# CONFIG
# -----------------------------
THIS_DIR = os.path.dirname(__file__)
FONTS_DIR = os.path.join(THIS_DIR, "fonts")  # asegúrate de colocar las .ttf aquí

# FIX: antes estas rutas apuntaban al home personal del autor
# (/home/ivana/ml_projects/...), lo que rompía el script en cualquier otra
# máquina. Ahora se calculan de forma dinámica, igual que en api.py, app.py
# y train_model.py.
PROJECT_ROOT = os.path.dirname(THIS_DIR)          # churn_project/
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Paleta azul corporativa (opción 1)
COLOR_PRIMARY = colors.HexColor("#0A3D62")
COLOR_ACCENT = colors.HexColor("#3C6382")
COLOR_LIGHT = colors.HexColor("#60A3BC")

# Fuentes esperadas
FONT_REGISTRY = {
    "DejaVuSans": "DejaVuSans.ttf",
    "DejaVuSans-Bold": "DejaVuSans-Bold.ttf",
    "DejaVuSans-Oblique": "DejaVuSans-Oblique.ttf",
}


# -----------------------------
# UTIL: Registrar fuentes
# -----------------------------
def _register_fonts():
    """Registra fuentes TTF requeridas. Lanza RuntimeError si faltan archivos."""
    for font_name, fname in FONT_REGISTRY.items():
        path = os.path.join(FONTS_DIR, fname)
        if not os.path.isfile(path):
            raise RuntimeError(
                f"Fuente requerida no encontrada: {path}\n"
                "Extrae DejaVuSans*.ttf en el directorio src/fonts/ o ajusta FONTS_DIR."
            )
        pdfmetrics.registerFont(TTFont(font_name, path))


# -----------------------------
# UTIL: Cargar Feature Importance & Mapeo
# -----------------------------
def _get_recommendation_text(feature_name: str) -> str:
    """Mapea una característica importante a una acción de negocio basada en el contexto Telco."""
    mapping = {
        'Contract_Month-to-month': 'El tipo de contrato Mes-a-Mes es el principal impulsor de churn. Urge migrar a planes de fidelización anuales.',
        'tenure': 'Clientes con poca antigüedad son críticos. Implementar programas de Onboarding intensivo y de seguimiento durante los primeros 6 meses.',
        'MonthlyCharges': 'Los cargos altos se correlacionan con insatisfacción. Revisar la estructura de precios y ofrecer paquetes o incentivos personalizados.',
        'InternetService_Fiber optic': 'La fibra óptica genera altos picos de abandono. Priorizar la calidad del servicio y soporte técnico para estos clientes.',
        'PaymentMethod_Electronic check': 'Este método de pago es un indicador de fricción. Ofrecer métodos de pago más convenientes y automatizados.',
    }
    return mapping.get(
        feature_name,
        'Realizar un análisis manual detallado sobre esta variable para definir una estrategia específica.'
    )


def _load_top_features(n=3):
    """Carga las N características más importantes desde el modelo."""
    importance_file = os.path.join(MODELS_DIR, "feature_importance.pkl")
    try:
        if os.path.exists(importance_file):
            feature_importances = joblib.load(importance_file)
            return feature_importances.head(n)['feature'].tolist()
        else:
            return ["Contract_Month-to-month", "tenure", "MonthlyCharges"]
    except Exception as e:
        print(f"Advertencia: Error cargando Feature Importance: {e}. Usando valores por defecto.")
        return ["Contract_Month-to-month", "tenure", "MonthlyCharges"]


# -----------------------------
# HEADER / FOOTER
# -----------------------------
def _header_footer(canvas, doc):
    """Dibuja header y footer para cada página (usado por SimpleDocTemplate.onPage)."""
    canvas.saveState()
    page_width, page_height = A4
    header_height = 40
    canvas.setFillColor(COLOR_PRIMARY)
    canvas.rect(0, page_height - header_height, page_width, header_height, fill=1, stroke=0)

    canvas.setFont("DejaVuSans-Bold", 16)
    canvas.setFillColor(colors.white)
    canvas.drawCentredString(page_width / 2, page_height - 26, "Reporte de Riesgo de Fuga (Churn)")

    canvas.setFont("DejaVuSans-Oblique", 8)
    fecha = datetime.now().strftime("%d/%m/%Y %H:%M")
    canvas.drawRightString(page_width - 20, page_height - 34, f"Generado: {fecha}")

    canvas.setFont("DejaVuSans", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawCentredString(page_width / 2, 15, f"Página {canvas.getPageNumber()}")

    canvas.restoreState()


# -----------------------------
# GRAFICOS (matplotlib -> BytesIO -> ReportLab Image)
# -----------------------------
def _make_pie_chart(results: pd.DataFrame) -> Image:
    """Genera un gráfico de pastel en memoria y devuelve un objeto Image para Platypus."""
    # FIX: los emojis de color (🔴🟡🟢) no renderizan con la fuente DejaVuSans
    # en matplotlib (glifo faltante -> aparece como cuadro vacío). El color
    # ya se transmite con el color de cada rebanada, así que basta con texto
    # plano en la etiqueta.
    etiquetas_limpias = {'Alto 🔴': 'Alto', 'Medio 🟡': 'Medio', 'Bajo 🟢': 'Bajo'}
    nivel_limpio = results['Nivel Riesgo'].map(etiquetas_limpias).fillna(results['Nivel Riesgo'])
    counts = nivel_limpio.value_counts()

    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    labels = counts.index.tolist()
    sizes = counts.values

    color_map = {'Bajo': '#2ecc71', 'Medio': '#f1c40f', 'Alto': '#e74c3c'}
    slice_colors = [color_map.get(l, '#999999') for l in labels]
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=slice_colors)
    ax.axis('equal')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)

    return Image(buf, width=120 * mm, height=90 * mm)


def _make_bar_chart(results: pd.DataFrame) -> Image:
    """Genera un gráfico de barras sencillo (top 10 probabilidades)."""
    top10 = results.sort_values('Probabilidad', ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)

    ax.barh(range(len(top10)), top10['Probabilidad'], tick_label=top10['CustomerID'].astype(str))
    ax.invert_yaxis()
    ax.set_xlabel('Probabilidad de Churn')
    ax.xaxis.set_major_formatter(lambda x, pos: f'{x:.0%}')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)

    return Image(buf, width=160 * mm, height=90 * mm)


# -----------------------------
# Construcción del documento
# -----------------------------
def generar_pdf(results: pd.DataFrame, resumen_metricas: Dict) -> bytes:
    """Genera el PDF ejecutivo y devuelve bytes.

    Args:
        results: DataFrame con columnas ['CustomerID', 'Probabilidad', 'Nivel Riesgo']
        resumen_metricas: dict con keys 'total', 'riesgo_alto', 'tasa'

    Returns:
        bytes del PDF listos para st.download_button
    """
    _register_fonts()

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=30 * mm,
        bottomMargin=20 * mm,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleCentered', parent=styles['Title'], alignment=TA_CENTER,
                              fontName='DejaVuSans-Bold', fontSize=20, textColor=COLOR_PRIMARY))
    styles.add(ParagraphStyle(name='SubTitle', parent=styles['Normal'], alignment=TA_CENTER,
                              fontName='DejaVuSans', fontSize=10, textColor=COLOR_ACCENT))
    styles.add(ParagraphStyle(name='Heading', parent=styles['Heading2'], fontName='DejaVuSans-Bold',
                              textColor=COLOR_PRIMARY))
    styles.add(ParagraphStyle(name='NormalDejaVu', parent=styles['Normal'], fontName='DejaVuSans', fontSize=10))

    flowables = []

    # Portada
    flowables.append(Spacer(1, 12 * mm))
    flowables.append(Paragraph('Reporte Ejecutivo: Riesgo de Fuga (Churn)', styles['TitleCentered']))
    flowables.append(Spacer(1, 4 * mm))
    flowables.append(Paragraph('Análisis automatizado de clientes y recomendaciones', styles['SubTitle']))
    flowables.append(Spacer(1, 12 * mm))

    metrics_table_data = [
        [Paragraph('<b>Total Clientes</b>', styles['NormalDejaVu']), str(resumen_metricas.get('total', 'N/A'))],
        [Paragraph('<b>En Riesgo Alto</b>', styles['NormalDejaVu']), str(resumen_metricas.get('riesgo_alto', 'N/A'))],
        [Paragraph('<b>Tasa de Churn Estimada</b>', styles['NormalDejaVu']), f"{resumen_metricas.get('tasa', 'N/A')}%"],
    ]

    metrics_table = Table(metrics_table_data, colWidths=[90 * mm, 60 * mm])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COLOR_LIGHT),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.grey),
        ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, -1), 'DejaVuSans'),
    ]))

    flowables.append(metrics_table)
    flowables.append(Spacer(1, 8 * mm))

    try:
        pie = _make_pie_chart(results)
        bar = _make_bar_chart(results)
        flowables.append(pie)
        flowables.append(Spacer(1, 6 * mm))
        flowables.append(bar)
    except Exception:
        flowables.append(Paragraph('Gráficos no disponibles', styles['NormalDejaVu']))

    flowables.append(PageBreak())

    # Sección: Top clientes en riesgo
    flowables.append(Paragraph('Top Clientes en Riesgo', styles['Heading']))
    flowables.append(Spacer(1, 4 * mm))

    top_risk = results.sort_values('Probabilidad', ascending=False).head(30)

    # FIX: el estilo se llama 'NormalDejaVu' (V mayúscula). Antes se buscaba
    # 'NormalDejavu' (v minúscula) -> nunca existía -> siempre caía al
    # fallback styles['Normal'], dejando esa celda con una fuente distinta
    # al resto del documento sin que se notara a simple vista.
    table_data = [[
        Paragraph('<b>CustomerID</b>', styles['NormalDejaVu']),
        Paragraph('<b>Probabilidad</b>', styles['NormalDejaVu']),
        Paragraph('<b>Nivel Riesgo</b>', styles['NormalDejaVu'])
    ]]

    for _, row in top_risk.iterrows():
        cust = '' if pd.isna(row['CustomerID']) else str(row['CustomerID'])
        try:
            prob_text = f"{row['Probabilidad']:.1%}"
        except Exception:
            prob_text = str(row['Probabilidad'])
        nivel = '' if pd.isna(row['Nivel Riesgo']) else str(row['Nivel Riesgo'])
        table_data.append([cust, prob_text, nivel])

    tbl = Table(table_data, colWidths=[70 * mm, 50 * mm, 50 * mm], repeatRows=1)
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COLOR_LIGHT),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ('FONTNAME', (0, 0), (-1, -1), 'DejaVuSans'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))

    flowables.append(tbl)
    flowables.append(Spacer(1, 6 * mm))

    # Sección: Recomendaciones (dinámicas)
    top_features = _load_top_features(n=3)

    flowables.append(Paragraph('Factores Clave de Riesgo y Estrategias', styles['Heading']))
    flowables.append(Spacer(1, 4 * mm))
    flowables.append(Paragraph('<b>Análisis Basado en el Modelo Predictivo:</b>', styles['NormalDejaVu']))

    for i, feature in enumerate(top_features):
        action_text = _get_recommendation_text(feature)
        flowables.append(Paragraph(
            f'• <b>Factor {i+1} ({feature}):</b> {action_text}',
            styles['NormalDejaVu']
        ))
        flowables.append(Spacer(1, 2 * mm))

    flowables.append(Spacer(1, 6 * mm))
    flowables.append(Paragraph('<b>Acciones Estratégicas Generales:</b>', styles['NormalDejaVu']))

    recomendaciones_generales = [
        'Contactar inmediatamente a los clientes listados en la tabla superior (Riesgo Alto).',
        'Analizar tickets de soporte abiertos y priorizar la resolución para clientes identificados como de riesgo.',
        'Lanzar una encuesta de satisfacción segmentada para identificar causas cualitativas no capturadas por el modelo.'
    ]

    for rec in recomendaciones_generales:
        flowables.append(Paragraph(f'• {rec}', styles['NormalDejaVu']))
        flowables.append(Spacer(1, 2 * mm))

    doc.build(flowables, onFirstPage=_header_footer, onLaterPages=_header_footer)

    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes