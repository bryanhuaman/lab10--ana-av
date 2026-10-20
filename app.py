"""DecisionLab | Analítica avanzada de campañas bancarias.

Datos: UCI Bank Marketing, versión bank-full.csv (45 211 llamadas).
Ejecutar: streamlit run app.py
"""

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "bank_data.zip"
SOURCE_URL = "https://archive.ics.uci.edu/dataset/222/bank+marketing"

NUMERIC = ["age", "balance", "campaign", "previous"]
CATEGORICAL = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "poutcome",
]
FEATURES = NUMERIC + CATEGORICAL
MONTH_ORDER = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
LABELS = {
    "management": "Gerencia", "technician": "Técnico", "entrepreneur": "Emprendedor",
    "blue-collar": "Operario", "admin.": "Administrativo", "services": "Servicios",
    "retired": "Jubilado", "student": "Estudiante", "self-employed": "Independiente",
    "unemployed": "Desempleado", "housemaid": "Trabajo doméstico", "unknown": "No informado",
    "single": "Soltero", "married": "Casado", "divorced": "Divorciado / viudo",
    "primary": "Primaria", "secondary": "Secundaria", "tertiary": "Superior",
    "yes": "Sí", "no": "No", "cellular": "Celular", "telephone": "Teléfono",
    "success": "Éxito", "failure": "Sin éxito", "other": "Otro resultado",
}


st.set_page_config(
    page_title="DecisionLab | Analítica bancaria",
    page_icon="DL",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"], [data-testid="stApp"] {font-family:'DM Sans',sans-serif;}
[data-testid="stApp"] {background:#0b111b;color:#e8eef7;}
[data-testid="stSidebar"] {background:#111b2a;border-right:1px solid #263448;}
[data-testid="stSidebar"] * {color:#d9e4f3;}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif!important;letter-spacing:-.035em;}
h1 {font-size:2.35rem!important;color:#f4f7fc!important;}
h2 {font-size:1.45rem!important;color:#e8eef7!important;}
[data-testid="stMetric"] {background:#141f30;border:1px solid #2b3950;padding:20px;border-radius:14px;}
[data-testid="stMetricLabel"] {color:#9fb0c7;}
[data-testid="stMetricValue"] {color:#f1f6ff;}
div.stButton>button[kind="primary"], div.stFormSubmitButton>button[kind="primary"] {
 background:#3d8df6;border:none;color:#fff;font-weight:700;border-radius:10px;
}
.topline {font-size:12px;letter-spacing:.16em;color:#70acff;font-weight:700;margin-bottom:4px;}
.subline {color:#a6b7cf;line-height:1.55;margin-top:-8px;margin-bottom:18px;}
.panel {background:#141f30;border:1px solid #2c3b50;border-radius:14px;padding:18px 20px;margin:14px 0;}
.panel strong {color:#edf4ff;}
.muted {color:#9fb0c7;font-size:.88rem;}
.pill {display:inline-block;border:1px solid #38608d;border-radius:99px;background:#172c44;color:#92c2ff;padding:5px 10px;font-size:12px;font-weight:600;}
[data-testid="stForm"] {background:#121e2d;border:1px solid #2b3b50;border-radius:16px;padding:17px;}
div[data-testid="stAlert"] {border-radius:12px;}
a {color:#83baff!important;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Cargando registros históricos...")
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Falta bank_data.zip junto a app.py. Súbelo al repositorio.")
    with ZipFile(DATA_PATH) as archive:
        df = pd.read_csv(archive.open("bank-full.csv"), sep=";")
    df["accepted"] = (df["y"] == "yes").astype(int)
    return df


@st.cache_resource(show_spinner="Entrenando el modelo predictivo...")
def train_model():
    df = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        df[FEATURES], df["accepted"], test_size=0.25, random_state=42,
        stratify=df["accepted"],
    )
    prep = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC),
        ("category", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    model = Pipeline([
        ("preparation", prep),
        ("classifier", LogisticRegression(max_iter=1200, solver="lbfgs")),
    ])
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predicted = (probabilities >= 0.5).astype(int)
    scores = {
        "auc": roc_auc_score(y_test, probabilities),
        "ap": average_precision_score(y_test, probabilities),
        "precision": precision_score(y_test, predicted, zero_division=0),
        "recall": recall_score(y_test, predicted, zero_division=0),
        "base_rate": float(y_test.mean()),
        "matrix": confusion_matrix(y_test, predicted),
        "test_size": len(y_test),
    }
    return model, scores


def plot_style(fig, height=350, percent=False):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#141f30", plot_bgcolor="#141f30",
        font=dict(family="DM Sans, sans-serif", color="#d7e2f2", size=12),
        margin=dict(l=18, r=22, t=54, b=28), height=height,
        colorway=["#61a5ff", "#40c2b4", "#dfae64", "#b38afa"],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(bgcolor="#26354b"),
    )
    fig.update_xaxes(showgrid=False, linecolor="#3b4a5e", automargin=True)
    fig.update_yaxes(gridcolor="#28384b", zeroline=False, automargin=True)
    if percent:
        fig.update_yaxes(tickformat=".0%")
    return fig


def rate_chart(source, col, title, min_count=0, ordering=None):
    summary = source.groupby(col, dropna=False)["accepted"].agg(["mean", "size"]).reset_index()
    summary = summary[summary["size"] >= min_count].copy()
    summary["etiqueta"] = summary[col].map(LABELS).fillna(summary[col].astype(str))
    summary = summary.sort_values("mean", ascending=True)
    fig = px.bar(
        summary, x="mean", y="etiqueta", orientation="h", color_discrete_sequence=["#61a5ff"],
        labels={"mean": "Tasa de aceptación", "etiqueta": ""},
        custom_data=["size"], title=title,
    )
    fig.update_traces(hovertemplate="%{y}<br>Aceptación: %{x:.1%}<br>Registros: %{customdata[0]:,}<extra></extra>")
    fig.update_xaxes(tickformat=".0%")
    return plot_style(fig, height=max(320, 42 * len(summary) + 110))


def header(section, title, subtitle):
    st.markdown(f'<div class="topline">DECISIONLAB / {section.upper()}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="subline">{subtitle}</div>', unsafe_allow_html=True)


try:
    data = load_data()
except Exception as exc:
    st.error(f"No se pudieron cargar los datos: {exc}")
    st.stop()

with st.sidebar:
    st.markdown("### DECISIONLAB")
    st.caption("ANALÍTICA PREDICTIVA · V1.0")
    st.divider()
    page = st.radio(
        "Navegación",
        ["Simulador predictivo", "Panel de indicadores", "Análisis de factores", "Modelo y metodología"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown('<span class="pill">UCI · DATOS REALES</span>', unsafe_allow_html=True)
    st.caption("45,211 registros históricos de campañas bancarias en Portugal. Datos antiguos; resultados demostrativos, no recomendaciones financieras.")


if page == "Simulador predictivo":
    header("Simulador", "Predicción de aceptación", "Introduce información disponible antes de contactar al cliente. El sistema estima la probabilidad de que acepte un depósito a plazo.")
    left, right = st.columns([1.18, 0.82], gap="large")
    with left:
        with st.form("prediction_form"):
            st.markdown("#### Perfil del cliente")
            a, b = st.columns(2)
            with a:
                age = st.number_input("Edad", min_value=18, max_value=100, value=35)
                job = st.selectbox("Ocupación", sorted(data.job.unique()), format_func=lambda x: LABELS.get(x, x), index=sorted(data.job.unique()).index("technician"))
                education = st.selectbox("Educación", sorted(data.education.unique()), format_func=lambda x: LABELS.get(x, x))
                balance = st.number_input("Saldo anual promedio (EUR)", value=1500, step=100)
            with b:
                marital = st.selectbox("Estado civil", sorted(data.marital.unique()), format_func=lambda x: LABELS.get(x, x))
                housing = st.selectbox("¿Tiene préstamo hipotecario?", ["no", "yes"], format_func=lambda x: LABELS[x])
                loan = st.selectbox("¿Tiene préstamo personal?", ["no", "yes"], format_func=lambda x: LABELS[x])
                default = st.selectbox("¿Registra impago crediticio?", ["no", "yes"], format_func=lambda x: LABELS[x])
            st.markdown("#### Contexto de la campaña")
            c, d = st.columns(2)
            with c:
                contact = st.selectbox("Canal de contacto", sorted(data.contact.unique()), format_func=lambda x: LABELS.get(x, x))
                campaign = st.number_input("Número de contacto en esta campaña", 1, 60, 1)
            with d:
                previous = st.number_input("Contactos en campañas anteriores", 0, 100, 0)
                poutcome = st.selectbox("Resultado de campaña anterior", ["unknown", "failure", "success", "other"], format_func=lambda x: LABELS.get(x, x))
            submit = st.form_submit_button("Calcular probabilidad", type="primary", width="stretch")
    with right:
        st.markdown("#### Resultado del modelo")
        if submit:
            sample = pd.DataFrame([{
                "age": age, "balance": balance, "campaign": campaign, "previous": previous,
                "job": job, "marital": marital, "education": education, "default": default,
                "housing": housing, "loan": loan, "contact": contact, "poutcome": poutcome,
            }])[FEATURES]
            model, _ = train_model()
            probability = float(model.predict_proba(sample)[0, 1])
            st.session_state["prediction"] = probability
            st.session_state["profile"] = sample
        if "prediction" not in st.session_state:
            st.info("Completa el formulario y selecciona «Calcular probabilidad» para obtener el resultado.")
        else:
            p = st.session_state["prediction"]
            baseline = float(data.accepted.mean())
            st.metric("Probabilidad estimada de aceptación", f"{p:.1%}", delta=f"{(p-baseline)*100:+.1f} puntos vs. promedio histórico")
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=p*100, number=dict(suffix=" %", font=dict(size=35)),
                gauge=dict(axis=dict(range=[0, 100], tickvals=[0, 25, 50, 75, 100]),
                           bar=dict(color="#61a5ff"), bgcolor="#243449", borderwidth=0,
                           steps=[dict(range=[0, 100], color="#1b2a3e")]),
            ))
            gauge.update_layout(height=250, margin=dict(l=30,r=30,t=25,b=12), paper_bgcolor="#141f30", font_color="#e5eefb")
            st.plotly_chart(gauge, width="stretch")
            if p >= baseline:
                st.success("Este perfil tiene una probabilidad estimada superior al promedio histórico del dataset.")
            else:
                st.warning("Este perfil tiene una probabilidad estimada inferior al promedio histórico del dataset.")
            st.markdown(f'<div class="panel"><strong>Referencia:</strong> en todo el conjunto histórico, el {baseline:.1%} de los contactos aceptó la oferta.<br><span class="muted">La probabilidad es una estimación estadística, no una garantía. Un resultado superior al promedio no implica causalidad.</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel"><strong>¿Qué calcula?</strong><br>Una regresión logística entrenada con campañas bancarias reales. No se utiliza la duración final de la llamada porque ese dato no está disponible antes del contacto.</div>', unsafe_allow_html=True)

elif page == "Panel de indicadores":
    header("Dashboard", "Panorama de campañas", "Explora los resultados históricos y compara tasas de aceptación entre segmentos.")
    a, b, c = st.columns(3)
    with a:
        selected_job = st.multiselect("Ocupación", sorted(data.job.unique()), format_func=lambda x: LABELS.get(x, x))
    with b:
        selected_education = st.multiselect("Nivel educativo", sorted(data.education.unique()), format_func=lambda x: LABELS.get(x, x))
    with c:
        selected_contact = st.multiselect("Canal", sorted(data.contact.unique()), format_func=lambda x: LABELS.get(x, x))
    df = data
    for column, selections in [("job", selected_job), ("education", selected_education), ("contact", selected_contact)]:
        if selections:
            df = df[df[column].isin(selections)]
    if df.empty:
        st.warning("No existen registros con los filtros seleccionados.")
        st.stop()
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Contactos analizados", f"{len(df):,}")
    k2.metric("Aceptaciones", f"{df.accepted.sum():,}")
    k3.metric("Tasa de aceptación", f"{df.accepted.mean():.1%}")
    k4.metric("Edad promedio", f"{df.age.mean():.1f} años")
    left, right = st.columns(2, gap="large")
    with left:
        monthly = df.groupby("month")["accepted"].agg(["size", "mean"]).reindex(MONTH_ORDER).fillna(0).reset_index()
        fig = px.bar(monthly, x="month", y="size", title="Volumen de contactos por mes", labels={"month":"Mes", "size":"Número de contactos"})
        fig.update_traces(marker_color="#61a5ff", hovertemplate="Mes: %{x}<br>Contactos: %{y:,}<extra></extra>")
        st.plotly_chart(plot_style(fig), width="stretch")
    with right:
        st.plotly_chart(rate_chart(df, "contact", "Tasa de aceptación por canal"), width="stretch")
    left, right = st.columns(2, gap="large")
    with left:
        st.plotly_chart(rate_chart(df, "education", "Tasa de aceptación por educación"), width="stretch")
    with right:
        counts = df["accepted"].value_counts().reindex([0,1], fill_value=0)
        fig = go.Figure(go.Pie(labels=["No aceptó", "Aceptó"], values=counts.values, hole=.64,
                               marker=dict(colors=["#354b64", "#40c2b4"]), textinfo="label+percent"))
        fig.update_layout(title="Distribución de resultados", showlegend=False)
        st.plotly_chart(plot_style(fig), width="stretch")
    st.caption("Las diferencias observadas son descriptivas. La fuente no representa una muestra reciente ni permite establecer causas por sí sola.")

elif page == "Análisis de factores":
    header("Exploración", "Factores relacionados con la aceptación", "Analiza asociaciones en los datos originales. Las diferencias entre grupos no significan que una variable cause la aceptación.")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.plotly_chart(rate_chart(data, "job", "Aceptación por ocupación", min_count=100), width="stretch")
    with col2:
        st.plotly_chart(rate_chart(data, "poutcome", "Aceptación según campaña anterior"), width="stretch")
    st.markdown("#### Edad y resultado")
    bins = [17, 25, 35, 45, 55, 65, 100]
    temp = data.assign(age_range=pd.cut(data.age, bins=bins, labels=["18–25", "26–35", "36–45", "46–55", "56–65", "66+"]))
    hist = temp.groupby("age_range", observed=True)["accepted"].agg(["mean", "size"]).reset_index()
    fig = px.bar(hist, x="age_range", y="mean", title="Tasa histórica de aceptación por rango de edad",
                 labels={"age_range":"Rango de edad", "mean":"Tasa de aceptación"}, custom_data=["size"])
    fig.update_traces(marker_color="#40c2b4", hovertemplate="Edad: %{x}<br>Tasa: %{y:.1%}<br>Registros: %{customdata[0]:,}<extra></extra>")
    st.plotly_chart(plot_style(fig, height=350, percent=True), width="stretch")
    st.markdown('<div class="panel"><strong>Lectura del análisis:</strong> compara la tasa de cada grupo junto con su cantidad de registros. Grupos pequeños pueden mostrar porcentajes inestables. No interpretes estos gráficos como efectos causales ni como reglas automáticas de decisión.</div>', unsafe_allow_html=True)

else:
    header("Metodología", "Transparencia del modelo", "Cómo se obtienen las predicciones, cuáles son sus límites y cómo evaluamos su desempeño.")
    model, scores = train_model()
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("ROC AUC", f"{scores['auc']:.3f}")
    m2.metric("PR AUC", f"{scores['ap']:.3f}")
    m3.metric("Precisión (umbral 50%)", f"{scores['precision']:.1%}")
    m4.metric("Recall (umbral 50%)", f"{scores['recall']:.1%}")
    st.markdown("#### Evaluación del conjunto de prueba")
    a,b = st.columns([0.9,1.1], gap="large")
    with a:
        mat = scores["matrix"]
        cm = go.Figure(data=go.Heatmap(z=mat, x=["Pred. no", "Pred. sí"], y=["Real no", "Real sí"],
                     text=mat, texttemplate="%{text:,}", colorscale=[[0,"#172537"],[1,"#61a5ff"]], showscale=False))
        cm.update_layout(title="Matriz de confusión · umbral 50%")
        st.plotly_chart(plot_style(cm, height=320), width="stretch")
    with b:
        st.markdown(f"""
        <div class="panel"><strong>Entrenamiento y validación</strong><br>
        Algoritmo: regresión logística con escalado de números y codificación de categorías.<br>
        División estratificada reproducible: 75% entrenamiento / 25% prueba ({scores['test_size']:,} casos).<br>
        Tasa positiva del conjunto de prueba: {scores['base_rate']:.1%}.<br><br>
        <strong>ROC AUC:</strong> capacidad de ordenar casos positivos por encima de negativos.<br>
        <strong>PR AUC:</strong> desempeño al identificar la clase menos frecuente.<br>
        <strong>Precisión:</strong> proporción de predicciones positivas que aciertan.<br>
        <strong>Recall:</strong> proporción de aceptaciones reales detectadas.</div>
        """, unsafe_allow_html=True)
    st.markdown("#### Variables usadas")
    st.write("Edad, saldo, número de contactos de la campaña actual, contactos previos, ocupación, estado civil, educación, impagos, hipoteca, préstamo personal, canal y resultado de una campaña anterior.")
    st.markdown("#### Decisiones de diseño")
    st.markdown("""
    - **Sin fuga de información:** se excluye `duration` (duración de la llamada), porque solo se conoce cuando la llamada termina.
    - **Separación de datos:** se ajusta el preprocesamiento únicamente con los datos de entrenamiento mediante un `Pipeline`.
    - **Sin causalidad:** el modelo descubre asociaciones, no demuestra causas ni garantiza el comportamiento de una persona.
    - **Limitación temporal:** se trata de campañas históricas; las predicciones pueden no generalizar a campañas actuales o a otros países.
    - **Validación académica:** la partición aleatoria mide desempeño en una muestra reservada, no sustituye una validación temporal externa.
    """)
    st.markdown("#### Procedencia de datos")
    st.markdown(f"**UCI Machine Learning Repository — Bank Marketing.** [Consultar dataset original]({SOURCE_URL}). Datos de campañas telefónicas de un banco portugués. Licencia CC BY 4.0; atribución a S. Moro, P. Rita y P. Cortez.")
    st.markdown("#### Descargar datos analizados")
    st.download_button("Descargar muestra CSV (primeros 5,000 registros)", data=data.head(5000).to_csv(index=False).encode("utf-8"), file_name="muestra_bank_marketing.csv", mime="text/csv")
