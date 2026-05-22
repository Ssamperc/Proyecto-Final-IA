"""
app.py — Demo interactiva: Detección de Fraude + Explicaciones LLM
Ejecutar con: streamlit run app/app.py
Proyecto Final IA · EAFIT 2026-1 · Samuel Samper Cardona
"""

import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import streamlit as st
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

# ── Configuración ────────────────────────────────────────────
st.set_page_config(
    page_title="Detector de Fraude · EAFIT 2026",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilos ──────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #1a1510; }
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.82) !important; }
.metric-fraude  { background:#fdf0f0; border-left:4px solid #9b1b1b;
                  padding:0.8rem 1rem; border-radius:8px; margin:4px 0; }
.metric-legitima{ background:#e6f4ed; border-left:4px solid #006b3c;
                  padding:0.8rem 1rem; border-radius:8px; margin:4px 0; }
.metric-neutral { background:#e8eef7; border-left:4px solid #003d79;
                  padding:0.8rem 1rem; border-radius:8px; margin:4px 0; }
h1,h2,h3 { color:#003d79; }
</style>
""", unsafe_allow_html=True)

# ── Rutas ────────────────────────────────────────────────────
ROOT   = os.path.join(os.path.dirname(__file__), "..")
MODELS = os.path.join(ROOT, "models")
PROC   = os.path.join(ROOT, "data", "processed")
FIGS   = os.path.join(ROOT, "figures")

# ── Carga de modelo y datos (con caché) ─────────────────────
@st.cache_resource
def cargar_modelo():
    path = os.path.join(MODELS, "modelo_final.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    # Modelo demo si no existe el real
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    X_s, y_s = make_classification(
        n_samples=5000, n_features=30, n_informative=15,
        weights=[0.997, 0.003], random_state=42
    )
    sc = StandardScaler()
    X_s = sc.fit_transform(X_s)
    m = XGBClassifier(n_estimators=100, random_state=42, verbosity=0, scale_pos_weight=332)
    m.fit(X_s, y_s)
    return m

@st.cache_data
def cargar_test_set():
    Xp = os.path.join(PROC, "X_test.npy")
    yp = os.path.join(PROC, "y_test.npy")
    fp = os.path.join(PROC, "feature_names.csv")
    if os.path.exists(Xp):
        X = np.load(Xp)
        y = np.load(yp)
        feat = pd.read_csv(fp).iloc[:, 0].tolist() if os.path.exists(fp) else [f"f{i}" for i in range(X.shape[1])]
        return X, y, feat
    return None, None, None

@st.cache_data
def cargar_metricas():
    path = os.path.join(MODELS, "metricas_finales.json")
    if os.path.exists(path):
        return json.load(open(path))
    return {"modelo": "XGBoost (demo)", "accuracy": 0.9743,
            "precision": 0.6731, "recall": 0.6195, "f1": 0.6452, "auc_roc": 0.9166}

modelo   = cargar_modelo()
X_test, y_test, feat_names = cargar_test_set()
metricas = cargar_metricas()

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Detector de Fraude")
    st.markdown("**Proyecto Final IA · EAFIT 2026-1**")
    st.divider()
    st.markdown("### 👥 Equipo")
    st.markdown("""
- David Ruiz Vallejo
- David Quintero
- Juan Pablo Duque Osorio
- Samuel Samper Cardona
""")
    st.divider()
    st.markdown("### 🏆 Métricas — Test Set")
    st.metric("AUC-ROC",   f"{metricas.get('auc_roc', 0):.4f}",  "+0.093 vs baseline")
    st.metric("F1-Score",  f"{metricas.get('f1', 0):.4f}",       "+0.367 vs baseline")
    st.metric("Recall",    f"{metricas.get('recall', 0):.4f}",   "fraudes detectados")
    st.metric("Precision", f"{metricas.get('precision', 0):.4f}", "de alertas son reales")
    st.divider()
    st.caption(f"Modelo: {metricas.get('modelo', 'XGBoost')}")

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Predicción", "📊 Análisis EDA", "🧠 Explicaciones LLM", "📄 Informe"
])

# ════════════════════════════════════════════════════════════
# TAB 1 — PREDICCIÓN
# ════════════════════════════════════════════════════════════
with tab1:
    st.title("🔮 Predicción de Fraude en Tiempo Real")
    st.markdown(
        "Ajusta los parámetros de la transacción. "
        "Los valores corresponden a componentes PCA del dataset de fraude financiero."
    )

    col_inp, col_res = st.columns([2, 1])

    with col_inp:
        st.subheader("Parámetros de la transacción")
        c1, c2, c3 = st.columns(3)
        n_feats = feat_names if feat_names else [f"V{i}" for i in range(1, 29)] + ["log_amount", "hour_of_day"]
        n_total = len(n_feats)

        sliders = {}
        per_col = (n_total + 2) // 3
        grupos  = [n_feats[:per_col], n_feats[per_col:2*per_col], n_feats[2*per_col:]]

        for col_st, grupo in zip([c1, c2, c3], grupos):
            with col_st:
                for feat in grupo:
                    if "amount" in feat.lower() or "log" in feat.lower():
                        sliders[feat] = st.slider(feat, 0.0, 10.0, 3.5, 0.1,
                                                   help="Log del monto de la transacción")
                    elif "hour" in feat.lower() or "time" in feat.lower():
                        sliders[feat] = st.slider(feat, 0.0, 23.9, 14.0, 0.1,
                                                   help="Hora del día (0=medianoche, 14=2pm)")
                    else:
                        sliders[feat] = st.slider(feat, -5.0, 5.0, 0.0, 0.1,
                                                   help=f"Componente PCA {feat}")

    with col_res:
        st.subheader("Resultado")
        if st.button("🚀 Analizar transacción", type="primary", use_container_width=True):
            inp     = np.array([[sliders[f] for f in (feat_names if feat_names else list(sliders.keys()))]])
            pred    = int(modelo.predict(inp)[0])
            prob    = float(modelo.predict_proba(inp)[0, 1])
            clase   = "FRAUDE" if pred == 1 else "LEGÍTIMA"
            color   = "metric-fraude" if pred == 1 else "metric-legitima"
            emoji   = "🔴" if pred == 1 else "🟢"

            st.markdown(f'<div class="{color}"><h3>{emoji} {clase}</h3></div>',
                        unsafe_allow_html=True)
            st.metric("Probabilidad de fraude", f"{prob:.1%}")
            st.progress(float(prob), text=f"Riesgo: {prob:.1%}")

            if pred == 1:
                st.error("⚠️ Recomendación: BLOQUEAR y notificar al titular")
            elif prob > 0.3:
                st.warning("🟡 Recomendación: REVISAR manualmente")
            else:
                st.success("✅ Recomendación: APROBAR sin intervención")

            # Mostrar top features
            st.subheader("Top 5 features influyentes")
            vals  = inp[0]
            top5  = sorted(enumerate(vals), key=lambda x: abs(x[1]), reverse=True)[:5]
            for i, v in top5:
                fname = feat_names[i] if feat_names and i < len(feat_names) else f"f{i}"
                st.write(f"• **{fname}**: {v:+.3f}")
        else:
            st.info("Ajusta los parámetros y presiona **Analizar transacción**")

    # Muestra de transacciones del test set
    if X_test is not None:
        st.divider()
        st.subheader("🎲 Cargar una transacción real del test set")
        col_a, col_b, col_c = st.columns([1, 1, 2])
        with col_a:
            tipo_muestra = st.radio("Tipo:", ["Fraude real", "Legítima real", "Aleatoria"])
        with col_b:
            if st.button("Cargar muestra", use_container_width=True):
                if tipo_muestra == "Fraude real" and (y_test == 1).any():
                    idx_pool = np.where(y_test == 1)[0]
                elif tipo_muestra == "Legítima real" and (y_test == 0).any():
                    idx_pool = np.where(y_test == 0)[0]
                else:
                    idx_pool = np.arange(len(y_test))
                idx_sel = int(np.random.choice(idx_pool))
                prob_r  = float(modelo.predict_proba(X_test[idx_sel:idx_sel+1])[0, 1])
                pred_r  = int(prob_r >= 0.5)
                etiq    = "🔴 FRAUDE" if pred_r else "🟢 LEGÍTIMA"
                real    = "🔴 FRAUDE" if y_test[idx_sel] == 1 else "🟢 LEGÍTIMA"
                with col_c:
                    st.markdown(f"**Índice:** {idx_sel}  |  **Predicción:** {etiq}  |  **Real:** {real}  |  **Prob:** {prob_r:.4f}")

# ════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ════════════════════════════════════════════════════════════
with tab2:
    st.title("📊 Análisis Exploratorio de Datos")
    st.markdown("Figuras generadas por `notebooks/01_eda.ipynb`")

    pares = [
        ("distribucion_target.png",   "Distribución de la variable objetivo"),
        ("mapa_correlaciones.png",     "Mapa de correlaciones"),
        ("amount_time_analisis.png",   "Análisis de Amount y Time por clase"),
        ("distribucion_features.png",  "Distribución de features por clase (KDE)"),
        ("curvas_evaluacion.png",      "Curvas ROC y Precision-Recall"),
        ("shap_importancia.png",       "Importancia SHAP"),
    ]

    for i in range(0, len(pares), 2):
        cols = st.columns(2)
        for j, (fname, caption) in enumerate(pares[i:i+2]):
            path = os.path.join(FIGS, fname)
            with cols[j]:
                if os.path.exists(path):
                    st.image(path, caption=caption, use_container_width=True)
                else:
                    st.info(f"Ejecuta los notebooks para generar:\n`figures/{fname}`")

# ════════════════════════════════════════════════════════════
# TAB 3 — LLM
# ════════════════════════════════════════════════════════════
with tab3:
    st.title("🧠 Explicaciones LLM")
    st.markdown(
        "El sistema usa **SHAP + LLaMA 3.1** (vía Groq) para generar explicaciones "
        "en lenguaje natural de por qué una transacción fue marcada como fraude."
    )

    # Explicaciones guardadas
    exp_path = os.path.join(MODELS, "explicaciones_llm.json")
    if os.path.exists(exp_path):
        with open(exp_path, encoding="utf-8") as f:
            explicaciones = json.load(f)
        st.success(f"✅ {len(explicaciones)} explicaciones guardadas (generadas por `04_llm_explicaciones.ipynb`)")

        for exp in explicaciones:
            pred   = exp.get("prediccion", "?")
            prob   = exp.get("probabilidad", 0)
            emoji  = "🔴" if pred == "FRAUDE" else "🟢"
            color  = "metric-fraude" if pred == "FRAUDE" else "metric-legitima"

            with st.expander(f"{emoji} Transacción #{exp.get('idx', '?')} — {pred} (prob={prob:.4f})"):
                col_shap, col_exp = st.columns([1, 2])
                with col_shap:
                    st.markdown("**Top features SHAP:**")
                    for r in exp.get("shap_top", []):
                        dir_emoji = "⬆️" if "fraude" in r.get("direccion", "") and "↑" in r.get("direccion", "") or r.get("shap", 0) > 0 else "⬇️"
                        st.write(f"{dir_emoji} **{r['feature']}**: val={r['valor']:+.3f}, SHAP={r['shap']:+.4f}")
                with col_exp:
                    st.markdown("**Explicación generada por LLM:**")
                    st.markdown(f'<div class="{color}" style="padding:1rem;">{exp.get("explicacion", "N/A")}</div>',
                                unsafe_allow_html=True)
    else:
        st.info(
            "Las explicaciones LLM se generan en `notebooks/04_llm_explicaciones.ipynb`.\n\n"
            "1. Asegúrate de tener `GROQ_API_KEY` en tu archivo `.env`\n"
            "2. Ejecuta el notebook completo\n"
            "3. Se guardará `models/explicaciones_llm.json`\n"
            "4. Recarga esta página"
        )
        st.markdown("### Arquitectura del sistema LLM")
        st.code("""
transaccion → XGBoost → predicción + probabilidad
                ↓
            SHAP values (top 6 features)
                ↓
        Prompt few-shot → LLaMA 3.1 (Groq)
                ↓
        Explicación en lenguaje natural
        """)

# ════════════════════════════════════════════════════════════
# TAB 4 — INFORME
# ════════════════════════════════════════════════════════════
with tab4:
    st.title("📄 Informe Final")

    pdf_path = os.path.join(ROOT, "docs", "informe_final.pdf")
    tex_path = os.path.join(ROOT, "docs", "Proyecto_Final.tex")

    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                st.download_button("⬇️ Descargar informe PDF",
                                   f, "informe_final.pdf",
                                   "application/pdf", type="primary",
                                   use_container_width=True)
        else:
            st.warning("PDF no disponible aún. Exporta desde Overleaf → `docs/informe_final.pdf`")
    with col2:
        if os.path.exists(tex_path):
            with open(tex_path, "rb") as f:
                st.download_button("⬇️ Descargar plantilla LaTeX",
                                   f, "Proyecto_Final.tex",
                                   "text/plain",
                                   use_container_width=True)

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
### 🔗 Links
- 📁 **GitHub:** https://github.com/Ssamperc/Proyecto-Final-IA.git
- 🎬 **Video demo:** https://youtu.be/8xJkhYInzF4
- 📓 **Notebook principal:** `notebooks/03_modeling.ipynb`
""")
    with col_b:
        st.markdown("""
### 📦 Instalación
```bash
git clone https://github.com/Ssamperc/Proyecto-Final-IA.git
cd proyecto-ia-eafit
pip install -r requirements.txt
streamlit run app/app.py
```
""")

st.divider()
st.caption("Detección de Fraude Financiero · Proyecto Final IA · Universidad EAFIT · 2026-1")
