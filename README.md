# 🤖 Detección de Fraude Financiero con ML + LLM Explicativo
### Proyecto Final · Inteligencia Artificial · EAFIT 2026-1

> **Pregunta de investigación:** ¿Puede un sistema basado en XGBoost + SHAP + LLM detectar
> transacciones fraudulentas con AUC-ROC > 0.90 y generar explicaciones en lenguaje natural?
>
> **Respuesta:** ✅ Sí. AUC-ROC = 0.9166, F1 = 0.6452, superando al baseline en +0.093 AUC y +0.367 F1.

---

## 👥 Equipo

| Integrante | Correo | Rama Git | Responsabilidad |
|---|---|---|---|
| David Ruiz | druizv1@eafit.edu.co | `feat/eda` | EDA, preprocesamiento, visualizaciones |
| David Quintero | dquinterg1@eafit.edu.co | `feat/modelo` | Arquitectura, entrenamiento, evaluación |
| Juan Pablo Duque | Jpduqueo@eafit.edu.co | `feat/llm` | LLM/SHAP, integración, app Streamlit |
| Samuel Samper | ssamperc@gmail.com | `feat/evaluacion` | Métricas, análisis, informe LaTeX |

---

## 📊 Resultados principales

| Modelo | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Baseline — Regresión Logística | 0.9430 | 0.2661 | 0.2920 | 0.2785 | 0.8236 |
| Experimento 1 — Random Forest | 0.9700 | 0.6211 | 0.5221 | 0.5673 | 0.9172 |
| **Experimento 2 — XGBoost ✓** | **0.9743** | **0.6731** | **0.6195** | **0.6452** | **0.9166** |

**CV 5-fold (XGBoost):** AUC-ROC = 0.912 ± 0.009  
**Evaluación LLM:** 17/20 casos correctos (85% coherencia SHAP-narrativa)

---

## 📦 Dataset

- **Nombre:** Credit Card Fraud Detection
- **Fuente:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Registros:** 284,807 transacciones (septiembre 2013)
- **Features:** V1–V28 (PCA anonimizada), Time, Amount
- **Desbalance:** 99.83% legítimas / 0.17% fraudes (ratio 284:1)

---

## ⚙️ Instalación rápida

```bash
# 1. Clonar el repositorio
git clone https://github.com/Ssamperc/proyecto-ia-eafit.git
cd proyecto-ia-eafit

# 2. Instalar dependencias (Python 3.10+)
pip install -r requirements.txt
```

### Variables de entorno (componente LLM)
```bash
cp .env.example .env
# Editar .env → poner tu GROQ_API_KEY (gratis en console.groq.com)
```

---

## 🚀 Cómo correr el proyecto

### 1. — Notebooks en orden (reproducción completa)
```bash
jupyter notebook
# Ejecutar en este orden:
# 01_eda.ipynb             → análisis exploratorio
# 02_preprocessing.ipynb  → limpieza, SMOTE, splits
# 03_modeling.ipynb        → entrenamiento y comparación de modelos
# 04_llm_explicaciones.ipynb → SHAP + explicaciones LLM
```

### 2. — Demo interactiva Streamlit
```bash
streamlit run app/app.py
# Abre automáticamente en http://localhost:8501
# Permite ingresar una transacción y obtener predicción + explicación LLM
```

> ⚠️ Para usar el componente LLM necesitas una `GROQ_API_KEY` gratuita.
> Regístrate en https://console.groq.com — no requiere tarjeta de crédito.

---

## 🗂️ Estructura del proyecto

```
proyecto-ia-eafit/
├── notebooks/
│   ├── 01_eda.ipynb                  ← EDA completo (David Ruiz)
│   ├── 02_preprocessing.ipynb        ← Limpieza, SMOTE, splits 70/15/15 (David Ruiz)
│   ├── 03_modeling.ipynb             ← Entrenamiento y comparación (David Quintero)
│   └── 04_llm_explicaciones.ipynb    ← SHAP + LLM narrativo (Juan Pablo Duque)
├── src/
│   ├── preprocessing.py              ← Pipeline sklearn reutilizable
│   ├── model.py                      ← Definición y entrenamiento de modelos
│   ├── evaluate.py                   ← Métricas y visualizaciones comparativas
│   └── llm_explainer.py              ← Generador de explicaciones LLM (few-shot)
├── app/
│   └── app.py                        ← Demo Streamlit interactiva
├── data/
│   └── README.md                     ← Instrucciones para descargar el dataset
├── figures/                          ← Figuras generadas (PNG, 180 dpi)
│   ├── fig1_eda.png                  ← Distribución objetivo + correlaciones
│   ├── fig2_arquitectura.png         ← Diagrama de arquitectura del sistema
│   ├── fig3_roc_pr.png               ← Curvas ROC y Precision-Recall
│   ├── fig4_confusion.png            ← Matriz de confusión (XGBoost)
│   ├── fig5_comparacion.png          ← Barras comparativas de modelos
│   └── fig6_shap.png                 ← Importancia SHAP features
├── models/
│   ├── modelo_final.joblib           ← XGBoost serializado
│   ├── preprocessor.joblib           ← Pipeline preprocesamiento
│   ├── metricas_finales.json         ← Métricas del test set
│   └── feature_names.csv             ← Nombres de features
├── docs/
│   ├── Proyecto_Final.tex            ← Informe LaTeX (compilar en Overleaf)
│   ├── informe_final.pdf             ← ← PDF compilado (entregable obligatorio)
│   └── README.md                     ← Instrucciones de compilación LaTeX
├── requirements.txt
├── .env.example
├── .gitignore
└── COMMITS.md                        ← Historial de ramas y commits del equipo
```

---

## 🎥 Entregables

- 📄 **Informe:** `docs/informe_final.pdf`
- 🎬 **Video demo (≤ 3 min):** https://youtu.be/8xJkhYInzF4 ← *actualizar antes de entregar*
- 🔗 **Repositorio:** https://github.com/Ssamperc/proyecto-ia-eafit ← *actualizar con usuario real*


---

## GUIA STREAMLIT

<img width="1918" height="1143" alt="image" src="https://github.com/user-attachments/assets/6edc27bd-3dd2-4787-bb84-c5d7e4206d94" />

Aqui se mueven los sliders para simular la transaccion, luego se presiona en analizar para que diga cuanto porcentaje tiene de ser fraudulenta.

<img width="1919" height="1134" alt="image" src="https://github.com/user-attachments/assets/242995ab-b607-4652-97ab-d573f100f664" />

Luego de ejecutar el primer notebook solo entrando a la seccion de analisis exploratorio de datos se van a ver todas las graficas.

<img width="1919" height="1139" alt="image" src="https://github.com/user-attachments/assets/153be933-80ad-459b-9164-87809e9bdb22" />

Luego de ejecutar el notebook #4 se debe recargar la pagina para poder ver las explicaciones en texto natural de llm.

<img width="1915" height="1031" alt="image" src="https://github.com/user-attachments/assets/be0aebfb-6fa3-4908-8b3b-70e7ded1424e" />

En esta ultima seccion se podra descargar el informe final tanto en PDF como en Latex, ademas estan los enlaces a este repositorio y al video demo.

## 🔬 Metodología resumida

1. **EDA:** Análisis de desbalance (284:1), correlaciones, nulos, outliers en `Amount`
2. **Preprocesamiento:** `log_amount`, `hour_of_day`, imputación mediana por clase, StandardScaler, SMOTE
3. **Modelado:** Baseline (LR) → Exp.1 (Random Forest) → Exp.2 (XGBoost con early stopping)
4. **Explicabilidad:** SHAP TreeExplainer + LLM few-shot (llama-3.1-8b vía Groq)
5. **Evaluación LLM:** 20 casos con ground truth → 85% coherencia SHAP-narrativa
6. **Demo:** App Streamlit con predicción interactiva, visualización SHAP y explicaciones LLM

---

*Universidad EAFIT · Escuela de Ciencias Aplicadas e Ingeniería · Inteligencia Artificial · 2026-1*
