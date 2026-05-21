# 📘 Guía de Usuario — Sistema de Detección de Fraude Financiero

**Proyecto Final · Inteligencia Artificial · EAFIT 2026-1**

---

## ¿Qué hace este sistema?

Este sistema detecta transacciones fraudulentas con tarjeta de crédito y genera una **explicación en lenguaje natural** de por qué la transacción es sospechosa, usando el modelo XGBoost + SHAP + LLM (llama-3.1-8b vía Groq).

---

## Requisitos previos

- Python 3.10 o superior
- Cuenta gratuita en [console.groq.com](https://console.groq.com) para obtener `GROQ_API_KEY`
- El dataset de Kaggle (ver `data/README.md`)

---

## Instalación paso a paso

```bash
# 1. Clonar el repositorio
git clone https://github.com/USUARIO/proyecto-ia-eafit.git
cd proyecto-ia-eafit

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar variables de entorno
cp .env.example .env
# Abrir .env con cualquier editor y pegar tu GROQ_API_KEY
```

---

## Opción 1 — Demo Streamlit (recomendada)

```bash
streamlit run app/app.py
```

Se abrirá automáticamente en `http://localhost:8501`.

### Flujo de la aplicación

1. **Ingresar los valores de la transacción** en el panel izquierdo (sliders para V1–V28, Time y Amount)
2. **Hacer clic en "Analizar transacción"**
3. El sistema muestra:
   - 🔴 / 🟢 Predicción (FRAUDE / LEGÍTIMA) con probabilidad
   - 📊 Gráfico de barras SHAP con las features más influyentes
   - 💬 Explicación del LLM en lenguaje natural
   - ✅ Recomendación: BLOQUEAR / REVISAR / APROBAR

---

## Opción 2 — Notebooks (reproducción completa)

Ejecutar en orden desde Jupyter:

| Notebook | Contenido | Tiempo estimado |
|---|---|---|
| `01_eda.ipynb` | Análisis exploratorio completo | ~5 min |
| `02_preprocessing.ipynb` | Limpieza, SMOTE, splits | ~3 min |
| `03_modeling.ipynb` | Entrenamiento 3 modelos + evaluación | ~10 min (con GPU) |
| `04_llm_explicaciones.ipynb` | SHAP + explicaciones LLM | ~5 min |

```bash
jupyter notebook
# Navegar a la carpeta notebooks/ y ejecutar en el orden indicado
```

---

## Opción 3 — Usar las funciones desde Python

```python
from src.model import load_model
from src.llm_explainer import generate_explanation
import pandas as pd

# Cargar modelo pre-entrenado
model, preprocessor = load_model()

# Crear una transacción de ejemplo
transaction = pd.DataFrame([{
    'V1': -1.36, 'V2': -0.07, 'V3': 2.54, 'V4': 1.38,
    'V5': -0.34, 'V6': 0.46, 'V7': 0.24, 'V8': 0.10,
    'V9': 0.36, 'V10': 0.09, 'V11': -0.55, 'V12': -0.62,
    'V13': -0.99, 'V14': -0.31, 'V15': 1.47, 'V16': -0.47,
    'V17': 0.21, 'V18': 0.03, 'V19': 0.40, 'V20': 0.25,
    'V21': -0.02, 'V22': 0.28, 'V23': -0.11, 'V24': 0.07,
    'V25': 0.13, 'V26': -0.19, 'V27': 0.13, 'V28': -0.02,
    'Time': 8000, 'Amount': 149.62
}])

# Predecir
X_proc = preprocessor.transform(transaction)
prob = model.predict_proba(X_proc)[0][1]
print(f"Probabilidad de fraude: {prob:.2%}")

# Generar explicación LLM
explanation = generate_explanation(model, preprocessor, transaction)
print(explanation)
```

---

## Preguntas frecuentes

**¿Necesito descargar el dataset para usar la demo Streamlit?**  
No. La app usa el modelo ya entrenado (`models/modelo_final.joblib`). Solo necesitas el dataset si quieres re-entrenar desde cero con los notebooks.

**¿Cuánto cuesta la API de Groq?**  
Es gratuita para uso académico. El plan free incluye suficientes llamadas para el proyecto.

**¿El sistema funciona en tiempo real?**  
La predicción XGBoost tarda <1 ms. La explicación LLM tarda ~1-3 segundos por transacción vía API Groq.

**¿Puedo cambiar el umbral de decisión?**  
Sí. En `app/app.py` hay un slider para ajustar el umbral entre 0.1 y 0.9. Por defecto está calibrado al valor que maximiza F1 en validación.

---

*Para problemas o preguntas: contactar al equipo vía los correos en el README.*
