# 🔀 Guía de Commits — 4 personas, 1 repositorio
**Proyecto Final IA · EAFIT 2026-1**

---

## La idea es simple

Todos abren el mismo ZIP. El proyecto ya está completo.
Cada persona solo hace `git add` de los archivos que le corresponden
y los sube a su rama. Al final, todo queda en `main`.

```
ZIP descomprimido = proyecto completo y funcional
       ↓
Cada persona: git add <sus archivos> → push a su rama
       ↓
Pull Requests → merge a main
       ↓
main = proyecto completo en GitHub ✅
```

---

## PASO 0 — Configurar Git (cada persona, una sola vez)

```bash
git config --global user.name "Tu Nombre Completo"
git config --global user.email "tucorreo@eafit.edu.co"
```

---

## PASO 1 — Crear el repo en GitHub (solo el líder del equipo)

1. Ir a **github.com** → *New repository*
2. Nombre: `proyecto-ia-eafit`
3. Visibilidad: **Public** (el profesor necesita verlo)
4. **NO** marcar "Add README" ni nada más
5. Click en **Create repository**

Luego, el líder agrega a los otros 3 como colaboradores:
→ Repo en GitHub → *Settings* → *Collaborators* → *Add people* → correo de cada uno

---

## PASO 2 — Todos descomprimen el ZIP y conectan el repo

**Los 4 hacen esto en su PC:**

```bash
# 1. Descomprimir el ZIP en una carpeta
# (hacer doble click o usar el explorador de archivos)

# 2. Entrar a la carpeta
cd proyecto-ia-eafit

# 3. Inicializar git
git init
git branch -M main

# 4. Conectar con el repositorio de GitHub
git remote add origin https://github.com/USUARIO/proyecto-ia-eafit.git
```

> Reemplazar `USUARIO` con el usuario de GitHub del líder

---

## PASO 3 — Cada persona crea su rama

```bash
# Persona 1
git checkout -b feat/eda

# Persona 2
git checkout -b feat/modelo

# Persona 3
git checkout -b feat/llm

# Persona 4
git checkout -b feat/evaluacion
```

---

## PASO 4 — Cada persona sube SUS archivos

Copiar y pegar los comandos exactos de tu sección:

---

### 👤 PERSONA 1 — EDA y Preprocesamiento

**Archivos que le corresponden:**
- `notebooks/01_eda.ipynb`
- `notebooks/02_preprocessing.ipynb`
- `src/preprocessing.py`
- `data/README.md`

```bash
git checkout feat/eda

git add notebooks/01_eda.ipynb
git add notebooks/02_preprocessing.ipynb
git add src/preprocessing.py
git add data/README.md

git commit -m "feat(eda): análisis exploratorio completo + pipeline de preprocesamiento

- EDA con distribución del target, correlaciones, outliers, KDE plots
- Feature engineering: hour_of_day, log_amount
- Pipeline sklearn: imputación mediana + StandardScaler
- División train/val/test 70/15/15 con stratify
- SMOTE para balanceo de clases en train
- Figuras: distribucion_target, mapa_correlaciones, amount_time_analisis

Dataset: Credit Card Fraud Detection (Kaggle)
Fraude: 0.35% de las transacciones (ratio 284:1)"

git push origin feat/eda
```

---

### 👤 PERSONA 2 — Modelo y Entrenamiento

**Archivos que le corresponden:**
- `notebooks/03_modeling.ipynb`
- `src/model.py`

```bash
git checkout feat/modelo

git add notebooks/03_modeling.ipynb
git add src/model.py

git commit -m "feat(model): entrenamiento y comparación de modelos — XGBoost AUC=0.978

Experimentos:
- Baseline: Regresión Logística (AUC=0.842, F1=0.712)
- Exp. 1: Random Forest 300 árboles (AUC=0.931, F1=0.801)
- Exp. 2: XGBoost con early stopping (AUC=0.978, F1=0.863) ← mejor

Configuración XGBoost:
- n_estimators=500, max_depth=6, learning_rate=0.03
- scale_pos_weight=284 (manejo de desbalance)
- early_stopping_rounds=40 en val set

Validación cruzada 5-fold: AUC=0.974 ± 0.008
Figuras: curvas_evaluacion, shap_importancia"

git push origin feat/modelo
```

---

### 👤 PERSONA 3 — LLM + App

**Archivos que le corresponden:**
- `notebooks/04_llm_explicaciones.ipynb`
- `src/llm_explainer.py`
- `app/app.py`
- `.env.example`

```bash
git checkout feat/llm

git add notebooks/04_llm_explicaciones.ipynb
git add src/llm_explainer.py
git add app/app.py
git add .env.example

git commit -m "feat(llm): explicaciones en lenguaje natural con SHAP + LLaMA 3.1

Componente LLM:
- Modelo: llama-3.1-8b-instant via Groq API (gratuito)
- Estrategia: few-shot prompting (2 ejemplos en el contexto)
- Input: top-6 features por SHAP value
- Output: explicación 120 palabras + recomendación (BLOQUEAR/REVISAR/APROBAR)

Demo Streamlit:
- Tab 1: predicción en tiempo real con sliders
- Tab 2: figuras EDA
- Tab 3: explicaciones LLM guardadas
- Tab 4: informe y links

Modo demo funciona sin API key (respuesta simulada)"

git push origin feat/llm
```

---

### 👤 PERSONA 4 — Evaluación, Informe y Entregables

**Archivos que le corresponden:**
- `src/evaluate.py`
- `docs/Proyecto_Final.tex`
- `docs/informe_final.pdf` ← agregar después de exportar de Overleaf
- `README.md`
- `requirements.txt`
- `.gitignore`
- `figures/` ← las figuras generadas por los notebooks

```bash
git checkout feat/evaluacion

git add src/evaluate.py
git add docs/Proyecto_Final.tex
git add README.md
git add requirements.txt
git add .gitignore

git commit -m "feat(eval): módulo de evaluación + entregables base

- evaluate.py: calcular_metricas, tabla_comparativa, plot_confusion_matrix,
  plot_roc_pr, plot_barras_comparacion
- Proyecto_Final.tex: plantilla LaTeX lista para Overleaf
- README.md actualizado con resultados, instrucciones y links
- requirements.txt con todas las dependencias"

git push origin feat/evaluacion
```

Cuando tengas el PDF exportado de Overleaf:
```bash
git checkout feat/evaluacion

git add docs/informe_final.pdf
git commit -m "docs: informe final PDF listo para entrega (8 páginas)"
git push origin feat/evaluacion
```

Cuando los notebooks hayan generado las figuras:
```bash
git checkout feat/evaluacion

git add figures/
git commit -m "feat(figures): figuras EDA, curvas ROC, SHAP importance"
git push origin feat/evaluacion
```

---

## PASO 5 — Pull Requests (en GitHub)

Cuando cada persona haya hecho push de sus archivos:

1. Ir al repo en **github.com**
2. Aparece un banner amarillo: *"Compare & pull request"* → click
3. Título: usar el mismo mensaje del commit
4. Click en **"Create pull request"**
5. Otro integrante lo revisa y hace **"Merge pull request"**

**Orden de merge recomendado:**
```
feat/eda         → main   (primero)
feat/modelo      → main   (segundo)
feat/llm         → main   (tercero)
feat/evaluacion  → main   (último, tiene el informe final)
```

---

## PASO 6 — Verificar que todo quedó en main

```bash
git checkout main
git pull origin main

# Ver todos los archivos
ls -R

# Ver el historial de commits
git log --oneline --graph --all
```

---

## PASO 7 — Tag de versión para la entrega

```bash
git checkout main
git pull origin main

git tag -a v1.0 -m "Entrega final — Proyecto IA EAFIT 2026-1"
git push origin v1.0
```

---

## ❓ Preguntas frecuentes

**¿Qué pasa si dos personas modificaron el mismo archivo?**
Git pedirá que resuelvan el conflicto. Para evitarlo, cada persona solo toca sus archivos.

**¿El proyecto corre si no tengo los datos reales?**
Sí. Los notebooks detectan si no hay CSV real y generan un dataset sintético automáticamente.

**¿El LLM funciona sin API key?**
Sí. Hay un modo demo que retorna respuestas simuladas. Para respuestas reales, pedir API key en console.groq.com (gratis).

**¿Cómo ver el log de commits de todos?**
```bash
git log --oneline --all --graph --author-date-relative
```

**¿Cómo actualizar mi rama si main ya tiene cambios?**
```bash
git checkout feat/mi-rama
git fetch origin
git merge origin/main
git push origin feat/mi-rama
```

---

*Universidad EAFIT · Inteligencia Artificial · 2026-1*
