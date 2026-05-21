# 🤖 Modelos serializados

## Archivos

| Archivo | Descripción | Tamaño |
|---|---|---|
| `modelo_final.joblib` | XGBoost (Exp. 2) — modelo principal serializado con joblib | ~5 MB |
| `preprocessor.joblib` | Pipeline sklearn: imputación + StandardScaler | ~50 KB |
| `metricas_finales.json` | Métricas del test set (JSON) | ~1 KB |
| `feature_names.csv` | Nombres de las 30 features en orden | ~1 KB |

## Métricas del modelo final (XGBoost)

```json
{
  "modelo": "XGBoost (Gradient Boosting)",
  "accuracy": 0.9743,
  "precision": 0.6731,
  "recall": 0.6195,
  "f1": 0.6452,
  "auc_roc": 0.9166
}
```

## Cargar y usar el modelo

```python
import joblib
import numpy as np

# Cargar preprocesador y modelo
preprocessor = joblib.load('models/preprocessor.joblib')
modelo = joblib.load('models/modelo_final.joblib')

# Predicción sobre nuevos datos (misma estructura que creditcard.csv)
# X_new: DataFrame con columnas V1-V28, log_amount, hour_of_day
X_proc = preprocessor.transform(X_new)
probabilidad_fraude = modelo.predict_proba(X_proc)[:, 1]
prediccion = (probabilidad_fraude >= 0.265).astype(int)  # umbral óptimo
```

## Nota sobre el .gitignore

Los archivos `.joblib` están en `.gitignore` por su tamaño.
Para compartir el modelo, usar una de estas alternativas:
- **Git LFS:** `git lfs track "*.joblib"`
- **Google Drive / HuggingFace Hub**
- Ejecutar `notebooks/03_modeling.ipynb` para regenerar
