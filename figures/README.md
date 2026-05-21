# 📊 Figuras del Proyecto

Todas las figuras están generadas programáticamente a 180 dpi.

| Archivo | Descripción | Sección informe |
|---|---|---|
| `fig1_eda.png` | Distribución variable objetivo + correlaciones con Class | §2 EDA |
| `fig2_arquitectura.png` | Diagrama de arquitectura general del sistema | §3 Arquitectura |
| `fig3_roc_pr.png` | Curvas ROC y Precision-Recall — 3 modelos comparados | §5.2 Resultados |
| `fig4_confusion.png` | Matriz de confusión normalizada — XGBoost (Exp. 2) | §5.3 Resultados |
| `fig5_comparacion.png` | Barras comparativas Accuracy/F1/AUC-ROC por modelo | Apéndice A |
| `fig6_shap.png` | Importancia media \|SHAP\| top-10 features | §5.3 Resultados |

## Regenerar figuras

```bash
python3 scripts/generar_figuras.py
```

O ejecutar las celdas de visualización en `notebooks/03_modeling.ipynb`
y `notebooks/04_llm_explicaciones.ipynb`.
