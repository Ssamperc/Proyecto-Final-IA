# 📄 Documentación — Instrucciones de compilación LaTeX

## Compilar el informe en Overleaf (recomendado)

1. Ir a [overleaf.com](https://overleaf.com) → crear cuenta gratuita
2. **New Project → Upload Project** → subir `Proyecto_Final.tex`
3. Subir **todas las imágenes** de la carpeta `figures/` (Upload → cada PNG)
4. Menú → Compiler → seleccionar **pdfLaTeX**
5. Presionar **Recompile** (botón verde)
6. Menú → **Download PDF** → guardar como `informe_final.pdf`
7. Subir `informe_final.pdf` a este repositorio en `docs/`

## Estructura de archivos necesarios en Overleaf

```
Proyecto_Final.tex          ← archivo principal
../figures/fig1_eda.png
../figures/fig2_arquitectura.png
../figures/fig3_roc_pr.png
../figures/fig4_confusion.png
../figures/fig5_comparacion.png
../figures/fig6_shap.png
```

> **Nota:** Las rutas `../figures/` en el `.tex` asumen la estructura
> del repositorio. En Overleaf, subir las imágenes al mismo nivel o
> ajustar las rutas a solo `fig1_eda.png`, etc.

## Compilar localmente (alternativa)

```bash
cd docs/
pdflatex Proyecto_Final.tex
pdflatex Proyecto_Final.tex  # segunda pasada para referencias
```

Requiere: TeX Live o MiKTeX con paquetes `babel`, `palatino`, `booktabs`,
`tabularx`, `listings`, `hyperref`, `natbib`, `pgfplots`.
