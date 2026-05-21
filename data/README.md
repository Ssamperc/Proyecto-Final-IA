# 📦 Dataset — Detección de Fraude Financiero

| Atributo | Descripción |
|---|---|
| **Nombre** | Credit Card Fraud Detection |
| **Fuente** | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| **Registros** | 284,807 transacciones |
| **Features** | V1–V28 (PCA), Time, Amount |
| **Target** | Class (0=legítima, 1=fraude) |
| **Tamaño** | ~150 MB |
| **Desbalance** | 99.65% / 0.35% |

## Cómo descargar los datos

### Opción A — Kaggle CLI
```bash
pip install kaggle
# Poner kaggle.json en ~/.kaggle/
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
```

### Opción B — Manual
1. Ir a https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Descargar `creditcard.csv`
3. Mover a `data/raw/creditcard.csv`

### Sin datos reales
Los notebooks generan un dataset sintético automáticamente si no hay CSV.

## Estructura esperada
```
data/
├── raw/
│   └── creditcard.csv
└── processed/          ← generado por 02_preprocessing.ipynb
    ├── X_train.npy
    ├── X_val.npy
    ├── X_test.npy
    └── ...
```
