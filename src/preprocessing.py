"""
preprocessing.py — Pipeline de preprocesamiento reutilizable
Proyecto: Detección de Fraude Financiero con ML + LLM Explicativo
Proyecto Final IA · EAFIT 2026-1
Responsable: Nombre Completo 1
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica transformaciones de feature engineering al dataset.
    - Convierte Time a hora del día
    - Aplica log1p a Amount
    - Elimina columnas originales reemplazadas
    """
    df = df.copy()
    df["hour_of_day"] = (df["Time"] % 86400) / 3600
    df["log_amount"]  = np.log1p(df["Amount"].fillna(df["Amount"].median()))
    df = df.drop(columns=["Time", "Amount"], errors="ignore")
    return df


def construir_pipeline(num_cols: list) -> ColumnTransformer:
    """
    Construye el pipeline de preprocesamiento numérico.
    Imputación con mediana + StandardScaler.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    return ColumnTransformer(
        transformers=[("num", num_pipeline, num_cols)],
        remainder="drop"
    )


def preparar_datos(df: pd.DataFrame, target: str = "Class"):
    """
    Aplica feature engineering y separa X e y.

    Returns:
        X (DataFrame), y (Series), num_cols (list)
    """
    df = feature_engineering(df)
    X  = df.drop(columns=[target])
    y  = df[target]
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    return X, y, num_cols


def guardar_pipeline(pipeline, path: str = "models/preprocessor.joblib") -> None:
    joblib.dump(pipeline, path)
    print(f"✅ Pipeline guardado en {path}")


def cargar_pipeline(path: str = "models/preprocessor.joblib"):
    pipeline = joblib.load(path)
    print(f"✅ Pipeline cargado desde {path}")
    return pipeline


def resumen_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna tabla con conteo y porcentaje de nulos por columna."""
    nulos = df.isnull().sum()
    pct   = (nulos / len(df) * 100).round(3)
    return (
        pd.DataFrame({"Nulos": nulos, "Porcentaje (%)": pct})
        .query("Nulos > 0")
        .sort_values("Porcentaje (%)", ascending=False)
    )


def detectar_outliers_iqr(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Detecta outliers por método IQR."""
    rows = []
    for col in cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR    = Q3 - Q1
        n_out  = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
        rows.append({"Feature": col, "N_outliers": n_out,
                     "Pct": round(n_out / len(df) * 100, 2)})
    return pd.DataFrame(rows).sort_values("Pct", ascending=False)
