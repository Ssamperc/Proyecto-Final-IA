"""
model.py — Definición y entrenamiento de modelos
Proyecto: Detección de Fraude Financiero con ML + LLM Explicativo
Proyecto Final IA · EAFIT 2026-1
Responsable: Nombre Completo 2
"""

import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


def crear_baseline(random_state: int = 42) -> LogisticRegression:
    """Baseline: Regresión Logística con manejo de desbalance."""
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=0.01,
        solver="lbfgs",
        random_state=random_state,
    )


def crear_random_forest(random_state: int = 42) -> RandomForestClassifier:
    """Experimento 1: Random Forest balanceado."""
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def crear_xgboost(scale_pos_weight: float = 1.0,
                  random_state: int = 42) -> XGBClassifier:
    """
    Experimento 2 (modelo principal): XGBoost.

    Args:
        scale_pos_weight: Ratio negativo/positivo.
                          Calcular: (y == 0).sum() / (y == 1).sum()
    """
    return XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight,
        reg_alpha=0.1,
        reg_lambda=1.5,
        eval_metric="aucpr",
        early_stopping_rounds=40,
        random_state=random_state,
        verbosity=0,
    )


def entrenar_xgboost(modelo: XGBClassifier,
                     X_train, y_train,
                     X_val,   y_val) -> XGBClassifier:
    """Entrena XGBoost con early stopping en validación."""
    modelo.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    best = modelo.best_iteration
    print(f"✅ XGBoost entrenado. Best iteration: {best}")
    return modelo


def validacion_cruzada(modelo, X, y,
                       cv: int = 5,
                       scoring: str = "roc_auc") -> dict:
    """K-fold estratificado. Retorna media y std del score."""
    skf    = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(modelo, X, y, cv=skf,
                             scoring=scoring, n_jobs=-1)
    result = {
        "scoring": scoring,
        "cv":      cv,
        "mean":    round(scores.mean(), 4),
        "std":     round(scores.std(), 4),
        "folds":   scores.tolist(),
    }
    print(f"  CV {scoring}: {result['mean']:.4f} ± {result['std']:.4f}")
    return result


def guardar_modelo(modelo, path: str = "models/modelo_final.joblib") -> None:
    joblib.dump(modelo, path)
    print(f"✅ Modelo guardado en {path}")


def cargar_modelo(path: str = "models/modelo_final.joblib"):
    modelo = joblib.load(path)
    print(f"✅ Modelo cargado desde {path}")
    return modelo
