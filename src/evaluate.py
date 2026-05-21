"""
evaluate.py — Métricas y visualizaciones de evaluación
Proyecto: Detección de Fraude Financiero con ML + LLM Explicativo
Proyecto Final IA · EAFIT 2026-1
Responsable: Samuel Samper Cardona
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score,
    classification_report, confusion_matrix,
    RocCurveDisplay, PrecisionRecallDisplay,
)


def calcular_metricas(y_true, y_pred, y_prob=None, nombre: str = "Modelo") -> dict:
    """Calcula métricas principales de clasificación binaria."""
    m = {
        "Modelo":     nombre,
        "Accuracy":   round(accuracy_score(y_true, y_pred), 4),
        "Precision":  round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":     round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1":         round(f1_score(y_true, y_pred, zero_division=0), 4),
        "AUC-ROC":    round(roc_auc_score(y_true, y_prob), 4) if y_prob is not None else None,
    }
    return m


def tabla_comparativa(lista_metricas: list) -> pd.DataFrame:
    """Genera DataFrame comparativo de múltiples modelos."""
    cols = ["Modelo", "Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
    df   = pd.DataFrame(lista_metricas)[cols]
    return df.set_index("Modelo")


def plot_confusion_matrix(y_true, y_pred,
                           labels=("Legítima", "Fraude"),
                           titulo: str = "Matriz de Confusión",
                           ax=None, save_path: str = None):
    """Matriz de confusión con valores absolutos y porcentajes."""
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    annot   = np.array([[f"{v}\n({p:.1%})" for v, p in zip(rv, rp)]
                        for rv, rp in zip(cm, cm_norm)])
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues", ax=ax,
                xticklabels=[f"Pred: {l}" for l in labels],
                yticklabels=[f"Real: {l}" for l in labels],
                linewidths=1.5, linecolor="white", vmin=0, vmax=1)
    ax.set_title(titulo, fontweight="bold")
    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()


def plot_roc_pr(y_true, y_prob_dict: dict, save_path: str = None):
    """
    Curvas ROC y PR para múltiples modelos.

    Args:
        y_prob_dict: {'Nombre modelo': array_probs, ...}
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#003d79", "#9b1b1b", "#b45309", "#006b3c"]

    for i, (nombre, y_prob) in enumerate(y_prob_dict.items()):
        auc = roc_auc_score(y_true, y_prob)
        RocCurveDisplay.from_predictions(
            y_true, y_prob, ax=axes[0],
            color=colors[i % len(colors)],
            name=f"{nombre} ({auc:.3f})"
        )
        PrecisionRecallDisplay.from_predictions(
            y_true, y_prob, ax=axes[1],
            color=colors[i % len(colors)],
            name=nombre
        )

    axes[0].plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="Random")
    axes[0].set_title("Curvas ROC — Test Set", fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[1].set_title("Curvas Precision-Recall", fontweight="bold")
    axes[1].legend(fontsize=8)

    plt.suptitle("Evaluación comparativa de modelos", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Curvas guardadas en {save_path}")
    plt.show()


def plot_barras_comparacion(lista_metricas: list, save_path: str = None):
    """Gráfico de barras comparando modelos."""
    df = tabla_comparativa(lista_metricas).reset_index()
    metricas_plot = ["Accuracy", "F1", "AUC-ROC"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x  = np.arange(len(df))
    w  = 0.22
    colors = ["#003d79", "#009650", "#b45309"]

    for i, (met, col) in enumerate(zip(metricas_plot, colors)):
        ax.bar(x + i * w, df[met], w, label=met, color=col,
               edgecolor="white", linewidth=1.2)

    ax.set_xticks(x + w)
    ax.set_xticklabels(df["Modelo"], rotation=10, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Comparación de modelos — Test Set", fontweight="bold", pad=12)
    ax.legend()
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Comparación guardada en {save_path}")
    plt.show()
