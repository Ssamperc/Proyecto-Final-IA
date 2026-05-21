"""
llm_explainer.py — Generador de explicaciones en lenguaje natural
Proyecto: Detección de Fraude Financiero con ML + LLM Explicativo
Proyecto Final IA · EAFIT 2026-1
Responsable: Juan Pablo Duque
"""

import os
import json
import numpy as np
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL    = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

SISTEMA = """Eres un analista experto en detección de fraude financiero.
Tu tarea es explicar en lenguaje claro y conciso (máximo 120 palabras) por qué
un modelo de machine learning clasificó una transacción como sospechosa.
Basa tu explicación ÚNICAMENTE en los valores SHAP proporcionados.
Usa un tono profesional pero accesible. No inventes datos adicionales.
Termina con una recomendación de acción (BLOQUEAR / REVISAR / APROBAR)."""

FEW_SHOT = """
=== EJEMPLO 1 (fraude) ===
Features: V4=-3.21 SHAP=+0.42 | V11=+2.87 SHAP=+0.31 | hour=2.3 SHAP=+0.18
Prob fraude: 0.94
Explicación: Esta transacción presenta señales claras de fraude. V4 inusualmente
bajo (-3.21) y V11 elevado, combinados con horario de madrugada, explican la alta
probabilidad. Recomendación: BLOQUEAR.

=== EJEMPLO 2 (legítima) ===
Features: V4=+1.12 SHAP=-0.15 | log_amount=3.22 SHAP=-0.12 | hour=14.5 SHAP=-0.09
Prob fraude: 0.03
Explicación: Patrón consistente con compras legítimas. Monto bajo a horario normal.
Recomendación: APROBAR.
"""


def llamar_llm(prompt: str, max_tokens: int = 350) -> str:
    """Llama al LLM via Groq o retorna respuesta demo."""
    if not GROQ_API_KEY:
        return (
            "[MODO DEMO — configura GROQ_API_KEY en .env para respuestas reales]\n"
            "Esta transacción fue marcada como sospechosa porque los componentes PCA "
            "principales (V4 y V11) muestran valores atípicos respecto al historial "
            "de transacciones legítimas. El horario y el monto contribuyen secundariamente. "
            "Recomendación: REVISAR con el equipo de fraude."
        )
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        resp   = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SISTEMA},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error LLM: {e}]"


def explicar_transaccion(
    idx: int,
    X: "np.ndarray",
    shap_values: "np.ndarray",
    feature_names: list,
    modelo,
    top_k: int = 6,
) -> dict:
    """
    Genera una explicación en lenguaje natural para la transacción idx.

    Returns:
        dict con keys: idx, prediccion, probabilidad, shap_top, explicacion
    """
    fila       = X[idx]
    shap_fila  = shap_values[idx]
    if shap_fila.ndim > 1:
        shap_fila = shap_fila[:, 1]
    prob       = float(modelo.predict_proba(X[idx : idx + 1])[0, 1])
    pred       = int(prob >= 0.5)

    top_idx  = np.argsort(np.abs(shap_fila))[::-1][:top_k]
    shap_top = [
        {
            "feature":    feature_names[i] if i < len(feature_names) else f"f{i}",
            "valor":      round(float(fila[i]), 4),
            "shap":       round(float(shap_fila[i]), 4),
            "direccion":  "↑ fraude" if shap_fila[i] > 0 else "↓ fraude",
        }
        for i in top_idx
    ]

    features_str = "\n".join(
        f"  - {r['feature']}: {r['valor']:+.3f}  SHAP={r['shap']:+.4f}  ({r['direccion']})"
        for r in shap_top
    )
    prompt = (
        f"{FEW_SHOT}\n"
        f"=== TRANSACCIÓN A EXPLICAR ===\n"
        f"Features relevantes:\n{features_str}\n"
        f"Probabilidad de fraude: {prob:.4f}\n"
        f"Predicción: {'FRAUDE' if pred else 'LEGÍTIMA'}\n\n"
        f"Explicación:"
    )

    return {
        "idx":          idx,
        "prediccion":   "FRAUDE" if pred else "LEGÍTIMA",
        "probabilidad": round(prob, 4),
        "shap_top":     shap_top,
        "explicacion":  llamar_llm(prompt),
    }


def guardar_explicaciones(explicaciones: list,
                           path: str = "models/explicaciones_llm.json") -> None:
    def serial(o):
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    with open(path, "w", encoding="utf-8") as f:
        json.dump(explicaciones, f, default=serial, ensure_ascii=False, indent=2)
    print(f"✅ Explicaciones guardadas en {path}")
