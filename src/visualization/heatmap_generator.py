"""
src/visualization/heatmap_generator.py
=========================================
Genera la figura Plotly para el heatmap de correlación de Pearson.

Responsabilidades:
  - Transformar la matriz 2D de coeficientes Pearson en un JSON Plotly.
  - Generar la leyenda de colores y anotaciones de texto.
  - Extraer los pares más y menos correlacionados.
  - Preparar versión compacta (sin anotaciones) para el dashboard.

NO calcula correlaciones — delega en CorrelationMatrix.
"""

import math
import logging
from typing import Any, Dict, List, Optional, Tuple

from src.visualization.visualization_utils import (
    COLORS,
    FONT_FAMILY,
    dark_plotly_layout,
    to_json_safe,
)

logger = logging.getLogger(__name__)

# Escala de colores para correlaciones en [-1, 1]:
#   -1 → rojo (movimientos contrarios)
#    0 → gris neutro
#   +1 → verde (movimientos paralelos)
CORR_COLORSCALE = [
    [0.00, "#f85149"],   # -1: rojo agresivo
    [0.25, "#d29922"],   # -0.5: ámbar
    [0.50, "#c9d1d9"],   # 0:  gris neutro
    [0.75, "#58a6ff"],   # +0.5: azul acento
    [1.00, "#3fb950"],   # +1: verde conservador
]


def build_heatmap_figure(
    tickers: List[str],
    matrix: List[List[Optional[float]]],
    compact: bool = False,
    title: str = "Matriz de Correlación de Pearson (retornos diarios)",
) -> Dict[str, Any]:
    """
    Construye el diccionario Plotly para un heatmap de correlación.

    Args:
        tickers: Lista de tickers en el orden de la matriz.
        matrix:  Lista 2D n×n de coeficientes de Pearson.
        compact: Si True, genera una versión sin anotaciones para el dashboard.
        title:   Título del gráfico.

    Returns:
        Dict con claves 'data' y 'layout', compatible con Plotly.js.
    """
    n = len(tickers)

    # ── Serializar valores (None → None, floats → float) ─────────────
    z_vals: List[List[Optional[float]]] = []
    for row in matrix:
        z_vals.append([
            round(v, 4) if (v is not None and not math.isnan(v)) else None
            for v in row
        ])

    # ── Anotaciones de texto (solo versión completa) ──────────────────
    if compact:
        text_template = ""
        annot_font_size = 0
    else:
        text_template = "%{text}"
        annot_font_size = 9

    text_matrix = [
        [f"{v:.2f}" if v is not None else "" for v in row]
        for row in z_vals
    ]

    trace = {
        "type":  "heatmap",
        "x":     tickers,
        "y":     tickers,
        "z":     z_vals,
        "text":  text_matrix if not compact else None,
        "texttemplate": text_template,
        "textfont":     {"size": annot_font_size, "color": COLORS["bg"]},
        "colorscale":   CORR_COLORSCALE,
        "zmid":  0.0,
        "zmin":  -1.0,
        "zmax":  1.0,
        "hovertemplate": (
            "<b>%{y}</b> → <b>%{x}</b><br>"
            "Pearson r = <b>%{z:.4f}</b><extra></extra>"
        ),
        "showscale": True,
        "colorbar": {
            "title":     {"text": "r", "side": "right", "font": {"color": COLORS["text_muted"]}},
            "tickvals":  [-1, -0.5, 0, 0.5, 1],
            "ticktext":  ["-1", "-0.5", "0", "+0.5", "+1"],
            "tickfont":  {"color": COLORS["text_muted"], "size": 10},
            "len":       0.8,
            "thickness": 12,
        },
    }

    height = 440 if compact else 680

    layout = dark_plotly_layout(
        title="" if compact else title,
        height=height,
        margin={"l": 80, "r": 80, "t": 50 if compact else 70, "b": 100},
        xaxis_extra={
            "tickangle": -45,
            "tickfont":  {"size": 9 if compact else 11, "color": COLORS["text_muted"]},
            "side":      "bottom",
        },
        yaxis_extra={
            "autorange": "reversed",
            "tickfont":  {"size": 9 if compact else 11, "color": COLORS["text_muted"]},
        },
    )
    layout["plot_bgcolor"] = COLORS["bg"]  # sin grid en heatmap

    return {"data": [trace], "layout": layout}


def extract_extremes(
    tickers: List[str],
    matrix: List[List[Optional[float]]],
    top_n: int = 5,
) -> Dict[str, Any]:
    """
    Extrae los pares con mayor y menor correlación (excluyendo la diagonal).

    Args:
        tickers: Lista de tickers.
        matrix:  Matriz de correlación.
        top_n:   Número de pares en cada extremo.

    Returns:
        Dict con:
          - most_correlated:  lista de top_n pares (valor más alto).
          - least_correlated: lista de top_n pares (valor más bajo).
          - top_pair:         par con mayor correlación absoluta.
          - bottom_pair:      par con menor correlación.
    """
    n = len(tickers)
    pairs: List[Tuple[float, str, str]] = []

    for i in range(n):
        for j in range(i + 1, n):
            v = matrix[i][j]
            if v is None:
                continue
            if isinstance(v, float) and math.isnan(v):
                continue
            pairs.append((v, tickers[i], tickers[j]))

    if not pairs:
        return {"most_correlated": [], "least_correlated": [], "top_pair": None, "bottom_pair": None}

    pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)

    def _fmt(triple):
        return {
            "ticker_a": triple[1],
            "ticker_b": triple[2],
            "value":    round(triple[0], 6),
        }

    most_list  = [_fmt(p) for p in pairs_sorted[:top_n]]
    least_list = [_fmt(p) for p in pairs_sorted[-top_n:]]

    return {
        "most_correlated":  most_list,
        "least_correlated": least_list,
        "top_pair":         most_list[0]  if most_list  else None,
        "bottom_pair":      least_list[0] if least_list else None,
    }
