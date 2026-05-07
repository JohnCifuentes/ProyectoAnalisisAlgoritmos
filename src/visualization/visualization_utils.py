"""
src/visualization/visualization_utils.py
==========================================
Utilidades compartidas para el módulo de visualización.

Provee:
  - Serialización de tipos numpy/pandas a Python nativo (para JSON).
  - Plantilla base de layout Plotly con tema oscuro consistente con la app.
  - Helpers de formateo de ejes y colores.
"""

import math
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ====================================================================== #
# Colores del tema oscuro (consistentes con styles.css)                    #
# ====================================================================== #

COLORS = {
    "bg":            "#0d1117",
    "bg_secondary":  "#161b22",
    "bg_tertiary":   "#21262d",
    "border":        "#30363d",
    "text":          "#c9d1d9",
    "text_strong":   "#f0f6fc",
    "text_muted":    "#8b949e",
    "accent":        "#58a6ff",
    "conservative":  "#3fb950",   # verde
    "moderate":      "#d29922",   # ámbar
    "aggressive":    "#f85149",   # rojo
    "sma_20":        "#f1c40f",   # amarillo
    "sma_50":        "#e67e22",   # naranja
    "sma_100":       "#9b59b6",   # morado
    "sma_200":       "#3498db",   # azul
}

FONT_FAMILY = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', "
    "Helvetica, Arial, sans-serif"
)


# ====================================================================== #
# Tema base de Plotly                                                       #
# ====================================================================== #

def dark_plotly_layout(
    title: str = "",
    height: int = 500,
    margin: Optional[Dict] = None,
    show_legend: bool = True,
    xaxis_extra: Optional[Dict] = None,
    yaxis_extra: Optional[Dict] = None,
) -> Dict:
    """
    Devuelve un diccionario de layout Plotly con el tema oscuro del proyecto.

    Todos los componentes de visualización deben usar esta función como base
    para garantizar consistencia visual a lo largo de la aplicación.

    Args:
        title:       Título del gráfico.
        height:      Altura en píxeles del gráfico.
        margin:      Márgenes personalizados {l, r, t, b}.
        show_legend: Si mostrar la leyenda.
        xaxis_extra: Configuración adicional del eje X.
        yaxis_extra: Configuración adicional del eje Y.

    Returns:
        Dict de layout compatible con Plotly.js y plotly Python.
    """
    base = {
        "title": {
            "text":  title,
            "font":  {"color": COLORS["text_strong"], "size": 16, "family": FONT_FAMILY},
            "x":     0.02,
        },
        "paper_bgcolor": COLORS["bg"],
        "plot_bgcolor":  COLORS["bg_secondary"],
        "font": {
            "color":  COLORS["text"],
            "family": FONT_FAMILY,
            "size":   12,
        },
        "height": height,
        "margin": margin or {"l": 60, "r": 40, "t": 60, "b": 60},
        "showlegend": show_legend,
        "legend": {
            "bgcolor":     COLORS["bg_tertiary"],
            "bordercolor": COLORS["border"],
            "borderwidth": 1,
            "font":        {"color": COLORS["text"], "size": 11},
        },
        "xaxis": {
            "gridcolor":    COLORS["bg_tertiary"],
            "linecolor":    COLORS["border"],
            "zerolinecolor": COLORS["border"],
            "tickfont":     {"color": COLORS["text_muted"], "size": 11},
            "title":        {"font": {"color": COLORS["text_muted"]}},
        },
        "yaxis": {
            "gridcolor":    COLORS["bg_tertiary"],
            "linecolor":    COLORS["border"],
            "zerolinecolor": COLORS["border"],
            "tickfont":     {"color": COLORS["text_muted"], "size": 11},
            "title":        {"font": {"color": COLORS["text_muted"]}},
        },
        "hovermode": "closest",
        "hoverlabel": {
            "bgcolor":     COLORS["bg_tertiary"],
            "bordercolor": COLORS["border"],
            "font":        {"color": COLORS["text"], "size": 12},
        },
    }

    if xaxis_extra:
        base["xaxis"].update(xaxis_extra)
    if yaxis_extra:
        base["yaxis"].update(yaxis_extra)

    return base


# ====================================================================== #
# Serialización JSON-segura                                                 #
# ====================================================================== #

def to_json_safe(obj: Any) -> Any:
    """
    Convierte recursivamente tipos numpy/pandas a Python nativo para jsonify().

    Numpy floats, integers y arrays no son serializables por defecto;
    esta función los convierte antes de que Flask intente serializar.

    Args:
        obj: Objeto a convertir (puede ser dict, list, numpy type, etc.)

    Returns:
        Versión JSON-segura del objeto.
    """
    import numpy as np

    if obj is None:
        return None

    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v

    if isinstance(obj, np.ndarray):
        return [to_json_safe(x) for x in obj.tolist()]

    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj

    if hasattr(obj, "isoformat"):   # datetime, pd.Timestamp
        return str(obj)

    return obj


def sma_color(period: int) -> str:
    """Devuelve el color asignado a una SMA según su período."""
    mapping = {
        20:  COLORS["sma_20"],
        50:  COLORS["sma_50"],
        100: COLORS["sma_100"],
        200: COLORS["sma_200"],
    }
    return mapping.get(period, COLORS["accent"])
