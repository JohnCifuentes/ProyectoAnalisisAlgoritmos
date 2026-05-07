"""
src/analytics/risk_classifier.py
==================================
Clasificación de activos financieros según su nivel de riesgo,
basada en la volatilidad anualizada histórica.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITERIO DE CLASIFICACIÓN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌──────────────┬──────────────────────┬─────────────────────────────┐
  │  Categoría   │  Umbral              │  Activos esperados          │
  ├──────────────┼──────────────────────┼─────────────────────────────┤
  │ Conservador  │ σ_anual < 20%        │ GLD, SPY, VOO, XLF          │
  │ Moderado     │ 20% ≤ σ_anual < 40%  │ AAPL, MSFT, AMZN, QQQ, IWM │
  │ Agresivo     │ σ_anual ≥ 40%        │ NVDA, TSLA, UBER, EC        │
  └──────────────┴──────────────────────┴─────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FUNDAMENTO DE LOS UMBRALES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Umbral Conservador → Moderado: 20% (0.20)
    La volatilidad histórica del S&P 500 oscila entre 15% y 20% en
    períodos "normales". Activos por debajo de ese nivel (GLD, bonos
    de renta fija) presentan menor riesgo que el mercado en conjunto.
    Este umbral es consistente con la clasificación de la SEC para
    fondos de bajo riesgo.

  Umbral Moderado → Agresivo: 40% (0.40)
    Activos con volatilidad entre 20% y 40% son acciones blue-chip y
    ETFs diversificados que superan el mercado pero mantienen un
    perfil de riesgo manejable. Por encima de 40%, la volatilidad
    supera 2× la del mercado; estos activos requieren alta tolerancia
    al riesgo (NVDA ~60%, TSLA ~70%, UBER ~50% en el período 2021-2026).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPLEJIDAD COMPUTACIONAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  classify_risk():            O(1) — dos comparaciones escalares.
  classify_risk_with_score(): O(1).
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


# ====================================================================== #
# Umbrales y categorías                                                    #
# ====================================================================== #

# Expresados en decimales (no porcentaje): 0.20 = 20%, 0.40 = 40%
THRESHOLD_CONSERVATIVE: float = 0.20   # σ_anual < 20%  → Conservador
THRESHOLD_MODERATE: float     = 0.40   # σ_anual < 40%  → Moderado
                                        # σ_anual ≥ 40%  → Agresivo

RISK_CONSERVATIVE: str = "Conservador"
RISK_MODERATE: str     = "Moderado"
RISK_AGGRESSIVE: str   = "Agresivo"

# Orden numérico ordinal (útil para ordenamientos y comparaciones)
RISK_ORDER: Dict[str, int] = {
    RISK_CONSERVATIVE: 1,
    RISK_MODERATE:     2,
    RISK_AGGRESSIVE:   3,
}

# Descripción de cada categoría (para reportes y dashboards futuros)
RISK_DESCRIPTIONS: Dict[str, str] = {
    RISK_CONSERVATIVE: (
        "Volatilidad anualizada < 20%. Activo de bajo riesgo, "
        "adecuado para perfiles de inversión conservadores."
    ),
    RISK_MODERATE: (
        "Volatilidad anualizada entre 20% y 40%. Activo de riesgo "
        "medio, adecuado para perfiles moderados."
    ),
    RISK_AGGRESSIVE: (
        "Volatilidad anualizada ≥ 40%. Activo de alto riesgo, "
        "adecuado solo para perfiles agresivos con alta tolerancia."
    ),
}


# ====================================================================== #
# Función principal de clasificación                                       #
# ====================================================================== #

def classify_risk(annualized_vol: float) -> str:
    """
    Clasifica un activo según su volatilidad anualizada.

    Lógica:
        if   σ_anual < 0.20  →  "Conservador"
        elif σ_anual < 0.40  →  "Moderado"
        else                 →  "Agresivo"

    Complejidad: O(1).

    Args:
        annualized_vol: Volatilidad anualizada en decimales
                        (ej. 0.25 para 25%).

    Returns:
        Una de las tres cadenas: "Conservador", "Moderado" o "Agresivo".

    Raises:
        ValueError: Si la volatilidad es negativa.
    """
    if annualized_vol < 0:
        raise ValueError(
            f"La volatilidad anualizada no puede ser negativa: {annualized_vol}"
        )

    if annualized_vol < THRESHOLD_CONSERVATIVE:
        return RISK_CONSERVATIVE
    elif annualized_vol < THRESHOLD_MODERATE:
        return RISK_MODERATE
    else:
        return RISK_AGGRESSIVE


def classify_risk_with_score(annualized_vol: float) -> Dict:
    """
    Clasifica el riesgo y devuelve también la puntuación ordinal (1-3)
    y la descripción de la categoría.

    Complejidad: O(1).

    Args:
        annualized_vol: Volatilidad anualizada en decimales.

    Returns:
        Dict con:
          - category (str):        nombre de la categoría de riesgo.
          - risk_score (int):      1 = conservador, 2 = moderado, 3 = agresivo.
          - annualized_vol (float): valor de entrada.
          - description (str):     explicación de la categoría.
    """
    category = classify_risk(annualized_vol)
    return {
        "category":       category,
        "risk_score":     RISK_ORDER[category],
        "annualized_vol": annualized_vol,
        "description":    RISK_DESCRIPTIONS[category],
    }
