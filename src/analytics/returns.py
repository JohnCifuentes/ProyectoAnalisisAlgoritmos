"""
src/analytics/returns.py
=========================
Cálculo manual de retornos financieros diarios.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FUNDAMENTO MATEMÁTICO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Retorno simple (porcentual):
    r_t = (P_t − P_{t-1}) / P_{t-1}

Retorno logarítmico (para uso futuro):
    r_t^{log} = ln(P_t / P_{t-1})

Por qué retornos y no precios:
  Los precios absolutos de activos con distinto nivel (ej. NVDA ≈ 800 USD
  vs AVAL ≈ 3 USD) hacen imposible comparar volatilidades sin normalización.
  Los retornos porcentuales eliminan ese sesgo de escala y permiten
  comparar cualquier par de activos directamente.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPLEJIDAD COMPUTACIONAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Temporal: O(n) — un recorrido lineal sobre los n precios.
  Espacial: O(n) — se construye una nueva serie de n-1 retornos.
"""

import logging
import math
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ====================================================================== #
# Retornos simples                                                         #
# ====================================================================== #

def compute_simple_returns(
    prices: pd.Series,
    drop_na: bool = True,
) -> pd.Series:
    """
    Calcula los retornos diarios simples de una serie de precios.

    Fórmula:
        r_t = (P_t − P_{t-1}) / P_{t-1}

    El primer elemento se descarta porque carece de P_{t-1}.
    Si P_{t-1} = 0, el retorno se registra como NaN (división indefinida).

    Complejidad temporal: O(n).
    Complejidad espacial: O(n).

    Args:
        prices: Serie de pandas con precios de cierre e índice datetime,
                ordenada ascendentemente por fecha.
        drop_na: Si True (predeterminado), elimina NaN del resultado.

    Returns:
        Serie de retornos diarios; su índice comienza en el segundo día
        de la serie original.

    Raises:
        ValueError: Si la serie tiene menos de 2 precios.
    """
    if prices is None:
        raise ValueError("La serie de precios no puede ser None.")
    if len(prices) < 2:
        raise ValueError(
            f"Se necesitan al menos 2 precios para calcular retornos. "
            f"Recibido: {len(prices)}"
        )

    n = len(prices)
    values = prices.values          # ndarray — acceso indexado O(1)
    index  = prices.index[1:]       # Índice del resultado (sin día 0)

    returns_list: list = []

    # Recorrido manual — O(n)
    for i in range(1, n):
        p_prev = float(values[i - 1])
        p_curr = float(values[i])

        if p_prev == 0.0:
            # División por cero: retorno indefinido
            returns_list.append(float("nan"))
        else:
            returns_list.append((p_curr - p_prev) / p_prev)

    result = pd.Series(returns_list, index=index, name=prices.name)

    if drop_na:
        result = result.dropna()

    return result


# ====================================================================== #
# Retornos logarítmicos (preparado para Req. #4 y análisis estadístico)   #
# ====================================================================== #

def compute_log_returns(
    prices: pd.Series,
    drop_na: bool = True,
) -> pd.Series:
    """
    Calcula los retornos logarítmicos diarios.

    Fórmula:
        r_t^{log} = ln(P_t / P_{t-1})

    Ventajas sobre retornos simples:
      - Son aditivos en el tiempo: r_{1→T} = Σ r_t^{log}
      - Su distribución es más simétrica (closer to normal).
      - Útiles para modelos estadísticos y de riesgo avanzados.

    Complejidad temporal: O(n).
    Complejidad espacial: O(n).

    Args:
        prices: Serie de pandas con precios de cierre e índice datetime.
        drop_na: Si True, elimina NaN del resultado.

    Returns:
        Serie de retornos logarítmicos.

    Raises:
        ValueError: Si la serie tiene menos de 2 precios.
    """
    if prices is None:
        raise ValueError("La serie de precios no puede ser None.")
    if len(prices) < 2:
        raise ValueError(
            f"Se necesitan al menos 2 precios para calcular retornos. "
            f"Recibido: {len(prices)}"
        )

    n = len(prices)
    values = prices.values
    index  = prices.index[1:]

    returns_list: list = []

    # Recorrido manual — O(n)
    for i in range(1, n):
        p_prev = float(values[i - 1])
        p_curr = float(values[i])

        if p_prev <= 0.0 or p_curr <= 0.0:
            # ln indefinido para precios no positivos
            returns_list.append(float("nan"))
        else:
            returns_list.append(math.log(p_curr / p_prev))

    result = pd.Series(returns_list, index=index, name=prices.name)

    if drop_na:
        result = result.dropna()

    return result
