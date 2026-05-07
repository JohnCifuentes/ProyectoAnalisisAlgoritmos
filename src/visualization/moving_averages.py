"""
src/visualization/moving_averages.py
======================================
Implementación manual de Medias Móviles Simples (SMA) para series de precios.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FUNDAMENTO MATEMÁTICO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

La Media Móvil Simple de período w en el instante t es:

    SMA_t = (1/w) · Σ_{i=0}^{w-1} P_{t-i}
          = (P_t + P_{t-1} + ... + P_{t-w+1}) / w

donde P_k es el precio de cierre en el día k.

Para t < w-1 (los primeros w-1 días) la SMA no está definida (None/NaN).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALGORITMO (VENTANA DESLIZANTE O(n))
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

En lugar del algoritmo de fuerza bruta O(n·w):
    sma[i] = sum(prices[i-w+1 : i+1]) / w     ← O(w) por iteración

Usamos una suma deslizante que mantiene el estado entre iteraciones:
    Para i en [0, n-1]:
        window_sum += prices[i]          ← agregar nuevo elemento
        if i >= w:                        ← quitar elemento saliente
            window_sum -= prices[i - w]
        if i >= w - 1:
            sma[i] = window_sum / w       ← registrar promedio

Cada elemento es procesado exactamente 2 veces: una al entrar y una al salir.
Complejidad total: O(n) tiempo, O(n) espacio.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESTRICCIONES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  NO se usa pandas.rolling().mean() ni ninguna función estadística de librería.
  La fórmula es explícita en el bucle.
"""

import logging
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def compute_sma(prices: List[float], period: int) -> List[Optional[float]]:
    """
    Calcula la Media Móvil Simple manualmente con ventana deslizante O(n).

    Algoritmo de suma deslizante (sliding window sum):
      - Se mantiene window_sum con la suma de los últimos `period` precios.
      - Al avanzar i: se suma prices[i] y se resta prices[i - period].
      - El promedio se registra cuando i >= period - 1.

    Args:
        prices: Lista de precios de cierre ordenada cronológicamente.
        period: Número de días de la ventana (w ≥ 1).

    Returns:
        Lista de longitud igual a `prices`.
        Posiciones 0 … period-2 contienen None (ventana incompleta).
        Posiciones period-1 … n-1 contienen el promedio flotante.

    Raises:
        ValueError: Si period < 1 o la lista está vacía.

    Complejidad:
        Temporal: O(n) — cada elemento es visitado exactamente 2 veces.
        Espacial: O(n) — lista de salida del mismo tamaño que la entrada.
    """
    if period < 1:
        raise ValueError(f"El período debe ser ≥ 1, se recibió: {period}")

    n = len(prices)
    if n == 0:
        raise ValueError("La lista de precios no puede estar vacía.")

    result: List[Optional[float]] = [None] * n
    window_sum: float = 0.0

    for i in range(n):
        # ── Agregar el elemento que entra en la ventana ──────────────
        window_sum += float(prices[i])

        # ── Quitar el elemento que sale de la ventana ────────────────
        # Cuando i >= period, el elemento prices[i - period] ya salió.
        if i >= period:
            window_sum -= float(prices[i - period])

        # ── Registrar SMA cuando la ventana está completa ────────────
        # La ventana está completa cuando tenemos exactamente `period` elementos,
        # es decir, cuando i >= period - 1.
        if i >= period - 1:
            result[i] = window_sum / period

    logger.debug(
        f"compute_sma(period={period}): {n} precios → "
        f"{sum(1 for v in result if v is not None)} valores SMA calculados"
    )
    return result


def compute_sma_series(price_series: pd.Series, period: int) -> pd.Series:
    """
    Versión de compute_sma que trabaja con pd.Series y preserva el índice.

    Convierte la serie a lista, aplica compute_sma y devuelve una nueva
    Series con el mismo índice datetime que la serie de entrada.

    Args:
        price_series: Serie de precios con índice datetime ordenado.
        period:       Número de días de la ventana.

    Returns:
        pd.Series con los valores SMA. Los primeros period-1 valores son NaN.
    """
    prices_list = price_series.values.tolist()
    sma_values  = compute_sma(prices_list, period)

    # Convertir None a float('nan') para compatibilidad con pandas
    sma_float = [v if v is not None else float("nan") for v in sma_values]

    return pd.Series(sma_float, index=price_series.index, name=f"SMA_{period}")


def compute_multiple_smas(
    price_series: pd.Series,
    periods: List[int],
) -> dict:
    """
    Calcula varias SMAs sobre la misma serie de precios en una sola función.

    Eficiencia: cada SMA se calcula en O(n), por lo que el total es O(n·k)
    donde k es el número de períodos (generalmente k ≤ 4).

    Args:
        price_series: Serie de precios con índice datetime.
        periods:      Lista de períodos (ej. [20, 50, 200]).

    Returns:
        Dict {period: pd.Series} con las SMAs calculadas.
    """
    return {p: compute_sma_series(price_series, p) for p in sorted(set(periods))}
