"""
src/analytics/pattern_detector.py
===================================
Detección de patrones de precio en series temporales financieras,
usando el algoritmo de ventana deslizante de analytics/sliding_window.py.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATRÓN #1 — "N días consecutivos al alza"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Definición:
    El precio de cierre sube durante n días consecutivos:
        close[i] > close[i-1]  ∀ i = 1, 2, ..., n

  Algoritmo:
    Ventana de tamaño (n + 1). Para cada ventana se verifica que
    todos los precios sean estrictamente crecientes.
    Si: vals[0] < vals[1] < ... < vals[n]  →  registrar ocurrencia.

  Aplicación financiera:
    Señal de momentum alcista a corto plazo. Frecuente en activos
    como NVDA, TSLA o SPY durante mercados alcistas.

  Complejidad:
    Temporal: O(n × min_days) ≈ O(n) para min_days constante y pequeño.
    Espacial: O(k) donde k = número de ocurrencias detectadas.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATRÓN #2 — "Caída fuerte seguida de recuperación"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Definición formal:
    Sean p_{i-1}, p_i, p_{i+1} tres precios consecutivos.

        r_caída        = (p_i   − p_{i-1}) / p_{i-1}
        r_recuperación = (p_{i+1} − p_i  ) / p_i

    El patrón se activa cuando:
        r_caída        ≤ −drop_threshold      (ej. ≤ −3%)
        r_recuperación ≥  recovery_threshold  (ej. ≥  +2%)

  Algoritmo:
    Ventana de tamaño 3 (tres precios consecutivos) para calcular
    ambos retornos sin pre-computar la serie completa.

  Aplicación financiera:
    Patrón de reversión a corto plazo ("V-recovery"). Indica que el
    mercado sobrereaccionó a la baja y se corrigió al día siguiente.
    Frecuente en reacciones a noticias macroeconómicas negativas o
    eventos de volatilidad extrema (ej. Flash Crash, earnings shocks).

  Umbrales por defecto (ajustables vía parámetros):
    drop_threshold:     0.03  (caída ≥ 3%)
    recovery_threshold: 0.02  (recuperación ≥ 2%)
    Elegidos empíricamente: -3% es el percentil ~5% de retornos diarios
    negativos para activos del portafolio; +2% asegura recuperación real.

  Complejidad:
    Temporal: O(n).
    Espacial: O(k) donde k = número de ocurrencias detectadas.
"""

import logging
from typing import Dict, List

import pandas as pd

from src.analytics.sliding_window import generate_windows

logger = logging.getLogger(__name__)

# ====================================================================== #
# Parámetros por defecto                                                   #
# ====================================================================== #

DEFAULT_CONSECUTIVE_DAYS: int   = 3       # días consecutivos al alza
DEFAULT_DROP_THRESHOLD: float   = 0.03    # caída mínima 3%
DEFAULT_RECOVERY_THRESHOLD: float = 0.02  # recuperación mínima 2%


# ====================================================================== #
# Patrón #1: N días consecutivos al alza                                  #
# ====================================================================== #

def detect_consecutive_gains(
    prices: pd.Series,
    min_days: int = DEFAULT_CONSECUTIVE_DAYS,
) -> Dict:
    """
    Detecta ocurrencias de `min_days` días consecutivos con precio al alza.

    Algoritmo:
        1. Genera ventanas de tamaño (min_days + 1).         — O(n)
        2. En cada ventana verifica precios estrictamente     — O(min_days)
           crecientes: vals[0] < vals[1] < ... < vals[min_days].
        3. Registra inicio, fin, precio de inicio, precio final y ganancia.

    Complejidad temporal: O(n × min_days) ≈ O(n) para min_days constante.
    Complejidad espacial: O(k) donde k = número de ocurrencias.

    Args:
        prices:   Serie de precios de cierre con índice datetime,
                  ordenada ascendentemente.
        min_days: Número mínimo de días consecutivos al alza (por defecto 3).

    Returns:
        Dict con:
          - pattern (str):       identificador del patrón.
          - description (str):   descripción legible.
          - min_days (int):      parámetro utilizado.
          - n_occurrences (int): total de ocurrencias encontradas.
          - total_windows (int): total de ventanas evaluadas.
          - frequency (float):   n_occurrences / total_windows ∈ [0, 1].
          - occurrences (list):  lista de dicts con detalles de cada evento.

    Raises:
        ValueError: Si min_days < 1 o la serie tiene < (min_days + 1) precios.
    """
    if min_days < 1:
        raise ValueError(
            f"min_days debe ser ≥ 1. Recibido: {min_days}"
        )

    ws = min_days + 1  # Necesitamos (min_days) diferencias → (min_days+1) precios

    if len(prices) < ws:
        raise ValueError(
            f"La serie debe tener al menos {ws} precios para detectar "
            f"{min_days} días consecutivos al alza. "
            f"Recibido: {len(prices)}"
        )

    occurrences: List[Dict] = []
    total_windows = 0

    for window in generate_windows(prices, ws):
        total_windows += 1
        vals  = window.values
        dates = window.index

        # Verificar tendencia estrictamente creciente — O(min_days)
        is_consecutive = True
        for i in range(1, len(vals)):
            if vals[i] <= vals[i - 1]:
                is_consecutive = False
                break

        if is_consecutive:
            start_price = float(vals[0])
            end_price   = float(vals[-1])
            gain_pct = (
                (end_price - start_price) / start_price * 100
                if start_price != 0.0 else float("nan")
            )
            occurrences.append({
                "start_date":     str(dates[0].date()) if hasattr(dates[0], "date") else str(dates[0]),
                "end_date":       str(dates[-1].date()) if hasattr(dates[-1], "date") else str(dates[-1]),
                "n_days":         min_days,
                "start_price":    round(start_price, 4),
                "end_price":      round(end_price, 4),
                "total_gain_pct": round(gain_pct, 4),
            })

    frequency = len(occurrences) / total_windows if total_windows > 0 else 0.0

    return {
        "pattern":       f"consecutive_gains_{min_days}d",
        "description":   f"{min_days} días consecutivos al alza",
        "min_days":      min_days,
        "n_occurrences": len(occurrences),
        "total_windows": total_windows,
        "frequency":     round(frequency, 6),
        "occurrences":   occurrences,
    }


# ====================================================================== #
# Patrón #2: Caída fuerte seguida de recuperación rápida                  #
# ====================================================================== #

def detect_drop_and_recovery(
    prices: pd.Series,
    drop_threshold: float     = DEFAULT_DROP_THRESHOLD,
    recovery_threshold: float = DEFAULT_RECOVERY_THRESHOLD,
) -> Dict:
    """
    Detecta el patrón "caída fuerte seguida de recuperación rápida".

    Definición formal:
        Sean p_{i-1}, p_i, p_{i+1} tres precios consecutivos.

            r_caída        = (p_i   − p_{i-1}) / p_{i-1}
            r_recuperación = (p_{i+1} − p_i  ) / p_i

        Patrón detectado cuando:
            r_caída        ≤ −drop_threshold       (ej. ≤ −3%)
            r_recuperación ≥  recovery_threshold   (ej. ≥  +2%)

    Algoritmo:
        Ventana de tamaño 3. Calcula los dos retornos directamente
        dentro de la ventana sin pre-computar la serie de retornos.
        Complejidad: O(n).

    Args:
        prices:             Serie de precios con índice datetime.
        drop_threshold:     Umbral de caída en decimal (ej. 0.03 para 3%).
        recovery_threshold: Umbral de recuperación en decimal (ej. 0.02 para 2%).

    Returns:
        Dict con:
          - pattern, description, drop_threshold, recovery_threshold.
          - n_occurrences, total_windows, frequency.
          - occurrences: lista de dicts con fechas y porcentajes del evento.

    Raises:
        ValueError: Si algún umbral es ≤ 0 o la serie tiene < 3 precios.
    """
    if drop_threshold <= 0:
        raise ValueError(
            f"drop_threshold debe ser positivo (ej. 0.03 para 3%). "
            f"Recibido: {drop_threshold}"
        )
    if recovery_threshold <= 0:
        raise ValueError(
            f"recovery_threshold debe ser positivo (ej. 0.02 para 2%). "
            f"Recibido: {recovery_threshold}"
        )
    if len(prices) < 3:
        raise ValueError(
            f"Se necesitan al menos 3 precios para detectar el patrón. "
            f"Recibido: {len(prices)}"
        )

    occurrences: List[Dict] = []
    total_windows = 0

    for window in generate_windows(prices, 3):
        total_windows += 1
        vals  = window.values
        dates = window.index

        p_prev    = float(vals[0])
        p_drop    = float(vals[1])
        p_recover = float(vals[2])

        # Evitar división por cero
        if p_prev == 0.0 or p_drop == 0.0:
            continue

        r_drop    = (p_drop    - p_prev) / p_prev
        r_recover = (p_recover - p_drop) / p_drop

        if r_drop <= -drop_threshold and r_recover >= recovery_threshold:
            occurrences.append({
                "reference_date":  str(dates[0].date()) if hasattr(dates[0], "date") else str(dates[0]),
                "drop_date":       str(dates[1].date()) if hasattr(dates[1], "date") else str(dates[1]),
                "recovery_date":   str(dates[2].date()) if hasattr(dates[2], "date") else str(dates[2]),
                "drop_pct":        round(r_drop    * 100, 4),
                "recovery_pct":    round(r_recover * 100, 4),
                "price_before":    round(p_prev,    4),
                "price_at_drop":   round(p_drop,    4),
                "price_recovered": round(p_recover, 4),
            })

    frequency = len(occurrences) / total_windows if total_windows > 0 else 0.0

    return {
        "pattern":            "drop_and_recovery",
        "description":        (
            f"Caída ≥{drop_threshold * 100:.0f}% seguida de "
            f"recuperación ≥{recovery_threshold * 100:.0f}%"
        ),
        "drop_threshold":     drop_threshold,
        "recovery_threshold": recovery_threshold,
        "n_occurrences":      len(occurrences),
        "total_windows":      total_windows,
        "frequency":          round(frequency, 6),
        "occurrences":        occurrences,
    }
