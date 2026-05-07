"""
src/analytics/sliding_window.py
================================
Algoritmo genérico de ventana deslizante (sliding window) para series
temporales financieras.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONCEPTO ALGORÍTMICO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Para una serie S = [s_0, s_1, ..., s_{n-1}] y ventana de tamaño w:

    Ventana 0:  [s_0,   s_1,   ..., s_{w-1}  ]
    Ventana 1:  [s_1,   s_2,   ..., s_w       ]
    ...
    Ventana k:  [s_k,   s_{k+1}, ..., s_{k+w-1}]

  Número total de ventanas: n - w + 1

  El puntero derecho avanza de 0 a (n-w) de forma continua; cada
  elemento de la serie es visitado por al menos una ventana (extremos)
  y como máximo por w ventanas (elementos internos).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPLEJIDAD COMPUTACIONAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  generate_windows():
    Temporal: O(n) amortizado — cada llamada a .iloc[i:i+w] es O(w),
              pero sobre el ciclo completo resulta O(n·w). Para w fijo
              y pequeño (ej. w=3 o w=5) se considera O(n) en la práctica.
    Espacial: O(w) — solo existe una ventana activa en memoria a la vez
              (patrón generador: no materializa todas las ventanas).

  get_all_windows():
    Temporal: O(n·w).
    Espacial: O(n·w) — almacena todas las ventanas simultáneamente.

  window_statistics():
    Temporal: O(n·w) — calcula min/max/mean manualmente para cada ventana.
    Espacial: O(n) — una fila por ventana en el DataFrame de salida.
"""

import logging
from typing import Generator, List

import pandas as pd

logger = logging.getLogger(__name__)


# ====================================================================== #
# Generador de ventanas (núcleo del módulo)                               #
# ====================================================================== #

def generate_windows(
    series: pd.Series,
    window_size: int,
) -> Generator[pd.Series, None, None]:
    """
    Genera ventanas deslizantes de tamaño fijo sobre una serie de pandas.

    Utiliza slicing .iloc[i : i+window_size] para preservar el índice
    datetime original en cada sub-serie generada.

    Este módulo es el núcleo que usa pattern_detector.py: en lugar de
    pre-computar índices manualmente, el detector de patrones simplemente
    itera sobre este generador.

    Complejidad temporal: O(n) amortizado.
    Complejidad espacial: O(w) — patrón generador lazy.

    Args:
        series:      Serie de pandas con índice datetime y valores numéricos.
        window_size: Número de elementos por ventana (w ≥ 1).

    Yields:
        pd.Series de longitud window_size, con su índice datetime original.

    Raises:
        ValueError: Si window_size ≤ 0 o window_size > len(series).
    """
    n = len(series)

    if window_size <= 0:
        raise ValueError(
            f"window_size debe ser un entero positivo. Recibido: {window_size}"
        )
    if window_size > n:
        raise ValueError(
            f"window_size ({window_size}) no puede superar la longitud "
            f"de la serie ({n})."
        )

    # Recorrido O(n): el generador nunca almacena más de una ventana
    for i in range(n - window_size + 1):
        yield series.iloc[i : i + window_size]


# ====================================================================== #
# Versión materializada (para casos que requieren acceso aleatorio)        #
# ====================================================================== #

def get_all_windows(
    series: pd.Series,
    window_size: int,
) -> List[pd.Series]:
    """
    Materializa todas las ventanas deslizantes en una lista.

    Úsese solo cuando se necesita acceso aleatorio a las ventanas
    (ej. visualización). Para iterar una sola vez, preferir
    generate_windows() para ahorrar memoria.

    Complejidad temporal: O(n·w).
    Complejidad espacial: O(n·w).

    Args:
        series:      Serie de pandas con índice datetime.
        window_size: Número de elementos por ventana.

    Returns:
        Lista de pd.Series, cada una de longitud window_size.
    """
    return list(generate_windows(series, window_size))


# ====================================================================== #
# Estadísticas por ventana                                                 #
# ====================================================================== #

def window_statistics(
    series: pd.Series,
    window_size: int,
) -> pd.DataFrame:
    """
    Calcula estadísticas básicas para cada ventana deslizante.

    Para cada ventana devuelve:
        start_date, end_date, first, last, min, max, mean, change_pct.

    El cálculo de min, max y mean se implementa manualmente con bucles
    explícitos (sin numpy.min/max ni pandas.mean).

    Complejidad temporal: O(n·w).
    Complejidad espacial: O(n).

    Args:
        series:      Serie de pandas con índice datetime y valores numéricos.
        window_size: Número de elementos por ventana.

    Returns:
        DataFrame con una fila por ventana.
    """
    rows: list = []

    for window in generate_windows(series, window_size):
        vals  = window.values
        first = float(vals[0])
        last  = float(vals[-1])

        # Media manual — O(w)
        total = 0.0
        for v in vals:
            total += float(v)
        mean_val = total / len(vals)

        # Min y max manual — O(w)
        min_val = float(vals[0])
        max_val = float(vals[0])
        for v in vals[1:]:
            fv = float(v)
            if fv < min_val:
                min_val = fv
            if fv > max_val:
                max_val = fv

        change_pct = (
            (last - first) / first * 100
            if first != 0.0 else float("nan")
        )

        rows.append({
            "start_date": window.index[0],
            "end_date":   window.index[-1],
            "first":      first,
            "last":       last,
            "min":        min_val,
            "max":        max_val,
            "mean":       mean_val,
            "change_pct": change_pct,
        })

    return pd.DataFrame(rows)
