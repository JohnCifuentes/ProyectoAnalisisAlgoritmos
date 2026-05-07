"""
src/similarity/dynamic_time_warping.py
========================================
Implementación manual de Dynamic Time Warping (DTW) con programación dinámica.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FUNDAMENTO MATEMÁTICO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DTW fue propuesto por Sakoe & Chiba (1978) para comparar secuencias de
habla. En análisis financiero, permite comparar activos cuyas tendencias
son similares pero están desfasadas en el tiempo (ej. un mercado emergente
que reacciona un día después de Wall Street).

Definición formal:
  Dadas dos series x = (x₁, ..., xₘ) y y = (y₁, ..., yₙ),
  (pueden tener longitudes distintas: m ≠ n),

  DTW busca el "camino de alineamiento" w = (w₁, ..., wₖ) que minimiza
  la suma de las distancias locales:

      DTW(x, y) = min_w  Σₖ d(wₖ)

  donde el camino w es una secuencia de pares de índices (i, j) que
  satisface:
    1. Monotonicity:   i y j no decrecen a lo largo del camino.
    2. Continuity:     cada paso solo avanza 1 posición en i o j (o ambos).
    3. Boundary:       el camino empieza en (1,1) y termina en (m,n).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALGORITMO DE PROGRAMACIÓN DINÁMICA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Se construye una matriz de costo acumulado D de tamaño (m+1) × (n+1):

  Inicialización:
    D[0][0] = 0
    D[i][0] = ∞  para i > 0
    D[0][j] = ∞  para j > 0

  Recurrencia:
    D[i][j] = |x[i-1] - y[j-1]|  +  min(
                                       D[i-1][j],    ← insertar (solo avanzar en y)
                                       D[i][j-1],    ← borrar   (solo avanzar en x)
                                       D[i-1][j-1]   ← emparejar (avanzar en ambos)
                                     )

  Resultado:
    DTW(x, y) = D[m][n]

La distancia DTW es el costo mínimo total del camino óptimo de
alineación entre las dos series.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPLEJIDAD COMPUTACIONAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Temporal: O(m × n)
    — Se llenan m×n celdas de la matriz D.
    — Para series de igual longitud n: O(n²).

  Espacial: O(m × n)
    — La implementación completa almacena toda la matriz D.
    — Existe una optimización O(n) con "banda de Sakoe-Chiba" o usando
      solo dos filas, pero aquí se almacena la matriz completa para
      permitir la recuperación del camino óptimo.

Diferencia con distancia euclidiana:
  La distancia euclidiana es O(n) pero requiere alineación exacta.
  DTW es O(n²) pero es invariante a desfases temporales, lo cual la hace
  más robusta para comparar series financieras con reacciones retardadas.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESTRICCIONES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  NO se usa fastdtw, tslearn, dtaidistance ni ninguna librería externa.
  El algoritmo se implementa íntegramente con bucles Python y arrays numpy.
"""

import logging
import math
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Valor que representa "infinito" en la inicialización de la matriz DTW
_INF: float = float("inf")


def dtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    normalize: bool = True,
) -> float:
    """
    Calcula la distancia DTW entre dos series de tiempo.

    Esta función es el punto de entrada principal. Llama a dtw_matrix()
    internamente y retorna solo la distancia final D[m][n].

    Args:
        x:         Array numpy de la primera serie (puede tener NaN; se limpian).
        y:         Array numpy de la segunda serie.
        normalize: Si True, normaliza la distancia final dividiendo por
                   (m + n), donde m y n son las longitudes de x e y.
                   Esto permite comparar pares con series de distinto largo.

    Returns:
        Distancia DTW ≥ 0. Valor cercano a 0 = series muy similares.

    Raises:
        ValueError: Si algún array está vacío.
    """
    x, y = _clean_nan(x, y, "dtw_distance")
    _validate_nonempty(x, y, "dtw_distance")

    m = len(x)
    n = len(y)

    # Construir la matriz de costo acumulado
    D = _build_cost_matrix(x, y)

    distance = D[m][n]

    if normalize and (m + n) > 0:
        distance = distance / (m + n)

    logger.debug(
        f"dtw_distance: m={m}, n={n}, "
        f"D[m][n]={D[m][n]:.6f}, normalized={distance:.6f}"
    )
    return distance


def dtw_with_path(
    x: np.ndarray,
    y: np.ndarray,
    normalize: bool = True,
) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Calcula la distancia DTW Y recupera el camino óptimo de alineación.

    El camino óptimo (warping path) muestra cómo se empareja cada punto
    de x con cada punto de y para minimizar el costo total.

    Recuperación del camino (backtracking):
      Partiendo de D[m][n], en cada paso se elige el predecesor con
      menor costo entre:
        - D[i-1][j-1]  (emparejar)
        - D[i-1][j]    (borrar en x)
        - D[i][j-1]    (insertar en y)
      Se retrocede hasta llegar a D[1][1].

    Args:
        x:         Primera serie numérica.
        y:         Segunda serie numérica.
        normalize: Si True, normaliza la distancia por (m + n).

    Returns:
        Tupla (distance, path) donde:
          distance: distancia DTW.
          path:     lista de tuplas (i, j) con índices 0-based,
                    desde (0,0) hasta (m-1, n-1).
    """
    x, y = _clean_nan(x, y, "dtw_with_path")
    _validate_nonempty(x, y, "dtw_with_path")

    m = len(x)
    n = len(y)

    D = _build_cost_matrix(x, y)
    distance = D[m][n]

    if normalize and (m + n) > 0:
        distance = distance / (m + n)

    path = _backtrack_path(D, m, n)

    logger.debug(
        f"dtw_with_path: distancia={distance:.6f}, "
        f"longitud del path={len(path)}"
    )
    return distance, path


def dtw_matrix_only(
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Retorna la matriz de costo acumulado completa D de tamaño (m+1) × (n+1).

    Útil para visualización de la matriz DTW (se usará en Req. 4).

    Args:
        x: Primera serie numérica.
        y: Segunda serie numérica.

    Returns:
        Matriz numpy (m+1) × (n+1) de costos acumulados.
        Las posiciones [0][j] para j>0 y [i][0] para i>0 son ∞.
    """
    x, y = _clean_nan(x, y, "dtw_matrix_only")
    _validate_nonempty(x, y, "dtw_matrix_only")
    return _build_cost_matrix(x, y)


# ================================================================== #
# Núcleo del algoritmo de programación dinámica                        #
# ================================================================== #

def _build_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Construye la matriz de costo acumulado D mediante programación dinámica.

    Pseudocódigo:
        D = matriz de (m+1) × (n+1) inicializada en ∞
        D[0][0] = 0

        para i = 1 hasta m:
            para j = 1 hasta n:
                local_cost = |x[i-1] - y[j-1]|
                D[i][j] = local_cost + min(D[i-1][j-1],   ← emparejar
                                           D[i-1][j],       ← avanzar en x
                                           D[i][j-1])       ← avanzar en y

    La distancia DTW es D[m][n].

    Args:
        x: Array de longitud m.
        y: Array de longitud n.

    Returns:
        Matriz numpy de forma (m+1, n+1).
    """
    m = len(x)
    n = len(y)

    # ── Inicialización ───────────────────────────────────────────────
    # D[m+1][n+1] con infinito: modela el "costo infinito" de llegar
    # a cualquier posición sin haber comenzado desde (0,0).
    D = np.full((m + 1, n + 1), _INF, dtype=float)
    D[0][0] = 0.0

    # ── Relleno de la matriz: O(m × n) ───────────────────────────────
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Costo local: distancia absoluta entre los dos puntos
            # (distancia L₁, común en DTW para robustez ante outliers)
            local_cost: float = abs(float(x[i - 1]) - float(y[j - 1]))

            # Elegir el predecesor de menor costo acumulado
            best_prev: float = min(
                D[i - 1][j - 1],   # emparejar x[i] con y[j]
                D[i - 1][j],       # saltar x[i] sin emparejar
                D[i][j - 1],       # saltar y[j] sin emparejar
            )

            D[i][j] = local_cost + best_prev

    return D


# ================================================================== #
# Recuperación del camino óptimo (backtracking)                        #
# ================================================================== #

def _backtrack_path(
    D: np.ndarray,
    m: int,
    n: int,
) -> List[Tuple[int, int]]:
    """
    Recupera el camino óptimo mediante backtracking sobre la matriz D.

    Comienza en la celda D[m][n] (esquina inferior derecha) y retrocede
    eligiendo en cada paso el predecesor de menor costo, hasta llegar
    a D[1][1] (que corresponde a los índices 0-based (0,0)).

    Complejidad: O(m + n) — el camino tiene como máximo m+n pasos.

    Args:
        D: Matriz de costo acumulado de forma (m+1, n+1).
        m: Longitud de la primera serie.
        n: Longitud de la segunda serie.

    Returns:
        Lista de tuplas (i, j) con índices 0-based, desde (0,0) hasta
        (m-1, n-1), representando el emparejamiento óptimo.
    """
    path: List[Tuple[int, int]] = []
    i, j = m, n

    while i > 0 and j > 0:
        # Registrar la posición actual (convertida a 0-based)
        path.append((i - 1, j - 1))

        # Elegir el paso de retroceso hacia el predecesor de menor costo
        diag = D[i - 1][j - 1]
        up   = D[i - 1][j]
        left = D[i][j - 1]

        min_cost = min(diag, up, left)

        if min_cost == diag:
            i -= 1
            j -= 1
        elif min_cost == up:
            i -= 1
        else:
            j -= 1

    # Agregar punto de origen
    path.append((0, 0))

    # El path fue construido de atrás hacia adelante; invertir
    path.reverse()

    return path


# ================================================================== #
# Limpieza y validación                                                #
# ================================================================== #

def _clean_nan(
    x: np.ndarray,
    y: np.ndarray,
    fn_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Limpia NaN de los arrays.

    A diferencia de los otros algoritmos (que requieren alineación por par),
    DTW puede manejar series de distinto largo, por lo que se limpian
    los NaN de cada array de forma independiente.

    Args:
        x, y:    Arrays a limpiar.
        fn_name: Nombre de la función (para logging).

    Returns:
        Tupla (x_clean, y_clean) sin NaN.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    nan_x = int(np.sum(np.isnan(x)))
    nan_y = int(np.sum(np.isnan(y)))

    if nan_x > 0:
        x = x[~np.isnan(x)]
        logger.debug(f"[{fn_name}] {nan_x} NaN eliminados de x")
    if nan_y > 0:
        y = y[~np.isnan(y)]
        logger.debug(f"[{fn_name}] {nan_y} NaN eliminados de y")

    return x, y


def _validate_nonempty(x: np.ndarray, y: np.ndarray, fn_name: str) -> None:
    """
    Verifica que ninguno de los arrays esté vacío.

    Raises:
        ValueError: Si alguno de los arrays está vacío.
    """
    if len(x) == 0 or len(y) == 0:
        raise ValueError(
            f"[{fn_name}] Los arrays no pueden estar vacíos. "
            f"Recibidos: len(x)={len(x)}, len(y)={len(y)}"
        )
