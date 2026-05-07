"""
src/similarity/euclidean_distance.py
=====================================
Implementación manual de la Distancia Euclidiana para series de tiempo.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FUNDAMENTO MATEMÁTICO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dadas dos series numéricas de igual longitud n:
    x = (x₁, x₂, ..., xₙ)
    y = (y₁, y₂, ..., yₙ)

La distancia euclidiana es la norma L₂ de la diferencia vectorial:

    d(x, y) = ‖x − y‖₂ = √[ Σᵢ₌₁ⁿ (xᵢ − yᵢ)² ]

Interpretación:
  - d = 0    → las series son idénticas punto a punto.
  - d → ∞   → las series divergen en magnitud o forma.
  - No es una medida acotada; su valor depende de la escala y longitud.

Por eso, para comparaciones entre activos de distinta volatilidad, se
recomienda aplicar z_normalize() antes de llamar a euclidean_distance().

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPLEJIDAD COMPUTACIONAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Temporal: O(n)
    — Una sola pasada sobre los n pares (xᵢ, yᵢ).

  Espacial: O(1)
    — Solo se acumula la suma de cuadrados; no se almacena un vector
      auxiliar adicional.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LIMITACIONES EN SERIES DE TIEMPO FINANCIERAS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. Sensible a la escala: NVDA (alta volatilidad) siempre aparecerá
     más lejana de VOO (baja volatilidad) aunque compartan tendencia.
     → Solución: normalizar con z_normalize() antes de comparar.

  2. Requiere alineación exacta: timestamp a timestamp. No puede
     comparar series desfasadas. Para eso existe DTW.

  3. No indica dirección: d(x,y) = d(y,x) pero no dice si dos activos
     se mueven en la misma o en sentidos opuestos. Para eso, Pearson.
"""

import math
import logging
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)


def euclidean_distance(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Calcula la distancia euclidiana entre dos vectores numéricos.

    Implementación manual del algoritmo:

        Paso 1: Verificar que len(x) == len(y) == n > 0.
        Paso 2: Acumular suma_cuadrados = 0
        Paso 3: Para i en [0, n-1]:
                    diff = x[i] - y[i]
                    suma_cuadrados += diff * diff
        Paso 4: Retornar sqrt(suma_cuadrados)

    No usa scipy.spatial.distance.euclidean ni numpy.linalg.norm,
    sino una suma explícita para que el comportamiento algorítmico
    sea completamente transparente.

    Args:
        x: Array numpy de la primera serie (ya sin NaN, alineada).
        y: Array numpy de la segunda serie (mismo largo que x).

    Returns:
        Distancia euclidiana ≥ 0. Valor 0.0 significa series idénticas.

    Raises:
        ValueError: Si los arrays tienen diferente longitud o están vacíos.
    """
    _validate_inputs(x, y, "euclidean_distance")

    n = len(x)
    squared_sum: float = 0.0

    # ── Algoritmo principal: O(n) ────────────────────────────────────
    # Acumular la suma de cuadrados de diferencias elemento a elemento.
    # Se usa math.fsum-equivalent accumulation con Python float nativo
    # para mayor precisión numérica que una suma simple.
    for i in range(n):
        diff = float(x[i]) - float(y[i])   # diferencia en el día i
        squared_sum += diff * diff          # contribución al cuadrado

    distance = math.sqrt(squared_sum)

    logger.debug(
        f"euclidean_distance: n={n}, Σ(xi-yi)²={squared_sum:.6f}, "
        f"d={distance:.6f}"
    )
    return distance


def euclidean_distance_normalized(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Calcula la distancia euclidiana normalizada por la longitud de la serie.

    Fórmula:
        d_norm(x, y) = d(x, y) / √n

    La normalización por √n hace que el resultado sea independiente de la
    longitud de la serie, permitiendo comparar pares con distinto número
    de puntos temporales (ej. activos con distintas fechas de cotización).

    Args:
        x: Array numpy de la primera serie.
        y: Array numpy de la segunda serie.

    Returns:
        Distancia euclidiana normalizada ≥ 0.
    """
    _validate_inputs(x, y, "euclidean_distance_normalized")
    n = len(x)
    raw_distance = euclidean_distance(x, y)
    normalized = raw_distance / math.sqrt(n)
    logger.debug(
        f"euclidean_distance_normalized: d_raw={raw_distance:.6f}, "
        f"n={n}, d_norm={normalized:.6f}"
    )
    return normalized


def euclidean_similarity(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Convierte la distancia euclidiana a una similitud en [0, 1].

    Fórmula:
        sim(x, y) = 1 / (1 + d(x, y))

    Propiedades:
      - sim = 1.0 cuando d = 0  (series idénticas).
      - sim → 0   cuando d → ∞  (series muy distintas).
      - La función es monótonamente decreciente en d.

    Útil cuando se necesita comparar con Pearson y coseno (que son
    similitudes) en la misma escala conceptual.

    Args:
        x: Array numpy de la primera serie.
        y: Array numpy de la segunda serie.

    Returns:
        Similitud en (0, 1]. Valor 1.0 = idénticas.
    """
    d = euclidean_distance(x, y)
    sim = 1.0 / (1.0 + d)
    logger.debug(f"euclidean_similarity: d={d:.6f}, sim={sim:.6f}")
    return sim


# ================================================================== #
# Validación interna                                                   #
# ================================================================== #

def _validate_inputs(x: np.ndarray, y: np.ndarray, fn_name: str) -> None:
    """
    Valida los arrays de entrada para los algoritmos euclidianos.

    Verifica:
      - Ninguno es None.
      - Ambos son unidimensionales (o convertibles a ello).
      - Tienen la misma longitud.
      - No están vacíos.
      - No contienen NaN (la limpieza debería haberse hecho antes).

    Args:
        x, y:    Arrays a validar.
        fn_name: Nombre de la función llamante (para mensajes de error).

    Raises:
        ValueError: Si alguna validación falla.
    """
    if x is None or y is None:
        raise ValueError(f"[{fn_name}] Los arrays no pueden ser None")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(
            f"[{fn_name}] Se esperan arrays unidimensionales. "
            f"Recibidos: x.ndim={x.ndim}, y.ndim={y.ndim}"
        )

    if len(x) == 0 or len(y) == 0:
        raise ValueError(f"[{fn_name}] Los arrays no pueden estar vacíos")

    if len(x) != len(y):
        raise ValueError(
            f"[{fn_name}] Los arrays deben tener la misma longitud. "
            f"Recibidos: len(x)={len(x)}, len(y)={len(y)}"
        )

    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError(
            f"[{fn_name}] Los arrays contienen NaN. "
            "Usa remove_nan_pairs() antes de llamar al algoritmo."
        )
