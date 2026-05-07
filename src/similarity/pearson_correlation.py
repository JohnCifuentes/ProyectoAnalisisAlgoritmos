"""
src/similarity/pearson_correlation.py
======================================
Implementación manual de la Correlación de Pearson para series de tiempo.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FUNDAMENTO MATEMÁTICO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dadas dos series x = (x₁, ..., xₙ) y y = (y₁, ..., yₙ) con medias:
    x̄ = (1/n) Σᵢ xᵢ
    ȳ = (1/n) Σᵢ yᵢ

El coeficiente de correlación de Pearson es:

         Σᵢ (xᵢ − x̄)(yᵢ − ȳ)
    r = ────────────────────────────────────────
        √[ Σᵢ (xᵢ − x̄)² · Σᵢ (yᵢ − ȳ)² ]

Numerador:   covarianza muestral no normalizada entre x e y.
Denominador: producto de las desviaciones estándar (sin el factor 1/n
             ya que aparece en numerador y denominador y se cancela).

Propiedades:
  - r ∈ [−1, 1]
  - r = +1 → correlación lineal perfecta positiva (mismo movimiento).
  - r = −1 → correlación lineal perfecta negativa (movimientos opuestos).
  - r =  0 → sin relación lineal.
  - Es invariante a cambios de escala y traslación.

Interpretación financiera:
  - r > 0.8:  activos muy correlacionados (poca diversificación).
  - r ∈ [0.3, 0.8]: correlación moderada.
  - r < 0.3:  activos poco o no correlacionados.
  - r < 0:    activos que tienden a moverse en sentidos opuestos (hedge).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPLEJIDAD COMPUTACIONAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Temporal: O(n)
    — Tres pasadas de O(n): una para medias, una para covarianza y
      varianzas, ninguna adicional.
    — Implementable en una sola pasada (algoritmo de Welford) pero
      aquí se usan dos pasadas para mayor claridad pedagógica.

  Espacial: O(1)
    — Solo se acumulan escalares; no se almacenan vectores auxiliares.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESTRICCIONES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  NO se usa numpy.corrcoef(), pandas.corr() ni scipy.stats.pearsonr().
  Toda la lógica matemática es explícita en esta implementación.
"""

import math
import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def pearson_correlation(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Calcula el coeficiente de correlación de Pearson entre dos series.

    Algoritmo (dos pasadas sobre los datos):

      Pasada 1 — Calcular medias:
        x̄ = (x₁ + x₂ + ... + xₙ) / n
        ȳ = (y₁ + y₂ + ... + yₙ) / n

      Pasada 2 — Calcular covarianza y varianzas acumuladas:
        Para i en [0, n-1]:
            dx_i = xᵢ − x̄
            dy_i = yᵢ − ȳ
            cov  += dx_i · dy_i
            var_x += dx_i²
            var_y += dy_i²

      Resultado:
        r = cov / sqrt(var_x · var_y)

    Args:
        x: Array numpy de la primera serie (sin NaN, alineada).
        y: Array numpy de la segunda serie (mismo largo que x).

    Returns:
        Coeficiente de Pearson r ∈ [−1.0, 1.0].
        Retorna 0.0 si alguna serie es constante (sin varianza),
        ya que la correlación es indefinida en ese caso.

    Raises:
        ValueError: Si los arrays tienen diferente longitud, están vacíos
                    o contienen NaN.
    """
    _validate_inputs(x, y, "pearson_correlation")
    n = len(x)

    # ── Pasada 1: calcular medias ────────────────────────────────────
    # x̄ = (1/n) Σᵢ xᵢ
    sum_x: float = 0.0
    sum_y: float = 0.0
    for i in range(n):
        sum_x += float(x[i])
        sum_y += float(y[i])
    mean_x: float = sum_x / n
    mean_y: float = sum_y / n

    # ── Pasada 2: calcular covarianza y varianzas ────────────────────
    # cov(x,y) = Σᵢ (xᵢ − x̄)(yᵢ − ȳ)  [sin dividir por n: se cancela]
    # var(x)   = Σᵢ (xᵢ − x̄)²
    # var(y)   = Σᵢ (yᵢ − ȳ)²
    cov_xy: float = 0.0
    var_x: float = 0.0
    var_y: float = 0.0

    for i in range(n):
        dx = float(x[i]) - mean_x   # desviación de xᵢ respecto a su media
        dy = float(y[i]) - mean_y   # desviación de yᵢ respecto a su media
        cov_xy += dx * dy
        var_x += dx * dx
        var_y += dy * dy

    # ── Calcular r ───────────────────────────────────────────────────
    # Caso especial: si alguna varianza es 0, la serie es constante.
    # La correlación de Pearson no está definida en ese caso.
    denominator = math.sqrt(var_x * var_y)
    if denominator == 0.0:
        logger.warning(
            "pearson_correlation: denominador = 0. "
            "Una o ambas series son constantes. Retornando r=0.0"
        )
        return 0.0

    r = cov_xy / denominator

    # Ajuste numérico: clampear en [-1, 1] por errores de punto flotante
    r = max(-1.0, min(1.0, r))

    logger.debug(
        f"pearson_correlation: n={n}, x̄={mean_x:.6f}, ȳ={mean_y:.6f}, "
        f"cov={cov_xy:.6f}, var_x={var_x:.6f}, var_y={var_y:.6f}, r={r:.6f}"
    )
    return r


def pearson_with_components(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, dict]:
    """
    Calcula Pearson y retorna también los componentes intermedios.

    Útil para análisis pedagógico y documentación técnica del proyecto.

    Args:
        x: Primera serie numérica.
        y: Segunda serie numérica.

    Returns:
        Tupla (r, components) donde:
          r: coeficiente de correlación en [-1, 1].
          components: dict con:
            - mean_x, mean_y: medias de cada serie.
            - covariance:     covarianza no normalizada (Σ dx·dy).
            - std_x, std_y:   desviaciones estándar muestrales (√(var/n)).
            - n:              número de puntos.
    """
    _validate_inputs(x, y, "pearson_with_components")
    n = len(x)

    # Medias
    sum_x = sum_y = 0.0
    for i in range(n):
        sum_x += float(x[i])
        sum_y += float(y[i])
    mean_x = sum_x / n
    mean_y = sum_y / n

    # Covarianza y varianzas
    cov_xy = var_x = var_y = 0.0
    for i in range(n):
        dx = float(x[i]) - mean_x
        dy = float(y[i]) - mean_y
        cov_xy += dx * dy
        var_x += dx * dx
        var_y += dy * dy

    denominator = math.sqrt(var_x * var_y)
    r = 0.0 if denominator == 0.0 else cov_xy / denominator
    r = max(-1.0, min(1.0, r))

    components = {
        "mean_x": mean_x,
        "mean_y": mean_y,
        "covariance": cov_xy / n,           # covarianza normalizada
        "std_x": math.sqrt(var_x / n),      # desviación estándar muestral
        "std_y": math.sqrt(var_y / n),
        "n": n,
    }
    return r, components


def interpret_pearson(r: float) -> str:
    """
    Retorna una interpretación cualitativa del coeficiente de Pearson.

    Escala de referencia estándar para análisis financiero:

    Args:
        r: Coeficiente de Pearson en [-1, 1].

    Returns:
        Cadena descriptiva de la correlación.
    """
    abs_r = abs(r)
    direction = "positiva" if r >= 0 else "negativa"

    if abs_r >= 0.9:
        strength = "muy alta"
    elif abs_r >= 0.7:
        strength = "alta"
    elif abs_r >= 0.5:
        strength = "moderada"
    elif abs_r >= 0.3:
        strength = "baja"
    else:
        strength = "muy baja o inexistente"

    return f"Correlación {direction} {strength} (r = {r:.4f})"


# ================================================================== #
# Validación interna                                                   #
# ================================================================== #

def _validate_inputs(x: np.ndarray, y: np.ndarray, fn_name: str) -> None:
    """
    Valida los arrays de entrada para las funciones de Pearson.

    Raises:
        ValueError: Si arrays son None, vacíos, de distinto tamaño o con NaN.
    """
    if x is None or y is None:
        raise ValueError(f"[{fn_name}] Los arrays no pueden ser None")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) == 0 or len(y) == 0:
        raise ValueError(f"[{fn_name}] Los arrays no pueden estar vacíos")

    if len(x) != len(y):
        raise ValueError(
            f"[{fn_name}] Longitudes distintas: len(x)={len(x)}, len(y)={len(y)}"
        )

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(
            f"[{fn_name}] Se esperan arrays 1D. "
            f"Recibidos: x.ndim={x.ndim}, y.ndim={y.ndim}"
        )

    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError(
            f"[{fn_name}] Arrays contienen NaN. "
            "Usa remove_nan_pairs() antes de llamar al algoritmo."
        )
