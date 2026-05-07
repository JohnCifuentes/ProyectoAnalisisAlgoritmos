"""
src/similarity/cosine_similarity.py
=====================================
Implementación manual de la Similitud por Coseno para series de tiempo.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FUNDAMENTO MATEMÁTICO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dados dos vectores A = (a₁, ..., aₙ) y B = (b₁, ..., bₙ), la
similitud coseno mide el ángulo θ entre ellos en el espacio ℝⁿ:

         A · B           Σᵢ aᵢ · bᵢ
    cos θ = ────── = ─────────────────────────
         ‖A‖ · ‖B‖   √(Σᵢ aᵢ²) · √(Σᵢ bᵢ²)

Donde:
  · A · B  = producto punto (dot product).
  · ‖A‖   = norma euclidiana (magnitud) del vector A.
  · ‖B‖   = norma euclidiana del vector B.

Propiedades:
  - cos θ ∈ [−1, 1]  cuando los vectores pueden tener componentes negativas
    (como los retornos diarios).
  - cos θ = +1 → vectores paralelos, misma dirección (activos idénticos).
  - cos θ = 0  → vectores ortogonales, sin relación.
  - cos θ = −1 → vectores antiparalelos (activos perfectamente opuestos).

Diferencia clave con Pearson:
  La similitud coseno NO resta la media antes de calcular. Pearson mide
  la correlación lineal "centrada", mientras que coseno mide la alineación
  de dirección en el espacio vectorial "sin centrar". En la práctica, con
  retornos financieros (cuya media es ≈ 0), los resultados suelen ser
  similares, pero difieren en series con tendencia significativa.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPLEJIDAD COMPUTACIONAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Temporal: O(n)
    — Una sola pasada acumula dot_product, norm_a y norm_b simultáneamente.

  Espacial: O(1)
    — Solo se almacenan tres acumuladores escalares.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESTRICCIONES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  NO se usa numpy.dot(), sklearn.metrics.pairwise.cosine_similarity()
  ni scipy.spatial.distance.cosine().
  El producto punto y las normas se calculan con bucles explícitos.
"""

import math
import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def cosine_similarity(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Calcula la similitud coseno entre dos vectores numéricos.

    Algoritmo (una sola pasada sobre los datos):

        dot_product = 0,  norm_x = 0,  norm_y = 0
        Para i en [0, n-1]:
            dot_product += x[i] · y[i]          ← contribución al producto punto
            norm_x      += x[i]²                  ← contribución a ‖x‖²
            norm_y      += y[i]²                  ← contribución a ‖y‖²

        denom = sqrt(norm_x) · sqrt(norm_y)
        cos_sim = dot_product / denom

    Args:
        x: Array numpy de la primera serie (sin NaN, alineada).
        y: Array numpy de la segunda serie (mismo largo que x).

    Returns:
        Similitud coseno ∈ [−1.0, 1.0].
        Retorna 0.0 si alguno de los vectores es el vector nulo (norma = 0).

    Raises:
        ValueError: Si los arrays tienen diferente longitud, están vacíos
                    o contienen NaN.
    """
    _validate_inputs(x, y, "cosine_similarity")
    n = len(x)

    # ── Algoritmo principal: O(n) — una sola pasada ──────────────────
    dot_product: float = 0.0   # A · B = Σᵢ aᵢ · bᵢ
    norm_sq_x: float = 0.0     # ‖x‖² = Σᵢ xᵢ²
    norm_sq_y: float = 0.0     # ‖y‖² = Σᵢ yᵢ²

    for i in range(n):
        xi = float(x[i])
        yi = float(y[i])
        dot_product += xi * yi    # acumular producto punto
        norm_sq_x += xi * xi      # acumular cuadrados de x
        norm_sq_y += yi * yi      # acumular cuadrados de y

    # ── Calcular normas ──────────────────────────────────────────────
    # ‖x‖ = √(Σᵢ xᵢ²),  ‖y‖ = √(Σᵢ yᵢ²)
    norm_x: float = math.sqrt(norm_sq_x)
    norm_y: float = math.sqrt(norm_sq_y)

    denominator: float = norm_x * norm_y

    # ── Caso especial: vector nulo ────────────────────────────────────
    # Si ‖x‖ = 0 o ‖y‖ = 0, el ángulo no está definido.
    # En contexto financiero esto ocurriría si una serie de retornos
    # fuera identicamente cero (activo que no se movió nunca).
    if denominator == 0.0:
        logger.warning(
            "cosine_similarity: norma = 0 en al menos un vector. "
            "El ángulo no está definido. Retornando cos_sim=0.0"
        )
        return 0.0

    cos_sim: float = dot_product / denominator

    # Ajuste numérico: clampear en [-1, 1] por errores de punto flotante
    cos_sim = max(-1.0, min(1.0, cos_sim))

    logger.debug(
        f"cosine_similarity: n={n}, dot={dot_product:.6f}, "
        f"‖x‖={norm_x:.6f}, ‖y‖={norm_y:.6f}, cos_sim={cos_sim:.6f}"
    )
    return cos_sim


def cosine_distance(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Calcula la distancia coseno entre dos vectores.

    La distancia coseno se define como el complemento de la similitud:

        d_cos(x, y) = 1 − cos(θ) ∈ [0, 2]

    Con retornos diarios (que incluyen negativos), el rango teórico
    es [0, 2], donde:
      - 0 → misma dirección (activos idénticos en movimiento).
      - 1 → ortogonales (sin relación).
      - 2 → direcciones opuestas.

    Args:
        x: Primera serie numérica.
        y: Segunda serie numérica.

    Returns:
        Distancia coseno ∈ [0, 2].
    """
    sim = cosine_similarity(x, y)
    dist = 1.0 - sim
    logger.debug(f"cosine_distance: sim={sim:.6f}, d={dist:.6f}")
    return dist


def cosine_with_components(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, dict]:
    """
    Calcula similitud coseno y retorna los componentes intermedios.

    Útil para análisis pedagógico y documentación del proyecto.

    Args:
        x: Primera serie numérica.
        y: Segunda serie numérica.

    Returns:
        Tupla (cos_sim, components) donde:
          cos_sim: similitud coseno en [-1, 1].
          components: dict con:
            - dot_product: producto punto A·B.
            - norm_x, norm_y: normas de cada vector.
            - angle_degrees: ángulo θ en grados entre los vectores.
            - n: número de puntos.
    """
    _validate_inputs(x, y, "cosine_with_components")
    n = len(x)

    dot_product = norm_sq_x = norm_sq_y = 0.0
    for i in range(n):
        xi, yi = float(x[i]), float(y[i])
        dot_product += xi * yi
        norm_sq_x += xi * xi
        norm_sq_y += yi * yi

    norm_x = math.sqrt(norm_sq_x)
    norm_y = math.sqrt(norm_sq_y)
    denominator = norm_x * norm_y

    cos_sim = 0.0 if denominator == 0.0 else dot_product / denominator
    cos_sim = max(-1.0, min(1.0, cos_sim))

    # Ángulo en grados (arccos del coseno)
    try:
        angle_rad = math.acos(cos_sim)
        angle_deg = math.degrees(angle_rad)
    except ValueError:
        angle_deg = float("nan")

    components = {
        "dot_product": dot_product,
        "norm_x": norm_x,
        "norm_y": norm_y,
        "angle_degrees": angle_deg,
        "n": n,
    }
    return cos_sim, components


def interpret_cosine(cos_sim: float) -> str:
    """
    Retorna una interpretación cualitativa de la similitud coseno.

    Args:
        cos_sim: Similitud coseno en [-1, 1].

    Returns:
        Cadena descriptiva.
    """
    if cos_sim >= 0.9:
        return f"Similitud coseno muy alta: activos con retornos casi idénticos (cos={cos_sim:.4f})"
    elif cos_sim >= 0.7:
        return f"Similitud coseno alta: movimientos muy similares (cos={cos_sim:.4f})"
    elif cos_sim >= 0.4:
        return f"Similitud coseno moderada: alguna alineación en retornos (cos={cos_sim:.4f})"
    elif cos_sim >= 0.0:
        return f"Similitud coseno baja: poca relación vectorial (cos={cos_sim:.4f})"
    elif cos_sim >= -0.4:
        return f"Similitud coseno negativa moderada: tendencia a moverse en sentidos opuestos (cos={cos_sim:.4f})"
    else:
        return f"Similitud coseno muy negativa: activos con retornos opuestos (cos={cos_sim:.4f})"


# ================================================================== #
# Validación interna                                                   #
# ================================================================== #

def _validate_inputs(x: np.ndarray, y: np.ndarray, fn_name: str) -> None:
    """
    Valida los arrays de entrada para las funciones de similitud coseno.

    Raises:
        ValueError: Si arrays son None, vacíos, de distinto tamaño o con NaN.
    """
    if x is None or y is None:
        raise ValueError(f"[{fn_name}] Los arrays no pueden ser None")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(
            f"[{fn_name}] Se esperan arrays 1D. "
            f"Recibidos: x.ndim={x.ndim}, y.ndim={y.ndim}"
        )

    if len(x) == 0 or len(y) == 0:
        raise ValueError(f"[{fn_name}] Los arrays no pueden estar vacíos")

    if len(x) != len(y):
        raise ValueError(
            f"[{fn_name}] Longitudes distintas: len(x)={len(x)}, len(y)={len(y)}"
        )

    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError(
            f"[{fn_name}] Arrays contienen NaN. "
            "Usa remove_nan_pairs() antes de llamar al algoritmo."
        )
