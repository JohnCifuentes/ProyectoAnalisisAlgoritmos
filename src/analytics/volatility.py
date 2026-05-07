"""
src/analytics/volatility.py
============================
Cálculo manual de volatilidad histórica de activos financieros.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FUNDAMENTO MATEMÁTICO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

La volatilidad histórica de un activo es la desviación estándar de sus
retornos diarios, usando la corrección de Bessel (varianza muestral):

  Paso 1 — Media de retornos:
    r̄ = (1/n) × Σᵢ₌₁ⁿ rᵢ

  Paso 2 — Varianza muestral (ddof = 1):
    s² = (1/(n-1)) × Σᵢ₌₁ⁿ (rᵢ − r̄)²

  Paso 3 — Desviación estándar (volatilidad diaria):
    σ_d = √s²

  Paso 4 — Volatilidad anualizada (raíz del tiempo):
    σ_a = σ_d × √252

  La fórmula de anualización proviene de que los retornos diarios son
  (aproximadamente) IID; por tanto su varianza acumula linealmente en
  T días: Var(R_T) = T × Var(R_1), luego σ_T = σ_1 × √T.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESTRICCIONES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  NO se usan numpy.std(), pandas.std() ni scipy.stats.
  Toda la lógica se implementa con bucles explícitos y math.sqrt().

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPLEJIDAD COMPUTACIONAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Temporal: O(n) — dos pasadas lineales: una para la media, otra para
            la varianza. La raíz cuadrada es O(1).
  Espacial: O(1) — solo se acumulan escalares; no se almacena el array
            de desviaciones.
"""

import logging
import math
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Número de días de trading por año (convención de mercados bursátiles)
TRADING_DAYS_PER_YEAR: int = 252


# ====================================================================== #
# Funciones estadísticas auxiliares (implementación manual)               #
# ====================================================================== #

def _compute_mean(values: np.ndarray) -> float:
    """
    Calcula la media aritmética de un array de valores.

    Fórmula:
        r̄ = (1/n) × Σᵢ rᵢ

    Complejidad temporal: O(n).
    Complejidad espacial: O(1).

    Args:
        values: Array de valores numéricos (sin NaN).

    Raises:
        ValueError: Si el array está vacío.
    """
    n = len(values)
    if n == 0:
        raise ValueError("No se puede calcular la media de un array vacío.")

    total = 0.0
    for v in values:
        total += float(v)

    return total / n


def _compute_variance(values: np.ndarray, mean: float) -> float:
    """
    Calcula la varianza muestral con corrección de Bessel (ddof = 1).

    Fórmula:
        s² = (1/(n-1)) × Σᵢ (rᵢ − r̄)²

    Se usa n-1 en el denominador (corrección de Bessel) porque los
    retornos históricos son una muestra del proceso subyacente real;
    el estimador insesgado de la varianza poblacional requiere n-1.

    Complejidad temporal: O(n).
    Complejidad espacial: O(1).

    Args:
        values: Array de valores numéricos (sin NaN).
        mean:   Media precalculada de los valores.

    Raises:
        ValueError: Si el array tiene menos de 2 elementos.
    """
    n = len(values)
    if n < 2:
        raise ValueError(
            f"Se necesitan al menos 2 observaciones para calcular la "
            f"varianza muestral. Recibido: {n}"
        )

    sum_sq = 0.0
    for v in values:
        diff = float(v) - mean
        sum_sq += diff * diff

    return sum_sq / (n - 1)


# ====================================================================== #
# Volatilidad diaria                                                       #
# ====================================================================== #

def compute_daily_volatility(returns: np.ndarray) -> float:
    """
    Calcula la volatilidad diaria histórica como desviación estándar
    de los retornos diarios.

    Pasos:
        1. r̄   = media(retornos)          — O(n)
        2. s²  = varianza_muestral(r̄)    — O(n)
        3. σ_d = √s²                      — O(1)

    Complejidad total: O(n).

    Args:
        returns: Array de retornos diarios (sin NaN).

    Returns:
        Volatilidad diaria σ_d ≥ 0.

    Raises:
        ValueError: Si el array tiene menos de 2 elementos.
    """
    if len(returns) < 2:
        raise ValueError(
            f"Se necesitan al menos 2 retornos para calcular volatilidad. "
            f"Recibido: {len(returns)}"
        )

    mean     = _compute_mean(returns)
    variance = _compute_variance(returns, mean)
    return math.sqrt(variance)


# ====================================================================== #
# Volatilidad anualizada                                                   #
# ====================================================================== #

def compute_annualized_volatility(
    daily_vol: float,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Convierte la volatilidad diaria a volatilidad anualizada.

    Fórmula:
        σ_a = σ_d × √trading_days

    La raíz del tiempo proviene del supuesto de que los retornos
    diarios son aproximadamente IID (independientes e idénticamente
    distribuidos), por lo que:
        Var(R_T) = T × Var(R_1)  →  σ_T = σ_1 × √T

    Complejidad: O(1).

    Args:
        daily_vol:     Volatilidad diaria σ_d.
        trading_days:  Días de trading por año (estándar: 252).

    Returns:
        Volatilidad anualizada σ_a.

    Raises:
        ValueError: Si daily_vol < 0 o trading_days ≤ 0.
    """
    if daily_vol < 0:
        raise ValueError(
            f"La volatilidad diaria no puede ser negativa: {daily_vol}"
        )
    if trading_days <= 0:
        raise ValueError(
            f"trading_days debe ser un entero positivo: {trading_days}"
        )

    return daily_vol * math.sqrt(trading_days)


# ====================================================================== #
# Función de alto nivel                                                    #
# ====================================================================== #

def compute_volatility(
    returns: pd.Series,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> Tuple[float, float]:
    """
    Calcula volatilidad diaria y anualizada a partir de una Serie
    de retornos diarios.

    Limpia NaN antes de computar (comportamiento seguro por defecto).

    Args:
        returns:       Serie de retornos diarios (pandas Series).
        trading_days:  Días de trading por año para anualización.

    Returns:
        Tupla (volatilidad_diaria, volatilidad_anualizada).

    Raises:
        ValueError: Si después de eliminar NaN quedan menos de 2 valores.
    """
    arr = returns.dropna().values

    if len(arr) < 2:
        raise ValueError(
            f"Se necesitan al menos 2 retornos válidos (sin NaN). "
            f"Disponibles: {len(arr)}"
        )

    daily_vol  = compute_daily_volatility(arr)
    annual_vol = compute_annualized_volatility(daily_vol, trading_days)

    return daily_vol, annual_vol
