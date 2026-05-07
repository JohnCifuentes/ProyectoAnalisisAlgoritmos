"""
src/visualization/correlation_matrix.py
=========================================
Construcción de la matriz de correlación de Pearson para todos los activos.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REUTILIZACIÓN DEL ALGORITMO MANUAL (Req #2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Este módulo NO reimplementa Pearson.
Importa directamente la función manual ya validada en el Requerimiento #2:

    from src.similarity.pearson_correlation import pearson_correlation

Para cada par (i, j) con i < j:
  1. Cargar retornos diarios de ambos activos (usando utils de Req #2).
  2. Alinear fechas comunes con align_series() de Req #2.
  3. Limpiar pares NaN con remove_nan_pairs() de Req #2.
  4. Llamar pearson_correlation(x, y) → r ∈ [-1, 1].

La matriz es simétrica, por lo que solo se calculan n·(n-1)/2 pares.
Para 20 activos: 190 pares.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPLEJIDAD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  - Carga de retornos: O(n) por ticker → O(20n) total.
  - Por par: O(n) para alinear + O(n) para Pearson → O(n).
  - Total: O(190n) donde n ≈ 1250 días.
  - Caché: la matriz se computa UNA vez por sesión Flask (lazy singleton).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESTRICCIONES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  NO se usa pandas.corr(), numpy.corrcoef() ni scipy.stats.pearsonr().
  El cálculo aritmético lo hace pearson_correlation() del Req #2.
"""

import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Importaciones del Req #2 ─────────────────────────────────────────
from src.similarity.pearson_correlation import pearson_correlation
from src.similarity.utils import (
    align_series,
    compute_returns,
    load_master,
    load_series,
    remove_nan_pairs,
)
from src.config.settings import PROCESSED_DIR, ASSET_MAP

logger = logging.getLogger(__name__)


class CorrelationMatrix:
    """
    Construye y almacena la matriz de correlación de Pearson.

    Diseño lazy-singleton: la matriz se computa solo la primera vez que se
    solicita y se guarda en caché para todas las requests subsiguientes.

    Attributes:
        _master_path: Ruta al dataset maestro.
        _tickers:     Lista de tickers en el orden de la matriz.
        _matrix:      Lista 2D (n × n) de coeficientes de Pearson.
        _computed:    Bandera de caché.
    """

    def __init__(self, master_path: Optional[Path] = None) -> None:
        self._master_path: Path = master_path or (PROCESSED_DIR / "master_dataset.csv")
        self._tickers:  Optional[List[str]] = None
        self._matrix:   Optional[List[List[Optional[float]]]] = None
        self._computed: bool = False

    # ------------------------------------------------------------------ #
    # API pública                                                          #
    # ------------------------------------------------------------------ #

    def get(self) -> Tuple[List[str], List[List[Optional[float]]]]:
        """
        Devuelve (tickers, matrix).

        La primera llamada computa la matriz completa (puede tardar ~1-2 s).
        Las siguientes devuelven el resultado en caché instantáneamente.

        Returns:
            tickers: Lista de símbolos en el orden de filas/columnas.
            matrix:  Lista 2D [n][n] de floats en [-1, 1]. La diagonal
                     es 1.0. Puede contener None si un par no tiene datos.
        """
        if not self._computed:
            self._build()
        return self._tickers, self._matrix

    def invalidate(self) -> None:
        """Invalida el caché y fuerza recomputación en la próxima llamada."""
        self._computed = False
        self._tickers  = None
        self._matrix   = None

    # ------------------------------------------------------------------ #
    # Construcción de la matriz                                            #
    # ------------------------------------------------------------------ #

    def _build(self) -> None:
        """Carga datos, computa los 190 pares y llena la matriz 20×20."""
        t_start = time.perf_counter()
        logger.info("CorrelationMatrix: iniciando construcción…")

        # 1. Cargar el dataset maestro
        master = load_master(self._master_path)
        tickers = sorted(master["ticker"].unique().tolist())
        n = len(tickers)

        # 2. Cachear series de retornos por ticker (una sola carga por activo)
        returns_cache: Dict[str, pd.Series] = {}
        for ticker in tickers:
            try:
                prices  = load_series(ticker, master=master)
                returns = compute_returns(prices, drop_first_nan=True)
                returns_cache[ticker] = returns
            except Exception as exc:
                logger.warning(f"  [{ticker}] No se pudo cargar: {exc}")
                returns_cache[ticker] = pd.Series(dtype=float)

        # 3. Inicializar la matriz con None
        matrix: List[List[Optional[float]]] = [
            [None] * n for _ in range(n)
        ]

        # Diagonal principal = 1.0 (correlación consigo mismo)
        for i in range(n):
            matrix[i][i] = 1.0

        # 4. Calcular el triángulo superior (i < j) y reflejar en el inferior
        computed_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                r = self._compute_pair(
                    returns_cache[tickers[i]],
                    returns_cache[tickers[j]],
                    ticker_a=tickers[i],
                    ticker_b=tickers[j],
                )
                matrix[i][j] = r
                matrix[j][i] = r  # simétrica
                computed_pairs += 1

        elapsed = time.perf_counter() - t_start
        logger.info(
            f"CorrelationMatrix: {computed_pairs} pares calculados "
            f"en {elapsed*1000:.0f} ms"
        )

        self._tickers  = tickers
        self._matrix   = matrix
        self._computed = True

    def _compute_pair(
        self,
        r_a: pd.Series,
        r_b: pd.Series,
        ticker_a: str = "?",
        ticker_b: str = "?",
    ) -> Optional[float]:
        """
        Calcula la correlación de Pearson para un par de series de retornos.

        Proceso:
          1. Alinear fechas comunes con align_series() (Req #2).
          2. Extraer arrays numpy.
          3. Limpiar pares NaN con remove_nan_pairs() (Req #2).
          4. Llamar pearson_correlation(x, y) (Req #2).

        Returns:
            Coeficiente r ∈ [-1, 1] o None si no hay datos suficientes.
        """
        if r_a.empty or r_b.empty:
            logger.debug(f"  Par ({ticker_a}, {ticker_b}): series vacías.")
            return None

        try:
            # ── Alinear por fechas comunes ────────────────────────────
            aligned_a, aligned_b = align_series(r_a, r_b)

            x = aligned_a.values.astype(float)
            y = aligned_b.values.astype(float)

            # ── Eliminar pares NaN ────────────────────────────────────
            x_clean, y_clean = remove_nan_pairs(x, y)

            if len(x_clean) < 10:
                logger.debug(
                    f"  Par ({ticker_a}, {ticker_b}): "
                    f"solo {len(x_clean)} puntos válidos."
                )
                return None

            # ── Correlación de Pearson MANUAL (Req #2) ────────────────
            # NO se usa pandas.corr() ni numpy.corrcoef()
            r = pearson_correlation(x_clean, y_clean)
            return round(r, 6)

        except Exception as exc:
            logger.warning(
                f"  Par ({ticker_a}, {ticker_b}): error al calcular → {exc}"
            )
            return None


# ── Singleton global ─────────────────────────────────────────────────
_instance: Optional[CorrelationMatrix] = None


def get_correlation_matrix() -> CorrelationMatrix:
    """Devuelve la instancia singleton de CorrelationMatrix."""
    global _instance
    if _instance is None:
        _instance = CorrelationMatrix()
    return _instance
