"""
src/api/services/visualization_service.py
==========================================
Capa web para heatmap y candlestick.

Delega todo el cálculo a src/visualization/.
Serializa resultados a tipos JSON-nativos.
Mantiene caché del heatmap (costoso de calcular, ~1-2 s la primera vez).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config.settings import PROCESSED_DIR
from src.visualization.correlation_matrix import get_correlation_matrix
from src.visualization.heatmap_generator import (
    build_heatmap_figure,
    extract_extremes,
)
from src.visualization.candlestick_chart import prepare_ohlcv
from src.visualization.visualization_utils import to_json_safe

logger = logging.getLogger(__name__)

_MASTER_CSV = PROCESSED_DIR / "master_dataset.csv"


class VisualizationService:
    """
    Servicio web de visualización (singleton).

    - Heatmap:     caché en memoria (calculado una vez por sesión).
    - Candlestick: calculado por request (datos filtrados por ticker/período).
    """

    def __init__(self) -> None:
        self._master: Optional[pd.DataFrame] = None

        # Caché del heatmap (figura + extremos)
        self._heatmap_cache: Optional[Dict] = None

        logger.info("VisualizationService inicializado.")

    # ------------------------------------------------------------------ #
    # Dataset maestro (lazy load + caché)                                 #
    # ------------------------------------------------------------------ #

    def _load_master(self) -> pd.DataFrame:
        """Carga el dataset maestro una sola vez en memoria."""
        if self._master is None:
            if not _MASTER_CSV.exists():
                raise FileNotFoundError(
                    f"Dataset maestro no encontrado: {_MASTER_CSV}. "
                    "Ejecuta: python main.py --etl"
                )
            self._master = pd.read_csv(_MASTER_CSV, parse_dates=["date"])
            logger.info(
                f"VisualizationService: dataset cargado — "
                f"{len(self._master):,} filas, "
                f"{self._master['ticker'].nunique()} activos"
            )
        return self._master

    # ------------------------------------------------------------------ #
    # Heatmap                                                              #
    # ------------------------------------------------------------------ #

    def get_heatmap(self, compact: bool = False) -> Dict[str, Any]:
        """
        Devuelve el heatmap de correlación de Pearson.

        El resultado se cachea tras la primera llamada.
        La correlación se calcula usando el algoritmo manual del Req #2.

        Args:
            compact: Si True, genera figura sin anotaciones de texto
                     (para uso embebido en el dashboard).

        Returns:
            Dict JSON-serializable con:
              - figure:      figura Plotly {data, layout}.
              - tickers:     lista de tickers.
              - matrix:      matriz 2D de coeficientes.
              - extremes:    pares más y menos correlacionados.
        """
        cm = get_correlation_matrix()
        tickers, matrix = cm.get()

        figure   = build_heatmap_figure(tickers, matrix, compact=compact)
        extremes = extract_extremes(tickers, matrix, top_n=5)

        result = {
            "figure":  figure,
            "tickers": tickers,
            "matrix":  matrix,
            "extremes": extremes,
        }
        return to_json_safe(result)

    # ------------------------------------------------------------------ #
    # Candlestick                                                          #
    # ------------------------------------------------------------------ #

    def get_candlestick(
        self,
        ticker: str,
        period: str = "1y",
        sma_periods: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Devuelve la figura Plotly de tipo candlestick para un activo.

        Incluye SMAs calculadas manualmente (SMA 20 y SMA 50 por defecto).

        Args:
            ticker:      Símbolo del activo (ej. "AAPL").
            period:      Período temporal: "1m"|"3m"|"6m"|"1y"|"3y"|"all".
            sma_periods: Períodos de las SMAs a superponer.

        Returns:
            Dict con figure (Plotly), stats (métricas del período), sma_current.

        Raises:
            ValueError: Si el ticker no existe.
        """
        if sma_periods is None:
            sma_periods = [20, 50]

        master = self._load_master()
        result = prepare_ohlcv(ticker, master, period=period, sma_periods=sma_periods)
        return to_json_safe(result)

    # ------------------------------------------------------------------ #
    # Tickers disponibles                                                  #
    # ------------------------------------------------------------------ #

    def get_tickers(self) -> List[str]:
        """Devuelve la lista de tickers disponibles en el dataset."""
        master = self._load_master()
        return sorted(master["ticker"].unique().tolist())


# ── Singleton global ─────────────────────────────────────────────────
_viz_instance: Optional[VisualizationService] = None


def get_visualization_service() -> VisualizationService:
    global _viz_instance
    if _viz_instance is None:
        _viz_instance = VisualizationService()
    return _viz_instance
