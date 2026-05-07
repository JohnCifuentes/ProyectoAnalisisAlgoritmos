"""
src/api/services/similarity_service.py
========================================
Capa web para el servicio de similitud.

Delega todo el cálculo a src.similarity.SimilarityService (Req #2).
Solo transforma el resultado a tipos JSON-serializables nativos de Python.

NO reimplementa ningún algoritmo.
"""

import logging
from typing import Any, Dict, Optional
from pathlib import Path

from src.similarity.similarity_service import (
    SimilarityService as _CoreService,
    AVAILABLE_METRICS,
)

logger = logging.getLogger(__name__)

# Singleton: se crea en la primera llamada (lazy loading del dataset)
_instance: Optional["SimilarityWebService"] = None


class SimilarityWebService:
    """
    Wrapper web sobre SimilarityService (Req #2).

    Responsabilidades:
      - Instanciar y cachear el servicio core.
      - Convertir resultados numpy/pandas a tipos JSON-nativos.
      - Proporcionar mensajes de error claros para respuestas HTTP.
    """

    def __init__(self) -> None:
        # El servicio core carga el master_dataset.csv una sola vez
        self._core = _CoreService()
        logger.info(
            f"SimilarityWebService listo. "
            f"Tickers disponibles: {len(self._core.available_tickers)}"
        )

    def compare(
        self,
        ticker_a: str,
        ticker_b: str,
        metric: str = "pearson",
        on: str = "returns",
    ) -> Dict[str, Any]:
        """
        Compara dos activos con la métrica especificada.

        Args:
            ticker_a: Símbolo del primer activo.
            ticker_b: Símbolo del segundo activo.
            metric:   "euclidean" | "pearson" | "cosine" | "dtw"
            on:       "returns" | "prices"

        Returns:
            Diccionario JSON-serializable con value, interpretation,
            n_points, date_start, date_end, elapsed_ms.

        Raises:
            ValueError: Si la métrica o algún ticker no son válidos.
        """
        result = self._core.compare_assets(
            ticker_a=ticker_a,
            ticker_b=ticker_b,
            metric=metric,
            on=on,
            include_path=False,  # El warping path es demasiado grande para JSON
        )
        return _to_json(result)

    def compare_all(
        self,
        ticker_a: str,
        ticker_b: str,
        on: str = "returns",
    ) -> Dict[str, Any]:
        """
        Ejecuta los 4 algoritmos sobre el mismo par.

        Returns:
            Dict con claves ticker_a, ticker_b, results (dict por métrica).
        """
        raw = self._core.compare_all_metrics(ticker_a, ticker_b, on=on)

        # compare_all_metrics devuelve claves top-level + por-métrica
        non_metric_keys = {"ticker_a", "ticker_b", "n_points", "date_start", "date_end", "on", "summary"}
        serializable_results = {}
        for key, value in raw.items():
            if key not in non_metric_keys and isinstance(value, dict):
                serializable_results[key] = _to_json(value)

        return {
            "ticker_a":         ticker_a,
            "ticker_b":         ticker_b,
            "on":               on,
            "n_points":         _to_json(raw.get("n_points")),
            "date_start":       _to_json(raw.get("date_start")),
            "date_end":         _to_json(raw.get("date_end")),
            "results":          serializable_results,
            "available_metrics": list(AVAILABLE_METRICS.keys()),
        }

    @property
    def available_tickers(self):
        return self._core.available_tickers


def _to_json(obj: Any) -> Any:
    """
    Convierte recursivamente tipos no-JSON (numpy, pandas Timestamps, etc.)
    a tipos Python nativos seguros para jsonify().
    """
    import numpy as np
    import math

    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, np.ndarray):
        return [_to_json(v) for v in obj.tolist()]
    if hasattr(obj, "isoformat"):  # pandas Timestamp, datetime
        return str(obj)
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    return obj
