"""
src/api/services/analytics_service.py
========================================
Capa web para los servicios de analytics y patrones.

Delega todo el cálculo a src.analytics.AnalyticsService (Req #3).
Lee los CSVs ya generados cuando están disponibles (respuesta rápida).
Si los CSVs no existen, los computa y guarda en data/processed/.

NO reimplementa ningún algoritmo.
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.analytics.analytics_service import AnalyticsService as _CoreService
from src.config.settings import PROCESSED_DIR

logger = logging.getLogger(__name__)

_RANKING_CSV  = PROCESSED_DIR / "risk_ranking.csv"
_VOL_CSV      = PROCESSED_DIR / "volatility_summary.csv"
_PATTERN_CSV  = PROCESSED_DIR / "pattern_analysis.csv"

# Singleton del servicio core
_core_instance: Optional[_CoreService] = None


def _get_core() -> _CoreService:
    global _core_instance
    if _core_instance is None:
        _core_instance = _CoreService()
    return _core_instance


class AnalyticsWebService:
    """
    Wrapper web sobre AnalyticsService (Req #3).

    Prioriza leer los CSVs ya generados para evitar recomputar.
    Computa on-demand solo si los archivos no existen.
    """

    # ------------------------------------------------------------------ #
    # Risk ranking                                                         #
    # ------------------------------------------------------------------ #

    def get_risk_ranking(self) -> Dict[str, Any]:
        """
        Devuelve el ranking de riesgo de todos los activos.

        Returns:
            Dict con 'ranking' (list) y 'summary' (conteo por categoría).
        """
        df = self._load_or_compute_ranking()
        records = _df_to_records(df)

        # Resumen de conteo por categoría
        summary: Dict[str, int] = {}
        for row in records:
            cat = row.get("risk_category", "")
            summary[cat] = summary.get(cat, 0) + 1

        return {"ranking": records, "summary": summary}

    def _load_or_compute_ranking(self) -> pd.DataFrame:
        if _RANKING_CSV.exists():
            return pd.read_csv(_RANKING_CSV)
        logger.info("risk_ranking.csv no encontrado — computando...")
        return _get_core().generate_risk_ranking()

    # ------------------------------------------------------------------ #
    # Volatility summary                                                   #
    # ------------------------------------------------------------------ #

    def get_volatility_summary(self) -> Dict[str, Any]:
        """
        Devuelve el resumen de volatilidad de todos los activos.
        """
        df = self._load_or_compute_volatility()
        return {"volatility": _df_to_records(df)}

    def _load_or_compute_volatility(self) -> pd.DataFrame:
        if _VOL_CSV.exists():
            return pd.read_csv(_VOL_CSV)
        logger.info("volatility_summary.csv no encontrado — computando...")
        return _get_core().generate_volatility_summary()

    # ------------------------------------------------------------------ #
    # Pattern analysis                                                     #
    # ------------------------------------------------------------------ #

    def get_pattern_analysis(
        self, ticker: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Devuelve el análisis de patrones para todos los activos (o uno específico).

        Args:
            ticker: Si se especifica, filtra por ese ticker.
        """
        df = self._load_or_compute_patterns()

        if ticker:
            df = df[df["ticker"] == ticker]
            if df.empty:
                raise ValueError(
                    f"Ticker '{ticker}' no encontrado en el análisis de patrones."
                )

        return {"patterns": _df_to_records(df)}

    def _load_or_compute_patterns(self) -> pd.DataFrame:
        if _PATTERN_CSV.exists():
            return pd.read_csv(_PATTERN_CSV)
        logger.info("pattern_analysis.csv no encontrado — computando...")
        return _get_core().generate_pattern_analysis()

    # ------------------------------------------------------------------ #
    # Análisis completo de un activo                                       #
    # ------------------------------------------------------------------ #

    def get_asset_analysis(self, ticker: str) -> Dict[str, Any]:
        """
        Devuelve el análisis completo de un activo (volatilidad + patrones).
        """
        result = _get_core().analyze_asset(ticker)
        return _clean(result)

    def get_pattern_detail(self, ticker: str) -> Dict[str, Any]:
        """
        Devuelve los patrones detallados (con lista de ocurrencias) para un ticker.
        """
        result = _get_core().analyze_asset(ticker)
        patterns = result.get("patterns", {})
        return {
            "ticker": ticker,
            "consecutive_gains": _clean(patterns.get("consecutive_gains", {})),
            "drop_and_recovery":  _clean(patterns.get("drop_and_recovery", {})),
        }

    # ------------------------------------------------------------------ #
    # Placeholders para Req #4                                             #
    # ------------------------------------------------------------------ #

    def get_ohlcv(self, ticker: str) -> Dict[str, Any]:
        """
        [REQ #4 PLACEHOLDER] Devuelve datos OHLCV de un activo.

        En Req #4 este método alimentará los candlestick charts de Plotly.
        Por ahora devuelve los datos en formato JSON sin gráfica.
        """
        from src.config.settings import PROCESSED_DIR
        master_path = PROCESSED_DIR / "master_dataset.csv"
        if not master_path.exists():
            raise FileNotFoundError("Dataset maestro no encontrado.")

        df = pd.read_csv(master_path, parse_dates=["date"])
        df = df[df["ticker"] == ticker].sort_values("date")

        if df.empty:
            raise ValueError(f"Ticker '{ticker}' no encontrado.")

        records = []
        for _, row in df.iterrows():
            records.append({
                "date":   str(row["date"].date()),
                "open":   round(float(row["open"]),  4),
                "high":   round(float(row["high"]),  4),
                "low":    round(float(row["low"]),   4),
                "close":  round(float(row["close"]), 4),
                "volume": int(row["volume"]) if pd.notna(row["volume"]) else None,
            })

        return {"ticker": ticker, "ohlcv": records}

    def get_correlation_matrix(
        self, metric: str = "pearson", on: str = "returns"
    ) -> Dict[str, Any]:
        """
        [REQ #4 PLACEHOLDER] Matriz de correlaciones para heatmap.

        En Req #4 este método alimentará el heatmap de correlaciones de Plotly.
        Requiere implementación completa con compare_all_pairs() en Req #4.
        """
        return {
            "note": "Placeholder — será implementado en Req #4 con Plotly heatmap.",
            "metric": metric,
            "on": on,
        }


# ====================================================================== #
# Helpers de serialización                                                 #
# ====================================================================== #

def _df_to_records(df: pd.DataFrame) -> List[Dict]:
    """Convierte un DataFrame a una lista de dicts JSON-seguros."""
    records = []
    for _, row in df.iterrows():
        records.append({k: _safe(v) for k, v in row.items()})
    return records


def _safe(v: Any) -> Any:
    """Convierte un valor a tipo Python JSON-compatible."""
    import numpy as np
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        fv = float(v)
        return None if math.isnan(fv) or math.isinf(fv) else fv
    if isinstance(v, np.ndarray):
        return [_safe(x) for x in v.tolist()]
    if hasattr(v, "isoformat"):
        return str(v)
    return v


def _clean(obj: Any) -> Any:
    """Limpia recursivamente un objeto (dict/list/value) para JSON."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return _safe(obj)
