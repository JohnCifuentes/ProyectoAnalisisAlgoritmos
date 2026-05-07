"""
src/api/services/dataset_service.py
=====================================
Capa de servicio web para acceso al dataset maestro.

Responsabilidades:
  - Cargar master_dataset.csv una sola vez (caché en memoria).
  - Exponer registros con paginación y filtrado por ticker.
  - Serializar a estructuras JSON-compatibles (Python dicts nativos).

NO duplica lógica ETL — solo lee el CSV ya generado.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config.settings import PROCESSED_DIR

logger = logging.getLogger(__name__)

_MASTER_PATH: Path = PROCESSED_DIR / "master_dataset.csv"
_SUMMARY_PATH: Path = PROCESSED_DIR / "dataset_summary.csv"


class DatasetService:
    """
    Servicio de acceso al dataset maestro con caché en memoria.

    El DataFrame se carga una sola vez en el primer uso y se reutiliza
    para todas las requests subsiguientes.
    """

    def __init__(self, master_path: Optional[Path] = None) -> None:
        self._master_path = master_path or _MASTER_PATH
        self._df: Optional[pd.DataFrame] = None

    def _load(self) -> pd.DataFrame:
        """Carga el dataset maestro si no está en caché."""
        if self._df is None:
            if not self._master_path.exists():
                raise FileNotFoundError(
                    f"Dataset maestro no encontrado: {self._master_path}. "
                    "Ejecuta: python main.py --etl"
                )
            self._df = pd.read_csv(self._master_path, parse_dates=["date"])
            logger.info(
                f"Dataset maestro cargado: {len(self._df):,} filas, "
                f"{self._df['ticker'].nunique()} activos"
            )
        return self._df

    # ------------------------------------------------------------------ #

    def get_tickers(self) -> List[str]:
        """Devuelve la lista de tickers disponibles, ordenada alfabéticamente."""
        df = self._load()
        return sorted(df["ticker"].unique().tolist())

    def get_records(
        self,
        ticker: Optional[str] = None,
        page: int = 1,
        per_page: int = 50,
    ) -> Dict[str, Any]:
        """
        Devuelve registros paginados del dataset maestro.

        Args:
            ticker:   Filtrar por ticker. None = todos los activos.
            page:     Número de página (1-based).
            per_page: Filas por página (máximo 200).

        Returns:
            Dict con data (lista de registros), page, per_page, total, total_pages.

        Raises:
            ValueError: Si el ticker no existe en el dataset.
        """
        df = self._load()

        if ticker:
            if ticker not in df["ticker"].values:
                available = sorted(df["ticker"].unique().tolist())
                raise ValueError(
                    f"Ticker '{ticker}' no encontrado. Disponibles: {available}"
                )
            df = df[df["ticker"] == ticker]

        df = df.sort_values(["date", "ticker"], ascending=[False, True])

        total = len(df)
        total_pages = max(1, (total + per_page - 1) // per_page)
        page = min(page, total_pages)

        start = (page - 1) * per_page
        end   = start + per_page
        page_df = df.iloc[start:end]

        records = []
        for _, row in page_df.iterrows():
            records.append({
                "date":            str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"]),
                "ticker":          str(row["ticker"]),
                "open":            round(float(row["open"]),  4) if pd.notna(row["open"])  else None,
                "high":            round(float(row["high"]),  4) if pd.notna(row["high"])  else None,
                "low":             round(float(row["low"]),   4) if pd.notna(row["low"])   else None,
                "close":           round(float(row["close"]), 4) if pd.notna(row["close"]) else None,
                "volume":          int(row["volume"]) if pd.notna(row["volume"]) else None,
                "instrument_type": str(row["instrument_type"]),
            })

        return {
            "data":        records,
            "ticker":      ticker,
            "page":        page,
            "per_page":    per_page,
            "total":       total,
            "total_pages": total_pages,
        }

    def get_summary(self) -> List[Dict]:
        """
        Devuelve estadísticas resumidas por activo.

        Lee dataset_summary.csv si existe; de lo contrario lo computa
        directamente del dataset maestro.
        """
        # Intentar leer el CSV de resumen ya generado por el ETL
        if _SUMMARY_PATH.exists():
            df = pd.read_csv(_SUMMARY_PATH)
            return df.to_dict(orient="records")

        # Fallback: computar desde el dataset maestro
        df = self._load()
        rows = []
        for ticker, grp in df.groupby("ticker"):
            dates = pd.to_datetime(grp["date"])
            rows.append({
                "ticker":       ticker,
                "n_records":    len(grp),
                "date_start":   str(dates.min().date()),
                "date_end":     str(dates.max().date()),
                "instrument_type": str(grp["instrument_type"].iloc[0]),
            })
        rows.sort(key=lambda x: x["ticker"])
        return rows
