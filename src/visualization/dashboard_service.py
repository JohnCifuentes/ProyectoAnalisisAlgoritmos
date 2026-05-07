"""
src/visualization/dashboard_service.py
=========================================
Servicio de cómputo de KPIs para el dashboard principal.

Agrega métricas de volatilidad, riesgo y correlación en una sola
estructura JSON que el dashboard consume para sus tarjetas de KPI.

NO duplica lógica — lee los CSVs ya generados por analytics/ y usa
la correlación ya computada por correlation_matrix.py.
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config.settings import PROCESSED_DIR

logger = logging.getLogger(__name__)

_RANKING_CSV = PROCESSED_DIR / "risk_ranking.csv"
_VOL_CSV     = PROCESSED_DIR / "volatility_summary.csv"
_MASTER_CSV  = PROCESSED_DIR / "master_dataset.csv"


def get_kpis(
    most_correlated_pair: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Construye el diccionario completo de KPIs para el dashboard.

    Args:
        most_correlated_pair: Dict con ticker_a, ticker_b, value (del heatmap).

    Returns:
        Dict con todos los KPIs listos para serializar a JSON.
    """
    kpis: Dict[str, Any] = {}

    # ── KPIs del dataset ──────────────────────────────────────────────
    if _MASTER_CSV.exists():
        try:
            df = pd.read_csv(_MASTER_CSV, usecols=["ticker", "date"])
            kpis["n_assets"]  = int(df["ticker"].nunique())
            kpis["n_records"] = int(len(df))
            kpis["date_start"] = str(df["date"].min())[:10]
            kpis["date_end"]   = str(df["date"].max())[:10]
        except Exception as exc:
            logger.warning(f"No se pudieron cargar KPIs del dataset: {exc}")
            kpis.update({"n_assets": 20, "n_records": 0,
                          "date_start": None, "date_end": None})
    else:
        kpis.update({"n_assets": 20, "n_records": 0,
                      "date_start": None, "date_end": None})

    # ── KPIs de riesgo/volatilidad ────────────────────────────────────
    if _RANKING_CSV.exists():
        try:
            ranking = pd.read_csv(_RANKING_CSV)
            _enrich_risk_kpis(kpis, ranking)
        except Exception as exc:
            logger.warning(f"No se pudieron cargar KPIs de riesgo: {exc}")
            _default_risk_kpis(kpis)
    elif _VOL_CSV.exists():
        try:
            vol = pd.read_csv(_VOL_CSV)
            _enrich_volatility_kpis(kpis, vol)
        except Exception as exc:
            logger.warning(f"No se pudieron cargar KPIs de volatilidad: {exc}")
            _default_risk_kpis(kpis)
    else:
        _default_risk_kpis(kpis)

    # ── KPI de correlación ────────────────────────────────────────────
    kpis["top_correlated_pair"] = most_correlated_pair

    return kpis


def _enrich_risk_kpis(kpis: Dict, ranking: pd.DataFrame) -> None:
    """Llena los KPIs de riesgo desde el risk_ranking.csv."""
    if ranking.empty:
        _default_risk_kpis(kpis)
        return

    # Volatilidad promedio
    if "annualized_vol_pct" in ranking.columns:
        kpis["avg_volatility_pct"] = round(
            float(ranking["annualized_vol_pct"].mean()), 2
        )
    else:
        kpis["avg_volatility_pct"] = None

    # Activo más agresivo (rank 1 = mayor volatilidad)
    if "risk_category" in ranking.columns:
        most_agg = ranking[ranking["risk_category"] == "Agresivo"]
        if not most_agg.empty:
            row = most_agg.iloc[0]
            kpis["most_aggressive"] = {
                "ticker":            str(row.get("ticker", "")),
                "annualized_vol_pct": round(float(row.get("annualized_vol_pct", 0)), 2),
                "risk_category":     "Agresivo",
            }
        else:
            kpis["most_aggressive"] = None

        # Activo más conservador
        most_cons = ranking[ranking["risk_category"] == "Conservador"]
        if not most_cons.empty:
            row = most_cons.iloc[-1]  # último en ranking = menor volatilidad
            kpis["most_conservative"] = {
                "ticker":            str(row.get("ticker", "")),
                "annualized_vol_pct": round(float(row.get("annualized_vol_pct", 0)), 2),
                "risk_category":     "Conservador",
            }
        else:
            kpis["most_conservative"] = None

        # Conteo por categoría
        cat_counts = ranking["risk_category"].value_counts().to_dict()
        kpis["risk_categories"] = {
            "Conservador": int(cat_counts.get("Conservador", 0)),
            "Moderado":    int(cat_counts.get("Moderado",    0)),
            "Agresivo":    int(cat_counts.get("Agresivo",    0)),
        }
    else:
        _default_risk_kpis(kpis)


def _enrich_volatility_kpis(kpis: Dict, vol: pd.DataFrame) -> None:
    """Llena los KPIs desde volatility_summary.csv (fallback)."""
    if vol.empty:
        _default_risk_kpis(kpis)
        return

    if "annualized_vol_pct" in vol.columns:
        kpis["avg_volatility_pct"] = round(
            float(vol["annualized_vol_pct"].mean()), 2
        )
        row_max = vol.loc[vol["annualized_vol_pct"].idxmax()]
        kpis["most_aggressive"] = {
            "ticker":            str(row_max.get("ticker", "")),
            "annualized_vol_pct": round(float(row_max["annualized_vol_pct"]), 2),
            "risk_category":     "Agresivo",
        }
        row_min = vol.loc[vol["annualized_vol_pct"].idxmin()]
        kpis["most_conservative"] = {
            "ticker":            str(row_min.get("ticker", "")),
            "annualized_vol_pct": round(float(row_min["annualized_vol_pct"]), 2),
            "risk_category":     "Conservador",
        }
    else:
        _default_risk_kpis(kpis)


def _default_risk_kpis(kpis: Dict) -> None:
    """Valores por defecto cuando los CSVs no están disponibles."""
    kpis.setdefault("avg_volatility_pct", None)
    kpis.setdefault("most_aggressive",    None)
    kpis.setdefault("most_conservative",  None)
    kpis.setdefault("risk_categories", {"Conservador": 0, "Moderado": 0, "Agresivo": 0})
