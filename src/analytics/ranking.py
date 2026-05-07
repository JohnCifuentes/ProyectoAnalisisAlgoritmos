"""
src/analytics/ranking.py
==========================
Construcción y persistencia del ranking de riesgo de activos financieros.

El ranking ordena los activos de mayor a menor volatilidad anualizada,
asignando una posición numérica (rank 1 = más volátil / más riesgoso).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMATO DE SALIDA  →  data/processed/risk_ranking.csv
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  rank | ticker | instrument_type | annualized_vol_pct | daily_vol_pct
       | risk_category | n_observations | date_start | date_end

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPLEJIDAD COMPUTACIONAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  build_risk_ranking(): O(k log k) — ordenamiento de k activos
                        (k = 20 en el portafolio actual).
  save_risk_ranking():  O(k) — escritura secuencial de k filas.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.config.settings import PROCESSED_DIR

logger = logging.getLogger(__name__)

# Ruta de salida del ranking
RISK_RANKING_PATH: Path = PROCESSED_DIR / "risk_ranking.csv"


# ====================================================================== #
# Construcción del ranking                                                 #
# ====================================================================== #

def build_risk_ranking(volatility_records: List[Dict]) -> pd.DataFrame:
    """
    Construye el DataFrame de ranking de riesgo a partir de registros
    de volatilidad por activo.

    Ordena descendentemente por volatilidad anualizada y asigna el
    número de ranking (1 = activo más volátil / riesgoso).

    Complejidad: O(k log k) — sort de k activos.

    Args:
        volatility_records: Lista de dicts, uno por activo. Cada dict
            debe contener al menos:
              - ticker (str)
              - instrument_type (str)
              - daily_volatility (float)
              - annualized_volatility (float)
              - risk_category (str)
              - n_observations (int)
              - date_start (str)
              - date_end (str)

    Returns:
        DataFrame ordenado descendente por annualized_volatility, con
        columna 'rank' (1 = mayor volatilidad).

    Raises:
        ValueError: Si la lista de registros está vacía.
    """
    if not volatility_records:
        raise ValueError(
            "La lista de registros de volatilidad está vacía. "
            "Ejecuta AnalyticsService.run_full_analysis() primero."
        )

    df = pd.DataFrame(volatility_records)

    # Ordenar descendente por volatilidad anualizada — O(k log k)
    df = df.sort_values("annualized_volatility", ascending=False)
    df = df.reset_index(drop=True)

    # Asignar rank 1-based
    df.insert(0, "rank", range(1, len(df) + 1))

    # Expresar volatilidades en porcentaje para legibilidad humana
    df["annualized_vol_pct"] = (df["annualized_volatility"] * 100).round(2)
    df["daily_vol_pct"]      = (df["daily_volatility"] * 100).round(4)

    # Columnas de salida en orden lógico
    output_cols = [
        "rank",
        "ticker",
        "instrument_type",
        "annualized_vol_pct",
        "daily_vol_pct",
        "risk_category",
        "n_observations",
        "date_start",
        "date_end",
    ]

    # Incluir solo las columnas que existen (robustez)
    available = [c for c in output_cols if c in df.columns]
    return df[available]


# ====================================================================== #
# Persistencia                                                             #
# ====================================================================== #

def save_risk_ranking(
    df: pd.DataFrame,
    path: Optional[Path] = None,
) -> Path:
    """
    Guarda el DataFrame de ranking en data/processed/risk_ranking.csv.

    Args:
        df:   DataFrame producido por build_risk_ranking().
        path: Ruta alternativa. Si es None, usa RISK_RANKING_PATH.

    Returns:
        Ruta del archivo CSV guardado.
    """
    target: Path = path or RISK_RANKING_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False, encoding="utf-8")
    logger.info(f"Ranking de riesgo guardado: {target}  ({len(df)} activos)")
    return target
