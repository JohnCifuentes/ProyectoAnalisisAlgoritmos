"""
src/analytics/__init__.py
==========================
Paquete de análisis financiero — Req. #3.

Exportaciones principales:
  - AnalyticsService:       Servicio orquestador completo.
  - analyze_asset():        Función de acceso rápido para un activo.
  - generate_risk_ranking(): Ranking de riesgo de todo el portafolio.
"""

from src.analytics.analytics_service import (
    AnalyticsService,
    analyze_asset,
    generate_risk_ranking,
)
from src.analytics.returns import compute_simple_returns, compute_log_returns
from src.analytics.volatility import (
    compute_daily_volatility,
    compute_annualized_volatility,
    compute_volatility,
)
from src.analytics.risk_classifier import (
    classify_risk,
    classify_risk_with_score,
    RISK_CONSERVATIVE,
    RISK_MODERATE,
    RISK_AGGRESSIVE,
    THRESHOLD_CONSERVATIVE,
    THRESHOLD_MODERATE,
)
from src.analytics.sliding_window import generate_windows, get_all_windows
from src.analytics.pattern_detector import (
    detect_consecutive_gains,
    detect_drop_and_recovery,
)
from src.analytics.ranking import build_risk_ranking, save_risk_ranking

__all__ = [
    # Servicio principal
    "AnalyticsService",
    "analyze_asset",
    "generate_risk_ranking",
    # Retornos
    "compute_simple_returns",
    "compute_log_returns",
    # Volatilidad
    "compute_daily_volatility",
    "compute_annualized_volatility",
    "compute_volatility",
    # Clasificación
    "classify_risk",
    "classify_risk_with_score",
    "RISK_CONSERVATIVE",
    "RISK_MODERATE",
    "RISK_AGGRESSIVE",
    "THRESHOLD_CONSERVATIVE",
    "THRESHOLD_MODERATE",
    # Sliding window
    "generate_windows",
    "get_all_windows",
    # Patrones
    "detect_consecutive_gains",
    "detect_drop_and_recovery",
    # Ranking
    "build_risk_ranking",
    "save_risk_ranking",
]
