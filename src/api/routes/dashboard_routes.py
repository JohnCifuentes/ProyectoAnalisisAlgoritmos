"""
src/api/routes/dashboard_routes.py
=====================================
Blueprint para el dashboard financiero principal.

Rutas:
  GET /dashboard               → template HTML del dashboard
  GET /api/dashboard-summary   → KPIs JSON del dashboard
"""

import logging

from flask import Blueprint, jsonify, render_template

from src.api.services.dashboard_service import get_dashboard_service

logger = logging.getLogger(__name__)

dashboard_bp = Blueprint("dashboard", __name__)

# Singleton del servicio
_svc = get_dashboard_service()


@dashboard_bp.route("/dashboard")
def dashboard_view():
    """Renderiza el dashboard financiero principal."""
    return render_template("dashboard.html")


@dashboard_bp.route("/api/dashboard-summary")
def api_dashboard_summary():
    """
    Devuelve el resumen de KPIs para el dashboard.

    Returns JSON:
        {
          "n_assets":           20,
          "n_records":          25120,
          "date_start":         "2021-05-07",
          "date_end":           "2026-05-07",
          "avg_volatility_pct": 25.4,
          "most_aggressive":    {"ticker": "TSLA", "annualized_vol_pct": 58.94, ...},
          "most_conservative":  {"ticker": "GLD",  "annualized_vol_pct": 12.3,  ...},
          "risk_categories":    {"Conservador": 5, "Moderado": 12, "Agresivo": 3},
          "top_correlated_pair": {"ticker_a": "SPY", "ticker_b": "VOO", "value": 0.998}
        }
    """
    try:
        data = _svc.get_summary()
        return jsonify(data)
    except Exception as exc:
        logger.exception("Error al obtener dashboard summary")
        return jsonify({"error": str(exc)}), 500
