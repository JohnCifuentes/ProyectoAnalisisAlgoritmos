"""
src/api/routes/patterns_routes.py
====================================
Blueprint para detección y visualización de patrones de precio.

Rutas:
  GET /patterns              → template HTML de la vista
  GET /api/patterns          → análisis de patrones de todos los activos
  GET /api/patterns/<ticker> → patrones de un activo específico (con ocurrencias)
"""

import logging

from flask import Blueprint, jsonify, render_template, request

from src.api.services.analytics_service import AnalyticsWebService

logger = logging.getLogger(__name__)

patterns_bp = Blueprint("patterns", __name__)

# Comparte el singleton con analytics (mismo dataset en caché)
_svc = AnalyticsWebService()


@patterns_bp.route("/patterns")
def patterns_view():
    """Renderiza la vista de detección de patrones."""
    return render_template("patterns.html")


@patterns_bp.route("/api/patterns")
def api_patterns():
    """
    Devuelve el análisis de patrones para todos los activos.

    Query params:
        ticker (str, optional): filtrar por ticker.

    Returns JSON:
        {
          "patterns": [
            {
              "ticker":          "TSLA",
              "instrument_type": "EQUITY",
              "pattern":         "consecutive_gains_3d",
              "description":     "3 días consecutivos al alza",
              "n_occurrences":   180,
              "total_windows":   1253,
              "frequency_pct":   14.37
            }, ...
          ]
        }
    """
    ticker = request.args.get("ticker", "").strip().upper() or None
    try:
        data = _svc.get_pattern_analysis(ticker=ticker)
        return jsonify(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Error al obtener análisis de patrones")
        return jsonify({"error": str(exc)}), 500


@patterns_bp.route("/api/patterns/<ticker>")
def api_patterns_ticker(ticker: str):
    """
    Devuelve el análisis detallado de patrones para un activo específico,
    incluyendo la lista completa de ocurrencias individuales.

    Args:
        ticker: Símbolo del activo (ej. NVDA).

    Returns JSON:
        {
          "ticker": "NVDA",
          "consecutive_gains": {
            "n_occurrences": 180,
            "frequency": 0.1437,
            "occurrences": [ { start_date, end_date, gain_pct }, ... ]
          },
          "drop_and_recovery": {
            "n_occurrences": 25,
            "frequency": 0.020,
            "occurrences": [ { drop_date, recovery_date, drop_pct, recovery_pct }, ... ]
          }
        }
    """
    ticker = ticker.upper().strip()
    try:
        data = _svc.get_pattern_detail(ticker=ticker)
        return jsonify(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception(f"Error al obtener patrones detallados para {ticker}")
        return jsonify({"error": str(exc)}), 500
