"""
src/api/routes/analytics_routes.py
=====================================
Blueprint para volatilidad y ranking de riesgo.

Rutas:
  GET /analytics           → template HTML de la vista
  GET /api/risk-ranking    → ranking de riesgo de todos los activos
  GET /api/volatility      → resumen de volatilidad
  GET /api/asset/<ticker>  → análisis completo de un activo

  — Preparado para Req #4 (dashboard con gráficas) —
  GET /api/ohlcv/<ticker>              → precios OHLCV (placeholder)
  GET /api/correlation-matrix          → matriz de correlaciones (placeholder)
"""

import logging

from flask import Blueprint, jsonify, render_template, request

from src.api.services.analytics_service import AnalyticsWebService

logger = logging.getLogger(__name__)

analytics_bp = Blueprint("analytics", __name__)

# Singleton del servicio
_svc = AnalyticsWebService()


@analytics_bp.route("/analytics")
def analytics_view():
    """Renderiza la vista de volatilidad y ranking de riesgo."""
    return render_template("analytics.html")


@analytics_bp.route("/api/risk-ranking")
def api_risk_ranking():
    """
    Devuelve el ranking de riesgo de todos los activos, ordenado
    descendentemente por volatilidad anualizada.

    Returns JSON:
        {
          "ranking": [
            {
              "rank": 1,
              "ticker": "TSLA",
              "instrument_type": "EQUITY",
              "annualized_vol_pct": 58.9,
              "daily_vol_pct": 3.7,
              "risk_category": "Agresivo",
              "n_observations": 1256,
              "date_start": "2021-05-07",
              "date_end": "2026-05-07"
            }, ...
          ],
          "summary": {
            "Conservador": 5,
            "Moderado": 12,
            "Agresivo": 3
          }
        }
    """
    try:
        data = _svc.get_risk_ranking()
        return jsonify(data)
    except Exception as exc:
        logger.exception("Error al obtener risk ranking")
        return jsonify({"error": str(exc)}), 500


@analytics_bp.route("/api/volatility")
def api_volatility():
    """
    Devuelve el resumen de volatilidad de todos los activos.

    Returns JSON:
        { "volatility": [ { ticker, annualized_vol_pct, daily_vol_pct, ... }, ... ] }
    """
    try:
        data = _svc.get_volatility_summary()
        return jsonify(data)
    except Exception as exc:
        logger.exception("Error al obtener volatility summary")
        return jsonify({"error": str(exc)}), 500


@analytics_bp.route("/api/asset/<ticker>")
def api_asset(ticker: str):
    """
    Devuelve el análisis completo de un activo específico.

    Args:
        ticker: Símbolo del activo (ej. NVDA).

    Returns JSON:
        {
          "ticker": "NVDA",
          "daily_volatility": 0.032,
          "annualized_volatility": 0.517,
          "risk_category": "Agresivo",
          "n_observations": 1256,
          "date_start": "2021-05-07",
          "date_end": "2026-05-07",
          "patterns": { ... }
        }
    """
    ticker = ticker.upper().strip()
    try:
        data = _svc.get_asset_analysis(ticker)
        return jsonify(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception(f"Error al analizar activo {ticker}")
        return jsonify({"error": str(exc)}), 500


# ── Placeholders para Req #4 (dashboard con gráficas) ──────────────────

@analytics_bp.route("/api/ohlcv/<ticker>")
def api_ohlcv(ticker: str):
    """
    [REQ #4 PLACEHOLDER] Datos OHLCV de un activo para candlestick charts.
    Será implementado en el Requerimiento #4.
    """
    ticker = ticker.upper().strip()
    try:
        data = _svc.get_ohlcv(ticker)
        return jsonify(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception(f"Error al obtener OHLCV {ticker}")
        return jsonify({"error": str(exc)}), 500


@analytics_bp.route("/api/correlation-matrix")
def api_correlation_matrix():
    """
    [REQ #4 PLACEHOLDER] Matriz de correlaciones para heatmap.
    Será implementado en el Requerimiento #4.

    Query params:
        metric (str, default="pearson"): métrica de similitud.
        on     (str, default="returns"): sobre retornos o precios.
    """
    metric = request.args.get("metric", "pearson")
    on     = request.args.get("on", "returns")
    try:
        data = _svc.get_correlation_matrix(metric=metric, on=on)
        return jsonify(data)
    except Exception as exc:
        logger.exception("Error al calcular correlation matrix")
        return jsonify({"error": str(exc)}), 500
