"""
src/api/routes/visualization_routes.py
=========================================
Blueprint para las vistas de heatmap y candlestick.

Rutas:
  GET /visualizations              → hub de visualizaciones
  GET /heatmap                     → vista del heatmap interactivo
  GET /api/heatmap                 → JSON figura Plotly
  GET /api/heatmap?compact=true    → versión compacta para el dashboard
  GET /candlestick                 → vista del candlestick
  GET /api/candlestick             → JSON figura + estadísticas
      ?ticker=AAPL
      &period=1y
      &sma=20,50
"""

import logging

from flask import Blueprint, jsonify, render_template, request

from src.api.services.visualization_service import get_visualization_service

logger = logging.getLogger(__name__)

visualization_bp = Blueprint("visualization", __name__)

# Singleton del servicio
_svc = get_visualization_service()


# ── Vistas HTML ───────────────────────────────────────────────────────

@visualization_bp.route("/visualizations")
def visualizations_view():
    """Hub de visualizaciones."""
    return render_template("visualizations.html")


@visualization_bp.route("/heatmap")
def heatmap_view():
    """Renderiza la vista del heatmap de correlación."""
    return render_template("heatmap.html")


@visualization_bp.route("/candlestick")
def candlestick_view():
    """Renderiza la vista del candlestick OHLC."""
    return render_template("candlestick.html")


# ── APIs JSON ─────────────────────────────────────────────────────────

@visualization_bp.route("/api/heatmap")
def api_heatmap():
    """
    Devuelve la figura Plotly del heatmap de correlación de Pearson.

    Query params:
        compact (bool, default=false): Si true, figura sin anotaciones de texto.

    Returns JSON:
        {
          "figure":   {data: [...], layout: {...}},
          "tickers":  [...],
          "matrix":   [[...], ...],
          "extremes": {
            "most_correlated":  [{ticker_a, ticker_b, value}, ...],
            "least_correlated": [{ticker_a, ticker_b, value}, ...],
            "top_pair":   {ticker_a, ticker_b, value},
            "bottom_pair": {ticker_a, ticker_b, value}
          }
        }
    """
    compact_str = request.args.get("compact", "false").lower()
    compact     = compact_str in ("1", "true", "yes")

    try:
        data = _svc.get_heatmap(compact=compact)
        return jsonify(data)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        logger.exception("Error al generar heatmap")
        return jsonify({"error": str(exc)}), 500


@visualization_bp.route("/api/candlestick")
def api_candlestick():
    """
    Devuelve la figura Plotly candlestick para un activo.

    Query params:
        ticker (str, default="AAPL"):    Símbolo del activo.
        period (str, default="1y"):       Período: 1m|3m|6m|1y|3y|all.
        sma    (str, default="20,50"):   Períodos SMA separados por coma.

    Returns JSON:
        {
          "figure":      {data, layout},
          "stats":       {ticker, name, n_candles, current_close, ...},
          "sma_current": {"20": 183.2, "50": 179.8}
        }
    """
    ticker     = request.args.get("ticker", "AAPL").strip().upper()
    period     = request.args.get("period", "1y").strip().lower()
    sma_str    = request.args.get("sma", "20,50").strip()

    # Parsear SMA periods
    sma_periods = []
    for part in sma_str.split(","):
        part = part.strip()
        if part.isdigit():
            p = int(part)
            if 1 <= p <= 500:
                sma_periods.append(p)
    if not sma_periods:
        sma_periods = [20, 50]

    try:
        data = _svc.get_candlestick(ticker, period=period, sma_periods=sma_periods)
        return jsonify(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        logger.exception(f"Error al generar candlestick para {ticker}")
        return jsonify({"error": str(exc)}), 500
