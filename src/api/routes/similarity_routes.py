"""
src/api/routes/similarity_routes.py
=====================================
Blueprint para comparación de similitud entre pares de activos.

Rutas:
  GET  /similarity          → template HTML de la vista
  POST /api/similarity      → ejecuta un algoritmo de similitud
  POST /api/similarity/all  → ejecuta los 4 algoritmos sobre el mismo par
"""

import logging

from flask import Blueprint, jsonify, render_template, request

from src.api.services.similarity_service import SimilarityWebService

logger = logging.getLogger(__name__)

similarity_bp = Blueprint("similarity", __name__)

# Singleton del servicio
_svc = SimilarityWebService()


@similarity_bp.route("/similarity")
def similarity_view():
    """Renderiza la vista de comparación de similitud."""
    return render_template("similarity.html")


@similarity_bp.route("/api/similarity", methods=["POST"])
def api_similarity():
    """
    Compara dos activos usando el algoritmo especificado.

    Request JSON:
        {
          "ticker_a": "AAPL",
          "ticker_b": "MSFT",
          "metric":   "pearson",   // euclidean | pearson | cosine | dtw
          "on":       "returns"    // returns | prices (optional, default: returns)
        }

    Returns JSON:
        {
          "ticker_a":       "AAPL",
          "ticker_b":       "MSFT",
          "metric":         "pearson",
          "value":          0.8742,
          "interpretation": "Correlación alta positiva",
          "n_points":       1249,
          "date_start":     "2021-05-07",
          "date_end":       "2026-05-07",
          "on":             "returns",
          "elapsed_ms":     2.3
        }
    """
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Se requiere un body JSON válido"}), 400

    ticker_a = str(body.get("ticker_a", "")).strip().upper()
    ticker_b = str(body.get("ticker_b", "")).strip().upper()
    metric   = str(body.get("metric", "pearson")).strip().lower()
    on       = str(body.get("on", "returns")).strip().lower()

    if not ticker_a or not ticker_b:
        return jsonify({"error": "ticker_a y ticker_b son requeridos"}), 400

    try:
        result = _svc.compare(ticker_a, ticker_b, metric, on)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception(f"Error en comparación {ticker_a} vs {ticker_b}")
        return jsonify({"error": str(exc)}), 500


@similarity_bp.route("/api/similarity/all", methods=["POST"])
def api_similarity_all():
    """
    Ejecuta los 4 algoritmos de similitud sobre el mismo par de activos.

    Request JSON: { "ticker_a": "AAPL", "ticker_b": "MSFT", "on": "returns" }

    Returns JSON:
        {
          "ticker_a": "AAPL",
          "ticker_b": "MSFT",
          "results": {
            "euclidean": { "value": ..., "interpretation": ... },
            "pearson":   { "value": ..., "interpretation": ... },
            "cosine":    { "value": ..., "interpretation": ... },
            "dtw":       { "value": ..., "interpretation": ... }
          }
        }
    """
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Se requiere un body JSON válido"}), 400

    ticker_a = str(body.get("ticker_a", "")).strip().upper()
    ticker_b = str(body.get("ticker_b", "")).strip().upper()
    on       = str(body.get("on", "returns")).strip().lower()

    if not ticker_a or not ticker_b:
        return jsonify({"error": "ticker_a y ticker_b son requeridos"}), 400

    try:
        result = _svc.compare_all(ticker_a, ticker_b, on)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception(f"Error en comparación completa {ticker_a} vs {ticker_b}")
        return jsonify({"error": str(exc)}), 500
