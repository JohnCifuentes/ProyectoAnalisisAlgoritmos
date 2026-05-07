"""
src/api/routes/dataset_routes.py
==================================
Blueprint para el acceso y visualización del dataset maestro.

Rutas:
  GET /dataset              → template HTML de la vista
  GET /api/dataset          → JSON paginado (opcional: ?ticker=AAPL&page=1&per_page=50)
  GET /api/tickers          → JSON con lista de tickers disponibles
  GET /api/dataset/summary  → JSON con estadísticas por activo
"""

import logging

from flask import Blueprint, jsonify, render_template, request

from src.api.services.dataset_service import DatasetService

logger = logging.getLogger(__name__)

dataset_bp = Blueprint("dataset", __name__)

# Singleton del servicio (carga master_dataset.csv una sola vez)
_svc = DatasetService()


@dataset_bp.route("/dataset")
def dataset_view():
    """Renderiza la vista del dataset maestro."""
    return render_template("dataset.html")


@dataset_bp.route("/api/tickers")
def api_tickers():
    """
    Devuelve la lista de tickers disponibles.

    Returns JSON:
        { "tickers": ["AAPL", "AMZN", ...] }
    """
    try:
        return jsonify({"tickers": _svc.get_tickers()})
    except Exception as exc:
        logger.exception("Error al obtener tickers")
        return jsonify({"error": str(exc)}), 500


@dataset_bp.route("/api/dataset")
def api_dataset():
    """
    Devuelve registros del dataset maestro con paginación.

    Query params:
        ticker   (str, optional): filtrar por ticker (ej. AAPL).
        page     (int, default=1): número de página.
        per_page (int, default=50): filas por página.

    Returns JSON:
        {
          "data":       [ {...}, ... ],
          "ticker":     "AAPL" | null,
          "page":       1,
          "per_page":   50,
          "total":      1256,
          "total_pages": 26
        }
    """
    ticker   = request.args.get("ticker", "").strip().upper() or None
    try:
        page     = max(1, int(request.args.get("page", 1)))
        per_page = min(200, max(1, int(request.args.get("per_page", 50))))
    except ValueError:
        return jsonify({"error": "Parámetros de paginación inválidos"}), 400

    try:
        result = _svc.get_records(ticker=ticker, page=page, per_page=per_page)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Error al obtener registros del dataset")
        return jsonify({"error": str(exc)}), 500


@dataset_bp.route("/api/dataset/summary")
def api_dataset_summary():
    """
    Devuelve estadísticas resumidas por activo.

    Returns JSON:
        { "summary": [ { ticker, n_records, date_start, date_end, ... }, ... ] }
    """
    try:
        return jsonify({"summary": _svc.get_summary()})
    except Exception as exc:
        logger.exception("Error al obtener resumen del dataset")
        return jsonify({"error": str(exc)}), 500
