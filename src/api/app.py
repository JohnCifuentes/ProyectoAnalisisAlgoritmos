"""
src/api/app.py
===============
Flask application factory para el proyecto de Análisis de Algoritmos.

Crea y configura la aplicación Flask con:
  - Carpetas de templates y estáticos apuntando a src/templates/ y src/static/.
  - Blueprints registrados para home, dataset, similarity, analytics y patterns.
  - Respuestas JSON consistentes para todos los errores HTTP.

Uso (desde main.py):
    from src.api.app import create_app
    app = create_app()
    app.run(host="127.0.0.1", port=5000)
"""

import logging
import os
from pathlib import Path

from flask import Flask, jsonify

logger = logging.getLogger(__name__)

# Directorio src/ (un nivel arriba de api/)
_SRC_DIR: Path = Path(__file__).resolve().parent.parent


def create_app(debug: bool = False) -> Flask:
    """
    Crea y configura la instancia de Flask.

    Diseño de Application Factory:
      - Permite crear múltiples instancias (útil para testing).
      - Los blueprints se registran dentro de la función, no en el módulo.
      - Los servicios se instancian lazy (primera request).

    Args:
        debug: Si True, activa el modo debug de Flask (recarga automática,
               traceback en el navegador). Nunca usar debug=True en producción.

    Returns:
        Instancia de Flask completamente configurada.
    """
    app = Flask(
        __name__,
        template_folder=str(_SRC_DIR / "templates"),
        static_folder=str(_SRC_DIR / "static"),
        static_url_path="/static",
    )

    # ── Configuración ───────────────────────────────────────────────────
    # SECRET_KEY: en producción (Render) debe setearse como variable de entorno.
    # Si no está definida, usa el fallback (solo aceptable en desarrollo).
    app.config["SECRET_KEY"]     = os.environ.get(
        "SECRET_KEY", "analisis-algoritmos-uq-2026-dev"
    )
    app.config["DEBUG"]          = debug
    app.config["JSON_SORT_KEYS"] = False          # Mantener orden de claves

    # ── Registrar blueprints ────────────────────────────────────────────
    from src.api.routes.home_routes            import home_bp
    from src.api.routes.dataset_routes         import dataset_bp
    from src.api.routes.similarity_routes      import similarity_bp
    from src.api.routes.analytics_routes       import analytics_bp
    from src.api.routes.patterns_routes        import patterns_bp
    from src.api.routes.visualization_routes   import visualization_bp
    from src.api.routes.dashboard_routes       import dashboard_bp

    app.register_blueprint(home_bp)
    app.register_blueprint(dataset_bp)
    app.register_blueprint(similarity_bp)
    app.register_blueprint(analytics_bp)
    app.register_blueprint(patterns_bp)
    app.register_blueprint(visualization_bp)
    app.register_blueprint(dashboard_bp)

    # ── Manejadores de error globales ────────────────────────────────────
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Recurso no encontrado", "status": 404}), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({"error": "Método no permitido", "status": 405}), 405

    @app.errorhandler(500)
    def internal_error(e):
        logger.exception("Error interno del servidor")
        return jsonify({"error": "Error interno del servidor", "status": 500}), 500

    logger.info("Flask app creada. Blueprints: home, dataset, similarity, analytics, patterns, visualization, dashboard")
    return app
