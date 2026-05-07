"""
src/api/routes/home_routes.py
==============================
Blueprint para la ruta principal (/).
"""

from flask import Blueprint, render_template

home_bp = Blueprint("home", __name__)


@home_bp.route("/")
def index():
    """Renderiza la página de inicio."""
    return render_template("index.html")
