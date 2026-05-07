"""
src/api/services/dashboard_service.py
========================================
Capa web para el dashboard principal.

Agrega KPIs del dataset, volatilidad, riesgo y correlación en una sola
llamada. Cachea el resultado para respuestas rápidas.

Delega en:
  - src.visualization.dashboard_service  → KPIs
  - src.api.services.visualization_service → correlación (heatmap)
  - src.config.settings.PROCESSED_DIR    → CSVs de analytics
"""

import logging
from typing import Any, Dict, Optional

from src.visualization.dashboard_service import get_kpis
from src.api.services.visualization_service import get_visualization_service
from src.visualization.visualization_utils import to_json_safe

logger = logging.getLogger(__name__)

# Caché en memoria
_summary_cache: Optional[Dict] = None


class DashboardApiService:
    """
    Servicio del dashboard principal.

    Construye el resumen completo de KPIs combinando:
    - Estadísticas del dataset maestro.
    - Ranking de riesgo.
    - Par más correlacionado (del heatmap).
    """

    def get_summary(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Devuelve el resumen de KPIs para el dashboard.

        Args:
            use_cache: Si True (default), devuelve el resumen en caché
                       para evitar recalcular el heatmap en cada request.

        Returns:
            Dict JSON-serializable con todos los KPIs del dashboard.
        """
        global _summary_cache

        if use_cache and _summary_cache is not None:
            return _summary_cache

        logger.info("DashboardApiService: calculando resumen…")

        # 1. Obtener par más correlacionado del heatmap
        top_pair = None
        try:
            viz = get_visualization_service()
            heatmap_data = viz.get_heatmap(compact=True)
            extremes     = heatmap_data.get("extremes", {})
            top_pair     = extremes.get("top_pair")
        except Exception as exc:
            logger.warning(f"No se pudo obtener par correlacionado: {exc}")

        # 2. Construir KPIs combinados
        kpis = get_kpis(most_correlated_pair=top_pair)

        result = to_json_safe(kpis)
        _summary_cache = result
        return result

    def invalidate_cache(self) -> None:
        """Invalida el caché para forzar recalculación."""
        global _summary_cache
        _summary_cache = None


# ── Singleton global ─────────────────────────────────────────────────
_dash_instance: Optional[DashboardApiService] = None


def get_dashboard_service() -> DashboardApiService:
    global _dash_instance
    if _dash_instance is None:
        _dash_instance = DashboardApiService()
    return _dash_instance
