"""
src/similarity/__init__.py
===========================
Paquete de algoritmos de similitud de series de tiempo.

Requerimiento 2 — Análisis de Algoritmos, Universidad del Quindío.

Algoritmos implementados:
  - Distancia Euclidiana      O(n)
  - Correlación de Pearson    O(n)
  - Similitud por Coseno      O(n)
  - Dynamic Time Warping DTW  O(n²)

Uso rápido:
    from src.similarity import compare_assets
    result = compare_assets("AAPL", "MSFT", metric="pearson")
"""

from src.similarity.similarity_service import (
    SimilarityService,
    compare_assets,
    AVAILABLE_METRICS,
)

__all__ = [
    "SimilarityService",
    "compare_assets",
    "AVAILABLE_METRICS",
]
