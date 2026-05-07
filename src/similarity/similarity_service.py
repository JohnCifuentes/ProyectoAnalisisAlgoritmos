"""
src/similarity/similarity_service.py
=====================================
Capa de servicio para el cálculo de similitud entre activos financieros.

Esta clase orquesta el flujo completo:
  1. Cargar el dataset maestro (una sola vez, en caché).
  2. Preparar las series (retornos, alineación, limpieza).
  3. Despachar al algoritmo correcto.
  4. Retornar un resultado estructurado con valor + metadatos.

Función principal:
    compare_assets(ticker_a, ticker_b, metric="pearson")

Algoritmos disponibles (metric):
    "euclidean"  → distancia euclidiana (normalizada por √n)
    "pearson"    → correlación de Pearson en [-1, 1]
    "cosine"     → similitud coseno en [-1, 1]
    "dtw"        → distancia DTW normalizada

Todas las comparaciones operan sobre retornos diarios por defecto,
aunque se puede cambiar a precios con on="prices".
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.similarity.utils import load_master, prepare_series
from src.similarity.euclidean_distance import (
    euclidean_distance_normalized,
    euclidean_similarity,
)
from src.similarity.pearson_correlation import (
    pearson_correlation,
    pearson_with_components,
    interpret_pearson,
)
from src.similarity.cosine_similarity import (
    cosine_similarity,
    cosine_with_components,
    interpret_cosine,
)
from src.similarity.dynamic_time_warping import dtw_distance, dtw_with_path

logger = logging.getLogger(__name__)

# Métricas disponibles y sus descripciones
AVAILABLE_METRICS: Dict[str, str] = {
    "euclidean": "Distancia Euclidiana normalizada — O(n)",
    "pearson":   "Correlación de Pearson — O(n)",
    "cosine":    "Similitud por Coseno — O(n)",
    "dtw":       "Dynamic Time Warping — O(n²)",
}


class SimilarityService:
    """
    Servicio de análisis de similitud entre pares de activos financieros.

    Proporciona una interfaz unificada para los cuatro algoritmos del
    Requerimiento 2. Gestiona internamente la carga del dataset maestro
    (con caché) para evitar releer el CSV en cada comparación.

    Uso básico:
        service = SimilarityService()
        result = service.compare_assets("AAPL", "MSFT", metric="pearson")
        print(result["value"])   # 0.87
        print(result["interpretation"])

    Uso avanzado (todas las métricas):
        results = service.compare_all_metrics("AAPL", "MSFT")

    Comparar múltiples pares:
        matrix = service.compare_all_pairs(["AAPL", "MSFT", "GOOGL"], metric="pearson")
    """

    def __init__(
        self,
        master_path: Optional[Path] = None,
    ) -> None:
        """
        Inicializa el servicio y carga el dataset maestro en memoria.

        Args:
            master_path: Ruta alternativa al CSV maestro.
                         Si es None, usa data/processed/master_dataset.csv.
        """
        logger.info("Inicializando SimilarityService...")
        self._master: pd.DataFrame = load_master(master_path)
        self._available_tickers: List[str] = sorted(
            self._master["ticker"].unique().tolist()
        )
        logger.info(
            f"Dataset maestro cargado: {len(self._master):,} registros, "
            f"{len(self._available_tickers)} activos disponibles"
        )

    # ------------------------------------------------------------------ #
    # API principal                                                        #
    # ------------------------------------------------------------------ #

    def compare_assets(
        self,
        ticker_a: str,
        ticker_b: str,
        metric: str = "pearson",
        on: str = "returns",
        normalize_euclidean: bool = True,
        dtw_normalize: bool = True,
        include_path: bool = False,
    ) -> Dict[str, Any]:
        """
        Compara dos activos financieros usando la métrica especificada.

        Flujo interno:
          1. Validar parámetros.
          2. Preparar series (retornos + alineación + limpieza NaN).
          3. Ejecutar el algoritmo.
          4. Construir resultado estructurado.

        Args:
            ticker_a:           Símbolo del primer activo.
            ticker_b:           Símbolo del segundo activo.
            metric:             Algoritmo a usar: 'euclidean', 'pearson',
                                'cosine' o 'dtw'.
            on:                 Datos de entrada: 'returns' (default) o 'prices'.
            normalize_euclidean: Si True, normaliza distancia euclidiana por √n.
            dtw_normalize:      Si True, normaliza distancia DTW por (m+n).
            include_path:       Si True y metric='dtw', incluye el warping path
                                en el resultado (puede ser grande: O(m+n)).

        Returns:
            Diccionario con:
              - ticker_a, ticker_b: símbolos comparados.
              - metric:             nombre del algoritmo.
              - value:              resultado numérico del algoritmo.
              - interpretation:     texto explicativo del resultado.
              - n_points:           puntos temporales comunes usados.
              - date_start/end:     rango temporal.
              - on:                 'returns' o 'prices'.
              - elapsed_ms:         tiempo de ejecución en milisegundos.
              - components:         detalles intermedios (solo Pearson, Coseno).
              - path:               warping path (solo DTW si include_path=True).

        Raises:
            ValueError: Si la métrica no es válida.
            KeyError:   Si algún ticker no existe en el dataset.
        """
        metric = metric.lower().strip()
        self._validate_metric(metric)
        self._validate_ticker(ticker_a)
        self._validate_ticker(ticker_b)

        if ticker_a == ticker_b:
            logger.warning(
                f"compare_assets: ticker_a == ticker_b == '{ticker_a}'. "
                "Comparando una serie consigo misma."
            )

        # ── Preparar series ─────────────────────────────────────────
        x, y, meta = prepare_series(
            ticker_a,
            ticker_b,
            master=self._master,
            on=on,
        )

        # ── Ejecutar algoritmo ──────────────────────────────────────
        t_start = time.perf_counter()
        result = self._dispatch(
            metric, x, y,
            normalize_euclidean=normalize_euclidean,
            dtw_normalize=dtw_normalize,
            include_path=include_path,
        )
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        result.update(meta)
        result["elapsed_ms"] = round(elapsed_ms, 3)

        logger.info(
            f"compare_assets: {ticker_a} vs {ticker_b} | "
            f"metric={metric} | value={result['value']:.6f} | "
            f"{elapsed_ms:.1f}ms | n={meta['n_points']}"
        )
        return result

    def compare_all_metrics(
        self,
        ticker_a: str,
        ticker_b: str,
        on: str = "returns",
    ) -> Dict[str, Any]:
        """
        Ejecuta los 4 algoritmos de similitud sobre el mismo par de activos.

        Útil para el análisis comparativo de los métodos solicitado en el PDF.

        Args:
            ticker_a: Símbolo del primer activo.
            ticker_b: Símbolo del segundo activo.
            on:       'returns' o 'prices'.

        Returns:
            Diccionario con una clave por métrica y el resultado completo.
            Incluye además 'summary': tabla comparativa.
        """
        self._validate_ticker(ticker_a)
        self._validate_ticker(ticker_b)

        # Preparar las series una sola vez para las 4 métricas O(n)
        x, y, meta = prepare_series(
            ticker_a, ticker_b, master=self._master, on=on
        )

        results: Dict[str, Any] = {
            "ticker_a": ticker_a,
            "ticker_b": ticker_b,
            "n_points": meta["n_points"],
            "date_start": meta["date_start"],
            "date_end": meta["date_end"],
            "on": on,
        }

        for metric in AVAILABLE_METRICS:
            t_start = time.perf_counter()
            r = self._dispatch(metric, x, y)
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            r["elapsed_ms"] = round(elapsed_ms, 3)
            results[metric] = r

        # Tabla resumen
        results["summary"] = self._build_summary_table(results)
        return results

    def compare_all_pairs(
        self,
        tickers: Optional[List[str]] = None,
        metric: str = "pearson",
        on: str = "returns",
    ) -> pd.DataFrame:
        """
        Construye la matriz de similitud completa para una lista de activos.

        Calcula el triángulo superior de la matriz (n × n activos) y lo
        refleja para obtener la matriz simétrica completa.

        Complejidad:
          - Euclidean/Pearson/Coseno: O(k² × n) donde k = número de activos.
          - DTW: O(k² × n²) — puede ser muy lento para k=20 y n=1250.

        Args:
            tickers: Lista de tickers a comparar. Si es None, usa los 20
                     activos del portafolio.
            metric:  Algoritmo a usar.
            on:      'returns' o 'prices'.

        Returns:
            DataFrame k × k con los valores de similitud.
            Los valores de la diagonal son:
              - 0.0 para distancias (euclidean, dtw).
              - 1.0 para correlaciones/similitudes (pearson, cosine).
        """
        self._validate_metric(metric)

        if tickers is None:
            tickers = self._available_tickers
        else:
            for t in tickers:
                self._validate_ticker(t)

        k = len(tickers)
        logger.info(
            f"compare_all_pairs: {k} activos × {k} = {k*k} comparaciones, "
            f"metric={metric}"
        )

        # Inicializar matriz cuadrada con NaN
        matrix = pd.DataFrame(
            np.full((k, k), np.nan),
            index=tickers,
            columns=tickers,
        )

        # Diagonal: distancia 0 o similitud 1
        diagonal_value = 0.0 if metric in ("euclidean", "dtw") else 1.0
        for t in tickers:
            matrix.loc[t, t] = diagonal_value

        # Triángulo superior: O(k*(k-1)/2) comparaciones
        for i in range(k):
            for j in range(i + 1, k):
                try:
                    result = self.compare_assets(
                        tickers[i], tickers[j],
                        metric=metric, on=on,
                    )
                    val = result["value"]
                    matrix.loc[tickers[i], tickers[j]] = val
                    matrix.loc[tickers[j], tickers[i]] = val  # simetría
                except Exception as exc:
                    logger.warning(
                        f"compare_all_pairs: fallo en "
                        f"({tickers[i]}, {tickers[j]}): {exc}"
                    )

        return matrix

    # ------------------------------------------------------------------ #
    # Propiedades de consulta                                              #
    # ------------------------------------------------------------------ #

    @property
    def available_tickers(self) -> List[str]:
        """Lista de tickers disponibles en el dataset maestro."""
        return self._available_tickers.copy()

    @property
    def available_metrics(self) -> Dict[str, str]:
        """Diccionario métrica → descripción."""
        return AVAILABLE_METRICS.copy()

    # ------------------------------------------------------------------ #
    # Despacho de algoritmos (privado)                                     #
    # ------------------------------------------------------------------ #

    def _dispatch(
        self,
        metric: str,
        x: np.ndarray,
        y: np.ndarray,
        normalize_euclidean: bool = True,
        dtw_normalize: bool = True,
        include_path: bool = False,
    ) -> Dict[str, Any]:
        """
        Llama al algoritmo correspondiente y estructura el resultado.

        Args:
            metric:             Nombre del algoritmo.
            x, y:               Arrays preparados para el algoritmo.
            normalize_euclidean: Normalizar distancia euclidiana.
            dtw_normalize:      Normalizar distancia DTW.
            include_path:       Incluir warping path en DTW.

        Returns:
            Dict con 'value', 'interpretation', 'metric', y campos extra
            específicos del algoritmo.
        """
        result: Dict[str, Any] = {"metric": metric}

        if metric == "euclidean":
            if normalize_euclidean:
                value = euclidean_distance_normalized(x, y)
            else:
                from src.similarity.euclidean_distance import euclidean_distance
                value = euclidean_distance(x, y)
            similarity_val = euclidean_similarity(x, y)
            result["value"] = value
            result["similarity"] = similarity_val
            result["interpretation"] = (
                f"Distancia euclidiana {'normalizada' if normalize_euclidean else 'bruta'}: "
                f"{value:.6f}  |  Similitud equivalente: {similarity_val:.4f}"
            )

        elif metric == "pearson":
            value, components = pearson_with_components(x, y)
            result["value"] = value
            result["components"] = components
            result["interpretation"] = interpret_pearson(value)

        elif metric == "cosine":
            value, components = cosine_with_components(x, y)
            result["value"] = value
            result["components"] = components
            result["interpretation"] = interpret_cosine(value)

        elif metric == "dtw":
            if include_path:
                value, path = dtw_with_path(x, y, normalize=dtw_normalize)
                result["path"] = path
            else:
                value = dtw_distance(x, y, normalize=dtw_normalize)
            result["value"] = value
            result["interpretation"] = (
                f"Distancia DTW {'normalizada' if dtw_normalize else 'bruta'}: "
                f"{value:.6f}. "
                + ("Series muy similares (con posible desfase)" if value < 0.005
                   else "Series moderadamente similares" if value < 0.02
                   else "Series poco similares")
            )

        return result

    # ------------------------------------------------------------------ #
    # Validaciones                                                         #
    # ------------------------------------------------------------------ #

    def _validate_metric(self, metric: str) -> None:
        if metric not in AVAILABLE_METRICS:
            raise ValueError(
                f"Métrica '{metric}' no válida. "
                f"Disponibles: {list(AVAILABLE_METRICS.keys())}"
            )

    def _validate_ticker(self, ticker: str) -> None:
        if ticker not in self._available_tickers:
            raise KeyError(
                f"Ticker '{ticker}' no encontrado en el dataset. "
                f"Disponibles: {self._available_tickers}"
            )

    # ------------------------------------------------------------------ #
    # Resumen comparativo                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_summary_table(results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Construye una tabla resumen comparando las 4 métricas.

        Returns:
            Lista de dicts con columnas: metric, value, complexity,
            interpretation (primera frase).
        """
        complexity_map = {
            "euclidean": "O(n)",
            "pearson":   "O(n)",
            "cosine":    "O(n)",
            "dtw":       "O(n²)",
        }
        summary = []
        for metric in AVAILABLE_METRICS:
            if metric in results:
                r = results[metric]
                summary.append({
                    "metric": metric,
                    "value": round(r["value"], 6),
                    "complexity": complexity_map[metric],
                    "description": AVAILABLE_METRICS[metric],
                    "interpretation": r.get("interpretation", ""),
                })
        return summary


# ====================================================================== #
# API funcional de alto nivel (acceso rápido)                             #
# ====================================================================== #

def compare_assets(
    ticker_a: str,
    ticker_b: str,
    metric: str = "pearson",
    on: str = "returns",
    master_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Función de conveniencia para comparar dos activos sin instanciar la clase.

    Para comparaciones únicas. Para múltiples comparaciones, usar directamente
    la clase SimilarityService (que cachea el dataset maestro).

    Ejemplo:
        from src.similarity.similarity_service import compare_assets

        result = compare_assets("AAPL", "MSFT", metric="pearson")
        print(result["value"])
        print(result["interpretation"])

    Args:
        ticker_a:    Símbolo del primer activo.
        ticker_b:    Símbolo del segundo activo.
        metric:      'euclidean', 'pearson', 'cosine' o 'dtw'.
        on:          'returns' (default) o 'prices'.
        master_path: Ruta alternativa al dataset maestro CSV.

    Returns:
        Resultado completo del cálculo de similitud.
    """
    service = SimilarityService(master_path=master_path)
    return service.compare_assets(ticker_a, ticker_b, metric=metric, on=on)
