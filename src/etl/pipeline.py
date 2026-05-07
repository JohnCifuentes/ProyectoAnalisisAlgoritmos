"""
src/etl/pipeline.py
===================
Orquestador del proceso ETL completo.

Flujo de ejecución para cada activo:
    1. Extracción  → YahooFinanceExtractor.fetch()
                     Petición HTTP a la API de Yahoo Finance.
    2. Parsing     → YahooFinanceParser.parse()
                     Construcción de registros tabulares desde el JSON.
    3. Limpieza    → DataTransformer.transform()
                     Validación, eliminación de nulos y coherencia financiera.
    4. Persistencia→ DataLoader.save_raw()
                     CSV individual en data/raw/{ticker}.csv

Al finalizar todos los activos:
    5. Dataset maestro → DataLoader.build_master_dataset()
    6. Guardar master  → DataLoader.save_master()
    7. Resumen         → DataLoader.save_summary()
"""

import time
import logging
from typing import Any, Dict, List

import pandas as pd

from src.config.settings import ASSETS, DELAY_BETWEEN_REQUESTS
from src.etl.extractor import YahooFinanceExtractor
from src.etl.loader import DataLoader
from src.etl.parser import YahooFinanceParser
from src.etl.transformer import DataTransformer

logger = logging.getLogger(__name__)


class ETLPipeline:
    """
    Orquesta el proceso ETL para el portafolio de 20 activos financieros.

    Gestiona el flujo completo desde la descarga hasta la persistencia,
    con manejo de errores por activo (un fallo no detiene el pipeline).

    Atributos:
        extractor:   Componente de descarga HTTP.
        parser:      Componente de parsing JSON.
        transformer: Componente de limpieza de datos.
        loader:      Componente de persistencia CSV.
    """

    def __init__(self) -> None:
        self.extractor = YahooFinanceExtractor()
        self.parser = YahooFinanceParser()
        self.transformer = DataTransformer()
        self.loader = DataLoader()

    def run(self) -> pd.DataFrame:
        """
        Ejecuta el pipeline ETL completo para el portafolio de 20 activos.

        El pipeline es tolerante a fallos por activo:
          - Si un activo falla en extracción, parsing o limpieza,
            se registra el error y se continúa con el siguiente.
          - Al final se construye el master dataset con los activos exitosos.

        Returns:
            DataFrame maestro consolidado. Puede tener menos de 20 activos
            si algunos fallaron durante la descarga.
        """
        logger.info("=" * 65)
        logger.info("  INICIO DEL PIPELINE ETL")
        logger.info(f"  Portafolio: {len(ASSETS)} activos | Horizonte: 5 años")
        logger.info("=" * 65)

        # Inicializar sesión HTTP (cookies + crumb) antes de cualquier descarga
        self.extractor.initialize_session()

        dataframes: List[pd.DataFrame] = []
        results: List[Dict[str, Any]] = []

        for idx, asset in enumerate(ASSETS, start=1):
            ticker = asset["ticker"]
            name = asset["name"]

            logger.info(
                f"\n[{idx:02d}/{len(ASSETS)}] {ticker:6s} — {name}"
            )

            df = self._process_asset(ticker)

            status = "OK" if not df.empty else "FALLIDO"
            results.append(
                {
                    "ticker": ticker,
                    "status": status,
                    "registros": len(df),
                }
            )

            if not df.empty:
                dataframes.append(df)

            # Respetar el rate limit de Yahoo Finance entre solicitudes
            if idx < len(ASSETS):
                time.sleep(DELAY_BETWEEN_REQUESTS)

        # ── Construir y guardar el dataset maestro ─────────────────────
        logger.info("\n" + "=" * 65)
        logger.info("  CONSOLIDANDO DATASET MAESTRO")
        logger.info("=" * 65)

        master = self.loader.build_master_dataset(dataframes)
        self.loader.save_master(master)
        self.loader.save_summary(master)

        self._print_final_report(results, master)

        return master

    # ------------------------------------------------------------------ #
    # Procesamiento por activo                                             #
    # ------------------------------------------------------------------ #

    def _process_asset(self, ticker: str) -> pd.DataFrame:
        """
        Ejecuta el pipeline ETL completo para un único activo.

        Cada paso produce un resultado o falla silenciosamente,
        retornando un DataFrame vacío en caso de error.

        Args:
            ticker: Símbolo del activo a procesar.

        Returns:
            DataFrame limpio del activo, o DataFrame vacío si algún paso falla.
        """
        # ── Paso 1: Extracción ──────────────────────────────────────────
        json_data = self.extractor.fetch(ticker)
        if json_data is None:
            logger.error(f"[{ticker}] ✗ Extracción fallida — omitiendo activo")
            return pd.DataFrame()

        # ── Paso 2: Parsing ─────────────────────────────────────────────
        records = self.parser.parse(json_data, ticker)
        if not records:
            logger.error(
                f"[{ticker}] ✗ Parsing no produjo registros — omitiendo activo"
            )
            return pd.DataFrame()

        # ── Paso 3: Limpieza ─────────────────────────────────────────────
        df = self.transformer.transform(records, ticker)
        if df.empty:
            logger.error(
                f"[{ticker}] ✗ Limpieza resultó en DataFrame vacío — omitiendo activo"
            )
            return pd.DataFrame()

        # ── Paso 4: Persistencia individual ──────────────────────────────
        self.loader.save_raw(df, ticker)

        return df

    # ------------------------------------------------------------------ #
    # Reporte final                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _print_final_report(
        results: List[Dict[str, Any]], master: pd.DataFrame
    ) -> None:
        """
        Imprime un resumen final del pipeline en el log.

        Args:
            results: Lista de resultados por activo (ticker, status, registros).
            master:  DataFrame maestro final.
        """
        ok = [r for r in results if r["status"] == "OK"]
        failed = [r for r in results if r["status"] == "FALLIDO"]

        logger.info("\n" + "=" * 65)
        logger.info("  REPORTE FINAL DEL PIPELINE ETL")
        logger.info("=" * 65)
        logger.info(
            f"  Activos exitosos : {len(ok):2d} / {len(results)}"
        )
        logger.info(
            f"  Activos fallidos : {len(failed):2d} / {len(results)}"
        )

        if failed:
            tickers_fallidos = [r["ticker"] for r in failed]
            logger.warning(
                f"  Activos sin datos: {tickers_fallidos}"
            )

        if not master.empty:
            logger.info(
                f"  Total registros  : {len(master):>10,}"
            )
            logger.info(
                f"  Rango temporal   : {master['date'].min()} → "
                f"{master['date'].max()}"
            )
            logger.info(
                f"  Activos únicos   : {master['ticker'].nunique()}"
            )
            logger.info(
                f"  Archivos raw     : data/raw/{{ticker}}.csv"
            )
            logger.info(
                f"  Dataset maestro  : data/processed/master_dataset.csv"
            )
            logger.info(
                f"  Resumen          : data/processed/dataset_summary.csv"
            )

        logger.info("=" * 65)
