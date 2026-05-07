"""
src/etl/loader.py
=================
Módulo de persistencia de datos (Load en ETL).

Responsabilidades:
- Guardar CSV individuales por activo en data/raw/.
- Construir el dataset maestro consolidado uniendo todos los activos.
- Guardar el dataset maestro en data/processed/master_dataset.csv.
- Generar un resumen estadístico del dataset en dataset_summary.csv.

Estructura de salida:
    data/
    ├── raw/
    │   ├── AAPL.csv
    │   ├── MSFT.csv
    │   └── ... (un CSV por activo)
    └── processed/
        ├── master_dataset.csv   ← dataset principal para análisis
        └── dataset_summary.csv  ← resumen por activo
"""

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.config.settings import MASTER_COLUMNS, PROCESSED_DIR, RAW_DIR

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Gestiona la persistencia del dataset financiero en disco.

    Crea los directorios necesarios si no existen y guarda los
    datos en formato CSV con codificación UTF-8.
    """

    def __init__(self) -> None:
        # Crear directorios si no existen (idempotente)
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        logger.debug(
            f"Directorios verificados: raw={RAW_DIR}, processed={PROCESSED_DIR}"
        )

    # ------------------------------------------------------------------ #
    # Persistencia individual                                              #
    # ------------------------------------------------------------------ #

    def save_raw(self, df: pd.DataFrame, ticker: str) -> Optional[Path]:
        """
        Guarda el CSV individual de un activo en data/raw/{ticker}.csv.

        El archivo contiene todos los registros diarios del activo
        con las columnas: date, ticker, open, high, low, close, volume,
        instrument_type.

        Args:
            df:     DataFrame limpio del activo.
            ticker: Símbolo del activo (usado como nombre de archivo).

        Returns:
            Ruta del archivo guardado, o None si el DataFrame estaba vacío.
        """
        if df.empty:
            logger.warning(
                f"[{ticker}] DataFrame vacío, no se guarda CSV individual"
            )
            return None

        path = RAW_DIR / f"{ticker}.csv"
        df.to_csv(path, index=False, encoding="utf-8")
        logger.info(f"[{ticker}] CSV guardado → {path}  ({len(df):,} filas)")
        return path

    # ------------------------------------------------------------------ #
    # Dataset maestro                                                      #
    # ------------------------------------------------------------------ #

    def build_master_dataset(
        self, dataframes: List[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Construye el dataset maestro consolidado a partir de todos los activos.

        Proceso:
          1. Filtrar DataFrames vacíos.
          2. Concatenar todos los DataFrames en uno.
          3. Seleccionar y ordenar columnas según MASTER_COLUMNS.
          4. Ordenar por (date, ticker) para garantizar consistencia temporal.

        El dataset resultante es la entrada para todos los algoritmos
        de los requerimientos 2, 3 y 4.

        Args:
            dataframes: Lista de DataFrames individuales (uno por activo).

        Returns:
            DataFrame maestro consolidado con esquema MASTER_COLUMNS,
            o DataFrame vacío si no hay datos válidos.
        """
        valid_dfs = [df for df in dataframes if not df.empty]

        if not valid_dfs:
            logger.error(
                "No hay DataFrames válidos para construir el dataset maestro"
            )
            return pd.DataFrame(columns=MASTER_COLUMNS)

        # Concatenar todos los activos en una sola tabla
        master = pd.concat(valid_dfs, ignore_index=True)

        # Añadir columnas faltantes con None (robustez ante cambios de esquema)
        for col in MASTER_COLUMNS:
            if col not in master.columns:
                master[col] = None

        # Seleccionar solo las columnas del esquema y en el orden correcto
        master = master[MASTER_COLUMNS].copy()

        # Ordenar cronológicamente: primero por fecha, luego por ticker
        master = master.sort_values(
            ["date", "ticker"]
        ).reset_index(drop=True)

        n_assets = master["ticker"].nunique()
        logger.info(
            f"Dataset maestro construido: {len(master):,} registros, "
            f"{n_assets} activos únicos"
        )
        return master

    def save_master(self, master: pd.DataFrame) -> Optional[Path]:
        """
        Persiste el dataset maestro en data/processed/master_dataset.csv.

        Este archivo es el punto de entrada para los requerimientos 2, 3 y 4.

        Args:
            master: DataFrame maestro consolidado.

        Returns:
            Ruta del archivo guardado, o None si el DataFrame estaba vacío.
        """
        if master.empty:
            logger.error("Dataset maestro vacío, no se guarda")
            return None

        path = PROCESSED_DIR / "master_dataset.csv"
        master.to_csv(path, index=False, encoding="utf-8")
        logger.info(
            f"Dataset maestro guardado → {path}  ({len(master):,} filas)"
        )
        return path

    # ------------------------------------------------------------------ #
    # Resumen estadístico                                                   #
    # ------------------------------------------------------------------ #

    def save_summary(self, master: pd.DataFrame) -> Optional[Path]:
        """
        Genera y guarda un resumen por activo del dataset maestro.

        El resumen incluye:
          - ticker
          - instrumento (EQUITY / ETF)
          - total de registros diarios
          - fecha de inicio de la serie
          - fecha de fin de la serie
          - porcentaje de cobertura respecto al activo más largo

        Args:
            master: DataFrame maestro consolidado.

        Returns:
            Ruta del archivo de resumen, o None si el DataFrame estaba vacío.
        """
        if master.empty:
            logger.warning("Dataset maestro vacío, no se genera resumen")
            return None

        summary = (
            master.groupby("ticker", sort=True)
            .agg(
                instrumento=("instrument_type", "first"),
                total_registros=("date", "count"),
                fecha_inicio=("date", "min"),
                fecha_fin=("date", "max"),
            )
            .reset_index()
        )

        # Calcular cobertura relativa (%) respecto al activo con más registros
        max_records = summary["total_registros"].max()
        summary["cobertura_pct"] = (
            (summary["total_registros"] / max_records * 100)
            .round(1)
        )

        path = PROCESSED_DIR / "dataset_summary.csv"
        summary.to_csv(path, index=False, encoding="utf-8")

        # Imprimir resumen en log para visibilidad inmediata
        logger.info(f"Resumen del dataset guardado → {path}")
        logger.info("\n" + summary.to_string(index=False))

        return path
