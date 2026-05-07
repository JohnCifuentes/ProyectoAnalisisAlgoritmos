"""
src/etl/transformer.py
======================
Módulo de limpieza y validación de datos financieros.

Yahoo Finance entrega datos relativamente limpios, por lo que se aplica
una limpieza ligera y razonable en lugar de transformaciones complejas.

Algoritmo de limpieza (aplicado secuencialmente):
    1. Eliminar filas con valores nulos en campos esenciales (OHLCV).
    2. Convertir columnas al tipo numérico apropiado.
    3. Eliminar filas que no pasan las validaciones de coherencia financiera:
         high >= low       (el máximo no puede ser menor que el mínimo)
         close > 0         (precio estrictamente positivo)
         volume >= 0       (volumen no negativo)
    4. Eliminar duplicados por clave (date, ticker).
    5. Ordenar cronológicamente por fecha.

Impacto algorítmico de las decisiones de limpieza:
    - Eliminación directa: garantiza que los arrays usados en los algoritmos
      de similitud (Req. 2) no contengan NaN que rompan distancias o DTW.
    - NO se usa interpolación: el dataset de Yahoo Finance tiene tasas de
      datos faltantes muy bajas (<0.1%). Interpolar introduciría valores
      artificiales que distorsionarían las métricas de volatilidad (Req. 3).
    - Ordenamiento cronológico: requisito fundamental para DTW y sliding window.
"""

import logging
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

# Columnas que deben tener valor para que un registro sea útil
_REQUIRED_COLS = ["date", "ticker", "open", "high", "low", "close", "volume"]

# Columnas numéricas de tipo float
_FLOAT_COLS = ["open", "high", "low", "close"]

# Columnas numéricas de tipo entero (volume puede ser float en la API)
_NUMERIC_COLS = ["volume"]


class DataTransformer:
    """
    Limpia y valida los registros diarios de un activo financiero.

    Recibe una lista de dicts crudos del parser y retorna un DataFrame
    limpio, tipado, sin duplicados y ordenado cronológicamente.
    """

    def transform(
        self, records: List[Dict[str, Any]], ticker: str
    ) -> pd.DataFrame:
        """
        Ejecuta la pipeline de limpieza sobre los registros de un activo.

        Complejidad: O(N) para cada paso de limpieza, O(N log N) para el sort.
        N = número de registros diarios (~1 250 para 5 años de datos diarios).

        Args:
            records: Lista de dicts crudos del YahooFinanceParser.
            ticker:  Símbolo del activo (para logging).

        Returns:
            DataFrame limpio con las columnas del esquema master,
            o DataFrame vacío si no quedan registros válidos.
        """
        if not records:
            logger.warning(f"[{ticker}] Sin registros para transformar")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        initial_count = len(df)

        # ── Paso 1: eliminar filas con nulos en campos esenciales ───────
        df = self._drop_null_rows(df, ticker)
        if df.empty:
            return df

        # ── Paso 2: convertir a tipos numéricos ─────────────────────────
        df = self._cast_numeric(df, ticker)
        if df.empty:
            return df

        # ── Paso 3: validaciones de coherencia financiera ───────────────
        df = self._validate_prices(df, ticker)
        if df.empty:
            return df

        # ── Paso 4: eliminar duplicados por (date, ticker) ───────────────
        df = self._drop_duplicates(df, ticker)

        # ── Paso 5: ordenar cronológicamente ────────────────────────────
        df = df.sort_values("date").reset_index(drop=True)

        final_count = len(df)
        removed = initial_count - final_count

        if removed > 0:
            logger.info(
                f"[{ticker}] Limpieza: {initial_count} → {final_count} registros "
                f"({removed} eliminados)"
            )
        else:
            logger.info(
                f"[{ticker}] Limpieza completa: {final_count} registros, "
                "ninguno eliminado"
            )

        return df

    # ------------------------------------------------------------------ #
    # Pasos de limpieza (privados)                                         #
    # ------------------------------------------------------------------ #

    def _drop_null_rows(
        self, df: pd.DataFrame, ticker: str
    ) -> pd.DataFrame:
        """
        Elimina filas donde algún campo esencial contiene None o NaN.

        Decisión de diseño: se elimina en lugar de interpolar porque:
          - Yahoo Finance rara vez produce gaps (< 0.1 % de los registros).
          - Interpolar precios introduciría valores sintéticos que
            distorsionarían volatilidad y similitud en fases posteriores.

        Args:
            df:     DataFrame con los registros.
            ticker: Símbolo del activo.

        Returns:
            DataFrame sin filas nulas en campos esenciales.
        """
        cols_present = [c for c in _REQUIRED_COLS if c in df.columns]
        before = len(df)
        df = df.dropna(subset=cols_present)
        dropped = before - len(df)
        if dropped:
            logger.debug(
                f"[{ticker}] Paso 1 (nulos): eliminadas {dropped} filas"
            )
        return df

    def _cast_numeric(
        self, df: pd.DataFrame, ticker: str
    ) -> pd.DataFrame:
        """
        Convierte las columnas de precio y volumen a tipos numéricos.

        Usa pd.to_numeric con errors='coerce' para convertir valores
        no convertibles a NaN y luego eliminarlos.

        Args:
            df:     DataFrame con los registros.
            ticker: Símbolo del activo.

        Returns:
            DataFrame con columnas correctamente tipadas.
        """
        all_numeric = _FLOAT_COLS + _NUMERIC_COLS
        for col in all_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Eliminar filas donde la conversión generó NaN
        cols_present = [c for c in all_numeric if c in df.columns]
        before = len(df)
        df = df.dropna(subset=cols_present)
        dropped = before - len(df)
        if dropped:
            logger.debug(
                f"[{ticker}] Paso 2 (tipos): eliminadas {dropped} filas "
                "por valores no numéricos"
            )
        return df

    def _validate_prices(
        self, df: pd.DataFrame, ticker: str
    ) -> pd.DataFrame:
        """
        Aplica reglas de coherencia financiera a cada registro.

        Reglas:
          1. high >= low:   El máximo diario no puede ser menor que el mínimo.
                            Indica dato corrupto o invertido.
          2. close > 0:     Precios estrictamente positivos (no hay precios
                            negativos en acciones o ETFs).
          3. volume >= 0:   El volumen no puede ser negativo.

        Args:
            df:     DataFrame con tipos numéricos ya aplicados.
            ticker: Símbolo del activo.

        Returns:
            DataFrame con solo registros financieramente coherentes.
        """
        before = len(df)

        mask_valid = (
            (df["high"] >= df["low"])   # Regla 1: coherencia high/low
            & (df["close"] > 0)          # Regla 2: precio positivo
            & (df["volume"] >= 0)        # Regla 3: volumen no negativo
        )

        df = df[mask_valid].copy()
        dropped = before - len(df)

        if dropped:
            logger.warning(
                f"[{ticker}] Paso 3 (validación): eliminadas {dropped} filas "
                "por incoherencia financiera (high<low, close≤0, volume<0)"
            )
        return df

    def _drop_duplicates(
        self, df: pd.DataFrame, ticker: str
    ) -> pd.DataFrame:
        """
        Elimina registros duplicados por la clave compuesta (date, ticker).

        Conserva el primer registro encontrado.
        En datos bien formados de Yahoo Finance esto raramente ocurre,
        pero puede aparecer si se combinan fuentes o se procesan re-descargas.

        Args:
            df:     DataFrame con los registros.
            ticker: Símbolo del activo.

        Returns:
            DataFrame sin duplicados.
        """
        before = len(df)
        df = df.drop_duplicates(subset=["date", "ticker"], keep="first")
        dropped = before - len(df)
        if dropped:
            logger.warning(
                f"[{ticker}] Paso 4 (duplicados): eliminados {dropped} "
                "registros duplicados (date, ticker)"
            )
        return df
