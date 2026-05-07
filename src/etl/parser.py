"""
src/etl/parser.py
=================
Módulo de parsing manual del JSON de Yahoo Finance.

Sigue exactamente la estructura documentada en API_JSON_Respuesta.txt:

    chart.result[0]
    ├── meta
    │   ├── symbol          → ticker del activo
    │   └── instrumentType  → "EQUITY" o "ETF"
    ├── timestamp[]         → fechas en formato UNIX (segundos desde epoch UTC)
    └── indicators
        └── quote[0]
            ├── open[]      ─┐
            ├── high[]       │ Todos alineados por índice:
            ├── low[]        │ open[i], high[i], ... corresponden al mismo día
            ├── close[]      │ que timestamp[i]
            └── volume[]    ─┘

Responsabilidades:
- Navegar la estructura JSON manualmente (sin atajos de alto nivel).
- Convertir timestamps UNIX a cadenas de fecha ISO 8601 (YYYY-MM-DD).
- Construir una lista de registros tabulares (dicts) listos para DataFrame.
- Manejar valores None que pueden aparecer en cualquier posición de los arrays.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class YahooFinanceParser:
    """
    Parsea manualmente el JSON de la API v8/finance/chart de Yahoo Finance.

    Implementa el algoritmo de extracción de series temporales:
        Para i = 0 hasta N-1:
            fecha  = convertir_unix(timestamp[i])
            open   = open[i]
            high   = high[i]
            low    = low[i]
            close  = close[i]
            volume = volume[i]
            →  Crear registro tabular

    Complejidad temporal: O(N) donde N = número de días en la serie.
    """

    def parse(
        self, json_data: Dict[str, Any], ticker: str
    ) -> List[Dict[str, Any]]:
        """
        Parsea la respuesta JSON y construye registros tabulares diarios.

        Args:
            json_data: Diccionario JSON de la respuesta de Yahoo Finance.
            ticker:    Símbolo del activo (usado como fallback si meta falta).

        Returns:
            Lista de dicts, donde cada elemento representa un día bursátil.
            Puede contener valores None en campos opcionales (el transformer
            se encarga de limpiarlos).
            Retorna lista vacía si el JSON no puede procesarse.
        """
        records: List[Dict[str, Any]] = []

        try:
            # ── Nivel 1: acceder a chart.result[0] ─────────────────────
            result = json_data["chart"]["result"][0]

            # ── Nivel 2: extraer metadatos ──────────────────────────────
            meta = result.get("meta", {})
            symbol: str = meta.get("symbol", ticker)
            instrument_type: str = meta.get("instrumentType", "UNKNOWN")

            # ── Nivel 3: extraer timestamps ─────────────────────────────
            # Cada elemento es un entero Unix (segundos desde 1970-01-01 UTC)
            timestamps: List[int] = result.get("timestamp", [])
            n: int = len(timestamps)

            if n == 0:
                logger.warning(f"[{ticker}] Array de timestamps vacío")
                return records

            # ── Nivel 4: extraer indicadores de precio ─────────────────
            # indicators.quote es un array; accedemos al primer elemento
            indicators: Dict[str, Any] = result.get("indicators", {})
            quote_list: List[Dict[str, Any]] = indicators.get("quote", [])

            if not quote_list:
                logger.warning(f"[{ticker}] indicators.quote vacío o ausente")
                return records

            quote: Dict[str, List[Optional[float]]] = quote_list[0]

            # Arrays de precios (pueden contener None en posiciones inválidas)
            opens: List[Optional[float]] = quote.get("open", [])
            highs: List[Optional[float]] = quote.get("high", [])
            lows: List[Optional[float]] = quote.get("low", [])
            closes: List[Optional[float]] = quote.get("close", [])
            volumes: List[Optional[float]] = quote.get("volume", [])

            # Verificar alineación de arrays (todos deben tener longitud n)
            self._log_alignment_warning(
                n, opens, highs, lows, closes, volumes, ticker
            )

            # ── Nivel 5: construir registros tabulares ─────────────────
            # Regla fundamental: todos los arrays están alineados por índice.
            # El elemento en la posición i de cada array corresponde al
            # mismo día bursátil definido por timestamp[i].
            null_timestamps = 0
            for i in range(n):
                date: Optional[str] = self._unix_to_date(timestamps[i])

                if date is None:
                    null_timestamps += 1
                    continue  # Timestamp inválido → omitir registro

                record: Dict[str, Any] = {
                    "date": date,
                    "ticker": symbol,
                    "open": self._safe_get(opens, i),
                    "high": self._safe_get(highs, i),
                    "low": self._safe_get(lows, i),
                    "close": self._safe_get(closes, i),
                    "volume": self._safe_get(volumes, i),
                    "instrument_type": instrument_type,
                }
                records.append(record)

            if null_timestamps:
                logger.debug(
                    f"[{ticker}] {null_timestamps} timestamps inválidos omitidos"
                )

            logger.info(
                f"[{ticker}] Parseados {len(records)} registros "
                f"({instrument_type})"
            )

        except (KeyError, IndexError, TypeError) as exc:
            logger.error(f"[{ticker}] Error crítico durante el parsing: {exc}")

        return records

    # ------------------------------------------------------------------ #
    # Métodos auxiliares (estáticos)                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _unix_to_date(unix_ts: Any) -> Optional[str]:
        """
        Convierte un timestamp UNIX (segundos desde epoch UTC) a YYYY-MM-DD.

        Se usa UTC explícitamente para garantizar consistencia entre
        activos de diferentes zonas horarias (NYSE, Colombia, etc.).

        Ejemplos:
            1577836800 → "2020-01-01"
            1704916200 → "2024-01-10"

        Args:
            unix_ts: Timestamp entero o flotante.

        Returns:
            Cadena de fecha o None si el valor es inválido.
        """
        try:
            return (
                datetime
                .fromtimestamp(int(unix_ts), tz=timezone.utc)
                .strftime("%Y-%m-%d")
            )
        except (ValueError, TypeError, OSError):
            return None

    @staticmethod
    def _safe_get(array: List[Any], index: int) -> Optional[Any]:
        """
        Accede de forma segura a un elemento de una lista por índice.

        Retorna None si:
          - El índice está fuera de rango (array más corto de lo esperado).
          - El valor en esa posición es None (dato faltante de Yahoo Finance).

        Args:
            array: Lista de valores numéricos (puede contener None).
            index: Índice a acceder (0-based).

        Returns:
            Valor en la posición dada, o None.
        """
        if index < len(array):
            return array[index]  # None es un valor válido aquí (el transformer lo limpia)
        return None

    @staticmethod
    def _log_alignment_warning(
        n: int,
        opens: list,
        highs: list,
        lows: list,
        closes: list,
        volumes: list,
        ticker: str,
    ) -> None:
        """
        Registra una advertencia si los arrays de precios no están alineados
        con el array de timestamps.

        La API de Yahoo Finance garantiza alineación, pero puede haber
        casos extremos de datos corruptos.

        Args:
            n:      Longitud esperada (= longitud de timestamps).
            opens, highs, lows, closes, volumes: Arrays de precios.
            ticker: Símbolo del activo (para logging).
        """
        lengths = {
            "open": len(opens),
            "high": len(highs),
            "low": len(lows),
            "close": len(closes),
            "volume": len(volumes),
        }
        mismatched = {k: v for k, v in lengths.items() if v != n}
        if mismatched:
            logger.warning(
                f"[{ticker}] Desalineación detectada. "
                f"timestamps={n}, arrays distintos={mismatched}"
            )
