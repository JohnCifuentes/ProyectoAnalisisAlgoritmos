"""
src/etl/extractor.py
====================
Módulo de extracción de datos financieros desde Yahoo Finance.

Responsabilidades:
- Inicializar la sesión HTTP (cookies + crumb) con Yahoo Finance.
- Construir URLs dinámicamente para cada ticker.
- Realizar peticiones HTTP con manejo de errores, timeouts y reintentos
  con backoff exponencial.
- Retornar el JSON crudo sin interpretar (eso lo hace parser.py).

RESTRICCIÓN: NO usa yfinance, pandas_datareader ni ninguna librería
que encapsule la descarga financiera. Solo usa `requests`.
"""

import time
import logging
from typing import Any, Dict, Optional

import requests

from src.config.settings import (
    DELAY_BETWEEN_REQUESTS,
    HTTP_HEADERS,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    RETRY_DELAY,
    YAHOO_BASE_URL,
    YAHOO_CRUMB_URL,
    YAHOO_INTERVAL,
    YAHOO_RANGE,
    YAHOO_SESSION_INIT_URL,
)

logger = logging.getLogger(__name__)


class YahooFinanceExtractor:
    """
    Extractor de datos financieros históricos desde la API pública de Yahoo Finance.

    Utiliza la API v8/finance/chart con peticiones HTTP explícitas.
    Gestiona la autenticación mediante crumb + cookies de sesión,
    que Yahoo Finance requiere para acceder a sus endpoints.

    Atributos:
        session: Sesión HTTP reutilizable con headers y cookies persistentes.
        crumb:   Token de autenticación obtenido de Yahoo Finance.
    """

    def __init__(self) -> None:
        self.session: requests.Session = requests.Session()
        self.session.headers.update(HTTP_HEADERS)
        self.crumb: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Inicialización de sesión                                             #
    # ------------------------------------------------------------------ #

    def initialize_session(self) -> bool:
        """
        Inicializa la sesión HTTP con Yahoo Finance.

        Proceso de dos pasos:
          1. GET a finance.yahoo.com para establecer cookies de sesión.
          2. GET al endpoint getcrumb para obtener el token de autenticación.

        El crumb es necesario para que el endpoint v8/finance/chart
        responda con datos en lugar de un 401/403.

        Returns:
            True si la sesión se inicializó correctamente, False en caso contrario.
            Si falla, el extractor intentará las descargas sin crumb como fallback.
        """
        logger.info("Inicializando sesión con Yahoo Finance...")

        try:
            # Paso 1: Visitar Yahoo Finance para obtener cookies de sesión.
            # Es necesario antes de solicitar el crumb.
            init_response = self.session.get(
                YAHOO_SESSION_INIT_URL,
                timeout=REQUEST_TIMEOUT,
            )
            logger.debug(
                f"Inicialización de sesión: HTTP {init_response.status_code}"
            )

            # Paso 2: Obtener el crumb de autenticación.
            crumb_response = self.session.get(
                YAHOO_CRUMB_URL,
                timeout=REQUEST_TIMEOUT,
            )

            if crumb_response.status_code == 200:
                self.crumb = crumb_response.text.strip()
                # Mostrar solo los primeros 6 caracteres del crumb por seguridad
                crumb_preview = self.crumb[:6] + "..." if len(self.crumb) > 6 else self.crumb
                logger.info(f"Sesión inicializada correctamente. Crumb: {crumb_preview}")
                return True
            else:
                logger.warning(
                    f"No se pudo obtener crumb (HTTP {crumb_response.status_code}). "
                    "Se intentará la descarga sin crumb."
                )
                self.crumb = None
                return False

        except requests.exceptions.RequestException as e:
            logger.warning(
                f"Error al inicializar sesión: {e}. "
                "Se intentará la descarga sin crumb."
            )
            self.crumb = None
            return False

    # ------------------------------------------------------------------ #
    # Construcción de URL                                                  #
    # ------------------------------------------------------------------ #

    def build_url(self, ticker: str) -> str:
        """
        Construye la URL del endpoint de Yahoo Finance para un ticker.

        Formato base:
            https://query1.finance.yahoo.com/v8/finance/chart/{TICKER}
                ?interval=1d&range=5y[&crumb=TOKEN]

        Args:
            ticker: Símbolo bursátil del activo (ej. 'AAPL', 'EC').

        Returns:
            URL completa lista para la petición HTTP.
        """
        url = (
            f"{YAHOO_BASE_URL}/{ticker}"
            f"?interval={YAHOO_INTERVAL}&range={YAHOO_RANGE}"
        )
        if self.crumb:
            url += f"&crumb={self.crumb}"
        return url

    # ------------------------------------------------------------------ #
    # Descarga principal                                                   #
    # ------------------------------------------------------------------ #

    def fetch(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Realiza la petición HTTP a Yahoo Finance y retorna el JSON crudo.

        Implementa reintentos con backoff exponencial:
          - Intento 1: espera RETRY_DELAY * 1  segundos ante fallo
          - Intento 2: espera RETRY_DELAY * 2  segundos ante fallo
          - Intento 3: espera RETRY_DELAY * 4  segundos ante fallo (429)

        Códigos HTTP manejados:
          - 200: éxito, retorna JSON
          - 429: rate limit → espera más tiempo y reintenta
          - 404: ticker no encontrado → aborta sin reintentos
          - otros: reintenta con backoff

        Args:
            ticker: Símbolo del activo financiero.

        Returns:
            Diccionario JSON con la respuesta de la API, o None si fallan
            todos los intentos.
        """
        url = self.build_url(ticker)
        logger.info(f"[{ticker}] Descargando → {url[:80]}...")

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.session.get(url, timeout=REQUEST_TIMEOUT)

                if response.status_code == 200:
                    json_data: Dict[str, Any] = response.json()
                    if self._is_valid_response(json_data, ticker):
                        logger.info(
                            f"[{ticker}] Descarga exitosa (intento {attempt}/{MAX_RETRIES})"
                        )
                        return json_data
                    else:
                        # La estructura JSON no es la esperada
                        logger.warning(
                            f"[{ticker}] JSON inválido o vacío en intento {attempt}"
                        )
                        return None

                elif response.status_code == 429:
                    # Rate limit: espera con backoff exponencial
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    logger.warning(
                        f"[{ticker}] Rate limit (429). "
                        f"Esperando {wait_time:.1f}s "
                        f"(intento {attempt}/{MAX_RETRIES})"
                    )
                    time.sleep(wait_time)

                elif response.status_code == 404:
                    # Ticker no existe en Yahoo Finance
                    logger.error(
                        f"[{ticker}] Ticker no encontrado (404). Abortando."
                    )
                    return None

                elif response.status_code in (401, 403):
                    # Problema de autenticación: reintentar reiniciando sesión
                    logger.warning(
                        f"[{ticker}] Error de autenticación ({response.status_code}). "
                        "Reiniciando sesión..."
                    )
                    self.initialize_session()
                    url = self.build_url(ticker)  # Reconstruir URL con nuevo crumb
                    time.sleep(RETRY_DELAY * attempt)

                else:
                    logger.warning(
                        f"[{ticker}] HTTP {response.status_code} "
                        f"(intento {attempt}/{MAX_RETRIES})"
                    )
                    time.sleep(RETRY_DELAY * attempt)

            except requests.exceptions.Timeout:
                logger.warning(
                    f"[{ticker}] Timeout ({REQUEST_TIMEOUT}s) "
                    f"(intento {attempt}/{MAX_RETRIES})"
                )
                time.sleep(RETRY_DELAY * attempt)

            except requests.exceptions.ConnectionError as exc:
                logger.warning(
                    f"[{ticker}] Error de conexión: {exc} "
                    f"(intento {attempt}/{MAX_RETRIES})"
                )
                time.sleep(RETRY_DELAY * attempt)

            except requests.exceptions.RequestException as exc:
                logger.error(f"[{ticker}] Error inesperado en la solicitud: {exc}")
                return None

        logger.error(
            f"[{ticker}] Fallaron todos los {MAX_RETRIES} intentos de descarga"
        )
        return None

    # ------------------------------------------------------------------ #
    # Validación de respuesta                                              #
    # ------------------------------------------------------------------ #

    def _is_valid_response(
        self, data: Dict[str, Any], ticker: str
    ) -> bool:
        """
        Valida que la respuesta JSON tenga la estructura mínima esperada.

        Estructura esperada (ver API_JSON_Respuesta.txt):
            data["chart"]["result"][0]["timestamp"]          → array
            data["chart"]["result"][0]["indicators"]["quote"][0] → dict

        Args:
            data:   Diccionario JSON de la respuesta.
            ticker: Símbolo (para logging).

        Returns:
            True si la estructura es válida y contiene datos, False si no.
        """
        try:
            chart = data.get("chart", {})

            # Verificar ausencia de error en la respuesta
            if chart.get("error"):
                logger.warning(f"[{ticker}] API reportó error: {chart['error']}")
                return False

            result_list = chart.get("result")
            if not result_list or not isinstance(result_list, list):
                logger.warning(f"[{ticker}] Campo 'result' ausente o vacío")
                return False

            result = result_list[0]

            # Verificar timestamps
            timestamps = result.get("timestamp")
            if not timestamps or len(timestamps) == 0:
                logger.warning(f"[{ticker}] Array 'timestamp' ausente o vacío")
                return False

            # Verificar indicadores de precio
            quote_list = result.get("indicators", {}).get("quote", [])
            if not quote_list or len(quote_list) == 0:
                logger.warning(f"[{ticker}] indicators.quote ausente o vacío")
                return False

            logger.debug(
                f"[{ticker}] Estructura JSON válida. "
                f"Timestamps: {len(timestamps)} registros"
            )
            return True

        except (KeyError, TypeError, IndexError) as exc:
            logger.warning(
                f"[{ticker}] Error validando estructura JSON: {exc}"
            )
            return False
