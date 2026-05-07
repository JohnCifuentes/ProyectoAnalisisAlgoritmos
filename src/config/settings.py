"""
src/config/settings.py
======================
Configuración central del proyecto.

Contiene:
- Portafolio de 20 activos financieros (tickers, nombres, tipos, sectores).
- Parámetros de la API de Yahoo Finance.
- Rutas de archivos y directorios.
- Parámetros de red (timeout, reintentos, delays).
- Definición de columnas del dataset maestro.
"""

from pathlib import Path
from typing import Dict, List

# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

# Directorio raíz: carpeta que contiene a src/, data/, etc.
BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

DATA_DIR: Path = BASE_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"          # CSVs individuales por activo
PROCESSED_DIR: Path = DATA_DIR / "processed"  # dataset maestro + resumen
EXPORTS_DIR: Path = DATA_DIR / "exports"      # reportes y exportaciones futuras
LOGS_DIR: Path = BASE_DIR / "logs"

# ============================================================
# CONFIGURACIÓN DE LA API DE YAHOO FINANCE
# ============================================================

# Endpoint v8/finance/chart — NO usar yfinance ni pandas_datareader.
# La URL se construye dinámicamente en el extractor.
YAHOO_BASE_URL: str = "https://query1.finance.yahoo.com/v8/finance/chart"

# URL para obtener el crumb de autenticación de Yahoo Finance
YAHOO_CRUMB_URL: str = "https://query1.finance.yahoo.com/v1/test/getcrumb"

# URL de inicialización de sesión (necesaria para obtener cookies antes del crumb)
YAHOO_SESSION_INIT_URL: str = "https://finance.yahoo.com"

# Parámetros de consulta
YAHOO_INTERVAL: str = "1d"   # Frecuencia diaria
YAHOO_RANGE: str = "5y"      # Horizonte de 5 años

# Headers HTTP que simulan un navegador para evitar bloqueos
HTTP_HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
}

# ============================================================
# PARÁMETROS DE RED
# ============================================================

MAX_RETRIES: int = 3            # Número máximo de reintentos por activo
RETRY_DELAY: float = 2.0        # Segundos base para backoff exponencial
REQUEST_TIMEOUT: int = 30       # Timeout por solicitud HTTP (segundos)
DELAY_BETWEEN_REQUESTS: float = 1.5  # Pausa entre activos (evitar rate limit)

# ============================================================
# PORTAFOLIO DE 20 ACTIVOS
# ============================================================
# Fuente: documento "20ActivosX5años.pdf" del proyecto.
# Todos los tickers han sido validados contra el endpoint de Yahoo Finance.

ASSETS: List[Dict[str, str]] = [
    {
        "ticker": "EC",
        "name": "Ecopetrol",
        "type": "EQUITY",
        "market": "Colombia/NYSE (ADR)",
        "sector": "Energía",
    },
    {
        "ticker": "UBER",
        "name": "Uber Technologies, Inc.",
        "type": "EQUITY",
        "market": "EE.UU.",
        "sector": "Tecnología/Transporte",
    },
    {
        "ticker": "BBVA",
        "name": "Banco Bilbao Vizcaya Argentaria, S.A.",
        "type": "EQUITY",
        "market": "Colombia/NYSE",
        "sector": "Financiero",
    },
    {
        "ticker": "CIB",
        "name": "Grupo Cibest S.A.",
        "type": "EQUITY",
        "market": "Colombia",
        "sector": "Financiero",
    },
    {
        "ticker": "AVAL",
        "name": "Grupo Aval Acciones y Valores S.A.",
        "type": "EQUITY",
        "market": "Colombia",
        "sector": "Utilities",
    },
    {
        "ticker": "AAPL",
        "name": "Apple Inc.",
        "type": "EQUITY",
        "market": "EE.UU.",
        "sector": "Tecnología",
    },
    {
        "ticker": "MSFT",
        "name": "Microsoft",
        "type": "EQUITY",
        "market": "EE.UU.",
        "sector": "Tecnología",
    },
    {
        "ticker": "GOOGL",
        "name": "Alphabet Inc.",
        "type": "EQUITY",
        "market": "EE.UU.",
        "sector": "Tecnología",
    },
    {
        "ticker": "AMZN",
        "name": "Amazon",
        "type": "EQUITY",
        "market": "EE.UU.",
        "sector": "Consumo Discrecional",
    },
    {
        "ticker": "TSLA",
        "name": "Tesla",
        "type": "EQUITY",
        "market": "EE.UU.",
        "sector": "Automotriz",
    },
    {
        "ticker": "JPM",
        "name": "JPMorgan Chase",
        "type": "EQUITY",
        "market": "EE.UU.",
        "sector": "Financiero",
    },
    {
        "ticker": "XOM",
        "name": "Exxon Mobil",
        "type": "EQUITY",
        "market": "EE.UU.",
        "sector": "Energía",
    },
    {
        "ticker": "NVDA",
        "name": "NVIDIA",
        "type": "EQUITY",
        "market": "EE.UU.",
        "sector": "Tecnología",
    },
    {
        "ticker": "VOO",
        "name": "Vanguard S&P 500 ETF",
        "type": "ETF",
        "market": "EE.UU.",
        "sector": "Índice S&P 500",
    },
    {
        "ticker": "SPY",
        "name": "SPDR S&P 500 ETF Trust",
        "type": "ETF",
        "market": "EE.UU.",
        "sector": "Índice S&P 500",
    },
    {
        "ticker": "QQQ",
        "name": "Invesco QQQ Trust",
        "type": "ETF",
        "market": "EE.UU.",
        "sector": "Nasdaq 100",
    },
    {
        "ticker": "IWM",
        "name": "iShares Russell 2000 ETF",
        "type": "ETF",
        "market": "EE.UU.",
        "sector": "Small Caps",
    },
    {
        "ticker": "EEM",
        "name": "iShares MSCI Emerging Markets ETF",
        "type": "ETF",
        "market": "EE.UU.",
        "sector": "Mercados Emergentes",
    },
    {
        "ticker": "GLD",
        "name": "SPDR Gold Shares",
        "type": "ETF",
        "market": "EE.UU.",
        "sector": "Oro",
    },
    {
        "ticker": "XLF",
        "name": "Financial Select Sector SPDR Fund",
        "type": "ETF",
        "market": "EE.UU.",
        "sector": "Sector Financiero",
    },
]

# Lista rápida de tickers
TICKERS: List[str] = [asset["ticker"] for asset in ASSETS]

# Mapa rápido ticker → metadata completa
ASSET_MAP: Dict[str, Dict[str, str]] = {asset["ticker"]: asset for asset in ASSETS}

# ============================================================
# COLUMNAS DEL DATASET MAESTRO
# ============================================================
# Define el esquema del archivo master_dataset.csv

MASTER_COLUMNS: List[str] = [
    "date",             # Fecha de negociación (YYYY-MM-DD)
    "ticker",           # Símbolo del activo
    "open",             # Precio de apertura
    "high",             # Precio máximo del día
    "low",              # Precio mínimo del día
    "close",            # Precio de cierre
    "volume",           # Volumen de negociación
    "instrument_type",  # Tipo: EQUITY o ETF
]
