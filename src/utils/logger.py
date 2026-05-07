"""
src/utils/logger.py
===================
Configuración del sistema de logging del proyecto.

Salidas configuradas:
  1. Consola (stdout): nivel INFO — muestra el progreso en tiempo real.
  2. Archivo de log (logs/etl_YYYYMMDD_HHMMSS.log): nivel DEBUG —
     registra todos los detalles para diagnóstico.

Uso:
    from src.utils.logger import setup_logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Mensaje")
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

from src.config.settings import LOGS_DIR


def setup_logging(level: int = logging.INFO) -> None:
    """
    Inicializa y configura el sistema de logging del proyecto.

    Crea dos handlers:
      - ConsoleHandler: escribe a stdout con nivel INFO (o el especificado).
      - FileHandler:    escribe a logs/ con nivel DEBUG (registro completo).

    Los loggers de terceros (urllib3, requests) se silencian a WARNING
    para evitar ruido en la salida.

    Args:
        level: Nivel mínimo de logging para la consola (default: INFO).
               El archivo de log siempre recibe hasta DEBUG.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file: Path = LOGS_DIR / f"etl_{timestamp}.log"

    # Formato común para ambos handlers
    fmt = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=date_fmt)

    # ── Handler de consola ──────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # ── Handler de archivo ──────────────────────────────────────────────
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # ── Root logger ─────────────────────────────────────────────────────
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # El nivel más bajo; cada handler filtra

    # Evitar handlers duplicados si setup_logging se llama más de una vez
    if root_logger.handlers:
        root_logger.handlers.clear()

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Silenciar librerías de terceros verbosas
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("charset_normalizer").setLevel(logging.WARNING)

    root_logger.info(f"Logging inicializado. Archivo de log: {log_file}")
