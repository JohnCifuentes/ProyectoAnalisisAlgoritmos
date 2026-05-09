"""
main.py
=======
Punto de entrada del proyecto de Análisis de Algoritmos.

Universidad del Quindío — Ingeniería de Sistemas y Computación
Asignatura: Análisis de Algoritmos

Uso:
    python main.py            → Inicia el servidor web Flask (local)
    python main.py --etl      → Ejecuta el pipeline ETL
    gunicorn main:app         → Inicia con Gunicorn (producción / Render)

Compatibilidad:
  - Gunicorn:  La variable global `app` es importada directamente.
               Comando: gunicorn main:app
  - Render:    Lee PORT del entorno (Render lo inyecta automáticamente).
               Start Command: gunicorn main:app
  - Local:     python main.py  arranca Flask en modo desarrollo.

Salida esperada (--etl):
    data/raw/{TICKER}.csv              — un archivo por activo (20 total)
    data/processed/master_dataset.csv  — dataset maestro consolidado
    data/processed/dataset_summary.csv — resumen estadístico por activo
    logs/etl_YYYYMMDD_HHMMSS.log       — log detallado de la ejecución
"""

import logging
import os
import sys

from src.utils.logger import setup_logging
from src.api.app import create_app


# ============================================================
# Instancia global de Flask
# Necesaria para Gunicorn y Render
# ============================================================
app = create_app(debug=False)


def run_etl() -> int:
    """
    Ejecuta el pipeline ETL completo para el portafolio de 20 activos.

    Returns:
        0 si el pipeline finalizó con datos válidos.
        1 si ocurrió un error crítico o el dataset está vacío.
    """
    from src.etl.pipeline import ETLPipeline

    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("=" * 65)
    logger.info("  ANÁLISIS DE ALGORITMOS — PROYECTO FINANCIERO")
    logger.info("  Universidad del Quindío")
    logger.info("  Fase 1: ETL — Extracción, Transformación y Carga")
    logger.info("=" * 65)

    try:
        pipeline = ETLPipeline()
        master = pipeline.run()

        if master.empty:
            logger.error(
                "El pipeline finalizó sin ningún dato. "
                "Revisa los logs para más detalles."
            )
            return 1

        logger.info(
            f"\nPipeline ETL completado exitosamente.\n"
            f"  Registros totales : {len(master):,}\n"
            f"  Activos           : {master['ticker'].nunique()}\n"
            f"  Archivo maestro   : data/processed/master_dataset.csv"
        )

        return 0

    except KeyboardInterrupt:
        logger.warning("\nEjecución interrumpida por el usuario (Ctrl+C)")
        return 1

    except Exception as exc:
        logger.exception(f"Error crítico no controlado: {exc}")
        return 1


def run_web() -> None:
    """
    Inicia el servidor web Flask.
    Compatible con Render y despliegue local.
    """
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("=" * 65)
    logger.info("  ANÁLISIS DE ALGORITMOS — PROYECTO FINANCIERO")
    logger.info("  Universidad del Quindío  |  Servidor Web")
    logger.info("=" * 65)

    # Puerto dinámico para Render
    port = int(os.environ.get("PORT", 5000))

    logger.info(f"Iniciando servidor Flask en puerto {port}")
    logger.info("Presiona Ctrl+C para detener el servidor.")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,     # Nunca True en producción (Render/Gunicorn)
        use_reloader=False
    )


if __name__ == "__main__":
    if "--etl" in sys.argv:
        sys.exit(run_etl())
    else:
        run_web()