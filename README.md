# Proyecto — Análisis de Algoritmos: Sistema Financiero ETL

**Universidad del Quindío — Ingeniería de Sistemas y Computación**  
**Asignatura:** Análisis de Algoritmos  
**Fase 1:** Requerimiento 1 — ETL (Extracción, Transformación y Carga)

---

## Descripción

Sistema automatizado de descarga, limpieza y unificación de datos financieros históricos para un portafolio de **20 activos** (acciones y ETFs) con **5 años** de historial diario.

Los datos se obtienen mediante **peticiones HTTP directas** a la API pública de Yahoo Finance (`query1.finance.yahoo.com/v8/finance/chart`), **sin usar** `yfinance` ni `pandas_datareader`.

---

## Portafolio de activos

| Ticker | Nombre | Tipo | Mercado |
|--------|--------|------|---------|
| EC | Ecopetrol | EQUITY | Colombia/NYSE |
| UBER | Uber Technologies | EQUITY | EE.UU. |
| BBVA | Banco Bilbao Vizcaya | EQUITY | NYSE |
| CIB | Grupo Cibest | EQUITY | Colombia |
| AVAL | Grupo Aval | EQUITY | Colombia |
| AAPL | Apple Inc. | EQUITY | EE.UU. |
| MSFT | Microsoft | EQUITY | EE.UU. |
| GOOGL | Alphabet Inc. | EQUITY | EE.UU. |
| AMZN | Amazon | EQUITY | EE.UU. |
| TSLA | Tesla | EQUITY | EE.UU. |
| JPM | JPMorgan Chase | EQUITY | EE.UU. |
| XOM | Exxon Mobil | EQUITY | EE.UU. |
| NVDA | NVIDIA | EQUITY | EE.UU. |
| VOO | Vanguard S&P 500 ETF | ETF | EE.UU. |
| SPY | SPDR S&P 500 ETF | ETF | EE.UU. |
| QQQ | Invesco QQQ Trust | ETF | EE.UU. |
| IWM | iShares Russell 2000 | ETF | EE.UU. |
| EEM | iShares MSCI Emerging | ETF | EE.UU. |
| GLD | SPDR Gold Shares | ETF | EE.UU. |
| XLF | Financial Select SPDR | ETF | EE.UU. |

---

## Requisitos previos

- **Python** 3.10 o superior
- Acceso a internet (para la descarga de datos de Yahoo Finance)

---

## Instalación

```bash
# 1. Clonar o descomprimir el proyecto
cd "03 Proyecto"

# 2. (Recomendado) Crear entorno virtual
python -m venv .venv

# En Windows:
.venv\Scripts\activate

# En Linux/macOS:
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## Ejecución

```bash
# Desde la raíz del proyecto (donde está main.py):
python main.py
```

El pipeline ejecuta automáticamente:
1. Inicialización de sesión HTTP con Yahoo Finance (cookies + crumb).
2. Descarga de datos para los 20 activos (~1–3 min según la red).
3. Parsing manual del JSON.
4. Limpieza y validación de datos.
5. Generación de CSVs individuales y dataset maestro.

---

## Archivos generados

```
data/
├── raw/
│   ├── AAPL.csv       ← datos limpios por activo
│   ├── MSFT.csv
│   └── ...            ← (20 archivos total)
└── processed/
    ├── master_dataset.csv    ← dataset maestro consolidado (~25 000 filas)
    └── dataset_summary.csv  ← resumen estadístico por activo

logs/
└── etl_YYYYMMDD_HHMMSS.log  ← log detallado de la ejecución
```

### Esquema del dataset maestro (`master_dataset.csv`)

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `date` | string | Fecha de negociación (YYYY-MM-DD) |
| `ticker` | string | Símbolo del activo |
| `open` | float | Precio de apertura |
| `high` | float | Precio máximo del día |
| `low` | float | Precio mínimo del día |
| `close` | float | Precio de cierre |
| `volume` | float | Volumen de negociación |
| `instrument_type` | string | Tipo: EQUITY o ETF |

---

## Ejecutar pruebas unitarias

```bash
pytest tests/ -v
```

---

## Estructura del proyecto

```
03 Proyecto/
│
├── data/
│   ├── raw/              ← CSVs individuales por activo
│   ├── processed/        ← dataset maestro y resumen
│   └── exports/          ← exportaciones futuras (reportes PDF, etc.)
│
├── logs/                 ← archivos de log por ejecución
│
├── src/
│   ├── config/
│   │   └── settings.py   ← configuración central (tickers, URLs, rutas)
│   ├── etl/
│   │   ├── extractor.py  ← peticiones HTTP a Yahoo Finance
│   │   ├── parser.py     ← parsing manual del JSON
│   │   ├── transformer.py← limpieza y validación de datos
│   │   ├── loader.py     ← persistencia en CSV
│   │   └── pipeline.py   ← orquestador del proceso ETL
│   ├── models/
│   │   └── asset.py      ← modelos de datos del dominio
│   └── utils/
│       └── logger.py     ← configuración de logging
│
├── tests/
│   └── test_etl.py       ← pruebas unitarias (parser, transformer, loader)
│
├── main.py               ← punto de entrada
├── requirements.txt
└── README.md
```

---

## Decisiones de diseño

### Descarga sin yfinance
Se usan peticiones HTTP directas con `requests`. La autenticación se realiza obteniendo un **crumb** de Yahoo Finance al inicio de la sesión, siguiendo el protocolo de la API v8.

### Parsing manual del JSON
El módulo `parser.py` navega explícitamente la estructura:
```
chart.result[0].timestamp[]              → fechas UNIX
chart.result[0].indicators.quote[0].open/high/low/close/volume[]
```
Todos los arrays están alineados por índice: `open[i]` corresponde al día `timestamp[i]`.

### Limpieza ligera (sin interpolación)
Yahoo Finance tiene tasas de datos faltantes < 0.1%. Se elimina en lugar de interpolar para no introducir precios sintéticos que distorsionarían los algoritmos de similitud y volatilidad en fases posteriores.

### Tolerancia a fallos
Si un activo falla durante la descarga, el pipeline continúa con los demás y reporta los fallos al final.

---

## Dependencias

| Librería | Versión mínima | Uso |
|----------|---------------|-----|
| `requests` | 2.31.0 | Peticiones HTTP a la API |
| `pandas` | 2.0.0 | Construcción y persistencia del dataset |
| `numpy` | 1.24.0 | Operaciones numéricas (fases siguientes) |
| `pytest` | 7.4.0 | Ejecución de pruebas unitarias |

> **No se usa** `yfinance`, `pandas_datareader` ni ninguna librería de trading.
