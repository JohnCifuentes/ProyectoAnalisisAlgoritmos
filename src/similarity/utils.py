"""
src/similarity/utils.py
=======================
Utilidades compartidas para los algoritmos de similitud de series de tiempo.

Proporciona las transformaciones previas necesarias antes de aplicar
cualquier métrica de similitud:

  1. Carga de series desde el dataset maestro (master_dataset.csv).
  2. Cálculo de retornos diarios porcentuales.
  3. Alineación temporal de dos series (intersección de fechas).
  4. Eliminación de pares NaN/NaN (limpieza defensiva).
  5. Normalización Z-score (opcional, para distancia euclidiana).

Por qué retornos y no precios:
  Los precios absolutos de activos con distinto nivel (ej. NVDA ≈ 800 USD
  vs AVAL ≈ 3 USD) crean un sesgo de magnitud que hace inútil la distancia
  euclidiana directa y distorsiona el coseno. Los retornos porcentuales
  normalizan la escala y hacen comparables series de cualquier activo.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.config.settings import PROCESSED_DIR

logger = logging.getLogger(__name__)

# Ruta por defecto del dataset maestro generado en el Req. 1
_MASTER_PATH: Path = PROCESSED_DIR / "master_dataset.csv"


# ====================================================================== #
# Carga de datos                                                           #
# ====================================================================== #

def load_master(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Carga el dataset maestro consolidado generado por el pipeline ETL.

    Args:
        path: Ruta al archivo CSV. Si es None, usa la ruta por defecto
              data/processed/master_dataset.csv.

    Returns:
        DataFrame con todas las columnas del dataset maestro,
        con la columna 'date' convertida a datetime.

    Raises:
        FileNotFoundError: Si el archivo no existe (el ETL no se ejecutó).
        ValueError: Si el CSV está vacío o carece de columnas esperadas.
    """
    target: Path = path or _MASTER_PATH

    if not target.exists():
        raise FileNotFoundError(
            f"Dataset maestro no encontrado: {target}\n"
            "Ejecuta primero: python main.py"
        )

    df = pd.read_csv(target, parse_dates=["date"])

    if df.empty:
        raise ValueError(f"El dataset maestro está vacío: {target}")

    required_cols = {"date", "ticker", "close"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Columnas faltantes en el dataset maestro: {missing}"
        )

    logger.debug(
        f"Dataset maestro cargado: {len(df):,} filas, "
        f"{df['ticker'].nunique()} activos únicos"
    )
    return df


def load_series(
    ticker: str,
    master: Optional[pd.DataFrame] = None,
    master_path: Optional[Path] = None,
) -> pd.Series:
    """
    Extrae la serie de precios de cierre diarios de un activo.

    Args:
        ticker:      Símbolo del activo (ej. 'AAPL', 'EC').
        master:      DataFrame maestro ya cargado (evita releer el CSV).
        master_path: Ruta al CSV maestro (si master es None).

    Returns:
        Series con índice de tipo datetime (fecha) y valores float (close),
        ordenada cronológicamente y sin duplicados de fecha.

    Raises:
        KeyError: Si el ticker no existe en el dataset.
    """
    if master is None:
        master = load_master(master_path)

    ticker_data = master[master["ticker"] == ticker].copy()

    if ticker_data.empty:
        available = sorted(master["ticker"].unique().tolist())
        raise KeyError(
            f"Ticker '{ticker}' no encontrado. "
            f"Disponibles: {available}"
        )

    series = (
        ticker_data
        .sort_values("date")
        .drop_duplicates(subset="date")
        .set_index("date")["close"]
    )

    logger.debug(
        f"[{ticker}] Serie cargada: {len(series)} días "
        f"({series.index.min().date()} → {series.index.max().date()})"
    )
    return series


# ====================================================================== #
# Retornos diarios                                                         #
# ====================================================================== #

def compute_returns(prices: pd.Series, drop_first_nan: bool = True) -> pd.Series:
    """
    Calcula los retornos porcentuales diarios de una serie de precios.

    Fórmula:
        r_t = (P_t - P_{t-1}) / P_{t-1}

    Equivalente aritmético de la diferencia relativa entre días consecutivos.
    El primer elemento siempre es NaN (no hay precio anterior) y se descarta
    cuando drop_first_nan=True.

    Complejidad: O(n) — una sola pasada sobre la serie.

    Args:
        prices:         Serie de precios de cierre (índice=fecha, valores=float).
        drop_first_nan: Si True, elimina el primer NaN resultante de la diferenciación.

    Returns:
        Serie de retornos diarios con el mismo índice de fechas (sin el primero
        si drop_first_nan=True).
    """
    n = len(prices)
    if n < 2:
        raise ValueError(
            f"Se necesitan al menos 2 precios para calcular retornos "
            f"(se recibieron {n})"
        )

    # Calcular retornos manualmente para evitar dependencia de .pct_change()
    # en términos de transparencia algorítmica.
    # r[i] = (prices[i] - prices[i-1]) / prices[i-1]
    prices_arr = prices.values.astype(float)
    returns_arr = np.empty(n)
    returns_arr[0] = np.nan  # No hay retorno para el primer día

    for i in range(1, n):
        prev = prices_arr[i - 1]
        curr = prices_arr[i]
        if prev == 0.0 or np.isnan(prev) or np.isnan(curr):
            returns_arr[i] = np.nan
        else:
            returns_arr[i] = (curr - prev) / prev

    returns = pd.Series(returns_arr, index=prices.index, name=prices.name)

    if drop_first_nan:
        returns = returns.iloc[1:]

    return returns


# ====================================================================== #
# Alineación temporal                                                      #
# ====================================================================== #

def align_series(
    series_a: pd.Series,
    series_b: pd.Series,
) -> Tuple[pd.Series, pd.Series]:
    """
    Alinea dos series temporales conservando solo las fechas comunes.

    Las series de activos de mercados distintos (Colombia vs NYSE) pueden
    tener calendarios bursátiles diferentes: días festivos, suspensiones o
    fechas de inicio distintas producen desalineación temporal. Para que
    los algoritmos de similitud sean válidos, ambas series deben cubrir
    exactamente el mismo conjunto de fechas.

    Proceso:
      1. Calcular la intersección de índices (fechas comunes a ambas series).
      2. Filtrar cada serie para conservar solo esas fechas.
      3. Ordenar cronológicamente.

    Complejidad: O(n log n) por la operación de intersección con índice ordenado.

    Args:
        series_a: Primera serie temporal (índice=datetime).
        series_b: Segunda serie temporal (índice=datetime).

    Returns:
        Tupla (a_aligned, b_aligned) con el mismo conjunto de fechas.

    Raises:
        ValueError: Si no hay fechas comunes entre las dos series.
    """
    common_dates = series_a.index.intersection(series_b.index)

    if len(common_dates) == 0:
        raise ValueError(
            f"No hay fechas comunes entre las series "
            f"'{series_a.name}' y '{series_b.name}'"
        )

    a_aligned = series_a.loc[common_dates].sort_index()
    b_aligned = series_b.loc[common_dates].sort_index()

    n_dropped_a = len(series_a) - len(a_aligned)
    n_dropped_b = len(series_b) - len(b_aligned)

    if n_dropped_a > 0 or n_dropped_b > 0:
        logger.debug(
            f"Alineación temporal: {len(common_dates)} fechas comunes. "
            f"Omitidas — A: {n_dropped_a}, B: {n_dropped_b}"
        )

    return a_aligned, b_aligned


# ====================================================================== #
# Limpieza de pares NaN                                                    #
# ====================================================================== #

def remove_nan_pairs(
    x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Elimina posiciones donde cualquiera de los dos vectores tiene NaN.

    Después de alinear y calcular retornos puede quedar algún NaN residual
    (ej. precios inválidos que sobrevivieron la limpieza, primer día de retornos).
    Los algoritmos de similitud requieren vectores densos sin valores faltantes.

    Complejidad: O(n).

    Args:
        x: Array numpy de la primera serie.
        y: Array numpy de la segunda serie.

    Returns:
        Tupla (x_clean, y_clean) sin posiciones NaN.

    Raises:
        ValueError: Si no quedan elementos válidos tras la limpieza.
    """
    # Máscara booleana: True donde NINGUNO de los dos es NaN
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]

    removed = int(np.sum(~valid_mask))
    if removed > 0:
        logger.debug(f"remove_nan_pairs: {removed} pares NaN eliminados")

    if len(x_clean) == 0:
        raise ValueError(
            "No quedan elementos válidos después de eliminar NaN. "
            "Verifica la calidad de las series."
        )

    return x_clean, y_clean


# ====================================================================== #
# Normalización                                                            #
# ====================================================================== #

def z_normalize(x: np.ndarray) -> np.ndarray:
    """
    Normalización Z-score de un vector numérico.

    Fórmula:
        z_i = (x_i - μ) / σ

    Donde μ es la media y σ es la desviación estándar.

    La normalización es útil para la distancia euclidiana cuando se quiere
    comparar patrones de forma sin considerar la magnitud absoluta de los
    retornos. Sin normalizar, activos más volátiles (TSLA, NVDA) siempre
    aparecerán más "distintos" que activos estables (VOO, GLD) por pura
    diferencia de escala.

    Complejidad: O(n).

    Args:
        x: Array numpy unidimensional.

    Returns:
        Array normalizado con media ≈ 0 y desviación ≈ 1.
        Si σ = 0 (serie constante), retorna un array de ceros.
    """
    mu = 0.0
    sigma = 0.0

    # Calcular media manualmente: μ = (1/n) * Σxi
    n = len(x)
    for val in x:
        mu += val
    mu /= n

    # Calcular desviación estándar muestral: σ = sqrt((1/n) * Σ(xi - μ)²)
    for val in x:
        sigma += (val - mu) ** 2
    sigma = (sigma / n) ** 0.5

    if sigma == 0.0:
        logger.warning(
            "z_normalize: σ = 0 (serie constante). "
            "Retornando array de ceros."
        )
        return np.zeros(n, dtype=float)

    return (x - mu) / sigma


# ====================================================================== #
# Pipeline completo de preparación                                         #
# ====================================================================== #

def prepare_series(
    ticker_a: str,
    ticker_b: str,
    master: Optional[pd.DataFrame] = None,
    master_path: Optional[Path] = None,
    on: str = "returns",
    normalize: bool = False,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Pipeline completo de preparación de dos series para comparación.

    Pasos:
      1. Cargar precios de cierre de ambos activos.
      2. Calcular retornos diarios (si on='returns') o usar precios (on='prices').
      3. Alinear temporalmente por fechas comunes.
      4. Eliminar pares NaN residuales.
      5. Normalizar Z-score si normalize=True.

    Args:
        ticker_a:    Símbolo del primer activo.
        ticker_b:    Símbolo del segundo activo.
        master:      DataFrame maestro (si ya está cargado).
        master_path: Ruta al CSV maestro (si master es None).
        on:          'returns' (default) o 'prices'. Define sobre qué serie operar.
        normalize:   Si True, aplica z_normalize a ambos vectores.

    Returns:
        Tupla (x, y, meta) donde:
          x:    Array numpy de la primera serie lista para los algoritmos.
          y:    Array numpy de la segunda serie lista para los algoritmos.
          meta: Dict con información contextual:
                  - ticker_a, ticker_b
                  - n_points: número de puntos comunes
                  - date_start, date_end: rango temporal
                  - on, normalized
    """
    if master is None:
        master = load_master(master_path)

    # Paso 1: Cargar precios
    prices_a = load_series(ticker_a, master=master)
    prices_b = load_series(ticker_b, master=master)

    # Paso 2: Transformar a retornos o usar precios directamente
    if on == "returns":
        series_a = compute_returns(prices_a)
        series_b = compute_returns(prices_b)
    elif on == "prices":
        series_a = prices_a
        series_b = prices_b
    else:
        raise ValueError(
            f"Parámetro 'on' inválido: '{on}'. "
            "Use 'returns' o 'prices'."
        )

    # Paso 3: Alinear por fechas comunes
    series_a, series_b = align_series(series_a, series_b)

    # Paso 4: Extraer arrays numpy y eliminar NaN
    x = series_a.values.astype(float)
    y = series_b.values.astype(float)
    x, y = remove_nan_pairs(x, y)

    # Paso 5: Normalización opcional
    if normalize:
        x = z_normalize(x)
        y = z_normalize(y)

    # Rango temporal de las series alineadas
    date_start = series_a.index.min().date().isoformat()
    date_end = series_a.index.max().date().isoformat()

    meta = {
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "n_points": len(x),
        "date_start": date_start,
        "date_end": date_end,
        "on": on,
        "normalized": normalize,
    }

    logger.info(
        f"Series preparadas: {ticker_a} vs {ticker_b} | "
        f"{len(x)} puntos | {date_start} → {date_end} | on={on}"
    )

    return x, y, meta
