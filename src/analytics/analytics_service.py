"""
src/analytics/analytics_service.py
====================================
Capa de servicio centralizada para el análisis financiero de activos.

Orquesta el pipeline completo del Req. #3:
    1. Cargar dataset maestro (master_dataset.csv).
    2. Para cada activo: extraer serie de precios → retornos → volatilidad
       → clasificación de riesgo → detección de patrones.
    3. Construir los tres reportes CSV de salida.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
API PRINCIPAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  service = AnalyticsService()

  # Análisis de un solo activo
  result = service.analyze_asset("NVDA")

  # Reportes completos (todos los tickers)
  vol_df     = service.generate_volatility_summary()   → volatility_summary.csv
  ranking_df = service.generate_risk_ranking()         → risk_ranking.csv
  pattern_df = service.generate_pattern_analysis()     → pattern_analysis.csv

  # Ejecutar todo de una sola vez
  paths = service.run_full_analysis()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISEÑO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  - El dataset maestro se carga una sola vez y se almacena en caché.
  - Los resultados por activo se calculan bajo demanda (lazy) pero
    run_full_analysis() los computa todos de una vez para evitar
    múltiples lecturas del CSV.
  - Este módulo NO importa de src.etl.* ni src.similarity.*.
  - Lee de src.config.settings solo para rutas y ASSET_MAP.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.config.settings import PROCESSED_DIR, ASSET_MAP
from src.analytics.returns import compute_simple_returns
from src.analytics.volatility import compute_volatility
from src.analytics.risk_classifier import classify_risk
from src.analytics.pattern_detector import (
    detect_consecutive_gains,
    detect_drop_and_recovery,
    DEFAULT_CONSECUTIVE_DAYS,
    DEFAULT_DROP_THRESHOLD,
    DEFAULT_RECOVERY_THRESHOLD,
)
from src.analytics.ranking import build_risk_ranking, save_risk_ranking

logger = logging.getLogger(__name__)

# Rutas de salida de los tres reportes
_VOLATILITY_SUMMARY_PATH: Path = PROCESSED_DIR / "volatility_summary.csv"
_RISK_RANKING_PATH: Path        = PROCESSED_DIR / "risk_ranking.csv"
_PATTERN_ANALYSIS_PATH: Path    = PROCESSED_DIR / "pattern_analysis.csv"

# Ruta por defecto del dataset maestro
_MASTER_PATH: Path = PROCESSED_DIR / "master_dataset.csv"


# ====================================================================== #
# Utilidad interna: carga del dataset maestro                             #
# ====================================================================== #

def _load_master_dataset(path: Path) -> pd.DataFrame:
    """
    Carga y valida el dataset maestro.

    Args:
        path: Ruta al archivo master_dataset.csv.

    Returns:
        DataFrame con columna 'date' como datetime.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el DataFrame está vacío o le faltan columnas clave.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset maestro no encontrado: {path}\n"
            "Ejecuta primero: python main.py"
        )

    df = pd.read_csv(path, parse_dates=["date"])

    if df.empty:
        raise ValueError(f"El dataset maestro está vacío: {path}")

    required = {"date", "ticker", "close", "instrument_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Columnas faltantes en el dataset maestro: {missing}"
        )

    logger.debug(
        f"Dataset maestro cargado: {len(df):,} filas, "
        f"{df['ticker'].nunique()} activos."
    )
    return df


# ====================================================================== #
# Clase principal                                                          #
# ====================================================================== #

class AnalyticsService:
    """
    Servicio centralizado de análisis financiero.

    Atributos:
        _master_path: Ruta al dataset maestro.
        _master:      DataFrame en caché (None hasta la primera carga).
    """

    def __init__(self, master_path: Optional[Path] = None) -> None:
        self._master_path: Path = master_path or _MASTER_PATH
        self._master: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ #
    # Acceso al dataset maestro (lazy loading con caché)                  #
    # ------------------------------------------------------------------ #

    def _get_master(self) -> pd.DataFrame:
        """Carga el dataset maestro si aún no está en caché."""
        if self._master is None:
            self._master = _load_master_dataset(self._master_path)
        return self._master

    def _get_close_series(self, ticker: str) -> pd.Series:
        """
        Extrae la serie de precios de cierre de un activo específico.

        Args:
            ticker: Símbolo del activo (ej. "NVDA").

        Returns:
            Serie de pandas con índice datetime y valores de cierre,
            ordenada ascendentemente por fecha.

        Raises:
            ValueError: Si el ticker no existe en el dataset maestro.
        """
        master = self._get_master()
        df = master[master["ticker"] == ticker]

        if df.empty:
            available = sorted(master["ticker"].unique().tolist())
            raise ValueError(
                f"Ticker '{ticker}' no encontrado en el dataset maestro. "
                f"Disponibles: {available}"
            )

        df = df.sort_values("date")
        series = df.set_index("date")["close"]
        series.name = ticker
        return series

    def _get_instrument_type(self, ticker: str) -> str:
        """Devuelve el tipo de instrumento (EQUITY/ETF) de un ticker."""
        master = self._get_master()
        rows = master[master["ticker"] == ticker]["instrument_type"]
        if rows.empty:
            return "UNKNOWN"
        return str(rows.iloc[0])

    # ------------------------------------------------------------------ #
    # Propiedades                                                          #
    # ------------------------------------------------------------------ #

    @property
    def available_tickers(self) -> List[str]:
        """Lista de tickers disponibles en el dataset maestro."""
        return sorted(self._get_master()["ticker"].unique().tolist())

    # ------------------------------------------------------------------ #
    # Análisis de un activo individual                                     #
    # ------------------------------------------------------------------ #

    def analyze_asset(
        self,
        ticker: str,
        min_consecutive_days: int   = DEFAULT_CONSECUTIVE_DAYS,
        drop_threshold: float        = DEFAULT_DROP_THRESHOLD,
        recovery_threshold: float    = DEFAULT_RECOVERY_THRESHOLD,
    ) -> Dict:
        """
        Ejecuta el análisis completo de un activo financiero.

        Pipeline:
            precios → retornos → volatilidad diaria/anual
            → clasificación riesgo
            → patrón #1 (N días consecutivos)
            → patrón #2 (caída + recuperación)

        Args:
            ticker:               Símbolo del activo (ej. "NVDA").
            min_consecutive_days: Días consecutivos al alza para patrón #1.
            drop_threshold:       Umbral de caída para patrón #2.
            recovery_threshold:   Umbral de recuperación para patrón #2.

        Returns:
            Dict con todas las métricas del activo.
        """
        prices        = self._get_close_series(ticker)
        returns       = compute_simple_returns(prices)
        daily_vol, annual_vol = compute_volatility(returns)
        risk_category = classify_risk(annual_vol)
        inst_type     = self._get_instrument_type(ticker)

        # Metadatos de la serie
        date_start = str(prices.index.min().date()) if hasattr(prices.index.min(), "date") else str(prices.index.min())
        date_end   = str(prices.index.max().date()) if hasattr(prices.index.max(), "date") else str(prices.index.max())

        # Detección de patrones
        pattern1 = detect_consecutive_gains(prices, min_consecutive_days)
        pattern2 = detect_drop_and_recovery(prices, drop_threshold, recovery_threshold)

        # Nombre legible del activo (desde ASSET_MAP en settings.py)
        asset_info = ASSET_MAP.get(ticker, {})

        return {
            "ticker":               ticker,
            "name":                 asset_info.get("name", ticker),
            "instrument_type":      inst_type,
            "n_observations":       len(prices),
            "n_return_observations": len(returns),
            "date_start":           date_start,
            "date_end":             date_end,
            "daily_volatility":     daily_vol,
            "annualized_volatility": annual_vol,
            "risk_category":        risk_category,
            "patterns": {
                "consecutive_gains": pattern1,
                "drop_and_recovery": pattern2,
            },
        }

    # ------------------------------------------------------------------ #
    # Generación de reportes                                               #
    # ------------------------------------------------------------------ #

    def generate_volatility_summary(
        self,
        min_consecutive_days: int = DEFAULT_CONSECUTIVE_DAYS,
        drop_threshold: float      = DEFAULT_DROP_THRESHOLD,
        recovery_threshold: float  = DEFAULT_RECOVERY_THRESHOLD,
    ) -> pd.DataFrame:
        """
        Calcula la volatilidad de todos los activos del portafolio.

        Returns:
            DataFrame con una fila por activo, ordenado descendente por
            volatilidad anualizada. Columnas:
              ticker, name, instrument_type, n_observations,
              daily_vol_pct, annualized_vol_pct, risk_category,
              date_start, date_end.
        """
        rows: List[Dict] = []

        for ticker in self.available_tickers:
            try:
                result = self.analyze_asset(
                    ticker,
                    min_consecutive_days,
                    drop_threshold,
                    recovery_threshold,
                )
                rows.append({
                    "ticker":             result["ticker"],
                    "name":               result["name"],
                    "instrument_type":    result["instrument_type"],
                    "n_observations":     result["n_observations"],
                    "daily_vol_pct":      round(result["daily_volatility"]     * 100, 4),
                    "annualized_vol_pct": round(result["annualized_volatility"] * 100, 2),
                    "risk_category":      result["risk_category"],
                    "date_start":         result["date_start"],
                    "date_end":           result["date_end"],
                })
            except Exception as exc:
                logger.warning(f"[{ticker}] Error en analyze_asset: {exc}")

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("annualized_vol_pct", ascending=False).reset_index(drop=True)

        return df

    def generate_risk_ranking(
        self,
        min_consecutive_days: int = DEFAULT_CONSECUTIVE_DAYS,
        drop_threshold: float      = DEFAULT_DROP_THRESHOLD,
        recovery_threshold: float  = DEFAULT_RECOVERY_THRESHOLD,
    ) -> pd.DataFrame:
        """
        Construye el ranking de riesgo de todos los activos.

        Returns:
            DataFrame ordenado descendente por volatilidad anualizada,
            con columna 'rank' (1 = más riesgoso).
        """
        records: List[Dict] = []

        for ticker in self.available_tickers:
            try:
                result = self.analyze_asset(
                    ticker,
                    min_consecutive_days,
                    drop_threshold,
                    recovery_threshold,
                )
                records.append({
                    "ticker":               result["ticker"],
                    "instrument_type":      result["instrument_type"],
                    "daily_volatility":     result["daily_volatility"],
                    "annualized_volatility": result["annualized_volatility"],
                    "risk_category":        result["risk_category"],
                    "n_observations":       result["n_observations"],
                    "date_start":           result["date_start"],
                    "date_end":             result["date_end"],
                })
            except Exception as exc:
                logger.warning(f"[{ticker}] Error al construir ranking: {exc}")

        return build_risk_ranking(records)

    def generate_pattern_analysis(
        self,
        min_consecutive_days: int = DEFAULT_CONSECUTIVE_DAYS,
        drop_threshold: float      = DEFAULT_DROP_THRESHOLD,
        recovery_threshold: float  = DEFAULT_RECOVERY_THRESHOLD,
    ) -> pd.DataFrame:
        """
        Ejecuta la detección de patrones para todos los activos.

        Returns:
            DataFrame con una fila por (ticker × patrón). Columnas:
              ticker, instrument_type, pattern, description,
              n_occurrences, total_windows, frequency_pct.
        """
        rows: List[Dict] = []

        for ticker in self.available_tickers:
            try:
                result = self.analyze_asset(
                    ticker,
                    min_consecutive_days,
                    drop_threshold,
                    recovery_threshold,
                )
                for pattern_data in result["patterns"].values():
                    rows.append({
                        "ticker":          result["ticker"],
                        "instrument_type": result["instrument_type"],
                        "pattern":         pattern_data["pattern"],
                        "description":     pattern_data["description"],
                        "n_occurrences":   pattern_data["n_occurrences"],
                        "total_windows":   pattern_data["total_windows"],
                        "frequency_pct":   round(pattern_data["frequency"] * 100, 2),
                    })
            except Exception as exc:
                logger.warning(f"[{ticker}] Error en detección de patrones: {exc}")

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # Ejecución completa del pipeline                                      #
    # ------------------------------------------------------------------ #

    def run_full_analysis(
        self,
        min_consecutive_days: int = DEFAULT_CONSECUTIVE_DAYS,
        drop_threshold: float      = DEFAULT_DROP_THRESHOLD,
        recovery_threshold: float  = DEFAULT_RECOVERY_THRESHOLD,
        print_summary: bool        = True,
    ) -> Dict[str, Path]:
        """
        Ejecuta el análisis completo del portafolio y guarda los tres CSVs.

        Pipeline:
            1. analyze_asset() para cada ticker.       — O(n) por ticker
            2. Construir volatility_summary.csv.
            3. Construir risk_ranking.csv.
            4. Construir pattern_analysis.csv.
            5. Guardar los tres archivos en data/processed/.

        Args:
            min_consecutive_days: Parámetro para patrón #1.
            drop_threshold:       Umbral de caída para patrón #2.
            recovery_threshold:   Umbral de recuperación para patrón #2.
            print_summary:        Si True, imprime resumen en consola.

        Returns:
            Dict con las rutas de los tres archivos generados:
              {
                "volatility_summary": Path,
                "risk_ranking":       Path,
                "pattern_analysis":   Path,
              }
        """
        t0 = time.time()
        logger.info("Iniciando análisis completo del portafolio...")

        # ── 1. Analizar todos los activos (un solo recorrido) ──────────
        all_results: List[Dict] = []

        for ticker in self.available_tickers:
            try:
                result = self.analyze_asset(
                    ticker,
                    min_consecutive_days,
                    drop_threshold,
                    recovery_threshold,
                )
                all_results.append(result)
                logger.debug(
                    f"[{ticker}] σ_anual={result['annualized_volatility']*100:.1f}%  "
                    f"riesgo={result['risk_category']}"
                )
            except Exception as exc:
                logger.warning(f"[{ticker}] Análisis fallido: {exc}")

        # ── 2. Volatility summary ──────────────────────────────────────
        vol_rows = [
            {
                "ticker":             r["ticker"],
                "name":               r["name"],
                "instrument_type":    r["instrument_type"],
                "n_observations":     r["n_observations"],
                "daily_vol_pct":      round(r["daily_volatility"]     * 100, 4),
                "annualized_vol_pct": round(r["annualized_volatility"] * 100, 2),
                "risk_category":      r["risk_category"],
                "date_start":         r["date_start"],
                "date_end":           r["date_end"],
            }
            for r in all_results
        ]
        vol_df = pd.DataFrame(vol_rows)
        if not vol_df.empty:
            vol_df = vol_df.sort_values(
                "annualized_vol_pct", ascending=False
            ).reset_index(drop=True)

        vol_path = _VOLATILITY_SUMMARY_PATH
        vol_path.parent.mkdir(parents=True, exist_ok=True)
        vol_df.to_csv(vol_path, index=False, encoding="utf-8")
        logger.info(f"volatility_summary.csv guardado: {vol_path}")

        # ── 3. Risk ranking ───────────────────────────────────────────
        rank_records = [
            {
                "ticker":                r["ticker"],
                "instrument_type":       r["instrument_type"],
                "daily_volatility":      r["daily_volatility"],
                "annualized_volatility": r["annualized_volatility"],
                "risk_category":         r["risk_category"],
                "n_observations":        r["n_observations"],
                "date_start":            r["date_start"],
                "date_end":              r["date_end"],
            }
            for r in all_results
        ]
        ranking_df = build_risk_ranking(rank_records)
        ranking_path = save_risk_ranking(ranking_df, _RISK_RANKING_PATH)

        # ── 4. Pattern analysis ───────────────────────────────────────
        pattern_rows = []
        for r in all_results:
            for pattern_data in r["patterns"].values():
                pattern_rows.append({
                    "ticker":          r["ticker"],
                    "instrument_type": r["instrument_type"],
                    "pattern":         pattern_data["pattern"],
                    "description":     pattern_data["description"],
                    "n_occurrences":   pattern_data["n_occurrences"],
                    "total_windows":   pattern_data["total_windows"],
                    "frequency_pct":   round(pattern_data["frequency"] * 100, 2),
                })
        pattern_df = pd.DataFrame(pattern_rows)
        pattern_path = _PATTERN_ANALYSIS_PATH
        pattern_df.to_csv(pattern_path, index=False, encoding="utf-8")
        logger.info(f"pattern_analysis.csv guardado: {pattern_path}")

        elapsed = time.time() - t0

        # ── 5. Resumen en consola ─────────────────────────────────────
        if print_summary:
            _print_analysis_report(vol_df, ranking_df, pattern_df, elapsed)

        return {
            "volatility_summary": vol_path,
            "risk_ranking":       ranking_path,
            "pattern_analysis":   pattern_path,
        }


# ====================================================================== #
# Reporte de consola                                                       #
# ====================================================================== #

def _print_analysis_report(
    vol_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    pattern_df: pd.DataFrame,
    elapsed: float,
) -> None:
    """Imprime un resumen del análisis en la consola."""
    print("\n" + "=" * 65)
    print("  ANÁLISIS DE VOLATILIDAD Y PATRONES — REQ. #3")
    print("=" * 65)

    if not vol_df.empty:
        print(f"\n{'ACTIVO':<8} {'VOL. ANUAL':>10} {'CATEGORÍA':<15}")
        print("-" * 35)
        for _, row in vol_df.iterrows():
            print(
                f"{row['ticker']:<8} "
                f"{row['annualized_vol_pct']:>9.1f}%  "
                f"{row['risk_category']}"
            )

    if not ranking_df.empty:
        cats = ranking_df["risk_category"].value_counts()
        print(f"\nCATEGORÍAS DE RIESGO:")
        for cat, count in cats.items():
            print(f"  {cat}: {count} activo(s)")

    if not pattern_df.empty:
        print(f"\nPATRONES DETECTADOS (frecuencia promedio por patrón):")
        for pattern, grp in pattern_df.groupby("pattern"):
            avg_freq = grp["frequency_pct"].mean()
            total_occ = grp["n_occurrences"].sum()
            print(f"  {pattern:<32} freq={avg_freq:.1f}%  ocurr.={total_occ}")

    print(f"\nTiempo de ejecución: {elapsed:.2f}s")
    print("=" * 65 + "\n")


# ====================================================================== #
# Función de conveniencia (nivel de módulo)                               #
# ====================================================================== #

def analyze_asset(
    ticker: str,
    master_path: Optional[Path] = None,
    **kwargs,
) -> Dict:
    """
    Función de acceso rápido para analizar un activo sin instanciar el servicio.

    Args:
        ticker:      Símbolo del activo.
        master_path: Ruta alternativa al dataset maestro.
        **kwargs:    Parámetros opcionales de analyze_asset().

    Returns:
        Dict con el análisis completo del activo.
    """
    return AnalyticsService(master_path).analyze_asset(ticker, **kwargs)


def generate_risk_ranking(master_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Función de acceso rápido para generar el ranking de riesgo completo.

    Args:
        master_path: Ruta alternativa al dataset maestro.

    Returns:
        DataFrame del ranking de riesgo.
    """
    return AnalyticsService(master_path).generate_risk_ranking()
