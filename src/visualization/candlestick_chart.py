"""
src/visualization/candlestick_chart.py
=========================================
Genera figuras Plotly tipo candlestick (OHLC) con SMAs superpuestas.

Responsabilidades:
  - Cargar datos OHLCV del dataset maestro para un ticker específico.
  - Filtrar el rango temporal solicitado.
  - Calcular SMAs manualmente (delegando a moving_averages.py).
  - Construir el dict Plotly para candlestick + SMAs + volumen.
  - Calcular estadísticas del período seleccionado.

NO usa librerías de análisis técnico ni indicadores automáticos.
Plotly se usa SOLO para la representación visual.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config.settings import ASSET_MAP, PROCESSED_DIR
from src.visualization.moving_averages import compute_multiple_smas
from src.visualization.visualization_utils import (
    COLORS,
    dark_plotly_layout,
    sma_color,
    to_json_safe,
)

logger = logging.getLogger(__name__)

# Períodos temporales disponibles → número de filas desde el final
PERIOD_MAP: Dict[str, Optional[int]] = {
    "1m":   21,
    "3m":   63,
    "6m":  126,
    "1y":  252,
    "3y":  756,
    "all": None,
}


def filter_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    Recorta el DataFrame al número de filas correspondiente al período.

    Args:
        df:     DataFrame OHLCV ordenado cronológicamente.
        period: "1m" | "3m" | "6m" | "1y" | "3y" | "all".

    Returns:
        DataFrame recortado. Si period no es válido, devuelve todo el df.
    """
    n_rows = PERIOD_MAP.get(period.lower())
    if n_rows is None:
        return df
    return df.iloc[-n_rows:].copy()


def build_candlestick_figure(
    ticker: str,
    df: pd.DataFrame,
    sma_periods: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Construye la figura Plotly candlestick para un activo.

    La figura incluye:
      - Traza principal: candlestick OHLC (verde=subida, rojo=bajada).
      - Trazas secundarias: una línea por cada período SMA solicitado.
      - Eje Y secundario: volumen en barras semitransparentes.

    Args:
        ticker:      Símbolo del activo.
        df:          DataFrame con columnas date, open, high, low, close, volume.
                     Debe estar ordenado cronológicamente.
        sma_periods: Lista de períodos SMA a superponer (ej. [20, 50]).
                     Si None, se usan [20, 50] por defecto.

    Returns:
        Dict {data, layout} compatible con Plotly.js.
    """
    if sma_periods is None:
        sma_periods = [20, 50]

    # ── Preparar listas de datos ──────────────────────────────────────
    dates  = [str(d)[:10] for d in df["date"].values]
    opens  = [round(float(v), 4) for v in df["open"].values]
    highs  = [round(float(v), 4) for v in df["high"].values]
    lows   = [round(float(v), 4) for v in df["low"].values]
    closes = [round(float(v), 4) for v in df["close"].values]
    vols   = [int(v) if pd.notna(v) else 0 for v in df["volume"].values]

    asset_info = ASSET_MAP.get(ticker, {})
    asset_name = asset_info.get("name", ticker)

    # ── Traza candlestick ─────────────────────────────────────────────
    candle_trace = {
        "type":       "candlestick",
        "name":       ticker,
        "x":          dates,
        "open":       opens,
        "high":       highs,
        "low":        lows,
        "close":      closes,
        "increasing": {
            "line":      {"color": COLORS["conservative"], "width": 1},
            "fillcolor": COLORS["conservative"],
        },
        "decreasing": {
            "line":      {"color": COLORS["aggressive"], "width": 1},
            "fillcolor": COLORS["aggressive"],
        },
        "hovertemplate": (
            "<b>%{x}</b><br>"
            "O: %{open:.2f}<br>"
            "H: %{high:.2f}<br>"
            "L: %{low:.2f}<br>"
            "C: %{close:.2f}<extra></extra>"
        ),
        "yaxis": "y",
    }

    data = [candle_trace]

    # ── Trazas SMA (cálculo manual delegado a moving_averages.py) ─────
    if sma_periods:
        price_series = pd.Series(closes, index=pd.to_datetime(dates))
        smas = compute_multiple_smas(price_series, sma_periods)

        for period, sma_series in smas.items():
            sma_vals = [
                round(float(v), 4) if pd.notna(v) else None
                for v in sma_series.values
            ]
            data.append({
                "type":  "scatter",
                "name":  f"SMA {period}",
                "x":     dates,
                "y":     sma_vals,
                "mode":  "lines",
                "line":  {
                    "color": sma_color(period),
                    "width": 1.5,
                    "dash":  "solid",
                },
                "hovertemplate": f"SMA {period}: %{{y:.2f}}<extra></extra>",
                "connectgaps": False,
                "yaxis": "y",
            })

    # ── Traza de volumen (eje Y secundario) ───────────────────────────
    vol_colors = [
        COLORS["conservative"] if c >= o else COLORS["aggressive"]
        for c, o in zip(closes, opens)
    ]
    data.append({
        "type":    "bar",
        "name":    "Volumen",
        "x":       dates,
        "y":       vols,
        "marker":  {"color": vol_colors, "opacity": 0.35},
        "hovertemplate": "Vol: %{y:,.0f}<extra></extra>",
        "yaxis":   "y2",
        "showlegend": False,
    })

    # ── Layout ────────────────────────────────────────────────────────
    layout = dark_plotly_layout(
        title=f"{asset_name} ({ticker}) — Candlestick OHLC",
        height=560,
        margin={"l": 50, "r": 80, "t": 60, "b": 30},
        show_legend=True,
        xaxis_extra={
            "type":        "category",
            "rangeslider": {"visible": False},
            "nticks":      12,
            "showgrid":    False,
        },
        yaxis_extra={
            "title":  {"text": "Precio", "font": {"color": COLORS["text_muted"], "size": 11}},
            "side":   "right",
            "domain": [0.20, 1.0],
        },
    )

    # Eje secundario para volumen
    layout["yaxis2"] = {
        "domain":        [0.0, 0.17],
        "showticklabels": False,
        "gridcolor":     COLORS["bg_tertiary"],
        "linecolor":     COLORS["border"],
    }

    # Selector de rango integrado en eje X
    layout["xaxis"]["rangeselector"] = {
        "buttons": [
            {"count": 21,  "label": "1M",  "step": "day", "stepmode": "backward"},
            {"count": 63,  "label": "3M",  "step": "day", "stepmode": "backward"},
            {"count": 126, "label": "6M",  "step": "day", "stepmode": "backward"},
            {"count": 252, "label": "1Y",  "step": "day", "stepmode": "backward"},
            {"step": "all", "label": "Todo"},
        ],
        "bgcolor":     COLORS["bg_tertiary"],
        "activecolor": COLORS["accent"],
        "bordercolor": COLORS["border"],
        "font":        {"color": COLORS["text"], "size": 11},
    }

    return {"data": data, "layout": layout}


def prepare_ohlcv(
    ticker: str,
    master: pd.DataFrame,
    period: str = "1y",
    sma_periods: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Punto de entrada principal para el endpoint candlestick.

    Carga, filtra y prepara todos los datos necesarios para renderizar
    el chart y las estadísticas.

    Args:
        ticker:      Símbolo del activo.
        master:      DataFrame maestro (ya cargado para evitar re-lectura).
        period:      Período temporal para el filtro.
        sma_periods: Períodos SMA a calcular.

    Returns:
        Dict con:
          - figure:    figura Plotly JSON-serializable.
          - stats:     estadísticas del período (precio actual, rango, etc.)
          - sma_current: último valor de cada SMA.

    Raises:
        ValueError: Si el ticker no existe o no tiene datos suficientes.
    """
    if sma_periods is None:
        sma_periods = [20, 50]

    # ── Filtrar activo ────────────────────────────────────────────────
    df = master[master["ticker"] == ticker].copy()
    if df.empty:
        available = sorted(master["ticker"].unique().tolist())
        raise ValueError(
            f"Ticker '{ticker}' no encontrado. Disponibles: {available}"
        )

    df = df.sort_values("date").reset_index(drop=True)

    # ── Filtrar período ───────────────────────────────────────────────
    df_period = filter_period(df, period)

    if len(df_period) < 5:
        raise ValueError(
            f"Insuficientes datos para '{ticker}' con período '{period}' "
            f"({len(df_period)} filas)."
        )

    # ── Construir figura ──────────────────────────────────────────────
    figure = build_candlestick_figure(ticker, df_period, sma_periods)

    # ── Calcular estadísticas ─────────────────────────────────────────
    closes = [float(v) for v in df_period["close"].values if pd.notna(v)]
    highs  = [float(v) for v in df_period["high"].values  if pd.notna(v)]
    lows   = [float(v) for v in df_period["low"].values   if pd.notna(v)]

    asset_info = ASSET_MAP.get(ticker, {})

    stats = {
        "ticker":          ticker,
        "name":            asset_info.get("name", ticker),
        "instrument_type": asset_info.get("type", ""),
        "sector":          asset_info.get("sector", ""),
        "n_candles":       len(df_period),
        "date_start":      str(df_period["date"].iloc[0])[:10],
        "date_end":        str(df_period["date"].iloc[-1])[:10],
        "current_close":   round(closes[-1], 2) if closes else None,
        "first_close":     round(closes[0],  2) if closes else None,
        "max_high":        round(max(highs),  2) if highs  else None,
        "min_low":         round(min(lows),   2) if lows   else None,
        "change_pct": (
            round(((closes[-1] - closes[0]) / closes[0]) * 100, 2)
            if len(closes) >= 2 and closes[0] != 0 else None
        ),
    }

    # ── Valor actual de cada SMA ──────────────────────────────────────
    price_series = pd.Series(closes)
    sma_dict = {}
    for p in sma_periods:
        from src.visualization.moving_averages import compute_sma
        sma_vals = compute_sma(closes, p)
        last_sma = next((v for v in reversed(sma_vals) if v is not None), None)
        sma_dict[str(p)] = round(last_sma, 2) if last_sma is not None else None

    return {
        "figure":      to_json_safe(figure),
        "stats":       stats,
        "sma_current": sma_dict,
    }
