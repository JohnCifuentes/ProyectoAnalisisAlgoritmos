"""
tests/test_analytics.py
========================
Pruebas unitarias para el paquete analytics (Req. #3).

Cobertura:
  - Returns:          cálculo de retornos simples y logarítmicos.
  - Volatility:       media, varianza, desviación estándar, anualización.
  - RiskClassifier:   umbrales, bordes, casos inválidos.
  - SlidingWindow:    generador, tamaños, validaciones.
  - PatternDetector:  patrón #1 (consecutivos), patrón #2 (drop+recovery).
  - Ranking:          construcción, ordenamiento, rank 1-based.
  - AnalyticsService: integración con dataset maestro real.

Uso:
    pytest tests/test_analytics.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math

import numpy as np
import pandas as pd
import pytest

# ── Imports del paquete analytics ────────────────────────────────────
from src.analytics.returns import compute_simple_returns, compute_log_returns
from src.analytics.volatility import (
    _compute_mean,
    _compute_variance,
    compute_daily_volatility,
    compute_annualized_volatility,
    compute_volatility,
    TRADING_DAYS_PER_YEAR,
)
from src.analytics.risk_classifier import (
    classify_risk,
    classify_risk_with_score,
    RISK_CONSERVATIVE,
    RISK_MODERATE,
    RISK_AGGRESSIVE,
    THRESHOLD_CONSERVATIVE,
    THRESHOLD_MODERATE,
)
from src.analytics.sliding_window import (
    generate_windows,
    get_all_windows,
    window_statistics,
)
from src.analytics.pattern_detector import (
    detect_consecutive_gains,
    detect_drop_and_recovery,
)
from src.analytics.ranking import build_risk_ranking
from src.analytics.analytics_service import AnalyticsService


# ====================================================================== #
# Fixtures compartidos                                                     #
# ====================================================================== #

@pytest.fixture
def price_series_up():
    """Serie de precios estrictamente creciente."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    return pd.Series([100.0, 101.0, 102.0, 103.0, 104.0,
                      105.0, 106.0, 107.0, 108.0, 109.0],
                     index=dates, name="TEST")


@pytest.fixture
def price_series_down():
    """Serie de precios estrictamente decreciente."""
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    return pd.Series([100.0, 99.0, 98.0, 97.0, 96.0],
                     index=dates, name="TEST")


@pytest.fixture
def price_series_mixed():
    """Serie de precios con movimientos mixtos."""
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    return pd.Series([100.0, 97.0, 99.5, 96.0, 98.5, 101.0, 99.0, 102.0],
                     index=dates, name="TEST")


@pytest.fixture
def price_series_drop_recovery():
    """
    Serie con un evento claro de drop + recovery:
      día 0→1: caída del 5%  (p: 100 → 95)
      día 1→2: recuperación del 4.2% (p: 95 → 99)
    """
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    return pd.Series([100.0, 95.0, 99.0, 100.0, 101.0],
                     index=dates, name="TEST")


@pytest.fixture
def returns_series():
    """Serie de retornos sintéticos para tests de volatilidad."""
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    return pd.Series([0.01, -0.02, 0.015, -0.005, 0.008],
                     index=dates, name="TEST")


@pytest.fixture
def volatility_records():
    """Lista de registros de volatilidad para tests de ranking."""
    return [
        {
            "ticker": "TSLA", "instrument_type": "EQUITY",
            "daily_volatility": 0.030, "annualized_volatility": 0.476,
            "risk_category": "Agresivo",
            "n_observations": 1200, "date_start": "2021-01-01", "date_end": "2026-01-01",
        },
        {
            "ticker": "SPY", "instrument_type": "ETF",
            "daily_volatility": 0.010, "annualized_volatility": 0.159,
            "risk_category": "Conservador",
            "n_observations": 1250, "date_start": "2021-01-01", "date_end": "2026-01-01",
        },
        {
            "ticker": "AAPL", "instrument_type": "EQUITY",
            "daily_volatility": 0.018, "annualized_volatility": 0.286,
            "risk_category": "Moderado",
            "n_observations": 1240, "date_start": "2021-01-01", "date_end": "2026-01-01",
        },
    ]


# ====================================================================== #
# Tests: compute_simple_returns                                            #
# ====================================================================== #

class TestSimpleReturns:

    def test_basic_length(self, price_series_up):
        r = compute_simple_returns(price_series_up)
        # n precios → n-1 retornos
        assert len(r) == len(price_series_up) - 1

    def test_known_return_value(self):
        """100 → 110: retorno = 0.10 exacto."""
        dates = pd.date_range("2020-01-01", periods=2, freq="D")
        s = pd.Series([100.0, 110.0], index=dates)
        r = compute_simple_returns(s)
        assert r.iloc[0] == pytest.approx(0.10, rel=1e-9)

    def test_constant_prices_zero_return(self):
        dates = pd.date_range("2020-01-01", periods=4, freq="D")
        s = pd.Series([50.0, 50.0, 50.0, 50.0], index=dates)
        r = compute_simple_returns(s)
        for v in r.values:
            assert v == pytest.approx(0.0, abs=1e-12)

    def test_division_by_zero_produces_nan(self):
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        s = pd.Series([0.0, 100.0, 110.0], index=dates)
        r = compute_simple_returns(s, drop_na=False)
        assert math.isnan(r.iloc[0])

    def test_drop_na_removes_nan(self):
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        s = pd.Series([0.0, 100.0, 110.0], index=dates)
        r = compute_simple_returns(s, drop_na=True)
        assert not r.isna().any()

    def test_index_preserves_dates(self, price_series_up):
        r = compute_simple_returns(price_series_up)
        # El índice comienza en el segundo día de la serie original
        assert r.index[0] == price_series_up.index[1]

    def test_decreasing_prices_negative_returns(self, price_series_down):
        r = compute_simple_returns(price_series_down)
        assert all(v < 0 for v in r.values)

    def test_fewer_than_2_prices_raises(self):
        dates = pd.date_range("2020-01-01", periods=1, freq="D")
        s = pd.Series([100.0], index=dates)
        with pytest.raises(ValueError, match="al menos 2"):
            compute_simple_returns(s)

    def test_none_raises(self):
        with pytest.raises((ValueError, AttributeError)):
            compute_simple_returns(None)


class TestLogReturns:

    def test_basic_length(self, price_series_up):
        r = compute_log_returns(price_series_up)
        assert len(r) == len(price_series_up) - 1

    def test_known_value(self):
        """100 → e^1 × 100: log return = 1.0."""
        import math
        dates = pd.date_range("2020-01-01", periods=2, freq="D")
        s = pd.Series([1.0, math.e], index=dates)
        r = compute_log_returns(s)
        assert r.iloc[0] == pytest.approx(1.0, rel=1e-9)

    def test_non_positive_price_nan(self):
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        s = pd.Series([0.0, 100.0, 110.0], index=dates)
        r = compute_log_returns(s, drop_na=False)
        assert math.isnan(r.iloc[0])

    def test_fewer_than_2_raises(self):
        dates = pd.date_range("2020-01-01", periods=1, freq="D")
        s = pd.Series([100.0], index=dates)
        with pytest.raises(ValueError, match="al menos 2"):
            compute_log_returns(s)


# ====================================================================== #
# Tests: volatilidad                                                       #
# ====================================================================== #

class TestVolatilityHelpers:

    def test_compute_mean_known_value(self):
        arr = np.array([2.0, 4.0, 6.0])
        assert _compute_mean(arr) == pytest.approx(4.0, abs=1e-10)

    def test_compute_mean_single(self):
        arr = np.array([7.0])
        assert _compute_mean(arr) == pytest.approx(7.0, abs=1e-10)

    def test_compute_mean_empty_raises(self):
        with pytest.raises(ValueError, match="vacío"):
            _compute_mean(np.array([]))

    def test_compute_variance_known_value(self):
        """[2, 4, 6]: varianza muestral = ((2-4)²+(4-4)²+(6-4)²)/(3-1) = 4."""
        arr = np.array([2.0, 4.0, 6.0])
        mean = _compute_mean(arr)
        assert _compute_variance(arr, mean) == pytest.approx(4.0, abs=1e-10)

    def test_compute_variance_requires_2(self):
        arr = np.array([5.0])
        with pytest.raises(ValueError, match="al menos 2"):
            _compute_variance(arr, 5.0)

    def test_compute_variance_non_negative(self):
        arr = np.array([1.0, 3.0, 5.0, 7.0])
        mean = _compute_mean(arr)
        assert _compute_variance(arr, mean) >= 0.0


class TestComputeDailyVolatility:

    def test_constant_returns_zero_volatility(self):
        """Si todos los retornos son iguales, std = 0."""
        arr = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        assert compute_daily_volatility(arr) == pytest.approx(0.0, abs=1e-10)

    def test_non_negative(self, returns_series):
        arr = returns_series.values
        assert compute_daily_volatility(arr) >= 0.0

    def test_requires_at_least_2(self):
        with pytest.raises(ValueError, match="al menos 2"):
            compute_daily_volatility(np.array([0.01]))

    def test_known_value(self):
        """
        arr = [0.0, 2.0]: media=1, varianza=(1+1)/1=2, std=√2 ≈ 1.4142.
        """
        arr = np.array([0.0, 2.0])
        assert compute_daily_volatility(arr) == pytest.approx(math.sqrt(2), rel=1e-9)


class TestAnnualizedVolatility:

    def test_formula(self):
        """σ_a = σ_d × √252."""
        daily = 0.01
        annual = compute_annualized_volatility(daily, 252)
        assert annual == pytest.approx(daily * math.sqrt(252), rel=1e-9)

    def test_negative_daily_vol_raises(self):
        with pytest.raises(ValueError, match="negativa"):
            compute_annualized_volatility(-0.01)

    def test_non_positive_trading_days_raises(self):
        with pytest.raises(ValueError, match="positivo"):
            compute_annualized_volatility(0.01, trading_days=0)

    def test_zero_daily_gives_zero_annual(self):
        assert compute_annualized_volatility(0.0) == pytest.approx(0.0, abs=1e-12)


class TestComputeVolatility:

    def test_returns_tuple(self, returns_series):
        result = compute_volatility(returns_series)
        assert isinstance(result, tuple) and len(result) == 2

    def test_daily_leq_annual(self, returns_series):
        daily, annual = compute_volatility(returns_series)
        assert annual >= daily

    def test_relationship_daily_annual(self, returns_series):
        """annual ≈ daily × √252."""
        daily, annual = compute_volatility(returns_series)
        assert annual == pytest.approx(daily * math.sqrt(252), rel=1e-9)

    def test_drops_nan_before_computing(self):
        dates = pd.date_range("2020-01-01", periods=6, freq="D")
        r = pd.Series([0.01, float("nan"), 0.02, -0.01, 0.005, 0.003], index=dates)
        # No debe lanzar excepción
        daily, annual = compute_volatility(r)
        assert daily >= 0.0


# ====================================================================== #
# Tests: clasificación de riesgo                                           #
# ====================================================================== #

class TestRiskClassifier:

    def test_below_conservative_threshold(self):
        assert classify_risk(THRESHOLD_CONSERVATIVE - 0.01) == RISK_CONSERVATIVE

    def test_exactly_conservative_threshold(self):
        """σ = 0.20 → Moderado (≥ 20%)."""
        assert classify_risk(THRESHOLD_CONSERVATIVE) == RISK_MODERATE

    def test_between_thresholds_is_moderate(self):
        mid = (THRESHOLD_CONSERVATIVE + THRESHOLD_MODERATE) / 2
        assert classify_risk(mid) == RISK_MODERATE

    def test_exactly_moderate_threshold(self):
        """σ = 0.40 → Agresivo (≥ 40%)."""
        assert classify_risk(THRESHOLD_MODERATE) == RISK_AGGRESSIVE

    def test_above_moderate_threshold(self):
        assert classify_risk(THRESHOLD_MODERATE + 0.10) == RISK_AGGRESSIVE

    def test_zero_volatility_is_conservative(self):
        assert classify_risk(0.0) == RISK_CONSERVATIVE

    def test_negative_volatility_raises(self):
        with pytest.raises(ValueError, match="negativa"):
            classify_risk(-0.01)

    def test_typical_gold_is_conservative(self):
        """GLD suele tener σ_anual ~12-15%."""
        assert classify_risk(0.13) == RISK_CONSERVATIVE

    def test_typical_aapl_is_moderate(self):
        """AAPL suele tener σ_anual ~25-30%."""
        assert classify_risk(0.27) == RISK_MODERATE

    def test_typical_tsla_is_aggressive(self):
        """TSLA suele tener σ_anual ~50-70%."""
        assert classify_risk(0.55) == RISK_AGGRESSIVE

    def test_classify_with_score_keys(self):
        result = classify_risk_with_score(0.25)
        assert "category" in result
        assert "risk_score" in result
        assert "annualized_vol" in result
        assert "description" in result

    def test_risk_score_ordering(self):
        """Conservador < Moderado < Agresivo en puntuación ordinal."""
        s_cons = classify_risk_with_score(0.10)["risk_score"]
        s_mod  = classify_risk_with_score(0.25)["risk_score"]
        s_aggr = classify_risk_with_score(0.50)["risk_score"]
        assert s_cons < s_mod < s_aggr


# ====================================================================== #
# Tests: sliding window                                                    #
# ====================================================================== #

class TestSlidingWindow:

    def test_number_of_windows(self, price_series_up):
        """Para n=10 y w=3: debe generar 10-3+1 = 8 ventanas."""
        n, w = len(price_series_up), 3
        windows = get_all_windows(price_series_up, w)
        assert len(windows) == n - w + 1

    def test_window_size(self, price_series_up):
        for window in generate_windows(price_series_up, 4):
            assert len(window) == 4

    def test_window_preserves_index(self, price_series_up):
        first_window = next(iter(generate_windows(price_series_up, 3)))
        assert first_window.index[0] == price_series_up.index[0]

    def test_last_window_ends_at_last_date(self, price_series_up):
        all_w = get_all_windows(price_series_up, 3)
        assert all_w[-1].index[-1] == price_series_up.index[-1]

    def test_window_size_zero_raises(self, price_series_up):
        with pytest.raises(ValueError, match="positivo"):
            list(generate_windows(price_series_up, 0))

    def test_window_size_exceeds_series_raises(self, price_series_up):
        with pytest.raises(ValueError, match="longitud"):
            list(generate_windows(price_series_up, len(price_series_up) + 1))

    def test_window_size_equals_series(self, price_series_up):
        """window_size = n → exactamente 1 ventana."""
        windows = get_all_windows(price_series_up, len(price_series_up))
        assert len(windows) == 1

    def test_generator_is_lazy(self, price_series_up):
        """generate_windows debe ser un generador (lazy evaluation)."""
        import types
        gen = generate_windows(price_series_up, 3)
        assert isinstance(gen, types.GeneratorType)

    def test_window_statistics_shape(self, price_series_up):
        ws = 3
        stats = window_statistics(price_series_up, ws)
        assert len(stats) == len(price_series_up) - ws + 1
        assert "min" in stats.columns and "max" in stats.columns

    def test_window_statistics_first_gt_zero(self, price_series_up):
        stats = window_statistics(price_series_up, 3)
        assert (stats["min"] > 0).all()


# ====================================================================== #
# Tests: detección de patrones                                             #
# ====================================================================== #

class TestPatternConsecutiveGains:

    def test_all_up_series_detects_pattern(self, price_series_up):
        result = detect_consecutive_gains(price_series_up, min_days=3)
        assert result["n_occurrences"] > 0

    def test_down_series_no_patterns(self, price_series_down):
        result = detect_consecutive_gains(price_series_down, min_days=3)
        assert result["n_occurrences"] == 0

    def test_frequency_in_zero_one(self, price_series_up):
        result = detect_consecutive_gains(price_series_up, min_days=3)
        assert 0.0 <= result["frequency"] <= 1.0

    def test_total_windows_correct(self, price_series_up):
        min_days = 3
        result = detect_consecutive_gains(price_series_up, min_days)
        expected_windows = len(price_series_up) - min_days  # n - (w-1) - 1 + 1 = n - w
        assert result["total_windows"] == expected_windows

    def test_known_single_occurrence(self):
        """
        [100, 101, 102, 103] → exactamente 1 ventana de 4 precios.
        Con min_days=3: w=4. Total ventanas = 4-4+1 = 1. Debe detectar 1.
        """
        dates = pd.date_range("2020-01-01", periods=4, freq="D")
        s = pd.Series([100.0, 101.0, 102.0, 103.0], index=dates)
        result = detect_consecutive_gains(s, min_days=3)
        assert result["n_occurrences"] == 1

    def test_flat_prices_no_pattern(self):
        dates = pd.date_range("2020-01-01", periods=6, freq="D")
        s = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0, 100.0], index=dates)
        result = detect_consecutive_gains(s, min_days=3)
        assert result["n_occurrences"] == 0

    def test_result_keys_present(self, price_series_up):
        result = detect_consecutive_gains(price_series_up)
        for key in ("pattern", "description", "n_occurrences", "total_windows",
                    "frequency", "occurrences"):
            assert key in result

    def test_min_days_zero_raises(self, price_series_up):
        with pytest.raises(ValueError, match="≥ 1"):
            detect_consecutive_gains(price_series_up, min_days=0)

    def test_series_too_short_raises(self):
        dates = pd.date_range("2020-01-01", periods=2, freq="D")
        s = pd.Series([100.0, 101.0], index=dates)
        with pytest.raises(ValueError, match="al menos"):
            detect_consecutive_gains(s, min_days=3)

    def test_occurrence_gain_pct_positive(self, price_series_up):
        result = detect_consecutive_gains(price_series_up, min_days=3)
        for occ in result["occurrences"]:
            assert occ["total_gain_pct"] > 0


class TestPatternDropAndRecovery:

    def test_detects_known_event(self, price_series_drop_recovery):
        """
        100 → 95 → 99:
          drop     = (95-100)/100 = -5%  ≤ -3% ✓
          recovery = (99-95)/95   ≈ +4.2% ≥ +2% ✓
        """
        result = detect_drop_and_recovery(
            price_series_drop_recovery,
            drop_threshold=0.03,
            recovery_threshold=0.02,
        )
        assert result["n_occurrences"] >= 1

    def test_no_event_in_up_series(self, price_series_up):
        result = detect_drop_and_recovery(
            price_series_up,
            drop_threshold=0.03,
            recovery_threshold=0.02,
        )
        assert result["n_occurrences"] == 0

    def test_frequency_in_zero_one(self, price_series_drop_recovery):
        result = detect_drop_and_recovery(price_series_drop_recovery)
        assert 0.0 <= result["frequency"] <= 1.0

    def test_total_windows_equals_n_minus_2(self, price_series_drop_recovery):
        n = len(price_series_drop_recovery)
        result = detect_drop_and_recovery(price_series_drop_recovery)
        assert result["total_windows"] == n - 2

    def test_result_keys_present(self, price_series_drop_recovery):
        result = detect_drop_and_recovery(price_series_drop_recovery)
        for key in ("pattern", "description", "drop_threshold",
                    "recovery_threshold", "n_occurrences",
                    "total_windows", "frequency", "occurrences"):
            assert key in result

    def test_occurrence_drop_is_negative(self, price_series_drop_recovery):
        result = detect_drop_and_recovery(price_series_drop_recovery)
        for occ in result["occurrences"]:
            assert occ["drop_pct"] < 0

    def test_occurrence_recovery_is_positive(self, price_series_drop_recovery):
        result = detect_drop_and_recovery(price_series_drop_recovery)
        for occ in result["occurrences"]:
            assert occ["recovery_pct"] > 0

    def test_high_thresholds_no_detection(self, price_series_drop_recovery):
        """Umbrales imposiblemente altos → sin detección."""
        result = detect_drop_and_recovery(
            price_series_drop_recovery,
            drop_threshold=0.99,
            recovery_threshold=0.99,
        )
        assert result["n_occurrences"] == 0

    def test_zero_drop_threshold_raises(self, price_series_drop_recovery):
        with pytest.raises(ValueError, match="positivo"):
            detect_drop_and_recovery(price_series_drop_recovery, drop_threshold=0.0)

    def test_series_too_short_raises(self):
        dates = pd.date_range("2020-01-01", periods=2, freq="D")
        s = pd.Series([100.0, 95.0], index=dates)
        with pytest.raises(ValueError, match="al menos 3"):
            detect_drop_and_recovery(s)


# ====================================================================== #
# Tests: ranking de riesgo                                                 #
# ====================================================================== #

class TestRiskRanking:

    def test_build_ranking_length(self, volatility_records):
        df = build_risk_ranking(volatility_records)
        assert len(df) == len(volatility_records)

    def test_rank_starts_at_one(self, volatility_records):
        df = build_risk_ranking(volatility_records)
        assert df["rank"].iloc[0] == 1

    def test_rank_is_sequential(self, volatility_records):
        df = build_risk_ranking(volatility_records)
        ranks = df["rank"].tolist()
        assert ranks == list(range(1, len(volatility_records) + 1))

    def test_sorted_descending_by_volatility(self, volatility_records):
        df = build_risk_ranking(volatility_records)
        vol = df["annualized_vol_pct"].tolist()
        assert vol == sorted(vol, reverse=True)

    def test_first_is_most_volatile(self, volatility_records):
        """TSLA (47.6%) debe ser rank 1."""
        df = build_risk_ranking(volatility_records)
        assert df.iloc[0]["ticker"] == "TSLA"

    def test_last_is_least_volatile(self, volatility_records):
        """SPY (15.9%) debe ser el último."""
        df = build_risk_ranking(volatility_records)
        assert df.iloc[-1]["ticker"] == "SPY"

    def test_empty_records_raises(self):
        with pytest.raises(ValueError, match="vacía"):
            build_risk_ranking([])

    def test_output_columns_present(self, volatility_records):
        df = build_risk_ranking(volatility_records)
        expected = {"rank", "ticker", "instrument_type",
                    "annualized_vol_pct", "daily_vol_pct",
                    "risk_category", "n_observations"}
        assert expected.issubset(set(df.columns))


# ====================================================================== #
# Tests: AnalyticsService (integración con dataset maestro real)           #
# ====================================================================== #

class TestAnalyticsService:
    """
    Tests de integración que requieren el dataset maestro real.
    Se omiten automáticamente si master_dataset.csv no existe.
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_master(self):
        from src.config.settings import PROCESSED_DIR
        master = PROCESSED_DIR / "master_dataset.csv"
        if not master.exists():
            pytest.skip("master_dataset.csv no encontrado — ejecuta python main.py")

    @pytest.fixture
    def service(self):
        return AnalyticsService()

    def test_available_tickers_not_empty(self, service):
        tickers = service.available_tickers
        assert len(tickers) > 0

    def test_available_tickers_contains_nvda(self, service):
        assert "NVDA" in service.available_tickers

    def test_analyze_asset_returns_dict(self, service):
        result = service.analyze_asset("NVDA")
        assert isinstance(result, dict)

    def test_analyze_asset_required_keys(self, service):
        result = service.analyze_asset("AAPL")
        for key in ("ticker", "instrument_type", "n_observations",
                    "daily_volatility", "annualized_volatility",
                    "risk_category", "date_start", "date_end", "patterns"):
            assert key in result, f"Clave faltante: {key}"

    def test_daily_volatility_positive(self, service):
        result = service.analyze_asset("SPY")
        assert result["daily_volatility"] > 0.0

    def test_annual_volatility_greater_than_daily(self, service):
        result = service.analyze_asset("SPY")
        assert result["annualized_volatility"] > result["daily_volatility"]

    def test_risk_category_valid(self, service):
        from src.analytics.risk_classifier import (
            RISK_CONSERVATIVE, RISK_MODERATE, RISK_AGGRESSIVE
        )
        result = service.analyze_asset("TSLA")
        assert result["risk_category"] in (
            RISK_CONSERVATIVE, RISK_MODERATE, RISK_AGGRESSIVE
        )

    def test_patterns_present(self, service):
        result = service.analyze_asset("AAPL")
        assert "consecutive_gains" in result["patterns"]
        assert "drop_and_recovery" in result["patterns"]

    def test_unknown_ticker_raises(self, service):
        with pytest.raises(ValueError, match="no encontrado"):
            service.analyze_asset("XYZXYZ_INVALID")

    def test_generate_volatility_summary_shape(self, service):
        df = service.generate_volatility_summary()
        assert len(df) == len(service.available_tickers)
        assert "annualized_vol_pct" in df.columns

    def test_generate_risk_ranking_sorted(self, service):
        df = service.generate_risk_ranking()
        vol = df["annualized_vol_pct"].tolist()
        assert vol == sorted(vol, reverse=True)

    def test_generate_pattern_analysis_rows(self, service):
        df = service.generate_pattern_analysis()
        # 20 activos × 2 patrones = 40 filas
        assert len(df) == len(service.available_tickers) * 2

    def test_run_full_analysis_creates_files(self, service, tmp_path):
        """Verifica que run_full_analysis() genera los tres CSVs."""
        from src.analytics.analytics_service import (
            _VOLATILITY_SUMMARY_PATH,
            _RISK_RANKING_PATH,
            _PATTERN_ANALYSIS_PATH,
        )
        paths = service.run_full_analysis(print_summary=False)
        for p in paths.values():
            assert p.exists(), f"Archivo no generado: {p}"
            content = pd.read_csv(p)
            assert not content.empty, f"Archivo vacío: {p}"
