"""
tests/test_similarity.py
=========================
Pruebas unitarias para los cuatro algoritmos de similitud del Req. 2.

Cobertura:
  - EuclideanDistance: casos numéricos, normalización, similitud, validaciones.
  - PearsonCorrelation: correlaciones perfectas, nulas, negativas, componentes.
  - CosineSimilarity: vectores paralelos, ortogonales, antiparalelos.
  - DynamicTimeWarping: casos base, series iguales, desfasadas.
  - SimilarityUtils: compute_returns, align_series, remove_nan_pairs, z_normalize.

Uso:
    pytest tests/test_similarity.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math

import numpy as np
import pytest

# ── Imports de algoritmos ─────────────────────────────────────────────
from src.similarity.euclidean_distance import (
    euclidean_distance,
    euclidean_distance_normalized,
    euclidean_similarity,
)
from src.similarity.pearson_correlation import (
    pearson_correlation,
    pearson_with_components,
    interpret_pearson,
)
from src.similarity.cosine_similarity import (
    cosine_similarity,
    cosine_distance,
    cosine_with_components,
    interpret_cosine,
)
from src.similarity.dynamic_time_warping import (
    dtw_distance,
    dtw_with_path,
    dtw_matrix_only,
)
from src.similarity.utils import (
    compute_returns,
    align_series,
    remove_nan_pairs,
    z_normalize,
)

import pandas as pd


# ====================================================================== #
# Fixtures                                                                #
# ====================================================================== #

@pytest.fixture
def identical_series():
    """Dos series idénticas: distancia 0, correlación 1, coseno 1."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    return x, x.copy()


@pytest.fixture
def perfectly_correlated():
    """y = 2x + 1: correlación Pearson = 1, pero euclidiana > 0."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 2.0 * x + 1.0
    return x, y


@pytest.fixture
def negatively_correlated():
    """y = -x: correlación Pearson = -1."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = -x
    return x, y


@pytest.fixture
def orthogonal_vectors():
    """Vectores ortogonales: coseno = 0."""
    x = np.array([1.0, 0.0, 0.0])
    y = np.array([0.0, 1.0, 0.0])
    return x, y


@pytest.fixture
def random_series():
    """Series aleatorias reproducibles para tests de integración."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100)
    y = rng.standard_normal(100)
    return x, y


@pytest.fixture
def prices_series():
    """Serie de precios para test de compute_returns."""
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    return pd.Series([100.0, 101.0, 99.0, 102.0, 100.0], index=dates, name="T")


# ====================================================================== #
# Tests: Distancia Euclidiana                                              #
# ====================================================================== #

class TestEuclideanDistance:

    def test_identical_series_distance_zero(self, identical_series):
        x, y = identical_series
        assert euclidean_distance(x, y) == pytest.approx(0.0, abs=1e-10)

    def test_known_distance(self):
        """d([0,0,0], [3,4,0]) = 5 (triángulo 3-4-5)."""
        x = np.array([0.0, 0.0, 0.0])
        y = np.array([3.0, 4.0, 0.0])
        assert euclidean_distance(x, y) == pytest.approx(5.0, rel=1e-9)

    def test_single_element(self):
        x = np.array([3.0])
        y = np.array([3.0])
        assert euclidean_distance(x, y) == pytest.approx(0.0, abs=1e-10)

    def test_non_negative(self, random_series):
        x, y = random_series
        assert euclidean_distance(x, y) >= 0.0

    def test_symmetry(self, random_series):
        """d(x,y) == d(y,x)."""
        x, y = random_series
        assert euclidean_distance(x, y) == pytest.approx(
            euclidean_distance(y, x), rel=1e-12
        )

    def test_different_length_raises(self):
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="misma longitud"):
            euclidean_distance(x, y)

    def test_empty_array_raises(self):
        with pytest.raises(ValueError, match="vacíos"):
            euclidean_distance(np.array([]), np.array([]))

    def test_nan_raises(self):
        x = np.array([1.0, np.nan, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="NaN"):
            euclidean_distance(x, y)

    def test_normalized_less_than_raw(self):
        """La distancia normalizada debe ser ≤ la distancia bruta para n > 1."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        raw = euclidean_distance(x, y)
        norm = euclidean_distance_normalized(x, y)
        assert norm <= raw

    def test_similarity_in_zero_one(self, random_series):
        x, y = random_series
        sim = euclidean_similarity(x, y)
        assert 0.0 < sim <= 1.0

    def test_similarity_one_for_identical(self, identical_series):
        x, y = identical_series
        assert euclidean_similarity(x, y) == pytest.approx(1.0, rel=1e-9)


# ====================================================================== #
# Tests: Correlación de Pearson                                            #
# ====================================================================== #

class TestPearsonCorrelation:

    def test_perfect_positive_correlation(self, perfectly_correlated):
        """y = 2x + 1 → Pearson = 1."""
        x, y = perfectly_correlated
        r = pearson_correlation(x, y)
        assert r == pytest.approx(1.0, abs=1e-9)

    def test_perfect_negative_correlation(self, negatively_correlated):
        """y = -x → Pearson = -1."""
        x, y = negatively_correlated
        r = pearson_correlation(x, y)
        assert r == pytest.approx(-1.0, abs=1e-9)

    def test_range_minus_one_to_one(self, random_series):
        x, y = random_series
        r = pearson_correlation(x, y)
        assert -1.0 <= r <= 1.0

    def test_symmetry(self, random_series):
        """Pearson es simétrico: r(x,y) == r(y,x)."""
        x, y = random_series
        assert pearson_correlation(x, y) == pytest.approx(
            pearson_correlation(y, x), rel=1e-12
        )

    def test_identical_series_returns_one(self, identical_series):
        x, y = identical_series
        r = pearson_correlation(x, y)
        assert r == pytest.approx(1.0, abs=1e-9)

    def test_constant_series_returns_zero(self):
        """Serie constante → sin varianza → Pearson = 0."""
        x = np.array([5.0, 5.0, 5.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        r = pearson_correlation(x, y)
        assert r == pytest.approx(0.0, abs=1e-9)

    def test_known_value(self):
        """Caso calculado a mano: x=[1,2,3], y=[1,2,3] → r=1."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        assert pearson_correlation(x, y) == pytest.approx(1.0, abs=1e-9)

    def test_different_length_raises(self):
        with pytest.raises(ValueError, match="[Ll]ongitudes"):
            pearson_correlation(np.array([1.0]), np.array([1.0, 2.0]))

    def test_nan_raises(self):
        x = np.array([1.0, np.nan])
        y = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="NaN"):
            pearson_correlation(x, y)

    def test_components_mean_x_correct(self):
        x = np.array([2.0, 4.0, 6.0])
        y = np.array([1.0, 2.0, 3.0])
        _, comp = pearson_with_components(x, y)
        assert comp["mean_x"] == pytest.approx(4.0, abs=1e-9)
        assert comp["mean_y"] == pytest.approx(2.0, abs=1e-9)

    def test_interpret_pearson_high(self):
        text = interpret_pearson(0.95)
        assert "alta" in text.lower() or "muy" in text.lower()

    def test_interpret_pearson_negative(self):
        text = interpret_pearson(-0.8)
        assert "negativa" in text.lower()


# ====================================================================== #
# Tests: Similitud por Coseno                                              #
# ====================================================================== #

class TestCosineSimilarity:

    def test_identical_vectors_similarity_one(self, identical_series):
        x, y = identical_series
        assert cosine_similarity(x, y) == pytest.approx(1.0, abs=1e-9)

    def test_orthogonal_vectors_similarity_zero(self, orthogonal_vectors):
        x, y = orthogonal_vectors
        assert cosine_similarity(x, y) == pytest.approx(0.0, abs=1e-9)

    def test_antiparallel_vectors_similarity_minus_one(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([-1.0, -2.0, -3.0])
        assert cosine_similarity(x, y) == pytest.approx(-1.0, abs=1e-9)

    def test_range(self, random_series):
        x, y = random_series
        sim = cosine_similarity(x, y)
        assert -1.0 <= sim <= 1.0

    def test_symmetry(self, random_series):
        x, y = random_series
        assert cosine_similarity(x, y) == pytest.approx(
            cosine_similarity(y, x), rel=1e-12
        )

    def test_scale_invariant(self):
        """
        cos([1,2,3], [2,4,6]) == cos([1,2,3], [1,2,3]) == 1.
        El coseno no depende de la magnitud, solo de la dirección.
        """
        x = np.array([1.0, 2.0, 3.0])
        y_scaled = np.array([2.0, 4.0, 6.0])
        assert cosine_similarity(x, y_scaled) == pytest.approx(1.0, abs=1e-9)

    def test_zero_vector_returns_zero(self):
        x = np.array([0.0, 0.0, 0.0])
        y = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(x, y) == pytest.approx(0.0, abs=1e-9)

    def test_distance_from_similarity(self):
        """cosine_distance = 1 - cosine_similarity."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        sim = cosine_similarity(x, y)
        dist = cosine_distance(x, y)
        assert dist == pytest.approx(1.0 - sim, abs=1e-12)

    def test_different_length_raises(self):
        with pytest.raises(ValueError):
            cosine_similarity(np.array([1.0, 2.0]), np.array([1.0]))

    def test_nan_raises(self):
        x = np.array([1.0, np.nan])
        y = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="NaN"):
            cosine_similarity(x, y)

    def test_components_dot_product(self):
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        _, comp = cosine_with_components(x, y)
        # dot = 1*3 + 2*4 = 11
        assert comp["dot_product"] == pytest.approx(11.0, abs=1e-9)

    def test_interpret_cosine_high(self):
        text = interpret_cosine(0.95)
        assert "alta" in text.lower() or "muy" in text.lower()


# ====================================================================== #
# Tests: Dynamic Time Warping                                              #
# ====================================================================== #

class TestDynamicTimeWarping:

    def test_identical_series_distance_zero(self, identical_series):
        x, y = identical_series
        d = dtw_distance(x, y, normalize=False)
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_non_negative(self, random_series):
        x, y = random_series
        assert dtw_distance(x, y) >= 0.0

    def test_symmetry(self, random_series):
        """DTW(x, y) == DTW(y, x)."""
        x, y = random_series
        d_xy = dtw_distance(x, y, normalize=False)
        d_yx = dtw_distance(y, x, normalize=False)
        assert d_xy == pytest.approx(d_yx, rel=1e-10)

    def test_different_length_series(self):
        """DTW puede manejar series de distinto largo."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.0, 3.0, 5.0])
        d = dtw_distance(x, y, normalize=False)
        assert d >= 0.0

    def test_shifted_series(self):
        """
        Una serie desplazada en el tiempo debe tener DTW < euclidiana
        porque DTW puede compensar el desfase.
        """
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 0.0, 1.0, 2.0, 3.0])  # x desfasado un período

        from src.similarity.euclidean_distance import euclidean_distance
        d_eucl = euclidean_distance(x, y)
        d_dtw  = dtw_distance(x, y, normalize=False)

        # DTW es más compacto porque puede "alinear" el desfase
        assert d_dtw <= d_eucl

    def test_known_dtw_value(self):
        """
        x = [1, 2, 3], y = [1, 2, 3]
        DTW bruta = 0 (alineación perfecta).
        """
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        d = dtw_distance(x, y, normalize=False)
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_normalized_leq_raw(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        raw  = dtw_distance(x, y, normalize=False)
        norm = dtw_distance(x, y, normalize=True)
        assert norm <= raw

    def test_empty_array_raises(self):
        with pytest.raises(ValueError, match="vacíos"):
            dtw_distance(np.array([]), np.array([1.0, 2.0]))

    def test_path_starts_at_origin(self, identical_series):
        x, y = identical_series
        _, path = dtw_with_path(x, y)
        assert path[0] == (0, 0)

    def test_path_ends_at_last_index(self, identical_series):
        x, y = identical_series
        n = len(x)
        _, path = dtw_with_path(x, y)
        assert path[-1] == (n - 1, n - 1)

    def test_path_monotonic(self, random_series):
        """El camino DTW debe ser monótonamente no decreciente."""
        x, y = random_series[:10], random_series[1][:10]  # subconjunto
        x_short = random_series[0][:10]
        y_short = random_series[1][:10]
        _, path = dtw_with_path(x_short, y_short)
        for k in range(1, len(path)):
            assert path[k][0] >= path[k-1][0]
            assert path[k][1] >= path[k-1][1]

    def test_matrix_shape(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        D = dtw_matrix_only(x, y)
        assert D.shape == (len(x) + 1, len(y) + 1)

    def test_matrix_origin_zero(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        D = dtw_matrix_only(x, y)
        assert D[0][0] == pytest.approx(0.0, abs=1e-10)

    def test_nan_values_cleaned(self):
        """DTW debe limpiar NaN de forma independiente para cada serie."""
        x = np.array([1.0, np.nan, 3.0, 4.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        # No debe lanzar excepción
        d = dtw_distance(x, y)
        assert d >= 0.0


# ====================================================================== #
# Tests: Utilidades de similitud                                           #
# ====================================================================== #

class TestSimilarityUtils:

    # ── compute_returns ────────────────────────────────────────────────

    def test_compute_returns_length(self, prices_series):
        returns = compute_returns(prices_series)
        # Con drop_first_nan=True: longitud = n - 1
        assert len(returns) == len(prices_series) - 1

    def test_compute_returns_known_value(self):
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        prices = pd.Series([100.0, 110.0, 99.0], index=dates)
        returns = compute_returns(prices)
        # r[1] = (110 - 100) / 100 = 0.1
        assert returns.iloc[0] == pytest.approx(0.10, rel=1e-9)
        # r[2] = (99 - 110) / 110 ≈ -0.1
        assert returns.iloc[1] == pytest.approx(-11.0 / 110.0, rel=1e-9)

    def test_compute_returns_not_enough_prices(self):
        dates = pd.date_range("2020-01-01", periods=1, freq="D")
        with pytest.raises(ValueError, match="al menos 2"):
            compute_returns(pd.Series([100.0], index=dates))

    def test_compute_returns_zero_price_produces_nan(self):
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        prices = pd.Series([0.0, 100.0, 110.0], index=dates)
        returns = compute_returns(prices)
        # r[1] = (100 - 0) / 0 → NaN
        assert np.isnan(returns.iloc[0])

    # ── align_series ───────────────────────────────────────────────────

    def test_align_series_same_index(self):
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=dates)
        b = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=dates)
        a_al, b_al = align_series(a, b)
        assert len(a_al) == 5
        assert len(b_al) == 5

    def test_align_series_partial_overlap(self):
        dates_a = pd.date_range("2020-01-01", periods=5, freq="D")
        dates_b = pd.date_range("2020-01-03", periods=5, freq="D")
        a = pd.Series(range(5), index=dates_a, dtype=float)
        b = pd.Series(range(5), index=dates_b, dtype=float)
        a_al, b_al = align_series(a, b)
        # Solo se conservan las fechas comunes
        assert len(a_al) == len(b_al)
        assert len(a_al) < 5

    def test_align_series_no_overlap_raises(self):
        dates_a = pd.date_range("2020-01-01", periods=3, freq="D")
        dates_b = pd.date_range("2021-01-01", periods=3, freq="D")
        a = pd.Series([1.0, 2.0, 3.0], index=dates_a)
        b = pd.Series([1.0, 2.0, 3.0], index=dates_b)
        with pytest.raises(ValueError, match="[Nn]o hay"):
            align_series(a, b)

    # ── remove_nan_pairs ───────────────────────────────────────────────

    def test_remove_nan_pairs_basic(self):
        x = np.array([1.0, np.nan, 3.0])
        y = np.array([4.0, 5.0, np.nan])
        xc, yc = remove_nan_pairs(x, y)
        assert len(xc) == 1
        assert xc[0] == pytest.approx(1.0)

    def test_remove_nan_pairs_no_nan(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        xc, yc = remove_nan_pairs(x, y)
        assert len(xc) == 3

    def test_remove_nan_pairs_all_nan_raises(self):
        x = np.array([np.nan, np.nan])
        y = np.array([np.nan, np.nan])
        with pytest.raises(ValueError, match="[Nn]o quedan"):
            remove_nan_pairs(x, y)

    # ── z_normalize ────────────────────────────────────────────────────

    def test_z_normalize_zero_mean(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        z = z_normalize(x)
        assert np.mean(z) == pytest.approx(0.0, abs=1e-10)

    def test_z_normalize_unit_std(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        z = z_normalize(x)
        # std population ≈ 1.0 después de z-score
        std = np.sqrt(np.mean((z - np.mean(z)) ** 2))
        assert std == pytest.approx(1.0, abs=1e-10)

    def test_z_normalize_constant_returns_zeros(self):
        x = np.array([5.0, 5.0, 5.0, 5.0])
        z = z_normalize(x)
        np.testing.assert_array_almost_equal(z, np.zeros(4))
