"""lipipy.numerics.special: parity with SciPy plus overflow and edge cases."""

import numpy as np
import pytest

from lipipy.bench import estimate_complexity
from lipipy.numerics import (
    erf,
    erfc,
    log_sigmoid,
    log_softmax,
    logsumexp,
    norm_cdf,
    norm_ppf,
    sigmoid,
    softmax,
    softplus,
)


def max_rel_err(got, ref):
    got, ref = np.asarray(got), np.asarray(ref)
    mask = ref != 0
    assert np.all(got[~mask] == 0)
    return float(np.max(np.abs(got[mask] - ref[mask]) / np.abs(ref[mask])))


# -- log-sum-exp / softmax -------------------------------------------------------
@pytest.mark.parametrize("axis", [None, 0, 1, -1, (0, 1)])
@pytest.mark.parametrize("keepdims", [False, True])
def test_logsumexp_matches_scipy(rng, axis, keepdims):
    sp = pytest.importorskip("scipy.special")
    a = rng.normal(scale=300, size=(4, 5))
    np.testing.assert_allclose(
        logsumexp(a, axis=axis, keepdims=keepdims),
        sp.logsumexp(a, axis=axis, keepdims=keepdims),
        rtol=1e-14,
    )


@pytest.mark.filterwarnings("ignore::RuntimeWarning:scipy")
def test_logsumexp_infinities_like_scipy():
    sp = pytest.importorskip("scipy.special")
    a = np.array([[0.0, -np.inf, 1.0], [-np.inf, -np.inf, -np.inf], [np.inf, 0.0, -np.inf]])
    np.testing.assert_array_equal(logsumexp(a, axis=1), sp.logsumexp(a, axis=1))
    assert logsumexp([-np.inf, -np.inf]) == -np.inf
    assert np.isnan(logsumexp([np.nan, 1.0]))


def test_logsumexp_no_overflow():
    assert logsumexp([1000.0, 1000.0]) == pytest.approx(1000.0 + np.log(2.0), rel=1e-15)
    assert logsumexp([-1000.0, -1000.0]) == pytest.approx(-1000.0 + np.log(2.0), rel=1e-15)
    assert np.ndim(logsumexp([1.0, 2.0])) == 0


@pytest.mark.parametrize("axis", [0, 1, -1, None])
def test_softmax_and_log_softmax_match_scipy(rng, axis):
    sp = pytest.importorskip("scipy.special")
    a = rng.normal(scale=50, size=(3, 6))
    np.testing.assert_allclose(softmax(a, axis=axis), sp.softmax(a, axis=axis), rtol=1e-13)
    np.testing.assert_allclose(
        log_softmax(a, axis=axis), sp.log_softmax(a, axis=axis), rtol=1e-13, atol=1e-13
    )


@pytest.mark.filterwarnings("ignore::RuntimeWarning:scipy")
def test_softmax_edge_cases():
    sp = pytest.importorskip("scipy.special")
    a = np.array([[1000.0, 1000.0, -np.inf], [-np.inf, -np.inf, -np.inf]])
    np.testing.assert_array_equal(softmax(a), sp.softmax(a, axis=-1))
    np.testing.assert_array_equal(log_softmax(a), sp.log_softmax(a, axis=-1))
    np.testing.assert_allclose(softmax(np.arange(5.0)).sum(), 1.0, rtol=1e-15)
    assert log_softmax([0.0, -800.0])[1] == -800.0  # log(softmax) would be -inf


@pytest.mark.parametrize("func", [logsumexp, softmax, log_softmax])
def test_reductions_reject_bad_input(func):
    with pytest.raises(ValueError, match="empty"):
        func([])
    with pytest.raises(TypeError):
        func("abc")
    with pytest.raises(TypeError, match="real"):
        func(np.array([1 + 1j]))
    with pytest.raises(ValueError, match="out of bounds"):
        func(np.ones((2, 2)), axis=2)
    with pytest.raises(TypeError, match="axis"):
        func(np.ones(2), axis=0.5)


# -- sigmoid family -------------------------------------------------------------
def test_sigmoid_family_matches_scipy():
    sp = pytest.importorskip("scipy.special")
    x = np.concatenate([np.linspace(-800, 800, 4001), [-1e300, 1e300, 0.0]])
    # SciPy flushes sigmoid(x < -745) to 0; ours keeps the subnormal value, hence atol.
    np.testing.assert_allclose(sigmoid(x), sp.expit(x), rtol=1e-14, atol=1e-307)
    np.testing.assert_allclose(log_sigmoid(x), sp.log_expit(x), rtol=1e-14, atol=0)
    np.testing.assert_allclose(softplus(x), np.logaddexp(0.0, x), rtol=1e-14, atol=0)


def test_sigmoid_family_no_warnings_on_extremes():
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        assert sigmoid(-1000.0) == 0.0
        assert sigmoid(1000.0) == 1.0
        assert softplus(1000.0) == 1000.0
        assert log_sigmoid(-1000.0) == -1000.0
    assert sigmoid(np.inf) == 1.0 and sigmoid(-np.inf) == 0.0
    assert np.isnan(sigmoid(np.nan))
    assert softplus(0.0) == pytest.approx(np.log(2.0), rel=1e-15)


# -- erf / erfc -----------------------------------------------------------------
def test_erf_relative_error_on_dense_grid():
    sp = pytest.importorskip("scipy.special")
    x = np.linspace(-6.0, 6.0, 240_001)
    x = np.concatenate([x, [1e-300, -1e-300, 1e-20, 0.46875, -0.46875, 4.0, -4.0]])
    assert max_rel_err(erf(x), sp.erf(x)) <= 1e-13
    assert max_rel_err(erfc(x), sp.erfc(x)) <= 1e-13


def test_erfc_relative_error_in_right_tail():
    sp = pytest.importorskip("scipy.special")
    x = np.linspace(-6.0, 26.0, 320_001)
    assert max_rel_err(erfc(x), sp.erfc(x)) <= 1e-12
    assert erfc(27.0) == 0.0 and erfc(1e300) == 0.0


def test_erf_special_values():
    np.testing.assert_array_equal(erf([-np.inf, 0.0, np.inf]), [-1.0, 0.0, 1.0])
    np.testing.assert_array_equal(erfc([-np.inf, 0.0, np.inf]), [2.0, 1.0, 0.0])
    assert np.isnan(erf(np.nan)) and np.isnan(erfc(np.nan))
    x = np.linspace(-5, 5, 101)
    np.testing.assert_allclose(erf(-x), -erf(x), rtol=0, atol=0)
    np.testing.assert_allclose(erf(x) + erfc(x), 1.0, rtol=1e-15, atol=2e-16)
    assert erf(np.ones((2, 3))).shape == (2, 3)
    with pytest.raises(TypeError):
        erf("x")


# -- normal distribution --------------------------------------------------------
def test_norm_cdf_matches_scipy_in_both_tails():
    sp = pytest.importorskip("scipy.special")
    x = np.linspace(-37.5, 9.0, 100_001)
    assert max_rel_err(norm_cdf(x), sp.ndtr(x)) <= 1e-12
    assert norm_cdf(0.0) == 0.5


def test_norm_ppf_matches_ndtri():
    sp = pytest.importorskip("scipy.special")
    p = np.concatenate(
        [
            np.logspace(-300, -1, 30_000),
            np.linspace(0.0, 1.0, 100_001)[1:-1],
            1.0 - np.logspace(-16, -1, 30_000),
            0.5 + np.logspace(-17, -2, 2_000),
            0.5 - np.logspace(-17, -2, 2_000),
            [0.02425, 1 - 0.02425, 0.5],
        ]
    )
    assert max_rel_err(norm_ppf(p), sp.ndtri(p)) <= 1e-12


@pytest.mark.filterwarnings("error")
def test_norm_ppf_subnormal_p():
    sp = pytest.importorskip("scipy.special")
    tiny = np.finfo(np.float64).tiny
    p = np.concatenate([np.geomspace(5e-324, 2.2e-308, 2_000), [1e-320, 1e-310, tiny]])
    p = np.unique(p)  # subnormal rounding makes some grid points coincide
    got = norm_ppf(p)
    assert max_rel_err(got, sp.ndtri(p)) <= 2e-9  # bound stated in the docstring
    # The symmetric upper tail cannot be subnormal (1 - p >= 2^-53), check it anyway.
    q = 1.0 - np.array([2.0**-53, 1e-16, 1e-15])
    assert max_rel_err(norm_ppf(q), sp.ndtri(q)) <= 1e-12
    assert np.all(np.diff(got) > 0)  # still monotone across the subnormal boundary


def test_norm_ppf_special_values_and_roundtrip():
    np.testing.assert_array_equal(norm_ppf([0.0, 1.0]), [-np.inf, np.inf])
    assert np.all(np.isnan(norm_ppf([-0.1, 1.1, np.nan, -np.inf])))
    assert norm_ppf(0.5) == 0.0
    # Lower tail only: norm_cdf(x) rounds towards 1 for large positive x.
    x = np.linspace(-37.0, 0.0, 371)
    np.testing.assert_allclose(norm_ppf(norm_cdf(x)), x, rtol=1e-12, atol=1e-15)
    p = np.linspace(0.001, 0.999, 999)
    np.testing.assert_allclose(norm_ppf(p), -norm_ppf(1.0 - p), rtol=1e-12)


# -- memory --------------------------------------------------------------------
def test_logsumexp_memory_is_linear_in_n():
    """Declared space O(n): one shifted temporary the size of the input."""
    est = estimate_complexity(
        lambda a: logsumexp(a),
        lambda n, rng: rng.normal(size=n),
        sizes=(50_000, 100_000, 200_000, 400_000),
        repeats=1,
    )
    assert est.space_exponent == pytest.approx(1.0, abs=0.15)
    assert max(est.peak_bytes) <= 4 * 8 * 400_000  # a few float64 temporaries at most


@pytest.mark.complexity
def test_logsumexp_time_is_linear_in_n():
    est = estimate_complexity(
        lambda a: logsumexp(a),
        lambda n, rng: rng.normal(size=n),
        sizes=(200_000, 400_000, 800_000, 1_600_000),
    )
    assert est.time_exponent < 1.35
