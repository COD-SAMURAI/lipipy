"""Tests for lipipy.finance.risk: VaR / ES conventions, parity with SciPy and scikit-learn."""

import numpy as np
import pytest

from lipipy.finance import (
    ewma_covariance,
    expected_shortfall,
    ledoit_wolf,
    sample_covariance,
    value_at_risk,
)
from lipipy.finance.risk import _cornish_fisher_tail_mean, _cornish_fisher_z

pytestmark = pytest.mark.filterwarnings("error")

SEEDS = range(10)
LEVELS = (0.9, 0.95, 0.975, 0.99)


def _returns(seed, T=500, A=3):
    rng = np.random.default_rng(seed)
    return rng.standard_t(5, size=(T, A)) * 0.01 + 0.0003


# ---------------------------------------------------------------------------
# historical
# ---------------------------------------------------------------------------
def test_historical_lower_empirical_quantile_hand_computed():
    r = np.arange(1, 101) / 1000.0 - 0.05  # -0.049, ..., 0.05 sorted
    # T p = 100 * 0.05 = 5 (up to rounding): the 5th smallest, -0.045, not the 6th.
    assert value_at_risk(r, 0.95) == pytest.approx(0.045, abs=1e-15)
    assert expected_shortfall(r, 0.95) == pytest.approx(0.047, abs=1e-15)  # mean of 5 worst
    # T p = 2.5 -> k = 3: VaR = -r_(3); ES = -(r_(1) + r_(2) + 0.5 r_(3)) / 2.5
    assert value_at_risk(r, 0.975) == pytest.approx(0.047, abs=1e-15)
    assert expected_shortfall(r, 0.975) == pytest.approx(
        (0.049 + 0.048 + 0.5 * 0.047) / 2.5, abs=1e-15
    )
    # T p < 1: the worst observation
    assert value_at_risk(r, 0.999) == pytest.approx(0.049, abs=1e-16)
    assert expected_shortfall(r, 0.999) == pytest.approx(0.049, abs=1e-16)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_historical_matches_numpy_inverted_cdf(seed, level):
    R = _returns(seed, T=403)  # T p never an integer here, so no rounding ambiguity
    np.testing.assert_allclose(
        value_at_risk(R, level),
        -np.quantile(R, 1 - level, axis=0, method="inverted_cdf"),
        rtol=0,
        atol=0,
    )
    es = expected_shortfall(R, level)
    var = value_at_risk(R, level)
    assert np.all(es >= var)
    # Acerbi-Tasche: (1/Tp) * integral of the empirical quantile function over (0, p)
    T, p = R.shape[0], 1 - level
    srt = np.sort(R, axis=0)
    u = (np.arange(T) + 1) / T
    mass = np.clip(np.minimum(u, p) - np.concatenate([[0.0], u[:-1]]), 0.0, None)
    np.testing.assert_allclose(es, -(mass @ srt) / p, rtol=1e-12)


# ---------------------------------------------------------------------------
# gaussian
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_gaussian_matches_scipy(seed, level):
    stats = pytest.importorskip("scipy.stats")
    R = _returns(seed)
    mu, sd = R.mean(axis=0), R.std(axis=0, ddof=1)
    p = 1 - level
    expected_var = -stats.norm.ppf(p, loc=mu, scale=sd)
    np.testing.assert_allclose(value_at_risk(R, level, "gaussian"), expected_var, rtol=1e-12)
    expected_es = np.array(
        [
            -stats.norm.expect(
                lambda x: x, loc=m, scale=s, ub=stats.norm.ppf(p, m, s), conditional=True
            )
            for m, s in zip(mu, sd, strict=True)
        ]
    )
    np.testing.assert_allclose(expected_shortfall(R, level, "gaussian"), expected_es, rtol=1e-8)
    closed = -mu + sd * stats.norm.pdf(stats.norm.ppf(p)) / p
    np.testing.assert_allclose(expected_shortfall(R, level, "gaussian"), closed, rtol=1e-12)


# ---------------------------------------------------------------------------
# Cornish-Fisher
# ---------------------------------------------------------------------------
def test_cornish_fisher_reduces_to_gaussian():
    for z in (-2.326, -1.645, 0.3):
        assert _cornish_fisher_z(z, 0.0, 0.0) == z
    stats = pytest.importorskip("scipy.stats")
    for p in (0.01, 0.05, 0.1):
        c = stats.norm.ppf(p)
        assert _cornish_fisher_tail_mean(c, 0.0, 0.0) == pytest.approx(
            -stats.norm.pdf(c) / p, rel=1e-13
        )


@pytest.mark.parametrize("skew,kurt", [(-0.8, 2.0), (0.5, 1.0), (-0.3, 4.0), (0.0, 0.5)])
@pytest.mark.parametrize("p", [0.01, 0.025, 0.05, 0.1])
def test_cornish_fisher_tail_mean_matches_quadrature(skew, kurt, p):
    stats = pytest.importorskip("scipy.stats")
    integrate = pytest.importorskip("scipy.integrate")
    c = stats.norm.ppf(p)
    num, _ = integrate.quad(
        lambda z: _cornish_fisher_z(z, skew, kurt) * stats.norm.pdf(z), -np.inf, c, epsabs=1e-14
    )
    assert _cornish_fisher_tail_mean(c, skew, kurt) == pytest.approx(num / p, rel=1e-9)


@pytest.mark.parametrize("seed", SEEDS)
def test_cornish_fisher_uses_scipy_moments(seed):
    stats = pytest.importorskip("scipy.stats")
    R = _returns(seed)
    level = 0.99
    z = stats.norm.ppf(1 - level)
    S, K = stats.skew(R, axis=0), stats.kurtosis(R, axis=0)
    g = z + (z**2 - 1) * S / 6 + (z**3 - 3 * z) * K / 24 - (2 * z**3 - 5 * z) * S**2 / 36
    expected = -(R.mean(axis=0) + R.std(axis=0, ddof=1) * g)
    np.testing.assert_allclose(value_at_risk(R, level, "cornish_fisher"), expected, rtol=1e-11)
    assert np.all(
        expected_shortfall(R, level, "cornish_fisher") >= value_at_risk(R, level, "cornish_fisher")
    )


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------
def test_zero_variance_returns():
    r = np.full(50, 0.002)
    for method in ("historical", "gaussian", "cornish_fisher"):
        assert value_at_risk(r, 0.95, method) == pytest.approx(-0.002, abs=1e-16)
        assert expected_shortfall(r, 0.95, method) == pytest.approx(-0.002, abs=1e-16)


def test_single_observation_and_shapes():
    assert value_at_risk([-0.03], 0.95) == pytest.approx(0.03)
    with pytest.raises(ValueError, match="at least 2"):
        value_at_risk([-0.03], 0.95, "gaussian")
    R = _returns(0)
    assert value_at_risk(R).shape == (3,)
    assert isinstance(value_at_risk(R[:, 0]), float)


def test_risk_validation():
    with pytest.raises(ValueError, match="level"):
        value_at_risk([0.01, 0.02], 1.0)
    with pytest.raises(ValueError, match="level"):
        expected_shortfall([0.01, 0.02], 0.0)
    with pytest.raises(ValueError, match="method"):
        value_at_risk([0.01, 0.02], 0.95, "monte_carlo")
    with pytest.raises(ValueError, match="NaN"):
        value_at_risk([0.01, np.nan])
    with pytest.raises(TypeError):
        value_at_risk([0.01, 0.02], "0.95")


# ---------------------------------------------------------------------------
# covariance estimators
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_sample_covariance_matches_numpy(seed):
    R = _returns(seed, T=60, A=5)
    np.testing.assert_allclose(sample_covariance(R), np.cov(R, rowvar=False), rtol=1e-12)
    np.testing.assert_allclose(sample_covariance(R, 0), np.cov(R, rowvar=False, ddof=0), 1e-12)
    with pytest.raises(ValueError, match=r"\(T, A\)"):
        sample_covariance(R[:, 0])


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("lam", [0.5, 0.94, 0.97])
def test_ewma_matches_explicit_loop(seed, lam):
    R = _returns(seed, T=200, A=4)
    S = np.zeros((4, 4))
    N = 0.0
    for t in range(R.shape[0]):
        S = lam * S + np.outer(R[t], R[t])
        N = lam * N + 1.0
    np.testing.assert_allclose(ewma_covariance(R, lam), S / N, rtol=1e-12, atol=1e-18)


def test_ewma_long_sample_equals_textbook_recursion():
    R = _returns(1, T=2000, A=2)
    lam = 0.94
    sigma = np.outer(R[0], R[0])
    for t in range(1, R.shape[0]):
        sigma = lam * sigma + (1 - lam) * np.outer(R[t], R[t])
    np.testing.assert_allclose(ewma_covariance(R, lam), sigma, rtol=1e-12)
    with pytest.raises(ValueError, match="lam"):
        ewma_covariance(R, 1.0)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("shape", [(50, 5), (20, 30), (200, 10), (8, 2), (3, 6)])
def test_ledoit_wolf_matches_sklearn(seed, shape):
    cov_mod = pytest.importorskip("sklearn.covariance")
    rng = np.random.default_rng(seed)
    X = rng.normal(size=shape) @ rng.normal(size=(shape[1], shape[1])) * 0.01 + 0.001
    ref = cov_mod.LedoitWolf().fit(X)
    res = ledoit_wolf(X)
    assert res.shrinkage == pytest.approx(ref.shrinkage_, rel=1e-10, abs=1e-14)
    np.testing.assert_allclose(res.covariance, ref.covariance_, rtol=1e-10, atol=1e-18)
    if res.shrinkage > 0:
        assert np.linalg.eigvalsh(res.covariance)[0] > 0  # well conditioned even if T < A


def test_ledoit_wolf_edge_cases():
    sk = pytest.importorskip("sklearn.covariance")
    X = np.random.default_rng(0).normal(size=(30, 1))
    res = ledoit_wolf(X)
    assert res.shrinkage == 0.0
    np.testing.assert_allclose(res.covariance, sk.LedoitWolf().fit(X).covariance_, rtol=1e-12)
    # sample covariance already a multiple of the identity -> no shrinkage needed
    Y = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    res = ledoit_wolf(Y)
    assert res.shrinkage == sk.LedoitWolf().fit(Y).shrinkage_ == 0.0
    # a zero-variance asset (constant column) is handled without warnings
    Z = np.column_stack([np.random.default_rng(1).normal(size=40), np.full(40, 0.01)])
    ref = sk.LedoitWolf().fit(Z)
    np.testing.assert_allclose(ledoit_wolf(Z).covariance, ref.covariance_, rtol=1e-10)
    assert set(ledoit_wolf(Z).to_dict()) == {"covariance", "shrinkage"}
    with pytest.raises(ValueError, match="at least 2"):
        ledoit_wolf(np.ones((1, 3)))
