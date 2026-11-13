"""Tests for lipipy.finance.returns: hand-computed series, closed forms, edge cases."""

import numpy as np
import pytest

from lipipy.finance import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    capm_alpha_beta,
    cumulative_returns,
    drawdown_series,
    information_ratio,
    log_returns,
    max_drawdown,
    sharpe_ratio,
    simple_returns,
    sortino_ratio,
    tracking_error,
)

pytestmark = pytest.mark.filterwarnings("error")

SEEDS = range(8)


def _panel(seed, T=250, A=3):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0005, 0.01, size=(T, A))


# ---------------------------------------------------------------------------
# simple / log / cumulative returns
# ---------------------------------------------------------------------------
def test_simple_and_log_returns_hand_computed():
    p = np.array([100.0, 110.0, 99.0, 99.0])
    np.testing.assert_allclose(simple_returns(p), [0.1, -0.1, 0.0], rtol=0, atol=1e-15)
    np.testing.assert_allclose(log_returns(p), np.log([1.1, 0.9, 1.0]), rtol=1e-15, atol=0)


@pytest.mark.parametrize("seed", SEEDS)
def test_returns_roundtrip_prices(seed):
    rng = np.random.default_rng(seed)
    p = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, size=(60, 4)), axis=0))
    R = simple_returns(p)
    r = log_returns(p)
    np.testing.assert_allclose(p[1:] / p[0] - 1, cumulative_returns(R), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        p[1:] / p[0] - 1, cumulative_returns(r, log=True), rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(r, np.log1p(R), rtol=1e-12, atol=1e-16)


def test_log_returns_tiny_moves_keep_relative_accuracy():
    p = np.array([1.0, 1.0 + 1e-12])
    assert log_returns(p)[0] == pytest.approx(1e-12, rel=1e-4)


def test_returns_validation():
    with pytest.raises(ValueError, match="NaN"):
        simple_returns([1.0, np.nan, 2.0])
    with pytest.raises(ValueError, match="positive"):
        log_returns([1.0, 0.0])
    with pytest.raises(ValueError, match="at least 2"):
        simple_returns([1.0])
    with pytest.raises(ValueError, match="shape"):
        simple_returns(np.ones((2, 2, 2)))
    with pytest.raises(TypeError):
        simple_returns("abc")
    with pytest.raises(ValueError, match="NaN"):
        sharpe_ratio([0.01, np.nan, 0.02])
    with pytest.raises(ValueError, match="NaN"):
        max_drawdown([0.01, np.inf])
    with pytest.raises(ValueError, match="-1"):
        cumulative_returns([0.1, -1.5])


# ---------------------------------------------------------------------------
# annualisation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_annualized_return_matches_definition(seed):
    R = _panel(seed)
    expected = np.prod(1 + R, axis=0) ** (252 / R.shape[0]) - 1
    np.testing.assert_allclose(annualized_return(R), expected, rtol=1e-12)
    r = np.log1p(R)
    np.testing.assert_allclose(annualized_return(r, log=True), expected, rtol=1e-12)
    for j in range(R.shape[1]):
        assert annualized_return(R[:, j]) == pytest.approx(expected[j], rel=1e-12)


def test_annualized_return_single_period_and_wipeout():
    assert annualized_return([0.05], periods_per_year=1) == pytest.approx(0.05, rel=1e-15)
    assert annualized_return([0.05], periods_per_year=4) == pytest.approx(1.05**4 - 1, rel=1e-14)
    assert annualized_return([0.1, -1.0, 0.2]) == -1.0
    out = annualized_return(np.array([[0.1, -1.0], [0.0, 0.5]]), 1)
    assert out[0] == pytest.approx(np.sqrt(1.1) - 1, rel=1e-14) and out[1] == -1.0
    with pytest.raises(ValueError):
        annualized_return([0.1], periods_per_year=0)
    with pytest.raises(ValueError):
        annualized_return([-1.2])


@pytest.mark.parametrize("seed", SEEDS)
def test_annualized_volatility(seed):
    R = _panel(seed)
    np.testing.assert_allclose(
        annualized_volatility(R, 12), R.std(axis=0, ddof=1) * np.sqrt(12), rtol=1e-13
    )
    assert annualized_volatility([0.01, 0.01, 0.01]) < 1e-15
    with pytest.raises(ValueError):
        annualized_volatility([0.01])  # T = 1 with ddof = 1


# ---------------------------------------------------------------------------
# ratios
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_sharpe_and_information_ratio_definitions(seed):
    R = _panel(seed)
    rf = np.random.default_rng(seed + 100).uniform(0, 1e-4, size=R.shape[0])
    ex = R - rf[:, None]
    expected = ex.mean(axis=0) / ex.std(axis=0, ddof=1) * np.sqrt(252)
    np.testing.assert_allclose(sharpe_ratio(R, rf), expected, rtol=1e-12)
    np.testing.assert_allclose(sharpe_ratio(R, 1e-4), sharpe_ratio(R, np.full(250, 1e-4)))
    bench = R[:, 0]
    active = R[:, 1:] - bench[:, None]
    np.testing.assert_allclose(
        information_ratio(R[:, 1:], bench),
        active.mean(axis=0) / active.std(axis=0, ddof=1) * np.sqrt(252),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        tracking_error(R[:, 1:], bench), active.std(axis=0, ddof=1) * np.sqrt(252), rtol=1e-12
    )


def test_sortino_hand_computed_and_definition():
    r = np.array([0.04, -0.02, 0.01, -0.01, 0.03])
    tau = 0.005
    dd = np.sqrt(np.mean(np.minimum(r - tau, 0) ** 2))  # all T observations
    assert sortino_ratio(r, tau, 12) == pytest.approx((r.mean() - tau) / dd * np.sqrt(12), 1e-14)
    # by hand: shortfalls -0.025, -0.015 -> DD^2 = (0.000625 + 0.000225) / 5 = 0.00017
    assert dd == pytest.approx(np.sqrt(0.00017), rel=1e-14)


def test_zero_dispersion_ratios_are_inf_or_nan_without_warnings():
    const = np.full(10, 0.01)
    assert sharpe_ratio(const) == np.inf
    assert sharpe_ratio(-const) == -np.inf
    assert np.isnan(sharpe_ratio(const, risk_free=0.01))
    assert sortino_ratio(const) == np.inf
    assert np.isnan(sortino_ratio(np.zeros(5)))
    assert information_ratio(const + 0.001, const) == np.inf
    np.testing.assert_array_equal(
        sharpe_ratio(np.column_stack([const, np.zeros(10)])), [np.inf, np.nan]
    )
    assert calmar_ratio(const) == np.inf
    assert np.isnan(calmar_ratio(np.zeros(4)))


def test_ratio_validation():
    with pytest.raises(ValueError, match="at least 2"):
        sharpe_ratio([0.01])
    with pytest.raises(ValueError, match="rows"):
        sharpe_ratio([0.01, 0.02], risk_free=[0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="rows"):
        information_ratio([0.01, 0.02], [0.01])
    with pytest.raises(ValueError, match="NaN"):
        information_ratio([0.01, 0.02], [0.01, np.nan])
    with pytest.raises(TypeError):
        sharpe_ratio([0.01, 0.02], periods_per_year="252")


# ---------------------------------------------------------------------------
# drawdowns
# ---------------------------------------------------------------------------
def test_drawdown_hand_computed_with_recovery():
    r = [0.1, -0.5, 0.2, 1.0, -0.1]
    # wealth: 1, 1.1, 0.55, 0.66, 1.32, 1.188
    np.testing.assert_allclose(
        drawdown_series(r), [0, 0, -0.5, -0.4, 0, 1.188 / 1.32 - 1], atol=1e-15
    )
    res = max_drawdown(r)
    assert res.max_drawdown == pytest.approx(-0.5, abs=1e-15)
    assert (res.peak_index, res.trough_index, res.recovery_index) == (1, 2, 4)
    assert res.duration == 3 and res.recovered
    assert res.to_dict()["recovery_index"] == 4


def test_drawdown_not_recovered_and_first_period_loss():
    res = max_drawdown([-0.2, 0.1, -0.3, 0.05])
    # wealth 1, 0.8, 0.88, 0.616, 0.6468: peak is the initial capital
    assert res.peak_index == 0 and res.trough_index == 3
    assert res.recovery_index is None and not res.recovered
    assert res.duration == 4
    assert res.max_drawdown == pytest.approx(0.616 - 1, abs=1e-15)


def test_drawdown_flat_top_uses_last_peak_and_single_period():
    res = max_drawdown([0.1, 0.0, -0.1, 0.2])  # wealth 1, 1.1, 1.1, 0.99, 1.188
    assert (res.peak_index, res.trough_index, res.recovery_index, res.duration) == (2, 3, 4, 2)
    assert max_drawdown([0.05]) == max_drawdown([0.0])
    assert max_drawdown([0.05]).max_drawdown == 0.0
    single = max_drawdown([-0.3])
    assert single.max_drawdown == pytest.approx(-0.3) and single.duration == 1


def test_drawdown_total_loss_and_log_returns():
    res = max_drawdown([0.1, -1.0, 0.5])
    assert res.max_drawdown == -1.0 and not res.recovered
    r = np.log([1.1, 0.5, 1.2])
    np.testing.assert_allclose(drawdown_series(r, log=True), drawdown_series(np.expm1(r)))
    with pytest.raises(ValueError, match=r"shape \(T,\)"):
        max_drawdown(np.zeros((3, 2)))


@pytest.mark.parametrize("seed", SEEDS)
def test_drawdown_matches_bruteforce(seed):
    R = _panel(seed, T=120, A=2) * 3
    D = drawdown_series(R)
    W = np.vstack([np.ones(2), np.cumprod(1 + R, axis=0)])
    for j in range(2):
        brute = [W[i, j] / W[: i + 1, j].max() - 1 for i in range(W.shape[0])]
        np.testing.assert_allclose(D[:, j], brute, rtol=1e-14, atol=1e-15)
        res = max_drawdown(R[:, j])
        assert res.max_drawdown == pytest.approx(min(brute), rel=1e-14)
        assert W[res.peak_index, j] == W[: res.trough_index + 1, j].max()
        if res.recovered:
            assert W[res.recovery_index, j] >= W[res.peak_index, j]
            assert np.all(W[res.trough_index : res.recovery_index, j] < W[res.peak_index, j])
    mdd = -D.min(axis=0)
    np.testing.assert_allclose(calmar_ratio(R), annualized_return(R) / mdd, rtol=1e-14)


# ---------------------------------------------------------------------------
# CAPM
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_capm_matches_scipy_linregress(seed):
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(seed)
    m = rng.normal(0.0004, 0.01, size=300)
    rf = rng.uniform(0, 1e-4, size=300)
    R = np.column_stack([0.0002 + 1.3 * m + rng.normal(0, 0.005, 300) for _ in range(3)])
    res = capm_alpha_beta(R, m, risk_free=rf, periods_per_year=252)
    for j in range(3):
        lr = stats.linregress(m - rf, R[:, j] - rf)
        assert res.beta[j] == pytest.approx(lr.slope, rel=1e-12)
        assert res.alpha_per_period[j] == pytest.approx(lr.intercept, rel=1e-10, abs=1e-15)
        assert res.alpha[j] == pytest.approx(252 * lr.intercept, rel=1e-10, abs=1e-13)
        assert res.r_squared[j] == pytest.approx(lr.rvalue**2, rel=1e-12)
    one = capm_alpha_beta(R[:, 0], m, rf)
    assert one.beta == pytest.approx(res.beta[0], rel=1e-14)


def test_capm_exact_line_and_errors():
    m = np.array([0.01, -0.02, 0.03, 0.0])
    res = capm_alpha_beta(0.001 + 0.5 * m, m, periods_per_year=12)
    assert res.beta == pytest.approx(0.5, rel=1e-14)
    assert res.alpha == pytest.approx(0.012, rel=1e-12)
    assert res.r_squared == pytest.approx(1.0, rel=1e-14)
    with pytest.raises(ValueError, match="constant"):
        capm_alpha_beta([0.01, 0.02, 0.03], [0.01, 0.01, 0.01])
    with pytest.raises(ValueError, match="rows"):
        capm_alpha_beta([0.01, 0.02, 0.03], [0.01, 0.02])
    with pytest.raises(ValueError, match="market"):
        capm_alpha_beta(np.zeros((3, 2)), np.ones((3, 2)))
    with pytest.raises(ValueError, match="risk_free"):
        capm_alpha_beta(np.zeros((4, 2)), m, risk_free=np.zeros((4, 2)))
