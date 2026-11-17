"""Tests for lipipy.finance.timeseries: closed forms in CI, statsmodels parity locally."""

import math

import numpy as np
import pytest

from lipipy.finance import acf, adf, ar_fit, engle_granger, ljung_box, pacf
from lipipy.finance.timeseries import (
    _chi2_sf,
    _durbin_levinson,
    _mackinnon_crit,
    _mackinnon_p,
)

pytestmark = pytest.mark.filterwarnings("error")


def ar_process(phi, T, seed, burn=500):
    rng = np.random.default_rng(seed)
    p = len(phi)
    x = np.zeros(T + burn)
    e = rng.standard_normal(T + burn)
    for t in range(p, T + burn):
        x[t] = sum(phi[i] * x[t - 1 - i] for i in range(p)) + e[t]
    return x[burn:]


# ---------------------------------------------------------------------------
# closed forms (run in CI)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("phi", [-0.8, -0.3, 0.2, 0.7, 0.95])
def test_durbin_levinson_on_exact_ar1_acf(phi):
    rho = phi ** np.arange(11)  # AR(1) ACF = phi^k exactly
    pac, coefs = _durbin_levinson(rho, 10)
    np.testing.assert_allclose(pac, [phi] + [0.0] * 9, atol=1e-13)
    np.testing.assert_allclose(coefs, [phi] + [0.0] * 9, atol=1e-13)


@pytest.mark.parametrize("phi1,phi2", [(0.5, -0.2), (1.2, -0.4), (-0.3, 0.4)])
def test_durbin_levinson_on_exact_ar2_acf(phi1, phi2):
    rho = np.empty(12)
    rho[0], rho[1] = 1.0, phi1 / (1 - phi2)
    for k in range(2, 12):
        rho[k] = phi1 * rho[k - 1] + phi2 * rho[k - 2]
    pac, coefs = _durbin_levinson(rho, 6)
    assert pac[0] == pytest.approx(rho[1], rel=1e-13)
    assert pac[1] == pytest.approx(phi2, rel=1e-12)
    np.testing.assert_allclose(pac[2:], 0.0, atol=1e-12)
    np.testing.assert_allclose(coefs[:2], [phi1, phi2], rtol=1e-12)


@pytest.mark.parametrize("phi", [-0.6, 0.3, 0.8])
def test_sample_acf_converges_to_phi_power_k(phi):
    T = 200_000
    x = ar_process([phi], T, seed=1)
    r = acf(x, 8)
    se = math.sqrt((1 + phi**2) / (1 - phi**2) / T)  # Bartlett, AR(1), lag 1 (largest k)
    k = np.arange(9)
    assert np.all(np.abs(r - phi**k) < 5 * se * np.sqrt(np.maximum(k, 1)))
    assert r[0] == 1.0


def test_acf_hand_computed_and_errors():
    np.testing.assert_allclose(acf([1.0, 2.0, 3.0, 4.0], 2), [1.0, 0.25, -0.3], atol=1e-15)
    with pytest.raises(ValueError, match="constant"):
        acf(np.full(10, 3.0), 3)
    with pytest.raises(ValueError, match="nlags"):
        acf(np.arange(5.0), 5)
    with pytest.raises(ValueError, match="NaN"):
        pacf([1.0, np.nan, 2.0, 3.0])
    with pytest.raises(ValueError, match=r"shape \(T,\)"):
        acf(np.ones((5, 2)))
    np.testing.assert_array_equal(pacf(np.arange(5.0), 0), [1.0])


@pytest.mark.parametrize("phi", [0.4, 0.9, 0.99])
def test_yule_walker_and_ols_recover_coefficients(phi):
    x = ar_process([phi], 50_000, seed=2) + 3.0
    for method in ("yule_walker", "ols"):
        res = ar_fit(x, 1, method)
        assert res.coefficients[0] == pytest.approx(phi, abs=5 * math.sqrt((1 - phi**2) / 50_000))
        assert res.sigma2 == pytest.approx(1.0, abs=0.03)
        assert res.intercept / (1 - res.coefficients[0]) == pytest.approx(3.0, abs=0.3)


def test_yule_walker_on_exact_autocovariances_is_exact():
    """Durbin-Levinson solves the Toeplitz system: compare with a dense solve."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal(300)
    res = ar_fit(x, 4)
    g = np.array([(x - x.mean())[k:] @ (x - x.mean())[: 300 - k] for k in range(5)]) / 300
    G = np.array([[g[abs(i - j)] for j in range(4)] for i in range(4)])
    np.testing.assert_allclose(res.coefficients, np.linalg.solve(G, g[1:]), rtol=1e-12)
    assert res.sigma2 == pytest.approx(g[0] - res.coefficients @ g[1:], rel=1e-12)


def test_ols_matches_dense_least_squares():
    x = ar_process([0.5, -0.25], 400, seed=5)
    res = ar_fit(x, 2, "ols")
    X = np.column_stack([np.ones(398), x[1:-1], x[:-2]])
    beta, *_ = np.linalg.lstsq(X, x[2:], rcond=None)
    np.testing.assert_allclose(np.r_[res.intercept, res.coefficients], beta, rtol=1e-11)
    with pytest.raises(ValueError, match="method"):
        ar_fit(x, 2, "burg")
    with pytest.raises(ValueError, match="T > 2 p"):
        ar_fit(x[:4], 2)


@pytest.mark.parametrize("df", [1, 2, 5, 10, 37])
def test_chi2_survival_matches_scipy(df):
    stats = pytest.importorskip("scipy.stats")
    for x in np.concatenate([[0.0, 1e-8], np.geomspace(1e-3, 300, 60)]):
        assert _chi2_sf(x, df) == pytest.approx(stats.chi2.sf(x, df), rel=1e-12, abs=1e-300)


def test_ljung_box_definition_and_size():
    x = np.random.default_rng(0).standard_normal(500)
    res = ljung_box(x, 10)
    r = acf(x, 10)[1:]
    q = 500 * 502 * np.sum(r**2 / (500 - np.arange(1, 11)))
    assert res.statistic == pytest.approx(q, rel=1e-14) and res.df == 10
    # under H0 the p-values are ~uniform: rejection rate at 5% close to 5%
    rejections = [
        ljung_box(np.random.default_rng(s).standard_normal(300), 5).pvalue < 0.05
        for s in range(400)
    ]
    assert 0.025 <= np.mean(rejections) <= 0.08
    assert ljung_box(ar_process([0.5], 500, 1), 5).pvalue < 1e-10
    with pytest.raises(ValueError, match="model_df"):
        ljung_box(x, 3, model_df=3)


# ---------------------------------------------------------------------------
# ADF / Engle-Granger
# ---------------------------------------------------------------------------
def test_mackinnon_tables_known_values():
    # asymptotic 5% critical value for a constant is -2.86154; T = 100 adjusts it
    assert _mackinnon_crit("c", 1, math.inf)["5%"] == pytest.approx(-2.86154)
    c100 = -2.86154 - 2.8903 / 100 - 4.234 / 100**2 - 40.04 / 100**3
    assert _mackinnon_crit("c", 1, 100)["5%"] == pytest.approx(c100, rel=1e-14)
    # p-value at the asymptotic 5% critical value is ~0.05, and monotone in tau
    for reg in ("n", "c", "ct"):
        crit = _mackinnon_crit(reg, 1, math.inf)
        assert _mackinnon_p(crit["5%"], reg, 1) == pytest.approx(0.05, abs=0.003)
        assert _mackinnon_p(crit["1%"], reg, 1) == pytest.approx(0.01, abs=0.002)
        taus = np.linspace(-6, 0.5, 200)
        assert np.all(np.diff([_mackinnon_p(t, reg, 1) for t in taus]) >= 0)
    assert _mackinnon_p(-50.0, "c", 1) == 0.0 and _mackinnon_p(5.0, "c", 1) == 1.0


@pytest.mark.parametrize("regression", ["n", "c", "ct"])
def test_adf_size_under_unit_root(regression):
    """Random walks: the 5% test rejects about 5% of the time."""
    rej = []
    for s in range(300):
        w = np.cumsum(np.random.default_rng(s).standard_normal(250))
        rej.append(adf(w, regression=regression, autolag=None, maxlag=1).pvalue < 0.05)
    assert 0.025 <= np.mean(rej) <= 0.085


def test_adf_power_and_near_unit_root():
    assert adf(ar_process([0.5], 500, 3)).pvalue < 1e-6
    near = adf(ar_process([0.999], 300, 4))  # near unit root: finite, no warnings
    assert np.isfinite(near.statistic) and 0.0 <= near.pvalue <= 1.0
    res = adf(ar_process([0.9], 2000, 5), regression="ct")
    assert res.statistic < res.critical_values["1%"] and res.pvalue < 0.01
    assert set(res.to_dict()) >= {"statistic", "pvalue", "critical_values"}


def test_adf_regression_matches_dense_ols():
    w = np.cumsum(np.random.default_rng(9).standard_normal(200))
    res = adf(w, maxlag=2, autolag=None, regression="c")
    dx = np.diff(w)
    y = dx[2:]
    X = np.column_stack([w[2:-1], dx[1:-1], dx[:-2], np.ones(197)])
    beta, ssr, *_ = np.linalg.lstsq(X, y, rcond=None)
    se = np.sqrt(ssr[0] / (197 - 4) * np.linalg.inv(X.T @ X)[0, 0])
    assert res.statistic == pytest.approx(beta[0] / se, rel=1e-10)
    assert res.nobs == 197 and res.used_lag == 2


def test_adf_errors():
    with pytest.raises(ValueError, match="regression"):
        adf(np.arange(50.0) ** 1.5, regression="ctt")
    with pytest.raises(ValueError, match="maxlag"):
        adf(np.random.default_rng(0).standard_normal(20), maxlag=15)
    with pytest.raises(ValueError, match="constant"):
        adf(np.ones(30))
    with pytest.raises(ValueError, match="autolag"):
        adf(np.random.default_rng(0).standard_normal(50), autolag="bic")


def test_engle_granger_cointegrated_and_spurious():
    rng = np.random.default_rng(10)
    x = np.cumsum(rng.standard_normal(800))
    y = 0.7 * x + 2.0 + rng.standard_normal(800)
    res = engle_granger(y, x)
    assert res.pvalue < 0.001
    assert res.coefficients[0] == pytest.approx(0.7, abs=0.01)
    np.testing.assert_allclose(res.residuals, y - res.coefficients[0] * x - res.intercept)
    rej = []
    for s in range(150):  # independent random walks: spurious regression, H0 true
        r = np.random.default_rng(1000 + s)
        a, b = np.cumsum(r.standard_normal(300)), np.cumsum(r.standard_normal(300))
        rej.append(engle_granger(a, b, autolag=None, maxlag=1).pvalue < 0.05)
    # the asymptotic MacKinnon p-values are conservative here (statsmodels agrees)
    assert np.mean(rej) <= 0.1
    with pytest.raises(ValueError, match="exactly"):
        engle_granger(2.0 * x + 1.0, x)
    with pytest.raises(ValueError, match="at most 2"):
        engle_granger(y, np.column_stack([x, x**2, x**3]))
    with pytest.raises(ValueError, match="trend"):
        engle_granger(y, x, trend="n")


# ---------------------------------------------------------------------------
# statsmodels parity (local)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(5))
def test_statsmodels_parity(seed):
    import warnings

    with warnings.catch_warnings():  # statsmodels' own FutureWarnings about return types
        warnings.simplefilter("ignore", FutureWarning)
        _statsmodels_parity(seed)


def _statsmodels_parity(seed):
    st = pytest.importorskip("statsmodels.tsa.stattools")
    diag = pytest.importorskip("statsmodels.stats.diagnostic")
    lm = pytest.importorskip("statsmodels.regression.linear_model")
    armod = pytest.importorskip("statsmodels.tsa.ar_model")
    x = ar_process([0.5, -0.2], 600, seed)
    np.testing.assert_allclose(acf(x, 15), st.acf(x, nlags=15, fft=False), atol=1e-13)
    np.testing.assert_allclose(pacf(x, 15), st.pacf(x, nlags=15, method="ldb"), atol=1e-12)
    lb = diag.acorr_ljungbox(x, lags=[8], model_df=2)
    ours = ljung_box(x, 8, 2)
    assert ours.statistic == pytest.approx(lb.lb_stat.iloc[0], rel=1e-12)
    assert ours.pvalue == pytest.approx(lb.lb_pvalue.iloc[0], rel=1e-9, abs=1e-300)
    try:  # newer statsmodels warns unless the return type is chosen explicitly
        rho, sig = lm.yule_walker(x, 2, method="mle", result_object=False)
    except TypeError:
        rho, sig = lm.yule_walker(x, 2, method="mle")
    np.testing.assert_allclose(ar_fit(x, 2).coefficients, rho, rtol=1e-12)
    assert ar_fit(x, 2).sigma2 == pytest.approx(sig**2, rel=1e-12)
    ref = armod.AutoReg(x, 2, trend="c").fit()
    ols = ar_fit(x, 2, "ols")
    np.testing.assert_allclose(np.r_[ols.intercept, ols.coefficients], ref.params, rtol=1e-10)
    assert ols.sigma2 == pytest.approx(ref.sigma2, rel=1e-12)
    w = np.cumsum(np.random.default_rng(seed).standard_normal(400))
    for reg in ("n", "c", "ct"):
        for al in ("aic", None):
            o = adf(w, regression=reg, autolag=al)
            s = st.adfuller(w, regression=reg, autolag="AIC" if al else None)
            assert o.statistic == pytest.approx(s[0], rel=1e-10)
            assert o.pvalue == pytest.approx(s[1], rel=1e-10)
            assert (o.used_lag, o.nobs) == (s[2], s[3])
            for k, v in s[4].items():
                assert o.critical_values[k] == pytest.approx(v, rel=1e-12)
    y = 1.5 * w + np.random.default_rng(seed + 50).standard_normal(400)
    for tr in ("c", "ct"):
        o = engle_granger(y, w, tr)
        s = st.coint(y, w, trend=tr)
        assert o.statistic == pytest.approx(s[0], rel=1e-10)
        assert o.pvalue == pytest.approx(s[1], rel=1e-9, abs=1e-25)
        np.testing.assert_allclose(list(o.critical_values.values()), s[2], rtol=1e-12)
