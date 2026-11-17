"""Tests for lipipy.finance.volatility: likelihood, recovery, forecasts, arch parity."""

import math

import numpy as np
import pytest

from lipipy.finance import garch, gjr_garch

pytestmark = pytest.mark.filterwarnings("error")


def simulate(seed, T=2000, omega=1e-6, alpha=(0.08,), beta=(0.9,), gamma=0.0, mu=0.0):
    rng = np.random.default_rng(seed)
    p, q = len(alpha), len(beta)
    uncond = omega / (1 - sum(alpha) - sum(beta) - gamma / 2)
    e2 = [uncond] * p
    neg = [uncond / 2] * p
    s2h = [uncond] * q
    r = np.empty(T)
    for t in range(T):
        s2 = omega + sum(a * x for a, x in zip(alpha, e2, strict=True))
        s2 += gamma * neg[0] + sum(b * x for b, x in zip(beta, s2h, strict=True))
        eps = math.sqrt(s2) * rng.standard_normal()
        r[t] = mu + eps
        e2 = [eps * eps, *e2][:p]
        neg = [eps * eps * (eps < 0), *neg][:p]
        s2h = [s2, *s2h][:q]
    return r


def direct_loglik(r, mu, omega, alpha, beta, gamma=0.0):
    """Independent recomputation of the likelihood with the documented backcast."""
    eps = r - mu
    tau = min(75, len(r))
    w = 0.94 ** np.arange(tau)
    e0 = r - r.mean() if mu != 0.0 else r  # backcast residuals do not depend on mu
    bc = np.sum(e0[:tau] ** 2 * w / w.sum())
    ll = 0.0
    hist_e2, hist_s2, hist_neg = [bc] * len(alpha), [bc] * len(beta), bc / 2
    sig = []
    for e in eps:
        s2 = omega + np.dot(alpha, hist_e2) + gamma * hist_neg + np.dot(beta, hist_s2)
        ll += -0.5 * (math.log(2 * math.pi) + math.log(s2) + e * e / s2)
        sig.append(s2)
        hist_e2 = [e * e, *hist_e2][: len(alpha)]
        hist_s2 = [s2, *hist_s2][: len(beta)]
        hist_neg = e * e if e < 0 else 0.0
    return ll, np.sqrt(sig)


@pytest.mark.parametrize("seed", range(4))
def test_loglik_and_volatility_equal_direct_recomputation(seed):
    r = simulate(seed, T=800, mu=2e-4)
    for res, g in ((garch(r, mean="constant"), 0.0), (gjr_garch(r, mean="constant"), None)):
        pr = res.params
        gamma = pr.get("gamma[1]", 0.0) if g is None else g
        ll, sig = direct_loglik(r, pr["mu"], pr["omega"], [pr["alpha[1]"]], [pr["beta[1]"]], gamma)
        assert res.loglik == pytest.approx(ll, rel=1e-12)
        np.testing.assert_allclose(res.conditional_volatility, sig, rtol=1e-12)
        np.testing.assert_allclose(res.residuals, r - pr["mu"], rtol=0, atol=0)


def test_garch_pq_likelihood_recomputation():
    r = simulate(5, T=1000, alpha=(0.05, 0.03), beta=(0.6, 0.25))
    res = garch(r, p=2, q=2)
    pr = res.params
    ll, _ = direct_loglik(
        r, 0.0, pr["omega"], [pr["alpha[1]"], pr["alpha[2]"]], [pr["beta[1]"], pr["beta[2]"]]
    )
    assert res.loglik == pytest.approx(ll, rel=1e-12)
    assert res.model == "GARCH(2,2)" and 0 < res.persistence < 1
    arch_only = garch(simulate(6, T=600, alpha=(0.3,), beta=()), p=1, q=0)
    assert set(arch_only.params) == {"mu", "omega", "alpha[1]"}


def test_parameter_recovery_within_standard_errors():
    """Across seeds the true parameters fall within 3 standard errors (normal MLE)."""
    true = {"omega": 1e-6, "alpha[1]": 0.08, "beta[1]": 0.9}
    z = {k: [] for k in true}
    for seed in range(10):
        res = garch(simulate(100 + seed, T=2500))
        assert res.converged and res.persistence < 1
        for k, v in true.items():
            z[k].append((res.params[k] - v) / res.std_errors[k])
    for k in true:
        zk = np.abs(np.array(z[k]))
        assert np.all(zk < 3.5), (k, zk)
        assert np.mean(zk < 2.0) >= 0.7, (k, zk)


def test_gjr_recovers_asymmetry():
    res = gjr_garch(simulate(7, T=4000, alpha=(0.02,), beta=(0.88,), gamma=0.15))
    assert abs(res.params["gamma[1]"] - 0.15) < 3 * res.std_errors["gamma[1]"]
    assert res.params["gamma[1]"] > res.params["alpha[1]"]
    assert res.persistence < 1


def test_forecast_recursion_and_long_run_limit():
    res = garch(simulate(3, T=1500))
    pr = res.params
    w, a, b = pr["omega"], pr["alpha[1]"], pr["beta[1]"]
    f = res.forecast(500)
    eps_T, s_T = res.residuals[-1], res.conditional_volatility[-1]
    assert f[0] == pytest.approx(w + a * eps_T**2 + b * s_T**2, rel=1e-14)
    np.testing.assert_allclose(f[1:], w + (a + b) * f[:-1], rtol=1e-13)
    u = w / (1 - a - b)  # closed form: geometric approach to the unconditional variance
    np.testing.assert_allclose(f, u + (a + b) ** np.arange(500) * (f[0] - u), rtol=1e-10)
    g = gjr_garch(simulate(4, T=1500, gamma=0.05, alpha=(0.04,)))
    gp = g.params
    fg = g.forecast(5)
    pers = gp["alpha[1]"] + gp["gamma[1]"] / 2 + gp["beta[1]"]
    np.testing.assert_allclose(fg[1:], gp["omega"] + pers * fg[:-1], rtol=1e-13)


def test_near_unit_root_persistence_stays_stationary():
    r = simulate(11, T=3000, omega=1e-8, alpha=(0.05,), beta=(0.949,))
    res = garch(r)
    assert res.persistence < 1
    assert res.persistence > 0.98


def test_scale_invariance():
    r = simulate(8, T=1200)
    a, b = garch(r), garch(r * 1000)
    assert b.params["alpha[1]"] == pytest.approx(a.params["alpha[1]"], rel=1e-5)
    assert b.params["omega"] == pytest.approx(a.params["omega"] * 1e6, rel=1e-4)
    assert b.loglik == pytest.approx(a.loglik - len(r) * math.log(1000), rel=1e-9)


def test_validation_and_to_dict():
    with pytest.raises(ValueError, match="constant"):
        garch(np.full(100, 0.01))
    with pytest.raises(ValueError, match="NaN"):
        garch(np.r_[np.zeros(20), np.nan])
    with pytest.raises(ValueError, match="mean"):
        garch(simulate(0, T=50), mean="ar")
    with pytest.raises(ValueError, match="dist"):
        garch(simulate(0, T=50), dist="t")
    with pytest.raises(ValueError, match="shape"):
        garch(np.ones((50, 2)))
    with pytest.raises(ValueError, match="at least 10"):
        garch(np.arange(5.0))
    with pytest.raises(ValueError, match="h"):
        garch(simulate(0, T=200)).forecast(0)
    d = garch(simulate(0, T=200)).to_dict()
    assert {"model", "params", "std_errors", "loglik"} <= set(d) and "_p" not in d


@pytest.mark.parametrize("seed", range(3))
def test_arch_package_parity(seed):
    arch = pytest.importorskip("arch")
    r = simulate(200 + seed, T=2000, gamma=0.08, alpha=(0.03,))
    ours = garch(r, mean="constant")
    ref = arch.arch_model(r * 100, mean="Constant", vol="GARCH", p=1, q=1, rescale=False)
    fit = ref.fit(disp="off")
    shift = len(r) * math.log(100)
    # the likelihood functions agree exactly at the same parameters
    fixed = ref.fix(
        [
            ours.params["mu"] * 100,
            ours.params["omega"] * 1e4,
            ours.params["alpha[1]"],
            ours.params["beta[1]"],
        ]
    )
    assert fixed.loglikelihood + shift == pytest.approx(ours.loglik, rel=1e-10)
    # and the maxima agree (ours is at least as high up to optimiser tolerance)
    assert ours.loglik >= fit.loglikelihood + shift - 1e-6 * abs(ours.loglik)
    assert ours.params["alpha[1]"] == pytest.approx(fit.params["alpha[1]"], rel=2e-3)
    assert ours.params["beta[1]"] == pytest.approx(fit.params["beta[1]"], rel=2e-4)
    gjr = gjr_garch(r, mean="constant")
    gref = arch.arch_model(r * 100, mean="Constant", vol="GARCH", p=1, o=1, q=1, rescale=False)
    gfit = gref.fit(disp="off")
    assert gjr.params["gamma[1]"] == pytest.approx(gfit.params["gamma[1]"], rel=2e-3)
    fc = gfit.forecast(horizon=5, reindex=False).variance.values[-1] / 1e4
    np.testing.assert_allclose(gjr.forecast(5), fc, rtol=1e-3)
