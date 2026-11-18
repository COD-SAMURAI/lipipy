"""Tests for lipipy.finance.exotics: Heston, Merton, Asian, barrier and lookback options."""

import math

import numpy as np
import pytest

from lipipy.finance import (
    arithmetic_asian_mc,
    barrier_mc,
    barrier_price,
    bs_price,
    geometric_asian_price,
    heston_price,
    lookback_mc,
    merton_jump_price,
)

pytestmark = pytest.mark.filterwarnings("error")

FO = dict(v0=0.0175, kappa=1.5768, theta=0.0398, sigma=0.5751, rho=-0.5711)  # Fang-Oosterlee


# ---------------------------------------------------------------------------
# Heston
# ---------------------------------------------------------------------------
def test_heston_published_reference_value():
    # Fang & Oosterlee (2008), SIAM J. Sci. Comput. 31, Heston test: reference 5.785155450
    assert heston_price(100.0, 100.0, 1.0, 0.0, **FO) == pytest.approx(5.785155450, abs=5e-8)


@pytest.mark.parametrize(
    "params",
    [
        dict(v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.7),
        dict(v0=0.02, kappa=0.5, theta=0.06, sigma=0.9, rho=-1.0),  # Feller violated, rho=-1
        dict(v0=0.09, kappa=3.0, theta=0.02, sigma=1.2, rho=1.0),  # Feller violated, rho=+1
        FO,
    ],
)
def test_heston_put_call_parity_and_bounds(params):
    for T in (0.1, 0.5, 3.0):
        for K in (60.0, 95.0, 100.0, 130.0):
            c = heston_price(100.0, K, T, 0.03, q=0.01, **params)
            p = heston_price(100.0, K, T, 0.03, q=0.01, kind="put", **params)
            fwd = 100.0 * math.exp(-0.01 * T) - K * math.exp(-0.03 * T)
            assert c - p == pytest.approx(fwd, abs=1e-8)
            assert max(fwd, 0.0) - 1e-8 <= c <= 100.0 * math.exp(-0.01 * T)


@pytest.mark.parametrize("v0,theta", [(0.04, 0.04), (0.09, 0.03), (0.01, 0.05)])
def test_heston_tends_to_black_scholes_as_vol_of_vol_vanishes(v0, theta):
    """sigma -> 0: variance is deterministic, BS with the integrated variance.

    With rho = 0 the gap is O(sigma^2): checked as |gap| <= 5 sigma^2 for shrinking sigma.
    """
    kappa, T = 2.0, 1.5
    avg_var = theta + (v0 - theta) * (1 - math.exp(-kappa * T)) / (kappa * T)
    for sig in (1e-2, 3e-3, 1e-3):
        for K in (80.0, 100.0, 125.0):
            h = heston_price(100.0, K, T, 0.02, v0, kappa, theta, sig, 0.0)
            b = bs_price(100.0, K, T, 0.02, math.sqrt(avg_var))
            assert abs(h - b) <= 5 * sig**2


def test_heston_continuity_in_rho_at_minus_one_and_short_maturity():
    p = dict(v0=0.02, kappa=0.5, theta=0.06, sigma=0.9)
    a = heston_price(100.0, 100.0, 1.0, 0.03, rho=-1.0, **p)
    b = heston_price(100.0, 100.0, 1.0, 0.03, rho=-0.9999, **p)
    assert a == pytest.approx(b, abs=1e-3)
    assert heston_price(100.0, 90.0, 0.0, 0.03, rho=-0.5, **p) == 10.0
    tiny = heston_price(100.0, 90.0, 1e-6, 0.03, rho=-0.5, **p)
    assert tiny == pytest.approx(100.0 - 90.0 * math.exp(-0.03e-6), abs=1e-6)


def test_heston_warns_when_quadrature_cannot_converge():
    """rho = -1 with a 1-week maturity: the integrand decays too slowly; this is reported."""
    from lipipy.core.errors import ConvergenceWarning

    with pytest.warns(ConvergenceWarning):
        heston_price(100.0, 60.0, 0.02, 0.03, 0.02, 0.5, 0.06, 0.9, -1.0, 0.01)


def test_heston_validation():
    with pytest.raises(ValueError, match="rho"):
        heston_price(100.0, 100.0, 1.0, 0.0, 0.04, 1.0, 0.04, 0.3, -1.5)
    with pytest.raises(ValueError, match="kappa"):
        heston_price(100.0, 100.0, 1.0, 0.0, 0.04, 0.0, 0.04, 0.3, 0.0)
    with pytest.raises(ValueError, match="sigma"):
        heston_price(100.0, 100.0, 1.0, 0.0, 0.04, 1.0, 0.04, -0.3, 0.0)


def test_heston_quantlib_parity():
    ql = pytest.importorskip("QuantLib")
    today = ql.Date(15, 1, 2025)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    spot = ql.QuoteHandle(ql.SimpleQuote(100.0))
    rts = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.03, dc))
    qts = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.01, dc))
    sets = [
        (0.04, 1.5, 0.04, 0.3, -0.7),
        (0.09, 3.0, 0.02, 1.2, 0.99),
        (0.02, 0.5, 0.06, 0.9, -0.99),
    ]
    for v0, ka, th, sg, rho in sets:
        model = ql.HestonModel(ql.HestonProcess(rts, qts, spot, v0, ka, th, sg, rho))
        engine = ql.AnalyticHestonEngine(model, 1e-12, 100_000)
        for days in (30, 365, 1095):
            for K in (70.0, 100.0, 140.0):
                for kind, t in (("call", ql.Option.Call), ("put", ql.Option.Put)):
                    opt = ql.VanillaOption(
                        ql.PlainVanillaPayoff(t, K), ql.EuropeanExercise(today + days)
                    )
                    opt.setPricingEngine(engine)
                    ours = heston_price(100.0, K, days / 365, 0.03, v0, ka, th, sg, rho, 0.01, kind)
                    assert ours == pytest.approx(opt.NPV(), abs=2e-8)


# ---------------------------------------------------------------------------
# Merton
# ---------------------------------------------------------------------------
def test_merton_limits_and_parity():
    for K in (80.0, 100.0, 120.0):
        assert merton_jump_price(100.0, K, 1.0, 0.05, 0.2, 0.0, -0.1, 0.1) == bs_price(
            100.0, K, 1.0, 0.05, 0.2
        )
        # deterministic jumps of size 0 (mu_j = sigma_j = 0) change nothing
        assert merton_jump_price(100.0, K, 1.0, 0.05, 0.2, 3.0, 0.0, 0.0) == pytest.approx(
            bs_price(100.0, K, 1.0, 0.05, 0.2), rel=1e-12
        )
        c = merton_jump_price(100.0, K, 2.0, 0.05, 0.2, 1.5, -0.15, 0.2, 0.02)
        p = merton_jump_price(100.0, K, 2.0, 0.05, 0.2, 1.5, -0.15, 0.2, 0.02, "put")
        assert c - p == pytest.approx(100 * math.exp(-0.04) - K * math.exp(-0.1), abs=1e-10)
    with pytest.raises(ValueError, match="non-negative"):
        merton_jump_price(100.0, 100.0, 1.0, 0.05, 0.2, -1.0, 0.0, 0.1)


def test_merton_matches_monte_carlo_of_the_jump_process():
    S, K, T, r, sig, lam, mj, sj = 100.0, 105.0, 1.0, 0.03, 0.2, 2.0, -0.1, 0.15
    k = math.exp(mj + sj**2 / 2) - 1
    rng = np.random.default_rng(0)
    n = 2_000_000
    N = rng.poisson(lam * T, n)
    jumps = mj * N + sj * np.sqrt(N) * rng.standard_normal(n)
    lnST = (
        math.log(S)
        + (r - lam * k - sig**2 / 2) * T
        + sig * math.sqrt(T) * rng.standard_normal(n)
        + jumps
    )
    pay = math.exp(-r * T) * np.maximum(np.exp(lnST) - K, 0)
    se = pay.std() / math.sqrt(n)
    assert abs(merton_jump_price(S, K, T, r, sig, lam, mj, sj) - pay.mean()) < 3.5 * se


# ---------------------------------------------------------------------------
# geometric Asian closed form
# ---------------------------------------------------------------------------
def test_geometric_asian_limits():
    assert geometric_asian_price(100.0, 95.0, 1.0, 0.05, 0.2, n_fixings=1) == pytest.approx(
        bs_price(100.0, 95.0, 1.0, 0.05, 0.2), rel=1e-13
    )
    cont = geometric_asian_price(100.0, 100.0, 1.0, 0.05, 0.2)
    assert geometric_asian_price(100.0, 100.0, 1.0, 0.05, 0.2, n_fixings=100_000) == pytest.approx(
        cont, rel=1e-4
    )
    # zero volatility: deterministic geometric average of the forward path
    fwd = math.exp(math.log(100.0) + 0.05 * 1.0 * 13 / 24)
    assert geometric_asian_price(100.0, 90.0, 1.0, 0.05, 0.0, n_fixings=12) == pytest.approx(
        math.exp(-0.05) * (fwd - 90.0), rel=1e-13
    )


def test_geometric_asian_matches_exact_simulation():
    rng = np.random.default_rng(1)
    n, m = 12, 1_000_000
    S, K, T, r, s = 100.0, 100.0, 1.0, 0.05, 0.3
    dt = T / n
    x = math.log(S) + np.cumsum(
        (r - s**2 / 2) * dt + s * math.sqrt(dt) * rng.standard_normal((m, n)), 1
    )
    pay = math.exp(-r * T) * np.maximum(np.exp(x.mean(1)) - K, 0)
    ref = geometric_asian_price(S, K, T, r, s, n_fixings=n)
    assert abs(ref - pay.mean()) < 3.5 * pay.std() / math.sqrt(m)


def test_geometric_asian_quantlib_parity():
    ql = pytest.importorskip("QuantLib")
    today = ql.Date(15, 1, 2025)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    proc = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(100.0)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, 0.01, dc)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, 0.03, dc)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), 0.25, dc)),
    )
    fixings = [today + 30 * i for i in range(1, 13)]  # equally spaced, T = 360/365
    for kind, t in (("call", ql.Option.Call), ("put", ql.Option.Put)):
        opt = ql.DiscreteAveragingAsianOption(
            ql.Average.Geometric,
            1.0,
            0,
            fixings,
            ql.PlainVanillaPayoff(t, 100.0),
            ql.EuropeanExercise(fixings[-1]),
        )
        opt.setPricingEngine(ql.AnalyticDiscreteGeometricAveragePriceAsianEngine(proc))
        ours = geometric_asian_price(100.0, 100.0, 360 / 365, 0.03, 0.25, 0.01, kind, 12)
        assert ours == pytest.approx(opt.NPV(), rel=1e-10)


# ---------------------------------------------------------------------------
# barriers
# ---------------------------------------------------------------------------
BARRIER_CASES = [
    (90.0, "down-and-out"), (90.0, "down-and-in"), (115.0, "up-and-out"), (115.0, "up-and-in"),
]  # fmt: skip


@pytest.mark.parametrize("K", [80.0, 95.0, 100.0, 110.0, 130.0])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_barrier_in_plus_out_equals_vanilla(K, kind):
    for H, direction in (("90", "down"), ("115", "up")):
        args = (100.0, K, 0.75, 0.04, 0.25)
        out = barrier_price(*args, float(H), f"{direction}-and-out", 0.01, kind)
        inn = barrier_price(*args, float(H), f"{direction}-and-in", 0.01, kind)
        assert out >= 0 and inn >= 0
        assert out + inn == pytest.approx(bs_price(*args, 0.01, kind), abs=1e-12)


def test_barrier_at_or_beyond_spot():
    v = bs_price(100.0, 100.0, 1.0, 0.05, 0.2)
    for H in (100.0, 105.0):  # down barrier at/above spot: already knocked
        assert barrier_price(100.0, 100.0, 1.0, 0.05, 0.2, H, "down-and-out") == 0.0
        assert barrier_price(100.0, 100.0, 1.0, 0.05, 0.2, H, "down-and-in") == v
        res = barrier_mc(100.0, 100.0, 1.0, 0.05, 0.2, H, "down-and-out", random_state=0)
        assert res.price == 0.0 and res.std_error == 0.0
    assert barrier_price(100.0, 100.0, 1.0, 0.05, 0.2, 100.0, "up-and-in") == v
    # barrier just below spot: the down-and-out is nearly worthless
    assert barrier_price(100.0, 100.0, 1.0, 0.05, 0.2, 99.999, "down-and-out") < 1e-2
    # far barrier: out ~ vanilla
    far = barrier_price(100.0, 100.0, 1.0, 0.05, 0.2, 1.0, "down-and-out")
    assert far == pytest.approx(v, abs=1e-12)
    # at expiry: no time to cross
    assert barrier_price(100.0, 90.0, 0.0, 0.05, 0.2, 80.0, "down-and-out") == 10.0
    with pytest.raises(ValueError, match="barrier_type"):
        barrier_price(100.0, 100.0, 1.0, 0.05, 0.2, 90.0, "double-knock-out")
    with pytest.raises(ValueError, match="positive"):
        barrier_price(100.0, 100.0, 1.0, 0.05, 0.2, -90.0)


def test_barrier_quantlib_parity():
    ql = pytest.importorskip("QuantLib")
    today = ql.Date(15, 1, 2025)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    proc = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(100.0)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, 0.01, dc)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, 0.03, dc)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), 0.2, dc)),
    )
    kinds = {"down-and-out": ql.Barrier.DownOut, "down-and-in": ql.Barrier.DownIn,
             "up-and-out": ql.Barrier.UpOut, "up-and-in": ql.Barrier.UpIn}  # fmt: skip
    for H, bt in BARRIER_CASES:
        for K in (85.0, 100.0, 120.0):
            for kind, t in (("call", ql.Option.Call), ("put", ql.Option.Put)):
                opt = ql.BarrierOption(
                    kinds[bt], H, 0.0, ql.PlainVanillaPayoff(t, K), ql.EuropeanExercise(today + 365)
                )
                opt.setPricingEngine(ql.AnalyticBarrierEngine(proc))
                ours = barrier_price(100.0, K, 1.0, 0.03, 0.2, H, bt, 0.01, kind)
                assert ours == pytest.approx(opt.NPV(), abs=1e-12)


# ---------------------------------------------------------------------------
# Monte Carlo within 3 standard errors of the closed forms
# ---------------------------------------------------------------------------
def _within(results, refs, max_z=4.0, frac3=0.9):
    z = np.abs(
        (np.array([r.price for r in results]) - np.array(refs)) / [r.std_error for r in results]
    )
    assert np.all(z < max_z), z
    assert np.mean(z < 3.0) >= frac3, z


@pytest.mark.parametrize("H,btype", BARRIER_CASES)
@pytest.mark.parametrize("kind", ["call", "put"])
def test_barrier_mc_bridge_is_unbiased_for_continuous_monitoring(H, btype, kind):
    ref = barrier_price(100.0, 100.0, 1.0, 0.03, 0.25, H, btype, 0.01, kind)
    runs = [
        barrier_mc(100.0, 100.0, 1.0, 0.03, 0.25, H, btype, 0.01, kind, n_steps=20,
                   n_paths=20_000, random_state=s)
        for s in range(10)
    ]  # fmt: skip
    _within(runs, [ref] * 10)


def test_barrier_mc_without_bridge_is_biased_towards_discrete_monitoring():
    ref = barrier_price(100.0, 100.0, 1.0, 0.03, 0.25, 90.0, "down-and-out")
    raw = barrier_mc(100.0, 100.0, 1.0, 0.03, 0.25, 90.0, n_steps=10, n_paths=200_000,
                     brownian_bridge=False, random_state=0)  # fmt: skip
    assert raw.price - ref > 5 * raw.std_error  # discrete monitoring knocks out less often


@pytest.mark.parametrize("kind", ["call", "put"])
def test_arithmetic_asian_control_variate(kind):
    args = (100.0, 105.0, 1.0, 0.04, 0.3, 0.01, kind)
    cv = [
        arithmetic_asian_mc(*args, n_fixings=12, n_paths=20_000, random_state=s) for s in range(8)
    ]
    plain = [
        arithmetic_asian_mc(*args, n_fixings=12, n_paths=200_000, antithetic=False,
                            control_variate=False, random_state=100 + s)
        for s in range(2)
    ]  # fmt: skip
    ref = np.mean([p.price for p in plain])
    ref_se = math.sqrt(sum(p.std_error**2 for p in plain)) / 2
    for r in cv:
        assert abs(r.price - ref) < 4 * math.hypot(r.std_error, ref_se)
        assert r.std_error < plain[0].std_error / 5  # CV with 10x fewer paths still wins
    # geometric closed form as the target of the control: with n_fixings = 1 the
    # arithmetic and geometric payoffs coincide, so the estimate is exact
    one = arithmetic_asian_mc(*args, n_fixings=1, n_paths=1000, random_state=0)
    assert one.price == pytest.approx(geometric_asian_price(*args, n_fixings=1), rel=1e-10)


def _lookback_closed_form(S, T, r, sigma, q, kind):
    """Goldman, Sosin & Gatto (1979) floating-strike lookback, extremum = S (Haug 4.15.1)."""
    from lipipy.numerics import norm_cdf

    b = r - q
    st = sigma * math.sqrt(T)
    a1 = (b + sigma**2 / 2) * T / st
    a2 = a1 - st
    N = lambda x: float(norm_cdf(x))  # noqa: E731
    w = sigma**2 / (2 * b)
    if kind == "call":
        return (S * math.exp(-q * T) * N(a1) - S * math.exp(-r * T) * N(a2)
                + S * math.exp(-r * T) * w * (N(-a1 + 2 * b * math.sqrt(T) / sigma)
                                               - math.exp(b * T) * N(-a1)))  # fmt: skip
    return (S * math.exp(-r * T) * N(-a2) - S * math.exp(-q * T) * N(-a1)
            + S * math.exp(-r * T) * w * (-N(a1 - 2 * b * math.sqrt(T) / sigma)
                                           + math.exp(b * T) * N(a1)))  # fmt: skip


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("n_steps", [1, 10, 50])
def test_lookback_mc_unbiased_for_continuous_monitoring(kind, n_steps):
    ref = _lookback_closed_form(100.0, 1.0, 0.05, 0.3, 0.01, kind)
    runs = [
        lookback_mc(100.0, 1.0, 0.05, 0.3, 0.01, kind, n_steps=n_steps, n_paths=20_000,
                    random_state=s)
        for s in range(8)
    ]  # fmt: skip
    _within(runs, [ref] * 8)


def test_mc_chunk_invariance_determinism_and_errors():
    a = arithmetic_asian_mc(
        100.0, 100.0, 1.0, 0.05, 0.2, n_paths=10_000, random_state=3, chunk_size=1200
    )
    b = arithmetic_asian_mc(100.0, 100.0, 1.0, 0.05, 0.2, n_paths=10_000, random_state=3)
    assert a.price == pytest.approx(b.price, rel=1e-12)
    assert lookback_mc(100.0, 1.0, 0.05, 0.2, n_paths=1000, random_state=1) == lookback_mc(
        100.0, 1.0, 0.05, 0.2, n_paths=1000, random_state=1
    )
    with pytest.raises(ValueError, match="sigma > 0"):
        arithmetic_asian_mc(100.0, 100.0, 1.0, 0.05, 0.0)
    with pytest.raises(ValueError, match="even"):
        barrier_mc(100.0, 100.0, 1.0, 0.05, 0.2, 90.0, n_paths=1001)
    with pytest.raises(ValueError, match="n_steps"):
        lookback_mc(100.0, 1.0, 0.05, 0.2, n_steps=0)
