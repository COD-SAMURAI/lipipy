"""Tests for lipipy.finance.options: textbook values, parity, Greeks, trees, Monte Carlo."""

import itertools

import numpy as np
import pytest

from lipipy.finance import (
    binomial_price,
    bs_greeks,
    bs_price,
    implied_volatility,
    monte_carlo_price,
)

pytestmark = pytest.mark.filterwarnings("error")

HULL = dict(S=42.0, K=40.0, T=0.5, r=0.10, sigma=0.20)  # Hull, example 15.6


def _grid():
    S, K, T, sig = np.meshgrid(
        [100.0],
        [50.0, 80.0, 95.0, 100.0, 105.0, 120.0, 200.0],
        [0.01, 0.1, 0.5, 1.0, 3.0],
        [0.05, 0.2, 0.5, 1.2],
        indexing="ij",
    )
    return S.ravel(), K.ravel(), T.ravel(), sig.ravel()


# ---------------------------------------------------------------------------
# Black-Scholes-Merton
# ---------------------------------------------------------------------------
def test_hull_textbook_values():
    assert round(bs_price(**HULL), 2) == 4.76
    assert round(bs_price(**HULL, kind="put"), 2) == 0.81
    assert bs_price(**HULL) == pytest.approx(4.759422392871528, rel=1e-13)


@pytest.mark.parametrize("q", [0.0, 0.03])
def test_matches_independent_scipy_formula(q):
    stats = pytest.importorskip("scipy.stats")
    S, K, T, sig = _grid()
    r = 0.04
    d1 = (np.log(S / K) + (r - q + sig**2 / 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    call = S * np.exp(-q * T) * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    put = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * np.exp(-q * T) * stats.norm.cdf(-d1)
    np.testing.assert_allclose(bs_price(S, K, T, r, sig, q), call, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(bs_price(S, K, T, r, sig, q, "put"), put, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("r,q", [(0.0, 0.0), (0.05, 0.0), (0.05, 0.02), (-0.01, 0.03)])
def test_put_call_parity(r, q):
    S, K, T, sig = _grid()
    lhs = bs_price(S, K, T, r, sig, q) - bs_price(S, K, T, r, sig, q, "put")
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    np.testing.assert_allclose(lhs, rhs, rtol=0, atol=1e-12)


def test_broadcasting_and_scalar_output():
    out = bs_price(100.0, [90.0, 100.0, 110.0], 1.0, 0.05, [[0.1], [0.2]])
    assert out.shape == (2, 3)
    assert isinstance(bs_price(100.0, 100.0, 1.0, 0.05, 0.2), float)
    assert np.all(np.diff(out, axis=1) < 0)  # call decreasing in strike
    assert np.all(np.diff(out, axis=0) > 0)  # increasing in volatility


def test_degenerate_limits_are_discounted_intrinsic_without_warnings():
    S, K, r, q = 100.0, np.array([80.0, 100.0, 120.0]), 0.05, 0.02
    np.testing.assert_array_equal(bs_price(S, K, 0.0, r, 0.3), [20.0, 0.0, 0.0])
    np.testing.assert_array_equal(bs_price(S, K, 0.0, r, 0.3, kind="put"), [0.0, 0.0, 20.0])
    T = 2.0
    fwd = S * np.exp(-q * T) - K * np.exp(-r * T)
    np.testing.assert_allclose(bs_price(S, K, T, r, 0.0, q), np.maximum(fwd, 0), atol=1e-13)
    np.testing.assert_allclose(bs_price(S, K, T, r, 0.0, q, "put"), np.maximum(-fwd, 0), atol=1e-13)
    # continuity: tiny sigma or T (down to subnormal sigma * sqrt(T)) approaches the limit
    for sigma, t in [(1e-12, T), (1e-300, T), (5e-324, T), (0.3, 1e-300), (0.3, 5e-324)]:
        lim = np.maximum(S * np.exp(-q * t) - K * np.exp(-r * t), 0)
        np.testing.assert_allclose(bs_price(S, K, t, r, sigma, q), lim, rtol=1e-10, atol=1e-10)
    # forward exactly at the money: price -> 0 as sigma -> 0
    Katm = 100.0 * np.exp(0.05 * 1.0)
    assert bs_price(100.0, Katm, 1.0, 0.05, 0.0) == pytest.approx(0.0, abs=1e-12)


def test_extreme_moneyness_no_warnings_and_bounds():
    S = 100.0
    K = np.array([1e-8, 1e-3, 1.0, 1e3, 1e8])
    for kind in ("call", "put"):
        v = bs_price(S, K, 1.0, 0.05, 0.25, 0.0, kind)
        assert np.all(np.isfinite(v)) and np.all(v >= 0)
    call = bs_price(S, K, 1.0, 0.05, 0.25)
    np.testing.assert_allclose(call[:2], S - K[:2] * np.exp(-0.05), rtol=1e-12)
    assert call[-1] == 0.0


def test_bs_validation():
    with pytest.raises(ValueError, match="positive"):
        bs_price(0.0, 100.0, 1.0, 0.05, 0.2)
    with pytest.raises(ValueError, match="non-negative"):
        bs_price(100.0, 100.0, -1.0, 0.05, 0.2)
    with pytest.raises(ValueError, match="sigma"):
        bs_price(100.0, 100.0, 1.0, 0.05, -0.2)
    with pytest.raises(ValueError, match="NaN"):
        bs_price(np.nan, 100.0, 1.0, 0.05, 0.2)
    with pytest.raises(ValueError, match="kind"):
        bs_price(100.0, 100.0, 1.0, 0.05, 0.2, kind="straddle")
    with pytest.raises(ValueError, match="broadcast"):
        bs_price([1.0, 2.0], [1.0, 2.0, 3.0], 1.0, 0.0, 0.2)
    with pytest.raises(TypeError):
        bs_price("100", 100.0, 1.0, 0.05, 0.2)


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("q", [0.0, 0.02])
def test_greeks_match_central_differences(kind, q):
    S, K, T, sig = _grid()
    keep = T >= 0.1  # keep finite differences well conditioned
    S, K, T, sig = S[keep], K[keep], T[keep], sig[keep]
    r = 0.04
    g = bs_greeks(S, K, T, r, sig, q, kind)

    def price(**kw):
        args = dict(S=S, K=K, T=T, r=r, sigma=sig, q=q, kind=kind)
        args.update(kw)
        return bs_price(**args)

    h = 1e-6 * S
    delta = (price(S=S + h) - price(S=S - h)) / (2 * h)
    hg = 1e-4 * S
    gamma = (price(S=S + hg) - 2 * price() + price(S=S - hg)) / hg**2
    hv = 1e-7
    vega = (price(sigma=sig + hv) - price(sigma=sig - hv)) / (2 * hv)
    ht = 1e-7
    theta = -(price(T=T + ht) - price(T=T - ht)) / (2 * ht)
    hr = 1e-7
    rho = (price(r=r + hr) - price(r=r - hr)) / (2 * hr)
    np.testing.assert_allclose(g.delta, delta, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(g.gamma, gamma, rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(g.vega, vega, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(g.theta, theta, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(g.rho, rho, rtol=1e-6, atol=1e-6)


def test_greeks_hull_example_and_parity_relations():
    g = bs_greeks(49.0, 50.0, 0.3846, 0.05, 0.20)  # Hull, ch. 19 example
    assert round(g.delta, 3) == 0.522
    assert round(g.theta, 2) == -4.31
    assert round(g.rho, 2) == 8.91
    gp = bs_greeks(49.0, 50.0, 0.3846, 0.05, 0.20, kind="put")
    assert g.delta - gp.delta == pytest.approx(1.0, abs=1e-14)
    assert g.gamma == gp.gamma and g.vega == gp.vega
    assert set(g.to_dict()) == {"delta", "gamma", "vega", "theta", "rho"}


def test_greeks_degenerate_limits_without_warnings():
    g = bs_greeks(100.0, [80.0, 100.0, 120.0], 0.0, 0.05, 0.3)
    np.testing.assert_array_equal(g.delta, [1.0, 0.5, 0.0])
    np.testing.assert_array_equal(g.gamma, [0.0, np.inf, 0.0])
    np.testing.assert_array_equal(g.vega, [0.0, 0.0, 0.0])
    assert g.theta[0] == pytest.approx(-0.05 * 80.0) and g.theta[1] == -np.inf
    # sigma = 0, T > 0: deterministic forward; ATM-forward vega is the sigma -> 0+ limit
    T, r = 1.0, 0.0
    K = 100.0  # forward exactly at the money
    g = bs_greeks(100.0, [80.0, K, 150.0], T, r, 0.0)
    np.testing.assert_allclose(g.delta, [1.0, 0.5, 0.0])
    assert g.vega[1] == pytest.approx(100.0 / np.sqrt(2 * np.pi), rel=1e-12)
    assert bs_greeks(100.0, K, T, r, 1e-9).vega == pytest.approx(g.vega[1], rel=1e-6)
    np.testing.assert_allclose(g.rho, [80.0 * T * np.exp(-r * T), K * T * np.exp(-r * T) / 2, 0.0])
    # subnormal volatility: no overflow warnings
    tiny = bs_greeks(100.0, 90.0, 1.0, 0.0, 5e-324)
    assert tiny.delta == 1.0 and tiny.gamma == 0.0


# ---------------------------------------------------------------------------
# implied volatility
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["call", "put"])
def test_implied_volatility_round_trip_grid(kind):
    S, K, T, sig = _grid()
    r, q = 0.03, 0.01
    price = bs_price(S, K, T, r, sig, q, kind)
    vega = bs_greeks(S, K, T, r, sig, q, kind).vega
    lower = np.maximum((S * np.exp(-q * T) - K * np.exp(-r * T)) * (1 if kind == "call" else -1), 0)
    ok = price > lower  # prices numerically at the lower bound carry no information
    iv = implied_volatility(price[ok], S[ok], K[ok], T[ok], r, q, kind)
    repriced = bs_price(S[ok], K[ok], T[ok], r, iv, q, kind)
    np.testing.assert_allclose(repriced, price[ok], rtol=1e-12, atol=1e-13)
    well = vega[ok] > 1e-3
    assert well.sum() > 0.7 * ok.sum()
    np.testing.assert_allclose(iv[well], sig[ok][well], rtol=0, atol=1e-9)


def test_implied_volatility_bisection_fallback(monkeypatch):
    import lipipy.finance.options as options

    def broken_brentq(*args, **kwargs):
        raise ZeroDivisionError("float division by zero")

    monkeypatch.setattr(options, "brentq", broken_brentq)
    sig = np.array([0.05, 0.3, 1.5])
    price = bs_price(100.0, 110.0, 0.5, 0.02, sig)
    np.testing.assert_allclose(
        implied_volatility(price, 100.0, 110.0, 0.5, 0.02), sig, rtol=0, atol=1e-12
    )


def test_implied_volatility_bounds_and_errors():
    c = bs_price(100.0, 100.0, 1.0, 0.05, 0.2)
    assert implied_volatility(c, 100.0, 100.0, 1.0, 0.05) == pytest.approx(0.2, abs=1e-12)
    lower = 100.0 - 80.0 * np.exp(-0.05)
    assert implied_volatility(lower, 100.0, 80.0, 1.0, 0.05) == 0.0
    with pytest.raises(ValueError, match="no-arbitrage"):
        implied_volatility(lower - 1e-6, 100.0, 80.0, 1.0, 0.05)
    with pytest.raises(ValueError, match="no-arbitrage"):
        implied_volatility(100.0, 100.0, 80.0, 1.0, 0.05)  # = upper bound S
    with pytest.raises(ValueError, match="no-arbitrage"):
        implied_volatility(100.0, 100.0, 100.0, 1.0, 0.05, kind="put")  # > K e^{-rT}
    with pytest.raises(ValueError, match="T > 0"):
        implied_volatility(1.0, 100.0, 100.0, 0.0, 0.05)
    with pytest.raises(ValueError, match="no-arbitrage"):  # whole array rejected atomically
        implied_volatility([c, -1.0], 100.0, 100.0, 1.0, 0.05)
    with pytest.raises(ValueError, match="NaN"):
        implied_volatility(np.nan, 100.0, 100.0, 1.0, 0.05)
    with pytest.raises(ValueError, match="broadcast"):
        implied_volatility([1.0, 2.0], 100.0, [90.0, 100.0, 110.0], 1.0, 0.05)
    # one ulp below the upper bound is still bracketed (the price reaches the bound
    # in floating point at sigma sqrt(T) ~ 17, well inside the search range)
    near = np.nextafter(100.0, 0.0)
    iv = implied_volatility(near, 100.0, 100.0, 1.0, 0.0)
    assert 10.0 < iv < 64.0 and bs_price(100.0, 100.0, 1.0, 0.0, iv) == pytest.approx(near)
    # but a very short-dated option legitimately needs a huge volatility
    huge = bs_price(100.0, 100.0, 1e-10, 0.0, 2e5)
    assert implied_volatility(huge, 100.0, 100.0, 1e-10, 0.0) == pytest.approx(2e5, rel=1e-8)
    # huge volatility is still bracketed
    big = bs_price(100.0, 100.0, 1.0, 0.0, 8.0)
    assert implied_volatility(big, 100.0, 100.0, 1.0, 0.0) == pytest.approx(8.0, rel=1e-9)


# ---------------------------------------------------------------------------
# binomial tree
# ---------------------------------------------------------------------------
def test_binomial_hull_american_put_example():
    # Hull, example 21.1: S=50, K=50, r=10%, sigma=40%, T=5 months, 5 steps -> 4.49
    v = binomial_price(50.0, 50.0, 5 / 12, 0.10, 0.40, kind="put", steps=5, american=True)
    assert round(v, 2) == 4.49
    fine = binomial_price(50.0, 50.0, 5 / 12, 0.10, 0.40, kind="put", steps=2000, american=True)
    assert round(fine, 2) == 4.28  # Hull quotes 4.278 for many steps


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("q", [0.0, 0.03])
def test_binomial_converges_to_bs_at_rate_one_over_n(kind, q):
    args = (100.0, 105.0, 1.0, 0.05, 0.25, q, kind)
    exact = bs_price(*args)
    steps = np.array([50, 75, 100, 150, 200, 400, 800, 1600, 3200])
    err = np.array([binomial_price(*args, steps=int(n)) - exact for n in steps])
    scaled = steps * np.abs(err)
    # error O(1/N): N |err| stays bounded (CRR oscillates, so it is not monotone)
    assert scaled.max() < 2.0 * scaled[:4].max()
    assert abs(err[-1]) < 1e-3
    assert np.abs(err[-3:]).max() < np.abs(err[:3]).max() / 4


def test_american_vs_european_tree():
    for K in (80.0, 100.0, 120.0):
        eu = binomial_price(100.0, K, 1.0, 0.05, 0.3, kind="put", steps=400)
        am = binomial_price(100.0, K, 1.0, 0.05, 0.3, kind="put", steps=400, american=True)
        assert am >= eu
        assert am >= max(K - 100.0, 0.0)
        # no dividends: early exercise of a call is never optimal
        call_eu = binomial_price(100.0, K, 1.0, 0.05, 0.3, steps=400)
        call_am = binomial_price(100.0, K, 1.0, 0.05, 0.3, steps=400, american=True)
        assert call_am == pytest.approx(call_eu, rel=1e-13)
        # ... and the tree price is within its O(1/N) error of Black-Scholes
        assert call_am == pytest.approx(bs_price(100.0, K, 1.0, 0.05, 0.3), abs=5.0 / 400)
    deep = binomial_price(100.0, 200.0, 1.0, 0.05, 0.2, kind="put", steps=200, american=True)
    assert deep == pytest.approx(100.0, abs=1e-12)  # exercise immediately


def test_binomial_degenerate_and_errors():
    assert binomial_price(100.0, 90.0, 0.0, 0.05, 0.2) == 10.0
    assert binomial_price(100.0, 90.0, 0.0, 0.05, 0.2, kind="put") == 0.0
    # sigma = 0: deterministic forward
    v = binomial_price(100.0, 90.0, 1.0, 0.05, 0.0, steps=10)
    assert v == pytest.approx(bs_price(100.0, 90.0, 1.0, 0.05, 0.0), rel=1e-13)
    put = binomial_price(100.0, 120.0, 1.0, 0.05, 0.0, kind="put", steps=10, american=True)
    assert put == pytest.approx(20.0)  # exercise now beats the shrinking discounted payoff
    with pytest.raises(ValueError, match="increase steps"):
        binomial_price(100.0, 100.0, 1.0, 0.5, 0.05, steps=10)
    with pytest.raises(ValueError, match="steps"):
        binomial_price(100.0, 100.0, 1.0, 0.05, 0.2, steps=0)
    with pytest.raises(ValueError, match="scalar"):
        binomial_price([100.0, 110.0], 100.0, 1.0, 0.05, 0.2)


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["call", "put"])
def test_monte_carlo_within_three_standard_errors(kind):
    args = (100.0, 110.0, 0.75, 0.04, 0.3, 0.01, kind)
    exact = bs_price(*args)
    z = []
    for seed in range(40):
        res = monte_carlo_price(*args, n_paths=20_000, random_state=seed)
        z.append((res.price - exact) / res.std_error)
    z = np.abs(np.array(z))
    assert np.all(z < 4.0)
    assert np.mean(z < 3.0) >= 0.95


@pytest.mark.parametrize("antithetic,control", list(itertools.product([True, False], repeat=2)))
def test_monte_carlo_each_estimator_is_unbiased(antithetic, control):
    args = (42.0, 40.0, 0.5, 0.10, 0.20)
    res = monte_carlo_price(
        *args, n_paths=400_000, antithetic=antithetic, control_variate=control, random_state=7
    )
    assert abs(res.price - bs_price(*args)) < 4 * res.std_error
    assert res.n_samples == (200_000 if antithetic else 400_000)
    assert (res.control_coefficient is not None) == control


def test_variance_reduction_lowers_standard_error():
    args = (100.0, 100.0, 1.0, 0.05, 0.2)
    for seed in range(10):
        plain = monte_carlo_price(
            *args, n_paths=50_000, antithetic=False, control_variate=False, random_state=seed
        )
        anti = monte_carlo_price(*args, n_paths=50_000, control_variate=False, random_state=seed)
        both = monte_carlo_price(*args, n_paths=50_000, random_state=seed)
        assert anti.std_error < plain.std_error
        assert both.std_error < anti.std_error
        assert both.std_error < plain.std_error / 3


def test_monte_carlo_standard_error_is_calibrated():
    """The reported SE matches the spread of the estimates across seeds."""
    args = (100.0, 100.0, 1.0, 0.05, 0.2)
    for antithetic, control in [(False, False), (True, True)]:
        runs = [
            monte_carlo_price(
                *args, n_paths=4_000, antithetic=antithetic, control_variate=control, random_state=s
            )
            for s in range(300)
        ]
        spread = np.std([r.price for r in runs], ddof=1)
        mean_se = np.mean([r.std_error for r in runs])
        assert spread == pytest.approx(mean_se, rel=0.15)


def test_monte_carlo_chunking_and_determinism():
    args = (42.0, 40.0, 0.5, 0.10, 0.20)
    a = monte_carlo_price(*args, n_paths=100_000, random_state=3, chunk_size=1_000)
    b = monte_carlo_price(*args, n_paths=100_000, random_state=3, chunk_size=100_000)
    c = monte_carlo_price(*args, n_paths=100_000, random_state=3, chunk_size=777)
    assert a.price == pytest.approx(b.price, rel=1e-13)
    assert a.std_error == pytest.approx(b.std_error, rel=1e-10)
    assert c.price == pytest.approx(b.price, rel=1e-13)
    assert monte_carlo_price(*args, n_paths=1000, random_state=5) == monte_carlo_price(
        *args, n_paths=1000, random_state=5
    )
    rng = np.random.default_rng(0)
    assert monte_carlo_price(*args, n_paths=1000, random_state=rng).n_paths == 1000


def test_monte_carlo_degenerate_and_errors():
    res = monte_carlo_price(100.0, 90.0, 0.0, 0.05, 0.2, random_state=0)
    assert res.price == 10.0 and res.std_error == 0.0
    res = monte_carlo_price(100.0, 90.0, 1.0, 0.05, 0.0, kind="put", random_state=0)
    assert res.price == 0.0 and res.std_error == 0.0
    assert res.to_dict()["control_coefficient"] is None
    with pytest.raises(ValueError, match="even"):
        monte_carlo_price(100.0, 90.0, 1.0, 0.05, 0.2, n_paths=1001)
    with pytest.raises(ValueError, match="at least 3"):
        monte_carlo_price(100.0, 90.0, 1.0, 0.05, 0.2, n_paths=4)
    with pytest.raises(ValueError, match="chunk_size"):
        monte_carlo_price(100.0, 90.0, 1.0, 0.05, 0.2, chunk_size=1)
    with pytest.raises(TypeError):
        monte_carlo_price(100.0, 90.0, 1.0, 0.05, 0.2, random_state="seed")


# ---------------------------------------------------------------------------
# optional QuantLib parity (not installed in CI)
# ---------------------------------------------------------------------------
def test_quantlib_parity():
    ql = pytest.importorskip("QuantLib")
    today = ql.Date(15, 1, 2025)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    S, r, q, sigma = 100.0, 0.04, 0.015, 0.27
    spot = ql.QuoteHandle(ql.SimpleQuote(S))
    rts = ql.YieldTermStructureHandle(ql.FlatForward(today, r, dc))
    qts = ql.YieldTermStructureHandle(ql.FlatForward(today, q, dc))
    vts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), sigma, dc))
    process = ql.BlackScholesMertonProcess(spot, qts, rts, vts)
    for days, K, kind in itertools.product([30, 182, 730], [80.0, 100.0, 125.0], ["call", "put"]):
        T = days / 365.0
        payoff = ql.PlainVanillaPayoff(ql.Option.Call if kind == "call" else ql.Option.Put, K)
        euro = ql.VanillaOption(payoff, ql.EuropeanExercise(today + days))
        euro.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        assert bs_price(S, K, T, r, sigma, q, kind) == pytest.approx(euro.NPV(), rel=1e-10)
        g = bs_greeks(S, K, T, r, sigma, q, kind)
        assert g.delta == pytest.approx(euro.delta(), rel=1e-9, abs=1e-12)
        assert g.gamma == pytest.approx(euro.gamma(), rel=1e-9)
        assert g.vega == pytest.approx(euro.vega(), rel=1e-9)
        assert g.theta == pytest.approx(euro.theta(), rel=1e-9)
        assert g.rho == pytest.approx(euro.rho(), rel=1e-9, abs=1e-12)
        # QuantLib's CRR tree uses the first-order probability
        # p = 1/2 + (r - q - sigma^2/2) sqrt(dt) / (2 sigma) instead of the exact
        # (e^{(r-q)dt} - d) / (u - d), so the trees agree only to O(1/N).
        amer = ql.VanillaOption(payoff, ql.AmericanExercise(today, today + days))
        amer.setPricingEngine(ql.BinomialVanillaEngine(process, "crr", 801))
        ours = binomial_price(S, K, T, r, sigma, q, kind, steps=801, american=True)
        assert ours == pytest.approx(amer.NPV(), rel=1e-5, abs=1e-8)


def test_implied_volatility_never_rejects_bs_prices_random_grid():
    """Review regression: deep-ITM prices 1 ulp below the bound used to be rejected."""
    rng = np.random.default_rng(0)
    n = 4000
    S, K = rng.uniform(20, 200, n), rng.uniform(20, 200, n)
    T, r = rng.uniform(0.01, 5, n), rng.uniform(-0.02, 0.1, n)
    sig, q = rng.uniform(0.05, 1, n), rng.uniform(0, 0.05, n)
    scale = np.maximum(S * np.exp(-q * T), K * np.exp(-r * T))
    for kind in ("call", "put"):
        price = bs_price(S, K, T, r, sig, q, kind)
        iv = implied_volatility(price, S, K, T, r, q, kind)
        err = np.abs(bs_price(S, K, T, r, iv, q, kind) - price)
        assert np.all(err <= 1e-12 * scale)
