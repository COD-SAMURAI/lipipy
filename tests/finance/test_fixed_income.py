"""Tests for lipipy.finance.fixed_income: textbook values, derivatives, day counts, curves."""

import math
from datetime import date

import numpy as np
import pytest

from lipipy.finance import (
    NelsonSiegelResult,
    accrued_interest,
    bond_analytics,
    bond_price,
    bond_yield,
    bootstrap_zero_curve,
    nelson_siegel_fit,
)

pytestmark = pytest.mark.filterwarnings("error")

GRID = [
    (c, y, T, f, comp)
    for c in (0.0, 0.03, 0.08)
    for y in (-0.005, 0.01, 0.05, 0.15)
    for T in (0.3, 1.0, 4.75, 10.0, 30.0)
    for f in (1, 2, 4)
    for comp in (None, "continuous", 12)
]


# ---------------------------------------------------------------------------
# textbook values
# ---------------------------------------------------------------------------
def test_hull_table_4_6():
    """Hull, Options Futures and Other Derivatives, Table 4.6: 3-year 10% semiannual bond,
    yield 12% continuously compounded: price 94.213, duration 2.653 years."""
    a = bond_analytics(100.0, 0.10, 0.12, 3.0, compounding="continuous")
    assert round(a.price, 3) == 94.213
    assert round(a.macaulay_duration, 3) == 2.653
    assert a.modified_duration == a.macaulay_duration  # continuous compounding


def test_annuity_formula_and_par_bonds():
    # 10-year 8% semiannual bond at 10%: coupon annuity plus discounted principal
    price = bond_price(1000.0, 0.08, 0.10, 10.0)
    annuity = 40 * (1 - 1.05**-20) / 0.05 + 1000 * 1.05**-20
    assert price == pytest.approx(annuity, rel=1e-13)
    assert round(price, 2) == 875.38
    for f in (1, 2, 4, 12):
        assert bond_price(100.0, 0.07, 0.07, 6.0, f) == pytest.approx(100.0, rel=1e-13)
    # zero-coupon bond
    assert bond_price(100.0, 0.0, 0.04, 5.0, 1) == pytest.approx(100 / 1.04**5, rel=1e-14)


def test_between_coupon_dates_dirty_and_clean():
    # 2.25 years, semiannual: next coupon in 0.25 y (w = 0.5), accrued half a coupon
    dirty = bond_price(100.0, 0.06, 0.05, 2.25)
    clean = bond_price(100.0, 0.06, 0.05, 2.25, clean=True)
    assert dirty - clean == pytest.approx(1.5, abs=1e-13)
    t = np.array([0.25, 0.75, 1.25, 1.75, 2.25])
    cf = np.array([3, 3, 3, 3, 103.0])
    assert dirty == pytest.approx(cf @ 1.025 ** (-2 * t), rel=1e-14)
    # clean price is continuous across a coupon date, dirty jumps by the coupon
    eps = 1e-9
    before = bond_price(100.0, 0.06, 0.05, 2.0 + eps, clean=True)
    after = bond_price(100.0, 0.06, 0.05, 2.0, clean=True)
    assert before == pytest.approx(after, abs=1e-6)


# ---------------------------------------------------------------------------
# yields, durations, convexity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("c,y,T,f,comp", GRID[::3])
def test_yield_round_trip(c, y, T, f, comp):
    p = bond_price(100.0, c, y, T, f, comp)
    y2 = bond_yield(p, 100.0, c, T, f, comp)
    assert bond_price(100.0, c, y2, T, f, comp) == pytest.approx(p, rel=1e-12)
    assert y2 == pytest.approx(y, abs=1e-10)
    pc = bond_price(100.0, c, y, T, f, comp, clean=True)
    assert bond_yield(pc, 100.0, c, T, f, comp, clean=True) == pytest.approx(y, abs=1e-10)


@pytest.mark.parametrize("c,y,T,f,comp", GRID)
def test_duration_and_convexity_equal_finite_differences(c, y, T, f, comp):
    a = bond_analytics(100.0, c, y, T, f, comp)
    h = 1e-5
    up, mid, dn = (bond_price(100.0, c, y + s, T, f, comp) for s in (h, 0.0, -h))
    assert a.price == pytest.approx(mid, rel=1e-14)
    assert a.modified_duration == pytest.approx(-(up - dn) / (2 * h) / mid, rel=1e-7)
    assert a.convexity == pytest.approx((up - 2 * mid + dn) / h**2 / mid, rel=2e-4, abs=1e-6)
    assert a.dv01 == pytest.approx(a.modified_duration * mid * 1e-4, rel=1e-14)
    assert 0 < a.macaulay_duration <= T + 1e-12  # at most the maturity (zero coupon: equal)
    if c == 0.0:
        assert a.macaulay_duration == pytest.approx(T, rel=1e-12)


def test_bond_validation():
    with pytest.raises(ValueError, match="ytm"):
        bond_price(100.0, 0.05, -2.5, 5.0)  # below -compounding
    with pytest.raises(ValueError, match="maturity"):
        bond_price(100.0, 0.05, 0.05, 0.0)
    with pytest.raises(ValueError, match="coupon_rate"):
        bond_price(100.0, -0.05, 0.05, 1.0)
    with pytest.raises(ValueError, match="price"):
        bond_yield(-1.0, 100.0, 0.05, 5.0)
    with pytest.raises(TypeError):
        bond_price("100", 0.05, 0.05, 5.0)
    with pytest.raises(ValueError, match="compounding"):
        bond_price(100.0, 0.05, 0.05, 5.0, compounding=0)


# ---------------------------------------------------------------------------
# accrued interest
# ---------------------------------------------------------------------------
def test_day_counts_hand_computed():
    last, settle, nxt = date(2024, 1, 31), date(2024, 3, 31), date(2024, 7, 31)
    # 30/360 bond basis: D1 31 -> 30; D2 31 -> 30 since D1 >= 30: 2 months = 60 days
    assert accrued_interest(100.0, 0.06, last, settle, nxt, day_count="30/360") == pytest.approx(
        1.0
    )
    # 30E/360: same here
    assert accrued_interest(100.0, 0.06, last, settle, nxt, day_count="30E/360") == pytest.approx(
        1.0
    )
    # actual: 60 days (2024 is a leap year: Feb has 29 days)
    assert accrued_interest(100.0, 0.06, last, settle, nxt, day_count="act/360") == pytest.approx(
        6 * 60 / 360
    )
    assert accrued_interest(100.0, 0.06, last, settle, nxt, day_count="act/365f") == pytest.approx(
        6 * 60 / 365
    )
    # act/act ICMA: 60 / 182 days of the period, times half a year
    assert accrued_interest(100.0, 0.06, last, settle, nxt, 2, "act/act-icma") == pytest.approx(
        6 * 60 / 182 / 2
    )


def test_30_360_rules_differ_where_they_should():
    last, nxt = date(2024, 1, 15), date(2024, 7, 15)
    # D2 = 31 with D1 = 15: bond basis keeps 31, 30E caps it at 30
    a = accrued_interest(360.0, 1.0, last, date(2024, 3, 31), nxt, day_count="30/360")
    b = accrued_interest(360.0, 1.0, last, date(2024, 3, 31), nxt, day_count="30E/360")
    assert (a, b) == (pytest.approx(76.0), pytest.approx(75.0))
    assert accrued_interest(100.0, 0.06, last, last, nxt) == 0.0
    with pytest.raises(ValueError, match="settlement"):
        accrued_interest(100.0, 0.06, last, nxt, nxt)
    with pytest.raises(ValueError, match="day_count"):
        accrued_interest(100.0, 0.06, last, last, nxt, day_count="bus/252")
    with pytest.raises(TypeError):
        accrued_interest(100.0, 0.06, "2024-01-15", last, nxt)


# ---------------------------------------------------------------------------
# bootstrapping
# ---------------------------------------------------------------------------
def _par_price(curve, T, c, f):
    times = np.arange(1, round(T * f) + 1) / f
    df = curve.discount(times)
    return c / f * df.sum() + df[-1]


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("f", [1, 2, 4])
def test_bootstrap_reprices_every_instrument(seed, f):
    rng = np.random.default_rng(seed)
    mats = np.array([0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30], dtype=float)
    mats = mats[np.abs(mats * f - np.round(mats * f)) < 1e-12]
    rates = 0.03 + 0.02 * (1 - np.exp(-mats / 5)) + rng.normal(0, 0.003, mats.size)
    curve = bootstrap_zero_curve(mats, rates, f)
    for T, c in zip(mats, rates, strict=True):
        assert _par_price(curve, T, c, f) == pytest.approx(1.0, abs=1e-10)


def test_bootstrap_flat_and_consecutive_closed_form():
    # flat par curve with annual coupons on every coupon date: DF_n = (1 - c sum DF) / (1 + c)
    c = 0.05
    curve = bootstrap_zero_curve([1, 2, 3, 4], [c] * 4, 1)
    np.testing.assert_allclose(curve.discount([1, 2, 3, 4]), 1.05 ** -np.arange(1, 5), rtol=1e-13)
    np.testing.assert_allclose(curve.zero_rates, math.log(1.05), rtol=1e-12)
    assert set(curve.to_dict()) == {"maturities", "zero_rates"}


def test_bootstrap_negative_rates_and_errors():
    curve = bootstrap_zero_curve([0.5, 1.0, 2.0], [-0.005, -0.002, 0.001])
    for T, c in zip([0.5, 1.0, 2.0], [-0.005, -0.002, 0.001], strict=True):
        assert _par_price(curve, T, c, 2) == pytest.approx(1.0, abs=1e-10)
    with pytest.raises(ValueError, match="increasing"):
        bootstrap_zero_curve([1.0, 1.0], [0.03, 0.03])
    with pytest.raises(ValueError, match="multiples"):
        bootstrap_zero_curve([0.3, 1.0], [0.03, 0.03])
    with pytest.raises(ValueError, match="NaN"):
        bootstrap_zero_curve([0.5, 1.0], [0.03, np.nan])


# ---------------------------------------------------------------------------
# Nelson-Siegel
# ---------------------------------------------------------------------------
T_NS = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])


@pytest.mark.parametrize(
    "b0,b1,b2,tau",
    [(0.05, -0.02, 0.01, 1.8), (0.03, 0.01, -0.02, 0.6), (0.04, -0.03, 0.05, 4.0)],
)
def test_nelson_siegel_recovers_noise_free_curves(b0, b1, b2, tau):
    true = NelsonSiegelResult(b0, b1, b2, tau, 0.0)
    fit = nelson_siegel_fit(T_NS, true(T_NS))
    assert fit.tau == pytest.approx(tau, rel=1e-6)
    np.testing.assert_allclose([fit.beta0, fit.beta1, fit.beta2], [b0, b1, b2], atol=1e-8)
    assert fit.rmse < 1e-10


def test_nelson_siegel_noisy_fit_is_a_least_squares_minimum():
    rng = np.random.default_rng(3)
    true = NelsonSiegelResult(0.045, -0.015, 0.02, 2.5, 0.0)
    y = true(T_NS) + rng.normal(0, 5e-4, T_NS.size)
    fit = nelson_siegel_fit(T_NS, y)
    sse = np.sum((fit(T_NS) - y) ** 2)
    for tau in np.geomspace(0.1, 20, 200):  # no tau does better with refitted betas
        X = np.column_stack([np.ones(10), -np.expm1(-T_NS / tau) / (T_NS / tau)])
        X = np.column_stack([X, X[:, 1] - np.exp(-T_NS / tau)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        assert np.sum((X @ beta - y) ** 2) >= sse * (1 - 1e-9)
    with pytest.raises(ValueError, match="at least 4"):
        nelson_siegel_fit([1.0, 2.0], [0.01, 0.02])
    with pytest.raises(ValueError, match="positive"):
        nelson_siegel_fit([0.0, 1, 2, 3], [0.01] * 4)


def test_yield_bracket_extremes():
    # yields close to -compounding (huge prices) and very high yields (tiny prices)
    for y in (-1.995, -0.9, 5.0, 40.0):
        p = bond_price(100.0, 0.05, y, 3.0)
        assert bond_yield(p, 100.0, 0.05, 3.0) == pytest.approx(y, rel=1e-9)
    p = bond_price(100.0, 0.05, -0.9, 3.0, compounding="continuous")
    assert bond_yield(p, 100.0, 0.05, 3.0, compounding="continuous") == pytest.approx(
        -0.9, rel=1e-9
    )
    with pytest.raises(ValueError, match="too low"):
        bond_yield(1e-300, 100.0, 0.05, 3.0)
    with pytest.raises(ValueError, match="too high"):
        bond_yield(1e300, 100.0, 0.05, 3.0)
