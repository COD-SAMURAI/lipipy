r"""Fixed income: bond price and yield, duration and convexity, accrued interest,
zero-curve bootstrapping and Nelson-Siegel fitting.

Conventions
-----------
* A bond pays ``coupon_rate * face / frequency`` every :math:`1/f` years and ``face``
  at maturity. ``maturity`` is the time to maturity in years; the next coupon is
  :math:`w/f` years away with :math:`w = \text{maturity}\cdot f - (n - 1) \in (0, 1]` and
  :math:`n = \lceil \text{maturity}\cdot f\rceil` coupons remaining, so cash flows are
  at :math:`t_k = (k - 1 + w)/f`, :math:`k = 1..n`.
* Prices are **dirty** (full) unless ``clean=True``: clean = dirty - accrued, with
  accrued interest :math:`(1 - w)\cdot\text{coupon}/f` (linear in time). For accrued
  interest between actual dates use :func:`accrued_interest`.
* Yields are quoted with compounding ``m`` (``compounding``; default = ``frequency``),
  discount factor :math:`(1 + y/m)^{-mt}`, or :math:`e^{-yt}` for
  ``compounding="continuous"``.

.. math::
    P = \sum_k CF_k\,DF(t_k),\quad
    D_{mac} = \frac1P\sum_k t_k CF_k DF(t_k),\quad
    D_{mod} = -\frac{1}{P}\frac{\partial P}{\partial y} = \frac{D_{mac}}{1 + y/m},\quad
    C = \frac1P\frac{\partial^2P}{\partial y^2} = \frac{1}{P}\sum_k
        \frac{t_k(t_k + 1/m)}{(1+y/m)^2}CF_k DF(t_k)

(continuous compounding: :math:`D_{mod} = D_{mac}`, :math:`C = \sum t_k^2 CF_kDF/P`).
The yield from a price solves :math:`P(y) = \text{price}` with
:func:`lipipy.numerics.brentq` on a bracket found by expanding from
:math:`[-0.99m, 1]` (:math:`P` is strictly decreasing in :math:`y`).

Day counts (:func:`accrued_interest`)
-------------------------------------
* ``"30/360"`` (ISDA 2006 §4.16(f), "30/360 bond basis"): :math:`D_1 \to 30` if 31;
  :math:`D_2 \to 30` if 31 and :math:`D_1 \ge 30`; days
  :math:`360\Delta Y + 30\Delta M + \Delta D`, year fraction days/360.
* ``"30E/360"`` (ISDA §4.16(g), Eurobond): both day numbers capped at 30.
* ``"act/360"``, ``"act/365f"``: actual days / 360 or 365.
* ``"act/act-icma"``: actual days since the last coupon / actual days in the coupon
  period, times :math:`1/f` (ICMA rule 251, regular periods).

Accrued interest = face x coupon_rate x year fraction (for act/act-icma the year
fraction is that ratio times :math:`1/f`).

Bootstrapping
-------------
Input: par yields of bullet bonds paying ``frequency`` coupons per year, maturities
that are multiples of :math:`1/f`, increasing. The curve stores continuously
compounded zero rates at the input maturities, **linear in the zero rate** between
knots and flat before the first one. Each instrument, in order, fixes the zero rate
at its own maturity: the par condition :math:`c/f\sum_k DF(t_k) + DF(T) = 1` with
earlier knots fixed is a monotone equation in the new knot (coupon dates between the
previous and the new knot are interpolated with it), solved by
:func:`lipipy.numerics.brentq` to :math:`10^{-15}`. Every input instrument is repriced
to par within :math:`10^{-10}`.

Nelson-Siegel
-------------
:math:`y(t) = \beta_0 + \beta_1\frac{1 - e^{-t/\tau}}{t/\tau} + \beta_2\Bigl(\frac{1 -
e^{-t/\tau}}{t/\tau} - e^{-t/\tau}\Bigr)`. For fixed :math:`\tau` the betas are a
linear least-squares problem (Householder QR); :math:`\tau` minimises the profile sum
of squares, found by a grid over :math:`\ln\tau` refined with
:func:`lipipy.numerics.minimize` (BFGS, numerical gradient).

References
----------
F. J. Fabozzi, *Bond Markets, Analysis, and Strategies*, 9th ed. (2016), ch. 2-4.
J. C. Hull, *Options, Futures and Other Derivatives*, 10th ed. (2018), ch. 4
(Table 4.6: price 94.213 and duration 2.653 years). ISDA, *2006 ISDA Definitions*,
§4.16. ICMA, *Primary Market Handbook* / rule 251 (actual/actual). C. R. Nelson &
A. F. Siegel, "Parsimonious modeling of yield curves", *J. Business* 60 (1987)
473-489.
"""

from __future__ import annotations

import datetime as _dt
import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from lipipy.core.complexity import Complexity
from lipipy.core.errors import ConvergenceWarning
from lipipy.core.registry import register
from lipipy.finance._common import (
    DictMixin,
    as_float_array,
    check_finite,
    check_int,
    check_positive_number,
    check_real,
)
from lipipy.numerics.linalg import lstsq_qr
from lipipy.numerics.optimize import brentq, minimize

__all__ = [
    "BondAnalytics",
    "NelsonSiegelResult",
    "ZeroCurve",
    "accrued_interest",
    "bond_analytics",
    "bond_price",
    "bond_yield",
    "bootstrap_zero_curve",
    "nelson_siegel_fit",
]

_FABOZZI = "F. J. Fabozzi, Bond Markets, Analysis, and Strategies, 9th ed. (2016)"
_HULL = "J. C. Hull, Options, Futures and Other Derivatives, 10th ed. (2018), ch. 4"
_DAY_COUNTS = ("30/360", "30E/360", "act/360", "act/365f", "act/act-icma")


# ---------------------------------------------------------------------------
# cash flows and discounting
# ---------------------------------------------------------------------------
def _schedule(face: Any, coupon_rate: Any, maturity: Any, frequency: Any):
    face = check_positive_number(face, "face")
    c = check_real(coupon_rate, "coupon_rate")
    if c < 0.0:
        raise ValueError(f"coupon_rate must be non-negative, got {coupon_rate!r}")
    mat = check_positive_number(maturity, "maturity")
    f = check_int(frequency, "frequency", 1)
    periods = mat * f
    n = max(1, math.ceil(periods - 1e-9))
    w = periods - (n - 1)
    t = (np.arange(n) + w) / f
    cf = np.full(n, face * c / f)
    cf[-1] += face
    accrued = (1.0 - w) * face * c / f
    return t, cf, accrued, f


def _compounding(compounding: Any, frequency: int) -> float | None:
    if compounding is None:
        return float(frequency)
    if compounding == "continuous":
        return None
    return float(check_int(compounding, "compounding", 1))


def _df(y: float, t: np.ndarray, m: float | None) -> np.ndarray:
    if m is None:
        return np.exp(-y * t)
    return np.exp(-m * t * math.log1p(y / m))


def _price(y: float, t: np.ndarray, cf: np.ndarray, m: float | None) -> float:
    if m is not None and y <= -m:
        return math.inf
    return float(cf @ _df(y, t, m))


@register(
    domain="finance",
    tasks=("fixed-income", "bond-pricing"),
    complexity=Complexity(time="O(n)", space="O(n)", notes="n remaining coupons"),
    references=(_FABOZZI + ", ch. 2", _HULL),
)
def bond_price(
    face: float,
    coupon_rate: float,
    ytm: float,
    maturity: float,
    frequency: int = 2,
    compounding: int | str | None = None,
    clean: bool = False,
) -> float:
    r"""Price of a fixed-coupon bullet bond from its yield to maturity.

    See the module docstring for the schedule, accrued interest and compounding.

    Parameters
    ----------
    face : float
        Principal (> 0).
    coupon_rate : float
        Annual coupon rate (>= 0), paid ``frequency`` times a year.
    ytm : float
        Yield to maturity.
    maturity : float
        Years to maturity (> 0).
    frequency : int, default=2
        Coupons per year.
    compounding : int or "continuous", optional
        Yield compounding frequency; default ``frequency``.
    clean : bool, default=False
        Return the clean price (dirty minus accrued interest).

    Returns
    -------
    float

    Complexity
    ----------
    Time and space O(n) for n remaining coupons.

    References
    ----------
    Fabozzi (2016), ch. 2; Hull (2018), ch. 4.

    Examples
    --------
    >>> round(bond_price(100.0, 0.10, 0.12, 3.0, compounding="continuous"), 3)  # Hull 4.6
    94.213
    >>> round(bond_price(100.0, 0.06, 0.06, 5.0), 10)  # a par bond on a coupon date
    100.0
    """
    t, cf, accrued, f = _schedule(face, coupon_rate, maturity, frequency)
    y = check_real(ytm, "ytm")
    m = _compounding(compounding, f)
    if m is not None and y <= -m:
        raise ValueError(f"ytm must exceed -compounding = {-m}, got {ytm!r}")
    dirty = _price(y, t, cf, m)
    return dirty - accrued if clean else dirty


@register(
    domain="finance",
    tasks=("fixed-income", "bond-pricing"),
    complexity=Complexity(time="O(n I)", space="O(n)", notes="I Brent iterations"),
    references=(_FABOZZI + ", ch. 3", "R. P. Brent (1973)"),
)
def bond_yield(
    price: float,
    face: float,
    coupon_rate: float,
    maturity: float,
    frequency: int = 2,
    compounding: int | str | None = None,
    clean: bool = False,
) -> float:
    r"""Yield to maturity from a bond price (Brent's method on a guaranteed bracket).

    Parameters
    ----------
    price : float
        Dirty price (clean if ``clean=True``), > 0.
    face, coupon_rate, maturity, frequency, compounding
        As in :func:`bond_price`.
    clean : bool, default=False
        Whether ``price`` is clean.

    Returns
    -------
    float
        :math:`y` with ``bond_price(..., y) == price`` to within
        :math:`10^{-12}` relative (the solver's ``xtol`` is :math:`10^{-15}`).

    Complexity
    ----------
    O(n) per evaluation, I Brent iterations.

    References
    ----------
    Fabozzi (2016), ch. 3.

    Examples
    --------
    >>> p = bond_price(100.0, 0.05, 0.07, 10.0)
    >>> round(bond_yield(p, 100.0, 0.05, 10.0), 12)
    0.07
    """
    t, cf, accrued, f = _schedule(face, coupon_rate, maturity, frequency)
    target = check_positive_number(price, "price") + (accrued if clean else 0.0)
    m = _compounding(compounding, f)
    g: Callable[[float], float] = lambda y: _price(y, t, cf, m) - target  # noqa: E731
    lo = -0.99 * (m if m is not None else 1.0)
    while g(lo) <= 0.0:  # price at the low end must exceed the target
        if m is not None:
            lo = -m + (lo + m) / 10.0
        else:
            lo *= 2.0
        if (m is not None and lo + m < 1e-12) or lo < -1e6:
            raise ValueError(f"price {price!r} is too high for any yield")
    hi = 1.0
    while g(hi) >= 0.0:
        hi *= 2.0
        if hi > 1e6:
            raise ValueError(f"price {price!r} is too low for any yield")
    return float(brentq(g, lo, hi, xtol=1e-15).root)


@dataclass(frozen=True)
class BondAnalytics(DictMixin):
    """Price and first/second-order yield sensitivities of a bond.

    Attributes
    ----------
    price : float
        Dirty price.
    macaulay_duration : float
        In years.
    modified_duration : float
        :math:`-\\frac{1}{P}\\partial P/\\partial y`.
    convexity : float
        :math:`\\frac{1}{P}\\partial^2 P/\\partial y^2`.
    dv01 : float
        Price change for a 1 bp fall in yield, :math:`D_{mod} P \\times 10^{-4}`.
    """

    price: float
    macaulay_duration: float
    modified_duration: float
    convexity: float
    dv01: float


@register(
    domain="finance",
    tasks=("fixed-income", "risk"),
    complexity=Complexity(time="O(n)", space="O(n)"),
    references=(_FABOZZI + ", ch. 4", _HULL),
)
def bond_analytics(
    face: float,
    coupon_rate: float,
    ytm: float,
    maturity: float,
    frequency: int = 2,
    compounding: int | str | None = None,
) -> BondAnalytics:
    r"""Macaulay and modified duration, convexity and DV01 (module docstring formulas).

    Parameters
    ----------
    face, coupon_rate, ytm, maturity, frequency, compounding
        As in :func:`bond_price`.

    Returns
    -------
    BondAnalytics

    Complexity
    ----------
    Time and space O(n).

    References
    ----------
    Fabozzi (2016), ch. 4; Hull (2018), §4.8.

    Examples
    --------
    >>> a = bond_analytics(100.0, 0.10, 0.12, 3.0, compounding="continuous")
    >>> round(a.macaulay_duration, 3)  # Hull, Table 4.6
    2.653
    """
    t, cf, _, f = _schedule(face, coupon_rate, maturity, frequency)
    y = check_real(ytm, "ytm")
    m = _compounding(compounding, f)
    if m is not None and y <= -m:
        raise ValueError(f"ytm must exceed -compounding = {-m}, got {ytm!r}")
    pv = cf * _df(y, t, m)
    P = float(pv.sum())
    mac = float(t @ pv) / P
    if m is None:
        mod = mac
        conv = float((t * t) @ pv) / P
    else:
        mod = mac / (1.0 + y / m)
        conv = float((t * (t + 1.0 / m)) @ pv) / (P * (1.0 + y / m) ** 2)
    return BondAnalytics(P, mac, mod, conv, mod * P * 1e-4)


@register(
    domain="finance",
    tasks=("fixed-income",),
    complexity=Complexity(time="O(1)", space="O(1)"),
    references=("ISDA, 2006 ISDA Definitions, §4.16", "ICMA rule 251 (actual/actual)"),
)
def accrued_interest(
    face: float,
    coupon_rate: float,
    last_coupon: _dt.date,
    settlement: _dt.date,
    next_coupon: _dt.date,
    frequency: int = 2,
    day_count: str = "30/360",
) -> float:
    r"""Accrued interest from the last coupon date to settlement.

    See the module docstring for the supported day-count conventions.

    Parameters
    ----------
    face, coupon_rate : float
        Principal and annual coupon rate.
    last_coupon, settlement, next_coupon : datetime.date
        ``last_coupon <= settlement < next_coupon``.
    frequency : int, default=2
        Coupons per year (used by act/act-icma).
    day_count : {"30/360", "30E/360", "act/360", "act/365f", "act/act-icma"}
        Day-count convention.

    Returns
    -------
    float

    Complexity
    ----------
    O(1).

    References
    ----------
    ISDA (2006), §4.16; ICMA rule 251.

    Examples
    --------
    >>> from datetime import date
    >>> round(accrued_interest(100.0, 0.06, date(2024, 1, 31), date(2024, 3, 31),
    ...                        date(2024, 7, 31)), 10)  # 30/360: 60 days
    1.0
    """
    face = check_positive_number(face, "face")
    c = check_real(coupon_rate, "coupon_rate")
    f = check_int(frequency, "frequency", 1)
    for name, d in (("last_coupon", last_coupon), ("settlement", settlement),
                    ("next_coupon", next_coupon)):  # fmt: skip
        if not isinstance(d, _dt.date):
            raise TypeError(f"{name} must be a datetime.date, got {type(d).__name__}")
    if not last_coupon <= settlement < next_coupon:
        raise ValueError("need last_coupon <= settlement < next_coupon")
    if day_count not in _DAY_COUNTS:
        raise ValueError(f"day_count must be one of {_DAY_COUNTS}, got {day_count!r}")
    if day_count in ("30/360", "30E/360"):
        d1, d2 = last_coupon.day, settlement.day
        if day_count == "30/360":
            d1 = 30 if d1 == 31 else d1
            d2 = 30 if d2 == 31 and d1 >= 30 else d2
        else:
            d1, d2 = min(d1, 30), min(d2, 30)
        days = (
            360 * (settlement.year - last_coupon.year)
            + 30 * (settlement.month - last_coupon.month)
            + (d2 - d1)
        )
        frac = days / 360.0
    elif day_count == "act/360":
        frac = (settlement - last_coupon).days / 360.0
    elif day_count == "act/365f":
        frac = (settlement - last_coupon).days / 365.0
    else:
        frac = (settlement - last_coupon).days / (next_coupon - last_coupon).days / f
    return face * c * frac


# ---------------------------------------------------------------------------
# zero curve
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ZeroCurve(DictMixin):
    """Continuously compounded zero curve, linear in the zero rate between knots.

    Attributes
    ----------
    maturities : ndarray of shape (N,)
        Knot times in years, increasing.
    zero_rates : ndarray of shape (N,)
        Continuously compounded zero rates at the knots.
    """

    maturities: np.ndarray
    zero_rates: np.ndarray

    def zero_rate(self, t: ArrayLike) -> np.ndarray:
        """Interpolated zero rate (flat before the first and after the last knot)."""
        tt = as_float_array(t, "t")
        return np.interp(tt, self.maturities, self.zero_rates)

    def discount(self, t: ArrayLike) -> np.ndarray:
        """Discount factor :math:`e^{-z(t)t}`."""
        tt = as_float_array(t, "t")
        return np.exp(-self.zero_rate(tt) * tt)


def _par_residual(
    knots_t: list[float], knots_z: list[float], T: float, c: float, f: int
) -> Callable[[float], float]:
    times = np.arange(1, round(T * f) + 1) / f

    def g(z_new: float) -> float:
        zr = np.interp(times, [*knots_t, T], [*knots_z, z_new])
        df = np.exp(-zr * times)
        return float(c / f * df.sum() + df[-1] - 1.0)

    return g


@register(
    domain="finance",
    tasks=("fixed-income", "yield-curve"),
    complexity=Complexity(
        time="O(N^2 f I)", space="O(N f)", notes="N instruments, I Brent iterations"
    ),
    references=(_HULL + " (§4.7, bootstrap)", _FABOZZI + ", ch. 5"),
)
def bootstrap_zero_curve(
    maturities: ArrayLike, par_rates: ArrayLike, frequency: int = 2
) -> ZeroCurve:
    r"""Bootstrap a zero curve from par yields of bullet bonds (module docstring).

    Parameters
    ----------
    maturities : array_like of shape (N, )
        Increasing maturities in years, each a multiple of ``1/frequency``.
    par_rates : array_like of shape (N, )
        Par coupon rates (annual, paid ``frequency`` times a year).
    frequency : int, default=2
        Coupons per year.

    Returns
    -------
    ZeroCurve

    Complexity
    ----------
    O(N^2 f I) for N instruments (each root solve reprices up to N f coupons).

    References
    ----------
    Hull (2018), §4.7; Fabozzi (2016), ch. 5.

    Examples
    --------
    >>> curve = bootstrap_zero_curve([0.5, 1.0, 2.0], [0.04, 0.04, 0.04])
    >>> curve.zero_rates.round(6)  # flat 4% semiannual par curve
    array([0.039605, 0.039605, 0.039605])
    """
    T = as_float_array(maturities, "maturities")
    R = as_float_array(par_rates, "par_rates")
    if T.ndim != 1 or T.shape != R.shape or T.size == 0:
        raise ValueError("maturities and par_rates must be non-empty 1-D arrays of equal length")
    check_finite(T, "maturities")
    check_finite(R, "par_rates")
    f = check_int(frequency, "frequency", 1)
    if np.any(np.diff(T) <= 0) or T[0] <= 0:
        raise ValueError("maturities must be positive and strictly increasing")
    if np.any(np.abs(T * f - np.round(T * f)) > 1e-9):
        raise ValueError("maturities must be multiples of 1 / frequency")
    if np.any(R <= -f):
        raise ValueError("par rates must exceed -frequency")
    kt: list[float] = []
    kz: list[float] = []
    for Ti, c in zip(T.tolist(), R.tolist(), strict=True):
        g = _par_residual(kt, kz, Ti, c, f)
        lo, hi = -1.0, 1.0
        while g(hi) > 0.0:  # residual decreases in the zero rate
            hi *= 2.0
            if hi > 1e3:
                raise ValueError(f"cannot bracket the zero rate at maturity {Ti}")
        while g(lo) < 0.0:
            lo *= 2.0
            if lo < -1e3:
                raise ValueError(f"cannot bracket the zero rate at maturity {Ti}")
        kz.append(float(brentq(g, lo, hi, xtol=1e-15).root))
        kt.append(Ti)
    return ZeroCurve(np.array(kt), np.array(kz))


# ---------------------------------------------------------------------------
# Nelson-Siegel
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NelsonSiegelResult(DictMixin):
    """Nelson-Siegel fit: ``beta0`` (level), ``beta1`` (slope), ``beta2`` (curvature),
    ``tau`` (> 0) and the root-mean-square fitting error ``rmse``."""

    beta0: float
    beta1: float
    beta2: float
    tau: float
    rmse: float

    def __call__(self, t: ArrayLike) -> np.ndarray:
        """Fitted yields at maturities ``t`` (> 0)."""
        X = _ns_design(as_float_array(t, "t").reshape(-1), self.tau)
        return X @ np.array([self.beta0, self.beta1, self.beta2])


def _ns_design(t: np.ndarray, tau: float) -> np.ndarray:
    x = t / tau
    load = -np.expm1(-x) / x
    return np.column_stack([np.ones_like(t), load, load - np.exp(-x)])


def _ns_profile(t: np.ndarray, y: np.ndarray, log_tau: float) -> tuple[float, np.ndarray]:
    X = _ns_design(t, math.exp(log_tau))
    beta = lstsq_qr(X, y)
    r = y - X @ beta
    return float(r @ r), beta


@register(
    domain="finance",
    tasks=("fixed-income", "yield-curve", "curve-fitting"),
    complexity=Complexity(time="O(n G + n I)", space="O(n)", notes="G grid points, I iterations"),
    references=("C. R. Nelson & A. F. Siegel, J. Business 60 (1987) 473-489",),
)
def nelson_siegel_fit(maturities: ArrayLike, yields: ArrayLike) -> NelsonSiegelResult:
    r"""Fit the Nelson-Siegel yield curve by profiled least squares (module docstring).

    Parameters
    ----------
    maturities : array_like of shape (n, )
        Positive maturities in years (n >= 4).
    yields : array_like of shape (n, )
        Observed yields.

    Returns
    -------
    NelsonSiegelResult
        Callable: ``result(t)`` evaluates the fitted curve.

    Complexity
    ----------
    O(n) per profile evaluation; G = 60 grid points plus I refinement iterations.

    References
    ----------
    Nelson & Siegel (1987).

    Examples
    --------
    >>> import numpy as np
    >>> t = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    >>> true = NelsonSiegelResult(0.05, -0.02, 0.01, 1.8, 0.0)
    >>> fit = nelson_siegel_fit(t, true(t))
    >>> round(fit.tau, 6), round(fit.rmse, 12)
    (1.8, 0.0)
    """
    t = as_float_array(maturities, "maturities")
    y = as_float_array(yields, "yields")
    if t.ndim != 1 or t.shape != y.shape or t.size < 4:
        raise ValueError("need at least 4 maturities and yields of equal length")
    check_finite(t, "maturities")
    check_finite(y, "yields")
    if np.any(t <= 0):
        raise ValueError("maturities must be positive")
    grid = np.linspace(math.log(0.05), math.log(30.0), 60)
    best = min(grid, key=lambda lt: _ns_profile(t, y, float(lt))[0])

    def obj(x: np.ndarray) -> float:
        return _ns_profile(t, y, float(x[0]))[0] / t.size

    def jac(x: np.ndarray) -> np.ndarray:
        h = 1e-6
        return np.array([(obj(x + h) - obj(x - h)) / (2 * h)])

    with warnings.catch_warnings():  # a flat profile (e.g. exact fit) is not a failure here
        warnings.simplefilter("ignore", ConvergenceWarning)
        res = minimize(obj, np.array([best]), jac, method="bfgs", tol=1e-14)
    lt = float(res.x[0]) if res.fun <= obj(np.array([best])) else float(best)
    ssr, beta = _ns_profile(t, y, lt)
    return NelsonSiegelResult(
        float(beta[0]), float(beta[1]), float(beta[2]), math.exp(lt), math.sqrt(ssr / t.size)
    )
