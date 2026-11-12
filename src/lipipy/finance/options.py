r"""European and American option pricing: Black-Scholes-Merton, binomial trees, Monte Carlo.

Conventions
-----------
* ``S`` spot, ``K`` strike, ``T`` time to expiry in **years**, ``r`` the
  continuously compounded risk-free rate, ``q`` the continuous dividend yield,
  ``sigma`` the annualised volatility; ``kind`` is ``"call"`` or ``"put"``.
* Prices and Greeks are per unit of the underlying.
* Greeks: ``vega`` per unit of volatility (divide by 100 for "per vol point"),
  ``theta`` = :math:`\partial V/\partial t = -\partial V/\partial T` per **year**
  (divide by 365 or 252 for per day), ``rho`` per unit of ``r``.

Black-Scholes-Merton
--------------------
With :math:`F = S e^{(r-q)T}`, :math:`v = \sigma\sqrt T`,
:math:`d_1 = \ln(F/K)/v + v/2`, :math:`d_2 = d_1 - v`:

.. math::
    C = S e^{-qT}\Phi(d_1) - K e^{-rT}\Phi(d_2),\qquad
    P = K e^{-rT}\Phi(-d_2) - S e^{-qT}\Phi(-d_1).

Put-call parity :math:`C - P = S e^{-qT} - K e^{-rT}` holds by construction.
**Degenerate limits** (:math:`v = 0`, i.e. :math:`T = 0` or :math:`\sigma = 0`): the
price is the discounted intrinsic value of the forward,
:math:`C = (S e^{-qT} - K e^{-rT})^+`, :math:`P = (K e^{-rT} - S e^{-qT})^+`, obtained
by setting :math:`d_1 = d_2 = \pm\infty` by the sign of :math:`\ln(F/K)` (0 at the money).
For tiny positive :math:`v`, :math:`d_1` may overflow to :math:`\pm\infty`, which is
its exact limit, so no warning is raised. The Greeks follow the same limits; at the
kink :math:`F = K` with :math:`v = 0`, gamma (and theta at :math:`T = 0`) are infinite.

Implied volatility
------------------
:math:`V(\sigma)` increases strictly from the lower no-arbitrage bound
:math:`V(0)` (above) to the upper bound (:math:`S e^{-qT}` for a call,
:math:`K e^{-rT}` for a put) as :math:`\sigma\to\infty`. So for a price strictly
between the bounds, :math:`f(\sigma) = V(\sigma) - \text{price}` has
:math:`f(0) < 0`, and doubling :math:`\sigma_{hi}` from 1 finds :math:`f(\sigma_{hi}) > 0`:
a guaranteed bracket for Brent's method (:func:`lipipy.numerics.brentq`).

Binomial tree (Cox-Ross-Rubinstein)
-----------------------------------
:math:`\Delta t = T/N`, :math:`u = e^{\sigma\sqrt{\Delta t}}`, :math:`d = 1/u`,
risk-neutral :math:`p = (e^{(r-q)\Delta t} - d)/(u - d)` (requires
:math:`0 \le p \le 1`, i.e. :math:`|r - q|\sqrt{\Delta t} \le \sigma`). Backward
induction :math:`V_{i,j} = e^{-r\Delta t}(p V_{i+1,j+1} + (1-p)V_{i+1,j})`, with
:math:`V_{i,j} \leftarrow \max(V_{i,j}, \text{payoff}(S_{i,j}))` for American
exercise. One array of :math:`N+1` values is updated in place: O(N^2) time,
O(N) memory. The European price converges to Black-Scholes with error O(1/N)
(oscillating in N).

Monte Carlo (geometric Brownian motion)
---------------------------------------
:math:`S_T = S\exp((r - q - \sigma^2/2)T + \sigma\sqrt T Z)`, price
:math:`e^{-rT}\mathbb E[\text{payoff}(S_T)]`. Variance reduction:

* **antithetic**: each sample averages the payoffs of :math:`Z` and :math:`-Z`;
* **control variate**: :math:`X = e^{-rT}S_T` (same pairing), with known mean
  :math:`\mathbb E X = S e^{-qT}`; the estimate is :math:`\bar Y - b(\bar X - \mathbb E X)`
  with :math:`b = \widehat{\mathrm{cov}}(X, Y)/\widehat{\mathrm{var}}(X)` estimated from
  the same samples (a small bias, Glasserman §4.1.2), and standard error
  :math:`\sqrt{(S_{YY} - b S_{XY})/((n-2)n)}`.

Paths are generated in chunks; per-chunk centred sums are merged with Chan et al.'s
pairwise formulas, so memory is O(chunk_size) for any number of paths. Results do
not depend on ``chunk_size`` beyond rounding (normals are drawn from one stream).

References
----------
F. Black & M. Scholes, *J. Political Economy* 81 (1973) 637-654. R. C. Merton,
*Bell J. Econ. Mgmt. Sci.* 4 (1973) 141-183. J. C. Hull, *Options, Futures and Other
Derivatives*, 10th ed. (2018), ch. 13, 15, 19, 21. J. C. Cox, S. A. Ross &
M. Rubinstein, *J. Financial Economics* 7 (1979) 229-263. P. Glasserman, *Monte Carlo
Methods in Financial Engineering* (2003), §4.1-4.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from lipipy.core.complexity import Complexity
from lipipy.core.random import RandomState, check_random_state
from lipipy.core.registry import register
from lipipy.finance._common import (
    DictMixin,
    as_float_array,
    check_finite,
    check_int,
    scalar_or_array,
)
from lipipy.numerics.optimize import brentq
from lipipy.numerics.special import norm_cdf

__all__ = [
    "Greeks",
    "MonteCarloResult",
    "binomial_price",
    "bs_greeks",
    "bs_price",
    "implied_volatility",
    "monte_carlo_price",
]

_KINDS = ("call", "put")
_BS = (
    "F. Black & M. Scholes, J. Political Economy 81 (1973) 637-654",
    "R. C. Merton, Bell J. Econ. Mgmt. Sci. 4 (1973) 141-183",
    "J. C. Hull, Options, Futures and Other Derivatives, 10th ed. (2018), ch. 15, 19",
)
_SQRT2PI = float(np.sqrt(2.0 * np.pi))
_ELEMENTWISE = Complexity(
    time="O(n)", space="O(n)", notes="n = number of broadcast option contracts"
)


# ---------------------------------------------------------------------------
# validation and the core formula
# ---------------------------------------------------------------------------
def _check_kind(kind: Any) -> str:
    if kind not in _KINDS:
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    return str(kind)


def _check_bs_inputs(S: Any, K: Any, T: Any, r: Any, sigma: Any, q: Any) -> list[np.ndarray]:
    arrays = []
    for name, value in (("S", S), ("K", K), ("T", T), ("r", r), ("sigma", sigma), ("q", q)):
        a = as_float_array(value, name)
        check_finite(a, name)
        arrays.append(a)
    S_, K_, T_, _, sig_, _ = arrays
    if np.any(S_ <= 0.0) or np.any(K_ <= 0.0):
        raise ValueError("S and K must be strictly positive")
    if np.any(T_ < 0.0):
        raise ValueError("T (time to expiry in years) must be non-negative")
    if np.any(sig_ < 0.0):
        raise ValueError("sigma must be non-negative")
    try:
        return list(np.broadcast_arrays(*arrays))
    except ValueError:
        raise ValueError("S, K, T, r, sigma and q cannot be broadcast together") from None


def _phi(x: np.ndarray) -> np.ndarray:
    # x * x may overflow to inf for |x| > 1e154; exp(-inf) = 0 is the exact limit.
    with np.errstate(over="ignore"):
        return np.exp(-0.5 * x * x) / _SQRT2PI


def _d1_d2(
    S: np.ndarray, K: np.ndarray, T: np.ndarray, r: np.ndarray, sigma: np.ndarray, q: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """d1, d2 and v = sigma sqrt(T), with the v = 0 limits (module docstring)."""
    v = sigma * np.sqrt(T)
    x = np.log(S / K) + (r - q) * T  # ln(F / K)
    degenerate = v == 0.0
    v_safe = np.where(degenerate, 1.0, v)
    # For tiny v, x / v overflows to +-inf, which is the exact limit of d1.
    with np.errstate(over="ignore"):
        d1 = x / v_safe + 0.5 * v_safe
    limit = np.where(x > 0.0, np.inf, np.where(x < 0.0, -np.inf, 0.0))
    d1 = np.where(degenerate, limit, d1)
    d2 = np.where(degenerate, d1, d1 - v)
    return d1, d2, v


def _bs(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray,
    q: np.ndarray,
    kind: str,
) -> np.ndarray:
    d1, d2, _ = _d1_d2(S, K, T, r, sigma, q)
    fwd_s = S * np.exp(-q * T)
    pv_k = K * np.exp(-r * T)
    if kind == "call":
        price = fwd_s * norm_cdf(d1) - pv_k * norm_cdf(d2)
    else:
        price = pv_k * norm_cdf(-d2) - fwd_s * norm_cdf(-d1)
    return np.maximum(price, 0.0)  # remove rounding below zero far out of the money


def _intrinsic(S: float, K: float, kind: str) -> float:
    return max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)


# ---------------------------------------------------------------------------
# Black-Scholes-Merton
# ---------------------------------------------------------------------------
@register(
    domain="finance",
    tasks=("option-pricing",),
    complexity=_ELEMENTWISE,
    references=_BS,
)
def bs_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    kind: str = "call",
) -> float | np.ndarray:
    r"""Black-Scholes-Merton price of a European call or put (vectorised).

    See the module docstring for the formula, conventions and the
    :math:`T \to 0`, :math:`\sigma \to 0` limits (discounted forward intrinsic value).

    Parameters
    ----------
    S, K : array_like
        Spot and strike, > 0.
    T : array_like
        Time to expiry in years, >= 0.
    r : array_like
        Continuously compounded risk-free rate.
    sigma : array_like
        Annualised volatility, >= 0.
    q : array_like, default=0.0
        Continuous dividend yield.
    kind : {"call", "put"}, default="call"
        Option type.

    Returns
    -------
    float or ndarray
        Price(s), broadcast over the inputs; a float when all inputs are scalars.

    Complexity
    ----------
    Time O(n), extra space O(n) for n broadcast contracts.

    References
    ----------
    Black & Scholes (1973); Merton (1973); Hull, *Options, Futures and Other
    Derivatives*, ch. 15 (example 15.6: S=42, K=40, r=10%, sigma=20%, T=0.5).

    Examples
    --------
    >>> round(bs_price(42.0, 40.0, 0.5, 0.10, 0.20), 4)
    4.7594
    >>> round(bs_price(42.0, 40.0, 0.5, 0.10, 0.20, kind="put"), 4)
    0.8086
    >>> bs_price(42.0, 40.0, 0.0, 0.10, 0.20)  # at expiry: intrinsic value
    2.0
    """
    kind = _check_kind(kind)
    S_, K_, T_, r_, s_, q_ = _check_bs_inputs(S, K, T, r, sigma, q)
    return scalar_or_array(_bs(S_, K_, T_, r_, s_, q_, kind))


@dataclass(frozen=True)
class Greeks(DictMixin):
    """Black-Scholes-Merton sensitivities (see the module docstring for units).

    Attributes
    ----------
    delta : float or ndarray
        :math:`\\partial V/\\partial S`.
    gamma : float or ndarray
        :math:`\\partial^2 V/\\partial S^2`.
    vega : float or ndarray
        :math:`\\partial V/\\partial\\sigma` (per unit volatility).
    theta : float or ndarray
        :math:`\\partial V/\\partial t = -\\partial V/\\partial T` (per year).
    rho : float or ndarray
        :math:`\\partial V/\\partial r` (per unit rate).
    """

    delta: Any
    gamma: Any
    vega: Any
    theta: Any
    rho: Any


@register(
    domain="finance",
    tasks=("option-pricing", "greeks"),
    complexity=_ELEMENTWISE,
    references=_BS,
)
def bs_greeks(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    kind: str = "call",
) -> Greeks:
    r"""Black-Scholes-Merton Greeks: delta, gamma, vega, theta and rho (vectorised).

    .. math::
        \Delta_C = e^{-qT}\Phi(d_1),\ \Delta_P = -e^{-qT}\Phi(-d_1),\quad
        \Gamma = \frac{e^{-qT}\varphi(d_1)}{S\sigma\sqrt T},\quad
        \mathcal V = S e^{-qT}\varphi(d_1)\sqrt T,

        \Theta_C = -\frac{S e^{-qT}\varphi(d_1)\sigma}{2\sqrt T} - rKe^{-rT}\Phi(d_2)
                   + qSe^{-qT}\Phi(d_1),\quad
        \Theta_P = -\frac{S e^{-qT}\varphi(d_1)\sigma}{2\sqrt T} + rKe^{-rT}\Phi(-d_2)
                   - qSe^{-qT}\Phi(-d_1),

        \rho_C = KTe^{-rT}\Phi(d_2),\quad \rho_P = -KTe^{-rT}\Phi(-d_2).

    Parameters
    ----------
    S, K, T, r, sigma, q, kind
        As in :func:`bs_price`.

    Returns
    -------
    Greeks
        Dataclass with ``delta``, ``gamma``, ``vega``, ``theta``, ``rho``.

    Complexity
    ----------
    Time O(n), extra space O(n) for n broadcast contracts.

    References
    ----------
    Hull, *Options, Futures and Other Derivatives*, ch. 19.

    Examples
    --------
    >>> g = bs_greeks(49.0, 50.0, 0.3846, 0.05, 0.20)  # Hull, ch. 19 running example
    >>> round(g.delta, 3), round(g.gamma, 3), round(g.vega, 2)
    (0.522, 0.066, 12.11)
    """
    kind = _check_kind(kind)
    S_, K_, T_, r_, s_, q_ = _check_bs_inputs(S, K, T, r, sigma, q)
    d1, d2, v = _d1_d2(S_, K_, T_, r_, s_, q_)
    dq = np.exp(-q_ * T_)
    dr = np.exp(-r_ * T_)
    pdf = _phi(d1)
    kink = pdf > 0.0  # only relevant where v == 0: then pdf > 0 iff F == K
    gamma = np.where(
        v > 0.0, dq * pdf / (S_ * np.where(v > 0.0, v, 1.0)), np.where(kink, np.inf, 0.0)
    )
    vega = S_ * dq * pdf * np.sqrt(T_)
    decay_num = S_ * dq * pdf * s_
    sqrt_t = np.sqrt(T_)
    decay = np.where(
        T_ > 0.0,
        decay_num / (2.0 * np.where(T_ > 0.0, sqrt_t, 1.0)),
        np.where(decay_num > 0.0, np.inf, 0.0),
    )
    if kind == "call":
        delta = dq * norm_cdf(d1)
        theta = -decay - r_ * K_ * dr * norm_cdf(d2) + q_ * S_ * dq * norm_cdf(d1)
        rho = K_ * T_ * dr * norm_cdf(d2)
    else:
        delta = -dq * norm_cdf(-d1)
        theta = -decay + r_ * K_ * dr * norm_cdf(-d2) - q_ * S_ * dq * norm_cdf(-d1)
        rho = -K_ * T_ * dr * norm_cdf(-d2)
    return Greeks(
        delta=scalar_or_array(delta),
        gamma=scalar_or_array(gamma),
        vega=scalar_or_array(vega),
        theta=scalar_or_array(theta),
        rho=scalar_or_array(rho),
    )


_XTOL = 1e-14


def _bracket(f: Any, target: float, upper: float, sqrt_t: float) -> tuple[float, float]:
    """[lo, hi] with f(lo) < 0 < f(hi) for increasing f with f(0) < 0 (module docstring).

    Doubling from 1 finds hi; halving from 1 moves lo off sigma = 0 so that f is not
    numerically flat at the lower end of the bracket. Beyond sigma sqrt(T) = 64 the
    price differs from the upper bound by less than K Phi(-32) ~ 1e-225 K.
    """
    lo, hi = 0.0, 1.0
    while f(hi) <= 0.0:
        lo, hi = hi, 2.0 * hi
        if hi * sqrt_t > 64.0:  # defensive: V(sigma) rounds to the bound well before
            raise ValueError(
                f"price {target!r} is within rounding of the upper bound {upper!r}; "
                "implied volatility is not identifiable"
            )
    if lo == 0.0:
        s = 0.5
        while s > 1e-8:
            if f(s) < 0.0:
                lo = s
                break
            hi, s = s, 0.5 * s
    return lo, hi


def _bisect(f: Any, lo: float, hi: float) -> float:
    """Plain bisection on f(lo) < 0 < f(hi) down to _XTOL (fallback for brentq)."""
    while hi - lo > _XTOL + 4.0 * np.finfo(np.float64).eps * hi:
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _solve_bracketed(f: Any, target: float, upper: float, sqrt_t: float) -> float:
    lo, hi = _bracket(f, target, upper, sqrt_t)
    try:
        return float(brentq(f, lo, hi, xtol=_XTOL).root)
    except ZeroDivisionError:
        # lipipy.numerics.brentq can divide by zero in its interpolation step when
        # f takes equal values at two points (numerically flat price); bisection is
        # always safe on a bracket.
        return _bisect(f, lo, hi)


@register(
    domain="finance",
    tasks=("option-pricing", "implied-volatility"),
    complexity=Complexity(
        time="O(n T)",
        space="O(n)",
        notes="n contracts, T Brent iterations each",
    ),
    references=_BS + ("R. P. Brent, Algorithms for Minimization without Derivatives (1973)",),
)
def implied_volatility(
    price: ArrayLike,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike = 0.0,
    kind: str = "call",
) -> float | np.ndarray:
    r"""Black-Scholes implied volatility by Brent's method on a guaranteed bracket.

    Solves :math:`V_{BS}(\sigma) = \text{price}` (module docstring). For in-the-money
    options (positive lower bound), a price within
    :math:`8\varepsilon\max(Se^{-qT}, Ke^{-rT})` of the lower no-arbitrage bound is
    treated as at the bound and gives :math:`\sigma = 0`: where vega underflows the
    returned volatility reproduces the price (to that tolerance) but is not unique.
    The root is found to an
    absolute tolerance of :math:`10^{-14}` in :math:`\sigma`; the repriced option
    then matches ``price`` to within rounding (how well :math:`\sigma` itself is
    determined depends on vega).

    Parameters
    ----------
    price : array_like
        Observed option price(s).
    S, K, T, r, q, kind
        As in :func:`bs_price`; ``T`` must be > 0.

    Returns
    -------
    float or ndarray
        Implied volatility (annualised), broadcast over the inputs.

    Raises
    ------
    ValueError
        If any price is outside the no-arbitrage bounds
        ``max(S e^{-qT} - K e^{-rT}, 0) <= C < S e^{-qT}`` (call) or
        ``max(K e^{-rT} - S e^{-qT}, 0) <= P < K e^{-rT}`` (put), or ``T == 0``.

    Complexity
    ----------
    Time O(n T) for n contracts and T Brent iterations each; space O(n).

    References
    ----------
    Hull, ch. 15; Brent (1973).

    Examples
    --------
    >>> c = bs_price(42.0, 40.0, 0.5, 0.10, 0.20)
    >>> round(implied_volatility(c, 42.0, 40.0, 0.5, 0.10), 12)
    0.2
    """
    kind = _check_kind(kind)
    p = as_float_array(price, "price")
    check_finite(p, "price")
    S_, K_, T_, r_, _, q_ = _check_bs_inputs(S, K, T, r, 0.0, q)
    try:
        p, S_, K_, T_, r_, q_ = np.broadcast_arrays(p, S_, K_, T_, r_, q_)
    except ValueError:
        raise ValueError("price cannot be broadcast with S, K, T, r and q") from None
    if np.any(T_ <= 0.0):
        raise ValueError("implied volatility needs T > 0 (an expired option has no time value)")
    fwd_s = S_ * np.exp(-q_ * T_)
    pv_k = K_ * np.exp(-r_ * T_)
    if kind == "call":
        lower, upper = np.maximum(fwd_s - pv_k, 0.0), fwd_s
    else:
        lower, upper = np.maximum(pv_k - fwd_s, 0.0), pv_k
    # Prices within a few ulps of the lower bound are at the bound: there sigma is
    # not identifiable (vega underflows) and sigma = 0 reproduces the price.
    # (A zero bound is exact, so out-of-the-money prices get no slack.)
    slack = np.where(lower > 0.0, 8.0 * np.finfo(np.float64).eps * np.maximum(fwd_s, pv_k), 0.0)
    bad = (p < lower - slack) | (p >= upper)
    if np.any(bad):
        i = np.flatnonzero(bad.ravel())[0]
        raise ValueError(
            f"price {p.ravel()[i]!r} violates the no-arbitrage bounds "
            f"[{lower.ravel()[i]!r}, {upper.ravel()[i]!r}) of a {kind}; no implied volatility"
        )

    out = np.empty(p.shape)
    for idx in np.ndindex(p.shape):
        target = float(p[idx])
        if target <= lower[idx] + slack[idx]:
            out[idx] = 0.0
            continue
        args = tuple(np.asarray(a[idx]) for a in (S_, K_, T_, r_))
        qi = np.asarray(q_[idx])

        def f(sig: float, target: float = target, args: tuple = args, qi: Any = qi) -> float:
            return float(_bs(*args, np.asarray(sig), qi, kind)) - target

        out[idx] = _solve_bracketed(f, target, float(upper[idx]), float(np.sqrt(T_[idx])))
    return scalar_or_array(out)


# ---------------------------------------------------------------------------
# binomial tree
# ---------------------------------------------------------------------------
def _scalar_inputs(S: Any, K: Any, T: Any, r: Any, sigma: Any, q: Any) -> tuple[float, ...]:
    vals = _check_bs_inputs(S, K, T, r, sigma, q)
    if any(v.ndim != 0 for v in vals):
        raise ValueError("binomial_price and monte_carlo_price take scalar inputs")
    return tuple(float(v) for v in vals)


@register(
    domain="finance",
    tasks=("option-pricing", "american-options"),
    complexity=Complexity(time="O(N^2)", space="O(N)", notes="N = steps"),
    references=(
        "J. C. Cox, S. A. Ross & M. Rubinstein, J. Financial Economics 7 (1979) 229-263",
        "J. C. Hull, Options, Futures and Other Derivatives, 10th ed. (2018), ch. 13, 21",
    ),
)
def binomial_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    kind: str = "call",
    steps: int = 500,
    american: bool = False,
) -> float:
    r"""Cox-Ross-Rubinstein binomial-tree price of a European or American option.

    See the module docstring for the tree. With ``T == 0`` the intrinsic value is
    returned; with ``sigma == 0`` the underlying follows its forward deterministically
    and an American option is exercised at the best grid date.

    Parameters
    ----------
    S, K, T, r, sigma, q, kind
        As in :func:`bs_price` (scalars).
    steps : int, default=500
        Number of time steps :math:`N` (>= 1).
    american : bool, default=False
        Allow early exercise at every node.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If the risk-neutral probability leaves [0, 1] (too few steps for the drift:
        increase ``steps``).

    Complexity
    ----------
    Time O(N^2), extra space O(N).

    References
    ----------
    Cox, Ross & Rubinstein (1979); Hull, ch. 13 and 21.

    Examples
    --------
    >>> round(binomial_price(50.0, 52.0, 2.0, 0.05, 0.30, kind="put", steps=2, american=True), 4)
    7.4284
    """
    kind = _check_kind(kind)
    S, K, T, r, sigma, q = _scalar_inputs(S, K, T, r, sigma, q)
    N = check_int(steps, "steps", 1)
    sign = 1.0 if kind == "call" else -1.0
    if T == 0.0:
        return _intrinsic(S, K, kind)
    dt = T / N
    if sigma == 0.0:
        t = np.arange(N + 1) * dt
        values = np.maximum(sign * (S * np.exp((r - q) * t) - K), 0.0) * np.exp(-r * t)
        return float(values.max() if american else values[-1])

    h = sigma * np.sqrt(dt)
    u, d = np.exp(h), np.exp(-h)
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 <= p <= 1.0:
        raise ValueError(
            f"risk-neutral probability {p:.4g} is outside [0, 1]; increase steps "
            f"(need |r - q| sqrt(T/steps) <= sigma)"
        )
    disc = np.exp(-r * dt)
    # Node prices S u^(2j - i): all exponents -N..N on one grid of 2N + 1 values.
    # (built in place: three float arrays of length ~N to 2N are all the memory used)
    grid = np.arange(-N, N + 1, dtype=np.float64)
    grid *= h
    np.exp(grid, out=grid)
    grid *= S
    V = grid[0 : 2 * N + 1 : 2] - K  # step N: exponents -N, -N+2, ..., N
    V *= sign
    np.maximum(V, 0.0, out=V)
    up, down = disc * p, disc * (1.0 - p)
    buf = np.empty(N)  # scratch space, so each step allocates nothing
    for i in range(N - 1, -1, -1):
        v, b = V[: i + 1], buf[: i + 1]
        np.multiply(V[1 : i + 2], up, out=b)  # read V_{i+1, j+1} before overwriting
        v *= down
        v += b
        if american:
            np.subtract(grid[N - i : N + i + 1 : 2], K, out=b)
            b *= sign
            np.maximum(v, b, out=v)  # exercise value (<= 0 parts never win: v >= 0)
    return float(V[0])


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MonteCarloResult(DictMixin):
    """Outcome of :func:`monte_carlo_price`.

    Attributes
    ----------
    price : float
        Estimated option price.
    std_error : float
        Standard error of ``price`` (0 for the deterministic limits).
    n_paths : int
        Number of simulated terminal prices (antithetic pairs count twice).
    n_samples : int
        Number of independent samples (``n_paths / 2`` with antithetic variates).
    control_coefficient : float or None
        Estimated control-variate coefficient :math:`b`, if used.
    """

    price: float
    std_error: float
    n_paths: int
    n_samples: int
    control_coefficient: float | None


class _PairMoments:
    """Streaming means and centred cross-products of (X, Y) (Chan et al. merge)."""

    def __init__(self) -> None:
        self.n = 0
        self.mx = self.my = self.sxx = self.syy = self.sxy = 0.0

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        nb = x.size
        mxb, myb = float(x.mean()), float(y.mean())
        xc, yc = x - mxb, y - myb
        sxx_b, syy_b, sxy_b = float(xc @ xc), float(yc @ yc), float(xc @ yc)
        n = self.n + nb
        dx, dy = mxb - self.mx, myb - self.my
        w = self.n * nb / n
        self.sxx += sxx_b + dx * dx * w
        self.syy += syy_b + dy * dy * w
        self.sxy += sxy_b + dx * dy * w
        self.mx += dx * nb / n
        self.my += dy * nb / n
        self.n = n


@register(
    domain="finance",
    tasks=("option-pricing", "monte-carlo"),
    complexity=Complexity(time="O(P)", space="O(chunk_size)", notes="P paths processed in chunks"),
    references=(
        "P. Glasserman, Monte Carlo Methods in Financial Engineering (2003), §4.1-4.2",
        "Chan, Golub & LeVeque, The American Statistician 37 (1983) 242-247",
    ),
)
def monte_carlo_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    kind: str = "call",
    n_paths: int = 100_000,
    antithetic: bool = True,
    control_variate: bool = True,
    random_state: RandomState = None,
    chunk_size: int = 65_536,
) -> MonteCarloResult:
    r"""Monte Carlo price of a European option under geometric Brownian motion.

    See the module docstring for the estimator, the antithetic and control variates
    and the chunked accumulation. With ``T == 0`` or ``sigma == 0`` the terminal price
    is deterministic and the Black-Scholes limit is returned with zero standard error.

    Parameters
    ----------
    S, K, T, r, sigma, q, kind
        As in :func:`bs_price` (scalars).
    n_paths : int, default=100000
        Number of simulated terminal prices (even when ``antithetic``).
    antithetic : bool, default=True
        Pair every normal draw with its negative.
    control_variate : bool, default=True
        Use the discounted terminal price as a control variate.
    random_state : None, int, SeedSequence or Generator
        Seed for reproducibility.
    chunk_size : int, default=65536
        Paths simulated at once; bounds the memory.

    Returns
    -------
    MonteCarloResult
        ``price``, ``std_error``, ``n_paths``, ``n_samples``, ``control_coefficient``.

    Complexity
    ----------
    Time O(P) for P paths, extra space O(chunk_size).

    References
    ----------
    P. Glasserman, *Monte Carlo Methods in Financial Engineering*, §4.1-4.2.

    Examples
    --------
    >>> res = monte_carlo_price(42.0, 40.0, 0.5, 0.10, 0.20, n_paths=200_000, random_state=0)
    >>> abs(res.price - bs_price(42.0, 40.0, 0.5, 0.10, 0.20)) < 4 * res.std_error
    True
    """
    kind = _check_kind(kind)
    S, K, T, r, sigma, q = _scalar_inputs(S, K, T, r, sigma, q)
    n_paths = check_int(n_paths, "n_paths", 2)
    chunk_size = check_int(chunk_size, "chunk_size", 2)
    if antithetic and n_paths % 2:
        raise ValueError(f"n_paths must be even with antithetic=True, got {n_paths}")
    n_samples = n_paths // 2 if antithetic else n_paths
    if control_variate and n_samples < 3:
        raise ValueError("control_variate needs at least 3 independent samples")
    rng = check_random_state(random_state)

    if T == 0.0 or sigma == 0.0:
        price = float(_bs(*(np.asarray(v) for v in (S, K, T, r, sigma, q)), kind))
        return MonteCarloResult(price, 0.0, n_paths, n_samples, None)

    sign = 1.0 if kind == "call" else -1.0
    drift = (r - q - 0.5 * sigma * sigma) * T
    vol = sigma * np.sqrt(T)
    disc = np.exp(-r * T)
    per_chunk = max(chunk_size // 2, 1) if antithetic else chunk_size

    acc = _PairMoments()
    remaining = n_samples
    while remaining > 0:
        m = min(per_chunk, remaining)
        z = rng.standard_normal(m)
        st = S * np.exp(drift + vol * z)
        y = np.maximum(sign * (st - K), 0.0)
        x = st
        if antithetic:
            st = S * np.exp(drift - vol * z)
            y = 0.5 * (y + np.maximum(sign * (st - K), 0.0))
            x = 0.5 * (x + st)
        acc.update(disc * x, disc * y)
        remaining -= m

    n = acc.n
    if control_variate:
        b = acc.sxy / acc.sxx if acc.sxx > 0.0 else 0.0
        price = acc.my - b * (acc.mx - S * np.exp(-q * T))
        resid = max(acc.syy - b * acc.sxy, 0.0)
        se = np.sqrt(resid / (n - 2) / n)
        return MonteCarloResult(float(price), float(se), n_paths, n_samples, float(b))
    se = np.sqrt(acc.syy / (n - 1) / n)
    return MonteCarloResult(float(acc.my), float(se), n_paths, n_samples, None)
