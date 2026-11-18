r"""Stochastic-volatility, jump-diffusion and path-dependent option pricing.

Conventions as in :mod:`lipipy.finance.options`: ``S`` spot, ``K`` strike, ``T`` years,
continuously compounded ``r`` and dividend yield ``q``, ``kind`` = ``"call"``/``"put"``;
scalar inputs. Monte Carlo functions return
:class:`~lipipy.finance.options.MonteCarloResult` and simulate geometric Brownian
motion exactly on the monitoring grid :math:`t_i = iT/n`, in chunks of paths (memory
O(chunk_size)), with optional antithetic variates. Normals and uniforms come from two
independent child streams of ``random_state``, so results do not depend on
``chunk_size`` beyond rounding.

Heston (1993)
-------------
:math:`dS = (r-q)S\,dt + \sqrt v S\,dW_1`, :math:`dv = \kappa(\theta - v)dt +
\sigma\sqrt v\,dW_2`, :math:`d\langle W_1, W_2\rangle = \rho\,dt`. Characteristic
function of :math:`\ln S_T` in the "little trap" form of Albrecher et al. (2007),
which avoids the branch-cut discontinuity of the complex logarithm:

.. math::
    d = \sqrt{(\rho\sigma iu - \kappa)^2 + \sigma^2(iu + u^2)},\quad
    g = \frac{\kappa - \rho\sigma iu - d}{\kappa - \rho\sigma iu + d},

    \ln\varphi(u) = iu(\ln S + (r-q)T) + \frac{\kappa\theta}{\sigma^2}\Bigl[(\kappa -
    \rho\sigma iu - d)T - 2\ln\frac{1 - ge^{-dT}}{1-g}\Bigr] + \frac{v_0}{\sigma^2}
    (\kappa - \rho\sigma iu - d)\frac{1 - e^{-dT}}{1 - ge^{-dT}} .

The call uses Lewis's (2001) contour-shifted formula, with
:math:`X = \ln S_T - \ln S - (r-q)T`, :math:`\varphi_X` its characteristic function and
:math:`k = \ln(S/K) + (r-q)T`:

.. math::  C = Se^{-qT} - \frac{\sqrt{SK}e^{-(r+q)T/2}}{\pi}\int_0^\infty
           \frac{\operatorname{Re}[e^{iuk}\varphi_X(u - i/2)]}{u^2 + 1/4}\,du .

Since :math:`|\varphi_X(u - i/2)| \le \mathbb E[e^{X/2}] \le 1`, the integrand is
bounded by :math:`1/(u^2+1/4)` even where :math:`\varphi_X` decays slowly (e.g.
:math:`\rho = \pm1`). It is integrated with :func:`lipipy.numerics.quad` on doubling
intervals :math:`[0, U], [U, 2U], \dots` until the tail bound or the contributions fall
below the tolerance. Puts follow from put-call parity. Valid for any
:math:`\rho\in[-1,1]`, with or without the Feller condition :math:`2\kappa\theta \ge \sigma^2`.

Merton (1976) jump diffusion
----------------------------
Poisson jumps of intensity :math:`\lambda` with log-normal sizes
:math:`\ln(1+J)\sim N(\mu_J, \sigma_J^2)`, :math:`k = e^{\mu_J + \sigma_J^2/2} - 1`:

.. math::  V = \sum_{n\ge0} e^{-\lambda'T}\frac{(\lambda'T)^n}{n!}\,
           V_{BS}\bigl(S, K, T, r_n, \sigma_n, q\bigr),\quad
           \lambda' = \lambda(1+k),\ \sigma_n^2 = \sigma^2 + n\sigma_J^2/T,\
           r_n = r - \lambda k + n\ln(1+k)/T,

summed until the bound :math:`w_n\,n/(n - \lambda'T)` on the remaining Poisson-weighted
terms (each Black-Scholes price is at most the current partial sum's order) falls below
:math:`10^{-16}` of the sum.

Closed forms used as references and control variates
----------------------------------------------------
* **Geometric Asian** with ``n`` equally spaced fixings :math:`t_i = iT/n`: :math:`\ln G`
  is normal with mean :math:`\ln S + (r - q - \sigma^2/2)T\frac{n+1}{2n}` and variance
  :math:`\sigma^2T\frac{(n+1)(2n+1)}{6n^2}` (continuous averaging: ``n=None``, the
  limits :math:`T/2` and :math:`T/3`).
* **Barriers** (continuous monitoring, no rebate): Reiner & Rubinstein (1991) in the
  form of Haug (2007) §4.17.1. If the barrier is already breached at inception
  (``S <= H`` for "down", ``S >= H`` for "up") the out-option is worth 0 and the
  in-option is the vanilla.

Monte Carlo estimators
----------------------
* **Arithmetic Asian**: payoff on :math:`\frac1n\sum_{i=1}^n S_{t_i}`; control variate
  = the geometric-Asian payoff on the same path, whose exact mean is the closed form
  above.
* **Barriers**: continuous monitoring through the **Brownian-bridge** correction:
  between grid points with log-prices :math:`x_i, x_{i+1}` on the safe side of
  :math:`h = \ln H`, the bridge crosses with probability
  :math:`\exp(-2(h - x_i)(h - x_{i+1})/(\sigma^2\Delta t))`; the knock-out payoff is
  weighted by the product of the survival probabilities (conditional expectation),
  which makes the estimator unbiased for the continuously monitored option. Knock-in
  = payoff x (1 - survival).
* **Floating-strike lookbacks**: the path minimum (call) or maximum (put) between grid
  points is sampled exactly from the Brownian bridge,
  :math:`m = \tfrac12\bigl(x_i + x_{i+1} - \sqrt{(x_{i+1}-x_i)^2 - 2\sigma^2\Delta t\ln U}\bigr)`
  (maximum with the + sign), so the estimator is unbiased for continuous monitoring;
  payoffs :math:`S_T - \min_t S_t` and :math:`\max_t S_t - S_T`.

References
----------
S. L. Heston, *Rev. Financial Studies* 6 (1993) 327-343. H. Albrecher, P. Mayer,
W. Schoutens & J. Tistaert, "The little Heston trap", *Wilmott* (Jan 2007) 83-92.
A. L. Lewis, *Option Valuation under Stochastic Volatility* (2001), ch. 2.
F. Fang & C. W. Oosterlee, *SIAM J. Sci. Comput.* 31 (2008) 826-848 (reference
values). R. C. Merton, *J. Financial Economics* 3 (1976) 125-144. A. G. Z. Kemna &
A. C. F. Vorst, *J. Banking & Finance* 14 (1990) 113-129. E. Reiner & M. Rubinstein,
"Breaking down the barriers", *Risk* 4 (1991) 28-35. E. G. Haug, *The Complete Guide
to Option Pricing Formulas*, 2nd ed. (2007), §4.17. P. Glasserman, *Monte Carlo
Methods in Financial Engineering* (2003), §4.1, §6.4.
"""

from __future__ import annotations

import cmath
import math
from typing import Any

import numpy as np

from lipipy.core.complexity import Complexity
from lipipy.core.random import RandomState, check_random_state, spawn
from lipipy.core.registry import register
from lipipy.finance._common import check_int, check_real
from lipipy.finance.options import (
    MonteCarloResult,
    _bs,
    _check_kind,
    _PairMoments,
    _scalar_inputs,
)
from lipipy.numerics.integrate import quad
from lipipy.numerics.special import norm_cdf

__all__ = [
    "arithmetic_asian_mc",
    "barrier_mc",
    "barrier_price",
    "geometric_asian_price",
    "heston_price",
    "lookback_mc",
    "merton_jump_price",
]

_GLASSERMAN = "P. Glasserman, Monte Carlo Methods in Financial Engineering (2003)"
_BARRIERS = ("down-and-out", "down-and-in", "up-and-out", "up-and-in")


def _Phi(x: float) -> float:
    return float(norm_cdf(x))


def _bs_scalar(S: float, K: float, T: float, r: float, sigma: float, q: float, kind: str) -> float:
    return float(_bs(*(np.asarray(v) for v in (S, K, T, r, sigma, q)), kind))


# ---------------------------------------------------------------------------
# Heston
# ---------------------------------------------------------------------------
def _heston_cf(
    u: complex, lnS: float, T: float, r: float, q: float, v0: float, kappa: float,
    theta: float, sigma: float, rho: float,
) -> complex:  # fmt: skip
    """Characteristic function of ln S_T, little-trap form (Albrecher et al. 2007)."""
    iu = 1j * u
    beta = kappa - rho * sigma * iu
    d = cmath.sqrt(beta * beta + sigma * sigma * (iu + u * u))
    g = (beta - d) / (beta + d)
    edt = cmath.exp(-d * T)
    C = iu * (lnS + (r - q) * T) + kappa * theta / sigma**2 * (
        (beta - d) * T - 2.0 * cmath.log((1.0 - g * edt) / (1.0 - g))
    )
    D = (beta - d) / sigma**2 * (1.0 - edt) / (1.0 - g * edt)
    return cmath.exp(C + D * v0)


@register(
    domain="finance",
    tasks=("option-pricing", "stochastic-volatility"),
    complexity=Complexity(time="O(T)", space="O(T)", notes="T quadrature subintervals"),
    references=(
        "S. L. Heston, Rev. Financial Studies 6 (1993) 327-343",
        "A. L. Lewis, Option Valuation under Stochastic Volatility (2001)",
        "H. Albrecher, P. Mayer, W. Schoutens & J. Tistaert, Wilmott (2007) 83-92",
    ),
)
def heston_price(
    S: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    q: float = 0.0,
    kind: str = "call",
) -> float:
    r"""European option price in the Heston (1993) stochastic-volatility model.

    Fourier inversion of the little-trap characteristic function (module
    docstring, Lewis's formula) with adaptive Gauss-Kronrod quadrature at tolerance
    :math:`10^{-10}\max(S, K)` (a :class:`~lipipy.core.ConvergenceWarning` is emitted
    if it is not met). With ``T == 0`` the intrinsic value is returned.

    Parameters
    ----------
    S, K, T, r, q, kind
        As in :func:`lipipy.finance.bs_price` (scalars).
    v0 : float
        Initial variance (> 0).
    kappa : float
        Mean-reversion speed (> 0).
    theta : float
        Long-run variance (> 0).
    sigma : float
        Volatility of variance (> 0).
    rho : float
        Correlation in [-1, 1].

    Returns
    -------
    float

    Complexity
    ----------
    O(T) characteristic-function evaluations for T adaptive subintervals.

    References
    ----------
    Heston (1993); Albrecher et al. (2007); Fang & Oosterlee (2008) for test values.

    Examples
    --------
    >>> p = heston_price(100.0, 100.0, 1.0, 0.0, 0.0175, 1.5768, 0.0398, 0.5751, -0.5711)
    >>> round(p, 6)  # Fang & Oosterlee (2008), reference 5.785155450
    5.785155
    """
    kind = _check_kind(kind)
    S, K, T, r, _, q = _scalar_inputs(S, K, T, r, 0.0, q)
    for name, val in (("v0", v0), ("kappa", kappa), ("theta", theta), ("sigma", sigma)):
        if not check_real(val, name) > 0.0:
            raise ValueError(f"{name} must be positive, got {val!r}")
    rho = check_real(rho, "rho")
    if not -1.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [-1, 1], got {rho!r}")
    if T == 0.0:
        return max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
    lnS = math.log(S)
    args = (lnS, T, r, q, float(v0), float(kappa), float(theta), float(sigma), rho)
    shift = lnS + (r - q) * T  # phi(u) = exp(iu shift) * phi_X(u), X = ln S_T - shift
    k = math.log(S / K) + (r - q) * T

    def integrand(u: float) -> float:
        z = u - 0.5j
        phi_x = _heston_cf(z, *args) * cmath.exp(-1j * z * shift)
        return (cmath.exp(1j * u * k) * phi_x).real / (u * u + 0.25)

    scale = math.sqrt(S * K) * math.exp(-0.5 * (r + q) * T) / math.pi
    tol = 1e-10 * max(S, K)
    total, lo, hi = 0.0, 0.0, 4.0 / math.sqrt(max(float(v0), float(theta)) * T)
    while True:
        piece = quad(integrand, lo, hi, epsabs=0.25 * tol / scale, epsrel=1e-12, limit=2000)
        total += piece.value
        # |phi_X(u - i/2)| <= 1, so the tail beyond hi is at most 1 / hi in the integral
        if scale / hi < tol or (abs(piece.value) * scale < 1e-3 * tol and hi > 1e3):
            break
        lo, hi = hi, 2.0 * hi
    call = S * math.exp(-q * T) - scale * total
    if kind == "call":
        return max(call, 0.0)
    return max(call - S * math.exp(-q * T) + K * math.exp(-r * T), 0.0)


# ---------------------------------------------------------------------------
# Merton jump diffusion
# ---------------------------------------------------------------------------
@register(
    domain="finance",
    tasks=("option-pricing", "jump-diffusion"),
    complexity=Complexity(time="O(N)", space="O(1)", notes="N series terms"),
    references=("R. C. Merton, J. Financial Economics 3 (1976) 125-144",),
)
def merton_jump_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    lam: float,
    mu_j: float,
    sigma_j: float,
    q: float = 0.0,
    kind: str = "call",
) -> float:
    r"""European option price under Merton's (1976) log-normal jump diffusion.

    Poisson-weighted series of Black-Scholes prices (module docstring), truncated
    once the tail bound falls below :math:`10^{-16}` of the sum.

    Parameters
    ----------
    S, K, T, r, sigma, q, kind
        As in :func:`lipipy.finance.bs_price` (scalars).
    lam : float
        Jump intensity per year (>= 0).
    mu_j, sigma_j : float
        Mean and standard deviation (>= 0) of the log jump size.

    Returns
    -------
    float

    Complexity
    ----------
    O(N) Black-Scholes evaluations for N terms (about :math:`\lambda T` + a few dozen).

    References
    ----------
    R. C. Merton (1976).

    Examples
    --------
    >>> round(merton_jump_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, -0.1, 0.1), 6)  # no jumps
    10.450584
    """
    kind = _check_kind(kind)
    S, K, T, r, sigma, q = _scalar_inputs(S, K, T, r, sigma, q)
    lam = check_real(lam, "lam")
    mu_j = check_real(mu_j, "mu_j")
    sigma_j = check_real(sigma_j, "sigma_j")
    if lam < 0.0 or sigma_j < 0.0:
        raise ValueError("lam and sigma_j must be non-negative")
    if T == 0.0 or lam == 0.0:
        return _bs_scalar(S, K, T, r, sigma, q, kind)
    k = math.exp(mu_j + 0.5 * sigma_j**2) - 1.0
    lt = lam * (1.0 + k) * T
    total, n = 0.0, 0
    log_w = -lt
    while True:
        w = math.exp(log_w)
        sig_n = math.sqrt(sigma**2 + n * sigma_j**2 / T)
        r_n = r - lam * k + n * math.log1p(k) / T
        total += w * _bs_scalar(S, K, T, r_n, sig_n, q, kind)
        n += 1
        # Poisson tail after term n-1 is below w * n / (n - lt) once n > lt
        if n > lt and w * n / (n - lt) < 1e-16 * max(total, 1e-300) or n > 100_000:
            break
        log_w += math.log(lt) - math.log(n)
    return total


# ---------------------------------------------------------------------------
# closed forms: geometric Asian and barriers
# ---------------------------------------------------------------------------
def _geometric_moments(S: float, T: float, r: float, sigma: float, q: float, n: int | None):
    if n is None:
        a, b = 0.5, 1.0 / 3.0
    else:
        a, b = (n + 1) / (2.0 * n), (n + 1) * (2 * n + 1) / (6.0 * n * n)
    mean = math.log(S) + (r - q - 0.5 * sigma**2) * T * a
    return mean, sigma**2 * T * b


@register(
    domain="finance",
    tasks=("option-pricing", "asian-options"),
    complexity=Complexity(time="O(1)", space="O(1)"),
    references=("A. G. Z. Kemna & A. C. F. Vorst, J. Banking & Finance 14 (1990) 113-129",),
)
def geometric_asian_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    kind: str = "call",
    n_fixings: int | None = None,
) -> float:
    r"""Closed-form price of a fixed-strike geometric-average Asian option.

    :math:`\ln G \sim N(m, s^2)` (module docstring), price
    :math:`e^{-rT}(e^{m + s^2/2}\Phi(d_1) - K\Phi(d_2))` for a call,
    :math:`d_1 = (m - \ln K + s^2)/s`, :math:`d_2 = d_1 - s`.

    Parameters
    ----------
    S, K, T, r, sigma, q, kind
        As in :func:`lipipy.finance.bs_price` (scalars).
    n_fixings : int, optional
        Number of equally spaced fixings :math:`t_i = iT/n`, ``i = 1..n``; None for
        continuous averaging.

    Returns
    -------
    float

    Complexity
    ----------
    O(1).

    References
    ----------
    Kemna & Vorst (1990).

    Examples
    --------
    >>> round(geometric_asian_price(100.0, 100.0, 1.0, 0.05, 0.2, n_fixings=1), 6)  # = BS
    10.450584
    """
    kind = _check_kind(kind)
    S, K, T, r, sigma, q = _scalar_inputs(S, K, T, r, sigma, q)
    n = None if n_fixings is None else check_int(n_fixings, "n_fixings", 1)
    m, v = _geometric_moments(S, T, r, sigma, q, n)
    disc = math.exp(-r * T)
    fwd = math.exp(m + 0.5 * v)
    if v == 0.0:
        return disc * max(fwd - K, 0.0) if kind == "call" else disc * max(K - fwd, 0.0)
    s = math.sqrt(v)
    d1 = (m - math.log(K) + v) / s
    d2 = d1 - s
    if kind == "call":
        return disc * (fwd * _Phi(d1) - K * _Phi(d2))
    return disc * (K * _Phi(-d2) - fwd * _Phi(-d1))


def _check_barrier(barrier_type: Any, S: float, H: Any) -> tuple[str, float, bool]:
    if barrier_type not in _BARRIERS:
        raise ValueError(f"barrier_type must be one of {_BARRIERS}, got {barrier_type!r}")
    H = check_real(H, "barrier")
    if H <= 0.0:
        raise ValueError(f"barrier must be positive, got {H!r}")
    down = barrier_type.startswith("down")
    breached = S <= H if down else S >= H
    return str(barrier_type), H, breached


@register(
    domain="finance",
    tasks=("option-pricing", "barrier-options"),
    complexity=Complexity(time="O(1)", space="O(1)"),
    references=(
        "E. Reiner & M. Rubinstein, Risk 4 (1991) 28-35",
        "E. G. Haug, The Complete Guide to Option Pricing Formulas, 2nd ed. (2007), §4.17.1",
    ),
)
def barrier_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    barrier: float,
    barrier_type: str = "down-and-out",
    q: float = 0.0,
    kind: str = "call",
) -> float:
    r"""Continuously monitored single-barrier option (Reiner-Rubinstein, no rebate).

    Parameters
    ----------
    S, K, T, r, sigma, q, kind
        As in :func:`lipipy.finance.bs_price` (scalars); ``sigma > 0`` and ``T > 0``
        unless the barrier is already breached.
    barrier : float
        Barrier level :math:`H` (> 0).
    barrier_type : {"down-and-out", "down-and-in", "up-and-out", "up-and-in"}
        Barrier direction and effect.

    Returns
    -------
    float
        In + out = vanilla (exactly, by construction of the formulas).

    Complexity
    ----------
    O(1).

    References
    ----------
    Reiner & Rubinstein (1991); Haug (2007), §4.17.1.

    Examples
    --------
    >>> out = barrier_price(100.0, 100.0, 1.0, 0.05, 0.2, 90.0, "down-and-out")
    >>> inn = barrier_price(100.0, 100.0, 1.0, 0.05, 0.2, 90.0, "down-and-in")
    >>> round(out + inn, 6)  # = Black-Scholes call
    10.450584
    """
    kind = _check_kind(kind)
    S, K, T, r, sigma, q = _scalar_inputs(S, K, T, r, sigma, q)
    btype, H, breached = _check_barrier(barrier_type, S, barrier)
    vanilla = _bs_scalar(S, K, T, r, sigma, q, kind)
    if breached:
        return 0.0 if btype.endswith("out") else vanilla
    if T == 0.0:
        return vanilla if btype.endswith("out") else 0.0
    if sigma == 0.0:
        raise ValueError("barrier_price needs sigma > 0 when the barrier is not breached")
    b = r - q
    st = sigma * math.sqrt(T)
    mu = (b - 0.5 * sigma**2) / sigma**2
    phi = 1.0 if kind == "call" else -1.0
    eta = 1.0 if btype.startswith("down") else -1.0
    x1 = math.log(S / K) / st + (1 + mu) * st
    x2 = math.log(S / H) / st + (1 + mu) * st
    y1 = math.log(H * H / (S * K)) / st + (1 + mu) * st
    y2 = math.log(H / S) / st + (1 + mu) * st
    fs, dk = S * math.exp((b - r) * T), K * math.exp(-r * T)
    hs = H / S
    A = phi * fs * _Phi(phi * x1) - phi * dk * _Phi(phi * x1 - phi * st)
    B = phi * fs * _Phi(phi * x2) - phi * dk * _Phi(phi * x2 - phi * st)
    C = phi * fs * hs ** (2 * (mu + 1)) * _Phi(eta * y1) - phi * dk * hs ** (2 * mu) * _Phi(
        eta * y1 - eta * st
    )
    D = phi * fs * hs ** (2 * (mu + 1)) * _Phi(eta * y2) - phi * dk * hs ** (2 * mu) * _Phi(
        eta * y2 - eta * st
    )
    above = K > H
    table = {
        ("down-and-in", "call"): C if above else A - B + D,
        ("up-and-in", "call"): A if above else B - C + D,
        ("down-and-in", "put"): B - C + D if above else A,
        ("up-and-in", "put"): A - B + D if above else C,
    }
    if btype.endswith("in"):
        value = table[(btype, kind)]
    else:  # out = vanilla - in keeps the parity exact
        value = vanilla - table[(btype.replace("out", "in"), kind)]
    return max(value, 0.0)


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
def _mc_setup(
    S: Any, K: Any, T: Any, r: Any, sigma: Any, q: Any, kind: Any, n_steps: Any,
    n_paths: Any, antithetic: bool, chunk_size: Any,
) -> tuple:  # fmt: skip
    kind = _check_kind(kind)
    S, K, T, r, sigma, q = _scalar_inputs(S, K, T, r, sigma, q)
    if T <= 0.0 or sigma <= 0.0:
        raise ValueError("Monte Carlo path pricing needs T > 0 and sigma > 0")
    n = check_int(n_steps, "n_steps", 1)
    n_paths = check_int(n_paths, "n_paths", 4)
    chunk = check_int(chunk_size, "chunk_size", 2)
    if antithetic and n_paths % 2:
        raise ValueError(f"n_paths must be even with antithetic=True, got {n_paths}")
    return kind, S, K, T, r, sigma, q, n, n_paths, chunk


def _log_paths(z: np.ndarray, lnS: float, drift: float, vol: float) -> np.ndarray:
    """Log-prices at t_1..t_n from normals z of shape (m, n)."""
    return lnS + np.cumsum(drift + vol * z, axis=1)


def _run(
    sample: Any, rng: np.random.Generator, n_paths: int, n: int, chunk: int, antithetic: bool,
    control_mean: float | None,
) -> MonteCarloResult:  # fmt: skip
    """Generic chunked driver: ``sample(z, u) -> (y, x)`` per path (x: control or None)."""
    n_samples = n_paths // 2 if antithetic else n_paths
    per_chunk = max(chunk // (2 * n) if antithetic else chunk // n, 1)
    acc = _PairMoments()
    remaining = n_samples
    rng_z, rng_u = spawn(rng, 2)  # separate streams: results do not depend on chunking
    while remaining > 0:
        m = min(per_chunk, remaining)
        z = rng_z.standard_normal((m, n))
        u = rng_u.random((m, n))
        y, x = sample(z, u)
        if antithetic:
            y2, x2 = sample(-z, u)
            y = 0.5 * (y + y2)
            x = None if x is None else 0.5 * (x + x2)
        acc.update(np.zeros_like(y) if x is None else x, y)
        remaining -= m
    nn = acc.n
    if control_mean is not None:
        b = acc.sxy / acc.sxx if acc.sxx > 0.0 else 0.0
        price = acc.my - b * (acc.mx - control_mean)
        se = math.sqrt(max(acc.syy - b * acc.sxy, 0.0) / (nn - 2) / nn)
        return MonteCarloResult(float(price), float(se), n_paths, n_samples, float(b))
    se = math.sqrt(acc.syy / (nn - 1) / nn)
    return MonteCarloResult(float(acc.my), float(se), n_paths, n_samples, None)


_MC_COMPLEXITY = Complexity(
    time="O(P n)", space="O(chunk_size)", notes="P paths, n monitoring steps per path"
)


@register(
    domain="finance",
    tasks=("option-pricing", "asian-options", "monte-carlo"),
    complexity=_MC_COMPLEXITY,
    references=(_GLASSERMAN + ", §4.1", "Kemna & Vorst, J. Banking & Finance 14 (1990) 113-129"),
)
def arithmetic_asian_mc(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    kind: str = "call",
    n_fixings: int = 12,
    n_paths: int = 100_000,
    antithetic: bool = True,
    control_variate: bool = True,
    random_state: RandomState = None,
    chunk_size: int = 262_144,
) -> MonteCarloResult:
    r"""Monte Carlo price of a fixed-strike arithmetic-average Asian option.

    Fixings at :math:`t_i = iT/n`; the geometric-Asian payoff with its closed-form
    mean (:func:`geometric_asian_price`) is the control variate.

    Parameters
    ----------
    S, K, T, r, sigma, q, kind
        As in :func:`lipipy.finance.bs_price`; ``T > 0`` and ``sigma > 0``.
    n_fixings : int, default=12
        Number of averaging dates :math:`n`.
    n_paths : int, default=100000
        Simulated paths (even with antithetic variates).
    antithetic, control_variate : bool, default=True
        Variance reduction switches.
    random_state : None, int, SeedSequence or Generator
        Seed.
    chunk_size : int, default=262144
        Normals per chunk (paths x fixings), bounding memory.

    Returns
    -------
    MonteCarloResult

    Complexity
    ----------
    Time O(P n), extra space O(chunk_size).

    References
    ----------
    Glasserman (2003), §4.1; Kemna & Vorst (1990).

    Examples
    --------
    >>> res = arithmetic_asian_mc(100.0, 100.0, 1.0, 0.05, 0.2, random_state=0)
    >>> g = geometric_asian_price(100.0, 100.0, 1.0, 0.05, 0.2, n_fixings=12)
    >>> res.price > g  # arithmetic mean >= geometric mean
    True
    """
    kind, S, K, T, r, sigma, q, n, n_paths, chunk = _mc_setup(
        S, K, T, r, sigma, q, kind, n_fixings, n_paths, antithetic, chunk_size
    )
    rng = check_random_state(random_state)
    dt = T / n
    drift, vol, disc = (r - q - 0.5 * sigma**2) * dt, sigma * math.sqrt(dt), math.exp(-r * T)
    sign = 1.0 if kind == "call" else -1.0
    lnS = math.log(S)

    def sample(z: np.ndarray, _u: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        x = _log_paths(z, lnS, drift, vol)
        arith = np.exp(x).mean(axis=1)
        y = disc * np.maximum(sign * (arith - K), 0.0)
        if not control_variate:
            return y, None
        geo = np.exp(x.mean(axis=1))
        return y, disc * np.maximum(sign * (geo - K), 0.0)

    control = geometric_asian_price(S, K, T, r, sigma, q, kind, n) if control_variate else None
    return _run(sample, rng, n_paths, n, chunk, antithetic, control)


@register(
    domain="finance",
    tasks=("option-pricing", "barrier-options", "monte-carlo"),
    complexity=_MC_COMPLEXITY,
    references=(_GLASSERMAN + ", §6.4", "Reiner & Rubinstein, Risk 4 (1991) 28-35"),
)
def barrier_mc(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    barrier: float,
    barrier_type: str = "down-and-out",
    q: float = 0.0,
    kind: str = "call",
    n_steps: int = 50,
    n_paths: int = 100_000,
    antithetic: bool = True,
    brownian_bridge: bool = True,
    random_state: RandomState = None,
    chunk_size: int = 262_144,
) -> MonteCarloResult:
    r"""Monte Carlo price of a single-barrier option with Brownian-bridge correction.

    With ``brownian_bridge=True`` the estimator is unbiased for the **continuously**
    monitored option (module docstring); with False the barrier is checked only at
    the ``n_steps`` grid dates (a discretely monitored option).

    Parameters
    ----------
    S, K, T, r, sigma, q, kind
        As in :func:`lipipy.finance.bs_price`; ``T > 0`` and ``sigma > 0``.
    barrier, barrier_type
        As in :func:`barrier_price`. A barrier breached at inception gives the exact
        value (0 or the vanilla) with zero standard error.
    n_steps : int, default=50
        Monitoring grid size.
    n_paths, antithetic, random_state, chunk_size
        As in :func:`arithmetic_asian_mc`.
    brownian_bridge : bool, default=True
        Continuous-monitoring correction.

    Returns
    -------
    MonteCarloResult

    Complexity
    ----------
    Time O(P n), extra space O(chunk_size).

    References
    ----------
    Glasserman (2003), §6.4.

    Examples
    --------
    >>> res = barrier_mc(100.0, 100.0, 1.0, 0.05, 0.2, 90.0, n_paths=40_000, random_state=1)
    >>> ref = barrier_price(100.0, 100.0, 1.0, 0.05, 0.2, 90.0)
    >>> abs(res.price - ref) < 4 * res.std_error
    True
    """
    kind, S, K, T, r, sigma, q, n, n_paths, chunk = _mc_setup(
        S, K, T, r, sigma, q, kind, n_steps, n_paths, antithetic, chunk_size
    )
    btype, H, breached = _check_barrier(barrier_type, S, barrier)
    rng = check_random_state(random_state)
    if breached:
        value = 0.0 if btype.endswith("out") else _bs_scalar(S, K, T, r, sigma, q, kind)
        return MonteCarloResult(value, 0.0, n_paths, n_paths // 2 if antithetic else n_paths, None)
    dt = T / n
    drift, vol, disc = (r - q - 0.5 * sigma**2) * dt, sigma * math.sqrt(dt), math.exp(-r * T)
    sign = 1.0 if kind == "call" else -1.0
    lnS, h = math.log(S), math.log(H)
    down = btype.startswith("down")
    knock_out = btype.endswith("out")

    def sample(z: np.ndarray, _u: np.ndarray) -> tuple[np.ndarray, None]:
        x = _log_paths(z, lnS, drift, vol)
        prev = np.concatenate([np.full((x.shape[0], 1), lnS), x[:, :-1]], axis=1)
        dist_now, dist_prev = (x - h, prev - h) if down else (h - x, h - prev)
        safe = np.all(dist_now > 0.0, axis=1)
        if brownian_bridge:
            with np.errstate(under="ignore"):
                cross = np.exp(
                    -2.0 * np.maximum(dist_now, 0.0) * np.maximum(dist_prev, 0.0) / (sigma**2 * dt)
                )
            survive = np.where(safe, np.prod(1.0 - cross, axis=1), 0.0)
        else:
            survive = safe.astype(float)
        payoff = disc * np.maximum(sign * (np.exp(x[:, -1]) - K), 0.0)
        return payoff * (survive if knock_out else 1.0 - survive), None

    return _run(sample, rng, n_paths, n, chunk, antithetic, None)


@register(
    domain="finance",
    tasks=("option-pricing", "lookback-options", "monte-carlo"),
    complexity=_MC_COMPLEXITY,
    references=(_GLASSERMAN + ", §6.4",),
)
def lookback_mc(
    S: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    kind: str = "call",
    n_steps: int = 50,
    n_paths: int = 100_000,
    antithetic: bool = True,
    random_state: RandomState = None,
    chunk_size: int = 262_144,
) -> MonteCarloResult:
    r"""Monte Carlo price of a floating-strike lookback option (continuous monitoring).

    Call payoff :math:`S_T - \min_{t\le T} S_t`, put :math:`\max_{t\le T} S_t - S_T`, with
    the extremum between grid dates sampled exactly from the Brownian bridge (module
    docstring), so the estimator is unbiased for continuous monitoring for any
    ``n_steps``. The running extremum starts at ``S``.

    Parameters
    ----------
    S, T, r, sigma, q, kind
        As in :func:`lipipy.finance.bs_price`; ``T > 0`` and ``sigma > 0``.
    n_steps : int, default=50
        Simulation grid size (affects variance, not bias).
    n_paths, antithetic, random_state, chunk_size
        As in :func:`arithmetic_asian_mc`.

    Returns
    -------
    MonteCarloResult

    Complexity
    ----------
    Time O(P n), extra space O(chunk_size).

    References
    ----------
    Glasserman (2003), §6.4.

    Examples
    --------
    >>> lookback_mc(100.0, 1.0, 0.05, 0.2, n_paths=10_000, random_state=0).price > 0
    True
    """
    kind, S, _, T, r, sigma, q, n, n_paths, chunk = _mc_setup(
        S, S, T, r, sigma, q, kind, n_steps, n_paths, antithetic, chunk_size
    )
    rng = check_random_state(random_state)
    dt = T / n
    drift, vol, disc = (r - q - 0.5 * sigma**2) * dt, sigma * math.sqrt(dt), math.exp(-r * T)
    lnS = math.log(S)
    call = kind == "call"

    def sample(z: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, None]:
        x = _log_paths(z, lnS, drift, vol)
        prev = np.concatenate([np.full((x.shape[0], 1), lnS), x[:, :-1]], axis=1)
        root = np.sqrt((x - prev) ** 2 - 2.0 * sigma**2 * dt * np.log(1.0 - u))
        if call:
            ext = np.min(0.5 * (x + prev - root), axis=1)
            return disc * (np.exp(x[:, -1]) - np.exp(ext)), None
        ext = np.max(0.5 * (x + prev + root), axis=1)
        return disc * (np.exp(ext) - np.exp(x[:, -1])), None

    return _run(sample, rng, n_paths, n, chunk, antithetic, None)
