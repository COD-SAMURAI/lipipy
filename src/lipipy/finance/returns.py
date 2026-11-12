r"""Returns and performance metrics.

Conventions
-----------
* **Shapes.** Prices and returns are arrays of shape ``(T,)`` (one asset) or
  ``(T, A)`` (``A`` assets in columns); rows are time, oldest first. Metrics reduce
  over time and return a ``float`` for ``(T,)`` input and an array of shape ``(A,)``
  for ``(T, A)`` input.
* **Returns are simple** (:math:`R_t = P_t/P_{t-1} - 1`) unless a function has a
  ``log=True`` option, in which case they are log returns
  :math:`r_t = \ln(P_t/P_{t-1})`.
* **Rates are per period.** ``risk_free``, ``target`` and ``benchmark`` are given in
  the same periodicity as the returns (e.g. daily); a scalar or a series aligned with
  the returns.
* **Annualisation** uses an explicit ``periods_per_year`` (default 252 trading days):
  geometric for returns, :math:`\sqrt{\text{periods\_per\_year}}` for volatilities and
  ratios.
* **Sample statistics** use ``ddof=1`` (unbiased variance) unless stated.
* **Wealth and drawdowns.** The wealth curve starts at :math:`W_0 = 1` *before* the
  first return and has ``T + 1`` points :math:`W_i = \prod_{t<i}(1 + R_t)`; for returns
  computed from ``T + 1`` prices, index ``i`` of the wealth curve is index ``i`` of the
  prices. The drawdown is :math:`D_i = W_i / \max_{j \le i} W_j - 1 \le 0`.
* **Undefined ratios.** A ratio whose denominator (a standard deviation, downside
  deviation or drawdown) is zero up to rounding (below :math:`64\,\varepsilon` times the
  data's magnitude) is :math:`\pm\infty`, or NaN for 0/0, without a warning.
* **Validation.** NaN or infinite inputs raise ``ValueError``; nothing is silently
  propagated. Simple returns below -100 % cannot be compounded and raise
  ``ValueError`` in the geometric functions.

Formulas
--------
.. math::
    \text{annualized return} = \Bigl(\prod_t (1 + R_t)\Bigr)^{N/T} - 1,\qquad
    \text{annualized volatility} = s\sqrt N,

    \text{Sharpe} = \frac{\overline{R - r_f}}{s(R - r_f)}\sqrt N,\qquad
    \text{Sortino} = \frac{\bar R - \tau}{\mathrm{DD}_\tau}\sqrt N,\qquad
    \mathrm{DD}_\tau = \sqrt{\tfrac1T\textstyle\sum_t \min(R_t - \tau, 0)^2},

    \text{information ratio} = \frac{\overline{R - B}}{s(R - B)}\sqrt N,\qquad
    \text{Calmar} = \frac{\text{annualized return}}{|\text{max drawdown}|},

with :math:`N` = ``periods_per_year`` and :math:`s` the sample standard deviation.
CAPM: ordinary least squares of :math:`R_t - r_f` on :math:`M_t - r_f`,
:math:`\beta = \widehat{\mathrm{cov}}/\widehat{\mathrm{var}}` and
:math:`\alpha = \overline{R - r_f} - \beta\,\overline{M - r_f}` per period, annualised
arithmetically as :math:`\alpha N`.

References
----------
W. F. Sharpe, "The Sharpe ratio", *J. Portfolio Management* 21 (1994) 49-58.
F. A. Sortino & L. N. Price, "Performance measurement in a downside risk framework",
*J. Investing* 3 (1994) 59-64. T. W. Young, "Calmar ratio: a smoother tool",
*Futures* 20 (1991). R. C. Grinold & R. N. Kahn, *Active Portfolio Management*, 2nd ed.
(2000), ch. 5. J. Y. Campbell, A. W. Lo & A. C. MacKinlay, *The Econometrics of
Financial Markets* (1997), ch. 5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from lipipy.core.complexity import Complexity
from lipipy.core.registry import register
from lipipy.finance._common import (
    DictMixin,
    check_aligned,
    check_int,
    check_positive_number,
    check_prices,
    check_rate,
    check_series,
    is_negligible,
    safe_ratio,
    scalar_or_array,
)

__all__ = [
    "CAPMResult",
    "DrawdownResult",
    "annualized_return",
    "annualized_volatility",
    "calmar_ratio",
    "capm_alpha_beta",
    "cumulative_returns",
    "drawdown_series",
    "information_ratio",
    "log_returns",
    "max_drawdown",
    "sharpe_ratio",
    "simple_returns",
    "sortino_ratio",
    "tracking_error",
]

_LINEAR = Complexity(time="O(T A)", space="O(T A)")
_REDUCE = Complexity(time="O(T A)", space="O(T A)", notes="one temporary of the input's size")
_BACON = "C. R. Bacon, Practical Portfolio Performance Measurement and Attribution, 2nd ed. (2008)"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _check_ppy(periods_per_year: Any) -> float:
    return check_positive_number(periods_per_year, "periods_per_year")


def _check_compoundable(r: np.ndarray) -> None:
    if np.any(r < -1.0):
        raise ValueError("simple returns below -1 (a loss above 100%) cannot be compounded")


def _log_growth(r: np.ndarray, log: bool) -> np.ndarray:
    """Per-period log growth ln(1 + R) (or r itself for log returns); -inf for R = -1."""
    if log:
        return r
    _check_compoundable(r)
    out = np.full(r.shape, -np.inf)
    alive = r > -1.0
    out[alive] = np.log1p(r[alive])
    return out


def _wealth(r: np.ndarray, log: bool) -> np.ndarray:
    """Wealth curve of T + 1 points, W_0 = 1."""
    if log:
        growth = np.exp(np.cumsum(r, axis=0))
    else:
        _check_compoundable(r)
        growth = np.cumprod(1.0 + r, axis=0)
    first = np.ones((1,) + r.shape[1:])
    return np.concatenate([first, growth], axis=0)


def _mean_over_std(x: np.ndarray, ddof: int = 1) -> np.ndarray:
    """mean(x) / std(x, ddof) per column, with a zero std mapped to +-inf / NaN."""
    if x.shape[0] <= ddof:
        raise ValueError(f"need at least {ddof + 1} observations, got {x.shape[0]}")
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=ddof)
    scale = np.max(np.abs(x), axis=0)
    return safe_ratio(mean, std, is_negligible(std, scale))


# ---------------------------------------------------------------------------
# returns
# ---------------------------------------------------------------------------
@register(
    domain="finance",
    tasks=("returns",),
    complexity=_LINEAR,
    references=(_BACON + ", ch. 2",),
)
def simple_returns(prices: ArrayLike) -> np.ndarray:
    r"""Simple (arithmetic) returns :math:`R_t = P_t / P_{t-1} - 1`.

    Parameters
    ----------
    prices : array_like of shape (T, ) or (T, A)
        Strictly positive prices, oldest first, ``T >= 2``.

    Returns
    -------
    ndarray of shape (T - 1, ) or (T - 1, A)

    Complexity
    ----------
    Time O(T A), extra space O(T A).

    References
    ----------
    C. R. Bacon, *Practical Portfolio Performance Measurement and Attribution*, ch. 2.

    Examples
    --------
    >>> simple_returns([100.0, 110.0, 99.0])
    array([ 0.1, -0.1])
    """
    p = check_prices(prices)
    return np.diff(p, axis=0) / p[:-1]


@register(
    domain="finance",
    tasks=("returns",),
    complexity=_LINEAR,
    references=(_BACON + ", ch. 2",),
)
def log_returns(prices: ArrayLike) -> np.ndarray:
    r"""Log (continuously compounded) returns :math:`r_t = \ln(P_t / P_{t-1})`.

    Computed as :math:`\operatorname{log1p}((P_t - P_{t-1})/P_{t-1})`, which keeps full
    relative accuracy for small price changes.

    Parameters
    ----------
    prices : array_like of shape (T, ) or (T, A)
        Strictly positive prices, oldest first, ``T >= 2``.

    Returns
    -------
    ndarray of shape (T - 1, ) or (T - 1, A)

    Complexity
    ----------
    Time O(T A), extra space O(T A).

    References
    ----------
    Campbell, Lo & MacKinlay, *The Econometrics of Financial Markets*, §1.4.

    Examples
    --------
    >>> import numpy as np
    >>> r = log_returns([100.0, 110.0, 99.0])
    >>> np.allclose(r, np.log([1.1, 0.9]))
    True
    """
    p = check_prices(prices)
    return np.log1p(np.diff(p, axis=0) / p[:-1])


@register(
    domain="finance",
    tasks=("returns",),
    complexity=_LINEAR,
    references=(_BACON + ", ch. 2",),
)
def cumulative_returns(returns: ArrayLike, log: bool = False) -> np.ndarray:
    r"""Cumulative return after each period, :math:`\prod_{s\le t}(1 + R_s) - 1`.

    Parameters
    ----------
    returns : array_like of shape (T, ) or (T, A)
        Per-period returns.
    log : bool, default=False
        If True the input holds log returns and the result is
        :math:`\exp(\sum_{s\le t} r_s) - 1` (still a simple cumulative return).

    Returns
    -------
    ndarray, same shape as ``returns``

    Complexity
    ----------
    Time O(T A), extra space O(T A).

    References
    ----------
    C. R. Bacon, *Practical Portfolio Performance Measurement and Attribution*, ch. 2.

    Examples
    --------
    >>> cumulative_returns([0.1, -0.1])
    array([ 0.1 , -0.01])
    """
    r = check_series(returns)
    if log:
        return np.expm1(np.cumsum(r, axis=0))
    _check_compoundable(r)
    return np.cumprod(1.0 + r, axis=0) - 1.0


@register(
    domain="finance",
    tasks=("returns", "performance"),
    complexity=_REDUCE,
    references=(_BACON + ", ch. 2",),
)
def annualized_return(
    returns: ArrayLike, periods_per_year: float = 252, log: bool = False
) -> float | np.ndarray:
    r"""Geometric annualised return (CAGR) :math:`(\prod_t(1+R_t))^{N/T} - 1`.

    Evaluated as :math:`\operatorname{expm1}(\tfrac NT\sum_t \operatorname{log1p} R_t)`.
    A period with :math:`R_t = -1` gives exactly -1.

    Parameters
    ----------
    returns : array_like of shape (T, ) or (T, A)
        Per-period simple returns (log returns if ``log=True``).
    periods_per_year : float, default=252
        Number of return periods per year, :math:`N`.
    log : bool, default=False
        Whether ``returns`` are log returns.

    Returns
    -------
    float or ndarray of shape (A, )

    Complexity
    ----------
    Time O(T A), extra space O(T A).

    References
    ----------
    C. R. Bacon, *Practical Portfolio Performance Measurement and Attribution*, ch. 2.

    Examples
    --------
    >>> round(annualized_return([0.21, 0.0], periods_per_year=1), 12)  # 1.21 over 2 years
    0.1
    """
    r = check_series(returns)
    n = _check_ppy(periods_per_year)
    g = _log_growth(r, log)
    total = g.sum(axis=0)
    wiped = np.isneginf(total)  # some R_t = -1: wealth is 0 for good
    out = np.expm1(np.where(wiped, 0.0, total) * (n / r.shape[0]))
    return scalar_or_array(np.where(wiped, -1.0, out))


@register(
    domain="finance",
    tasks=("returns", "risk"),
    complexity=_REDUCE,
    references=(_BACON + ", ch. 4",),
)
def annualized_volatility(
    returns: ArrayLike, periods_per_year: float = 252, ddof: int = 1
) -> float | np.ndarray:
    r"""Annualised volatility :math:`s\sqrt N` (sample standard deviation times :math:`\sqrt N`).

    Parameters
    ----------
    returns : array_like of shape (T, ) or (T, A)
        Per-period returns, ``T > ddof``.
    periods_per_year : float, default=252
        Number of return periods per year, :math:`N`.
    ddof : int, default=1
        Delta degrees of freedom of the standard deviation.

    Returns
    -------
    float or ndarray of shape (A, )

    Complexity
    ----------
    Time O(T A), extra space O(T A).

    References
    ----------
    C. R. Bacon, *Practical Portfolio Performance Measurement and Attribution*, ch. 4.

    Examples
    --------
    >>> round(annualized_volatility([0.01, -0.01], periods_per_year=4), 12)
    0.028284271247
    """
    r = check_series(returns)
    n = _check_ppy(periods_per_year)
    ddof = check_int(ddof, "ddof", 0)
    if r.shape[0] <= ddof:
        raise ValueError(f"need more than ddof={ddof} observations, got {r.shape[0]}")
    return scalar_or_array(r.std(axis=0, ddof=ddof) * np.sqrt(n))


@register(
    domain="finance",
    tasks=("performance",),
    complexity=_REDUCE,
    references=("W. F. Sharpe, J. Portfolio Management 21 (1994) 49-58",),
)
def sharpe_ratio(
    returns: ArrayLike, risk_free: ArrayLike = 0.0, periods_per_year: float = 252
) -> float | np.ndarray:
    r"""Annualised Sharpe ratio :math:`\sqrt N\,\overline{R - r_f} / s(R - r_f)`.

    Parameters
    ----------
    returns : array_like of shape (T, ) or (T, A)
        Per-period simple returns, ``T >= 2``.
    risk_free : float or array_like of shape (T, ), default=0.0
        Per-period risk-free return (same periodicity as ``returns``).
    periods_per_year : float, default=252
        Number of return periods per year, :math:`N`.

    Returns
    -------
    float or ndarray of shape (A, )
        ``+-inf`` (or NaN) when the excess returns have zero dispersion.

    Complexity
    ----------
    Time O(T A), extra space O(T A).

    References
    ----------
    W. F. Sharpe, "The Sharpe ratio", *J. Portfolio Management* 21 (1994) 49-58.

    Examples
    --------
    >>> round(sharpe_ratio([0.01, 0.03], periods_per_year=1), 12)  # mean 0.02, s 0.01414
    1.414213562373
    """
    r = check_series(returns)
    rf = check_rate(risk_free, r, "risk_free")
    n = _check_ppy(periods_per_year)
    return scalar_or_array(_mean_over_std(r - rf) * np.sqrt(n))


@register(
    domain="finance",
    tasks=("performance",),
    complexity=_REDUCE,
    references=("F. A. Sortino & L. N. Price, J. Investing 3 (1994) 59-64",),
)
def sortino_ratio(
    returns: ArrayLike, target: ArrayLike = 0.0, periods_per_year: float = 252
) -> float | np.ndarray:
    r"""Annualised Sortino ratio :math:`\sqrt N\,(\bar R - \tau) / \mathrm{DD}_\tau`.

    The downside deviation below the target :math:`\tau` uses **all** ``T``
    observations (Sortino & Price 1994):

    .. math::  \mathrm{DD}_\tau = \sqrt{\tfrac1T\textstyle\sum_t \min(R_t - \tau, 0)^2}.

    Parameters
    ----------
    returns : array_like of shape (T, ) or (T, A)
        Per-period simple returns.
    target : float or array_like of shape (T, ), default=0.0
        Per-period minimum acceptable return :math:`\tau`.
    periods_per_year : float, default=252
        Number of return periods per year, :math:`N`.

    Returns
    -------
    float or ndarray of shape (A, )
        ``+inf`` when no return falls below the target and the mean excess is positive.

    Complexity
    ----------
    Time O(T A), extra space O(T A).

    References
    ----------
    F. A. Sortino & L. N. Price, *J. Investing* 3 (1994) 59-64.

    Examples
    --------
    >>> round(sortino_ratio([0.02, -0.01, 0.03, -0.02], periods_per_year=1), 12)
    0.4472135955
    """
    r = check_series(returns)
    tau = check_rate(target, r, "target")
    n = _check_ppy(periods_per_year)
    excess = r - tau
    downside = np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2, axis=0))
    scale = np.max(np.abs(excess), axis=0)
    ratio = safe_ratio(excess.mean(axis=0), downside, is_negligible(downside, scale))
    return scalar_or_array(ratio * np.sqrt(n))


# ---------------------------------------------------------------------------
# drawdowns
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DrawdownResult(DictMixin):
    """Maximum drawdown of one return series (see :func:`max_drawdown`).

    Indices refer to the wealth curve :math:`W_0 = 1, W_1, \\dots, W_T` (index ``i`` =
    after ``i`` returns, i.e. price index ``i``).

    Attributes
    ----------
    max_drawdown : float
        :math:`\\min_i D_i \\le 0` (e.g. -0.25 for a 25 % peak-to-trough loss).
    peak_index : int
        Last index of the running maximum before the trough.
    trough_index : int
        Index of the lowest point of the worst drawdown (first one if tied).
    recovery_index : int or None
        First index after the trough where wealth regains the peak; None if never.
    duration : int
        Periods from peak to recovery, or to the last index if not recovered.
    recovered : bool
        Whether the peak was regained.
    """

    max_drawdown: float
    peak_index: int
    trough_index: int
    recovery_index: int | None
    duration: int
    recovered: bool


@register(
    domain="finance",
    tasks=("drawdown", "risk"),
    complexity=_LINEAR,
    references=(_BACON + ", ch. 4",),
)
def drawdown_series(returns: ArrayLike, log: bool = False) -> np.ndarray:
    r"""Drawdown of the wealth curve, :math:`D_i = W_i / \max_{j\le i} W_j - 1`.

    The wealth curve has ``T + 1`` points and starts at :math:`W_0 = 1` (module
    docstring), so a loss in the first period already counts as a drawdown and
    :math:`D_0 = 0`.

    Parameters
    ----------
    returns : array_like of shape (T, ) or (T, A)
        Per-period simple returns (log returns if ``log=True``).
    log : bool, default=False
        Whether ``returns`` are log returns.

    Returns
    -------
    ndarray of shape (T + 1, ) or (T + 1, A)
        Non-positive drawdowns.

    Complexity
    ----------
    Time O(T A), extra space O(T A).

    References
    ----------
    C. R. Bacon, *Practical Portfolio Performance Measurement and Attribution*, ch. 4.

    Examples
    --------
    >>> drawdown_series([0.5, -0.2, -0.25, 0.8])
    array([ 0. ,  0. , -0.2, -0.4,  0. ])
    """
    r = check_series(returns)
    w = _wealth(r, log)
    return w / np.maximum.accumulate(w, axis=0) - 1.0


@register(
    domain="finance",
    tasks=("drawdown", "risk"),
    complexity=_LINEAR,
    references=(_BACON + ", ch. 4",),
)
def max_drawdown(returns: ArrayLike, log: bool = False) -> DrawdownResult:
    r"""Maximum drawdown with its peak, trough, recovery and duration.

    Parameters
    ----------
    returns : array_like of shape (T, )
        Per-period simple returns of one series (log returns if ``log=True``).
        For several assets, call it per column or use :func:`drawdown_series`.
    log : bool, default=False
        Whether ``returns`` are log returns.

    Returns
    -------
    DrawdownResult
        ``max_drawdown`` (<= 0), ``peak_index``, ``trough_index``, ``recovery_index``,
        ``duration``, ``recovered``; indices on the ``T + 1``-point wealth curve. With
        no drawdown at all every index is 0.

    Complexity
    ----------
    Time O(T), extra space O(T).

    References
    ----------
    C. R. Bacon, *Practical Portfolio Performance Measurement and Attribution*, ch. 4.

    Examples
    --------
    >>> res = max_drawdown([0.5, -0.2, -0.25, 0.8])
    >>> round(res.max_drawdown, 12), res.peak_index, res.trough_index, res.recovery_index
    (-0.4, 1, 3, 4)
    >>> res.duration, res.recovered
    (3, True)
    """
    r = check_series(returns)
    if r.ndim != 1:
        raise ValueError(f"max_drawdown takes one series of shape (T,), got {r.shape}")
    w = _wealth(r, log)
    running = np.maximum.accumulate(w)
    dd = w / running - 1.0
    trough = int(np.argmin(dd))
    mdd = float(dd[trough])
    if mdd == 0.0:
        return DrawdownResult(0.0, 0, 0, 0, 0, True)
    peak_value = running[trough]
    peak = int(np.flatnonzero(w[: trough + 1] == peak_value)[-1])
    after = np.flatnonzero(w[trough + 1 :] >= peak_value)
    if after.size:
        recovery = trough + 1 + int(after[0])
        return DrawdownResult(mdd, peak, trough, recovery, recovery - peak, True)
    return DrawdownResult(mdd, peak, trough, None, w.shape[0] - 1 - peak, False)


@register(
    domain="finance",
    tasks=("performance", "drawdown"),
    complexity=_LINEAR,
    references=("T. W. Young, Calmar ratio: a smoother tool, Futures 20 (1991)",),
)
def calmar_ratio(
    returns: ArrayLike, periods_per_year: float = 252, log: bool = False
) -> float | np.ndarray:
    r"""Calmar ratio: annualised return divided by the magnitude of the maximum drawdown.

    Computed over the whole sample (the original definition uses the last 36 months;
    pass that window yourself). ``+inf`` when there is no drawdown and the return is
    positive, NaN for zero over zero.

    Parameters
    ----------
    returns : array_like of shape (T, ) or (T, A)
        Per-period simple returns (log returns if ``log=True``).
    periods_per_year : float, default=252
        Number of return periods per year, :math:`N`.
    log : bool, default=False
        Whether ``returns`` are log returns.

    Returns
    -------
    float or ndarray of shape (A, )

    Complexity
    ----------
    Time O(T A), extra space O(T A).

    References
    ----------
    T. W. Young, "Calmar ratio: a smoother tool", *Futures* 20 (1991).

    Examples
    --------
    >>> round(calmar_ratio([0.5, -0.2, -0.25, 0.8], periods_per_year=4), 12)  # 62% / 40%
    1.55
    """
    r = check_series(returns)
    ann = np.asarray(annualized_return(r, periods_per_year, log=log))
    mdd = -np.min(drawdown_series(r, log=log), axis=0)
    return scalar_or_array(safe_ratio(ann, mdd, mdd == 0.0))


@register(
    domain="finance",
    tasks=("performance",),
    complexity=_REDUCE,
    references=("Grinold & Kahn, Active Portfolio Management, 2nd ed. (2000), ch. 5",),
)
def information_ratio(
    returns: ArrayLike, benchmark: ArrayLike, periods_per_year: float = 252
) -> float | np.ndarray:
    r"""Annualised information ratio :math:`\sqrt N\,\overline{R - B} / s(R - B)`.

    :math:`s(R - B)\sqrt N` is the annualised tracking error.

    Parameters
    ----------
    returns : array_like of shape (T, ) or (T, A)
        Per-period simple returns of the portfolio(s), ``T >= 2``.
    benchmark : array_like of shape (T, ) or (T, A)
        Per-period benchmark returns (one benchmark is shared by all columns).
    periods_per_year : float, default=252
        Number of return periods per year, :math:`N`.

    Returns
    -------
    float or ndarray of shape (A, )

    Complexity
    ----------
    Time O(T A), extra space O(T A).

    References
    ----------
    Grinold & Kahn, *Active Portfolio Management*, 2nd ed., ch. 5.

    Examples
    --------
    >>> round(information_ratio([0.02, 0.04], [0.01, 0.02], periods_per_year=1), 12)
    2.12132034356
    """
    r = check_series(returns)
    b = check_aligned(benchmark, r, "benchmark")
    n = _check_ppy(periods_per_year)
    return scalar_or_array(_mean_over_std(r - b) * np.sqrt(n))


@register(
    domain="finance",
    tasks=("performance", "risk"),
    complexity=_REDUCE,
    references=("Grinold & Kahn, Active Portfolio Management, 2nd ed. (2000), ch. 5",),
)
def tracking_error(
    returns: ArrayLike, benchmark: ArrayLike, periods_per_year: float = 252
) -> float | np.ndarray:
    r"""Annualised tracking error :math:`s(R - B)\sqrt N` (``ddof=1``).

    Parameters
    ----------
    returns : array_like of shape (T, ) or (T, A)
        Per-period simple returns of the portfolio(s), ``T >= 2``.
    benchmark : array_like of shape (T, ) or (T, A)
        Per-period benchmark returns.
    periods_per_year : float, default=252
        Number of return periods per year, :math:`N`.

    Returns
    -------
    float or ndarray of shape (A, )

    Complexity
    ----------
    Time O(T A), extra space O(T A).

    References
    ----------
    Grinold & Kahn, *Active Portfolio Management*, 2nd ed., ch. 5.

    Examples
    --------
    >>> round(tracking_error([0.02, 0.04], [0.01, 0.02], periods_per_year=1), 12)
    0.007071067812
    """
    r = check_series(returns, min_rows=2)
    b = check_aligned(benchmark, r, "benchmark")
    n = _check_ppy(periods_per_year)
    return scalar_or_array((r - b).std(axis=0, ddof=1) * np.sqrt(n))


@dataclass(frozen=True)
class CAPMResult(DictMixin):
    """CAPM regression :math:`R - r_f = \\alpha + \\beta (M - r_f) + \\varepsilon`.

    Attributes
    ----------
    alpha : float or ndarray
        Annualised (arithmetic) alpha, ``alpha_per_period * periods_per_year``.
    beta : float or ndarray
        Market beta.
    alpha_per_period : float or ndarray
        OLS intercept, in return units per period.
    r_squared : float or ndarray
        Coefficient of determination of the regression.
    """

    alpha: Any
    beta: Any
    alpha_per_period: Any
    r_squared: Any


@register(
    domain="finance",
    tasks=("performance", "regression"),
    complexity=_REDUCE,
    references=("Campbell, Lo & MacKinlay, The Econometrics of Financial Markets (1997), ch. 5",),
)
def capm_alpha_beta(
    returns: ArrayLike,
    market: ArrayLike,
    risk_free: ArrayLike = 0.0,
    periods_per_year: float = 252,
) -> CAPMResult:
    r"""CAPM alpha and beta by ordinary least squares on excess returns.

    With :math:`y = R - r_f`, :math:`x = M - r_f` (per period):
    :math:`\beta = \sum (x-\bar x)(y - \bar y) / \sum (x - \bar x)^2`,
    :math:`\alpha = \bar y - \beta\bar x`, annualised as :math:`\alpha N`.

    Parameters
    ----------
    returns : array_like of shape (T, ) or (T, A)
        Per-period asset returns, ``T >= 2``.
    market : array_like of shape (T, )
        Per-period market returns; must not be constant.
    risk_free : float or array_like of shape (T, ), default=0.0
        Per-period risk-free return.
    periods_per_year : float, default=252
        Number of return periods per year, :math:`N`.

    Returns
    -------
    CAPMResult
        ``alpha`` (annualised), ``beta``, ``alpha_per_period``, ``r_squared``.

    Complexity
    ----------
    Time O(T A), extra space O(T A).

    References
    ----------
    Campbell, Lo & MacKinlay, *The Econometrics of Financial Markets*, ch. 5.

    Examples
    --------
    >>> res = capm_alpha_beta([0.03, 0.05, 0.01], [0.01, 0.02, 0.0], periods_per_year=12)
    >>> round(res.beta, 12), round(res.alpha_per_period, 12), round(res.alpha, 12)
    (2.0, 0.01, 0.12)
    """
    r = check_series(returns, min_rows=2)
    m = check_series(market, "market", min_rows=2)
    if m.ndim != 1:
        raise ValueError(f"market must have shape (T,), got {m.shape}")
    if m.shape[0] != r.shape[0]:
        raise ValueError(f"market has {m.shape[0]} rows but returns has {r.shape[0]}")
    rf = check_rate(risk_free, r, "risk_free")
    n = _check_ppy(periods_per_year)
    y = r - rf
    if np.ndim(rf) == 2 and np.shape(rf)[1] != 1:
        raise ValueError("risk_free must be a scalar or a series of shape (T,) for CAPM")
    rf_x = rf if np.ndim(rf) == 0 else np.asarray(rf).reshape(r.shape[0])
    x = m - rf_x
    xc = x - x.mean()
    sxx = float(xc @ xc)
    if is_negligible(np.sqrt(sxx / x.size), np.max(np.abs(x))):
        raise ValueError("market excess returns are constant; beta is undefined")
    yc = y - y.mean(axis=0)
    beta = (xc @ yc) / sxx
    alpha = y.mean(axis=0) - beta * x.mean()
    resid = yc - np.multiply.outer(xc, beta) if yc.ndim == 2 else yc - xc * beta
    syy = np.sum(yc**2, axis=0)
    ssr = np.sum(resid**2, axis=0)
    r2 = safe_ratio(syy - ssr, syy, syy == 0.0)
    return CAPMResult(
        alpha=scalar_or_array(alpha * n),
        beta=scalar_or_array(beta),
        alpha_per_period=scalar_or_array(alpha),
        r_squared=scalar_or_array(r2),
    )
