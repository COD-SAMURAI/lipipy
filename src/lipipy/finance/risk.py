r"""Value at risk, expected shortfall and covariance estimation.

Conventions
-----------
* Returns have shape ``(T,)`` or ``(T, A)``, rows are time (see
  :mod:`lipipy.finance.returns`); risk measures are computed per column.
* **VaR and ES are reported as positive losses** in return units: at confidence
  ``level`` (e.g. 0.95) with tail probability :math:`p = 1 - \text{level}`,
  :math:`\mathrm{VaR} = -q_p(R)` and :math:`\mathrm{ES} = -\mathbb E[R \mid R \le q_p]`.
  A negative VaR means even the tail quantile is a gain.
* One-period horizon: no square-root-of-time scaling is applied.

Methods
-------
``"historical"``
    Lower empirical quantile: with returns sorted ascending,
    :math:`r_{(1)} \le \dots \le r_{(T)}`, :math:`q_p = r_{(k)}` with
    :math:`k = \lceil Tp \rceil` (the smallest observation whose empirical CDF is
    at least :math:`p`; NumPy's ``method="inverted_cdf"``). :math:`Tp` within
    :math:`10^{-9}` of an integer is rounded to it, so e.g. ``level=0.95`` with
    ``T=100`` uses the 5th smallest return, not the 6th. ES is the Acerbi-Tasche
    estimator, the average over the worst :math:`Tp` observations with the
    :math:`k`-th one weighted fractionally:

    .. math::  \mathrm{ES} = -\frac{1}{Tp}\Bigl(\sum_{i<k} r_{(i)}
               + (Tp - k + 1)\, r_{(k)}\Bigr).

    It equals the mean of the :math:`k` worst returns when :math:`Tp` is an integer,
    and always satisfies :math:`\mathrm{ES} \ge \mathrm{VaR}`.
``"gaussian"``
    :math:`R \sim N(\mu, \sigma^2)` with the sample mean and standard deviation
    (``ddof=1``), :math:`z_p = \Phi^{-1}(p)`:
    :math:`\mathrm{VaR} = -(\mu + \sigma z_p)`,
    :math:`\mathrm{ES} = -\mu + \sigma\,\varphi(z_p)/p`.
``"cornish_fisher"``
    The normal quantile is corrected with the sample skewness :math:`S` and excess
    kurtosis :math:`K` (biased moment ratios :math:`m_3/m_2^{3/2}` and
    :math:`m_4/m_2^2 - 3`, as ``scipy.stats.skew`` / ``kurtosis`` with defaults):

    .. math::  g(z) = z + \frac{(z^2-1)S}{6} + \frac{(z^3-3z)K}{24}
               - \frac{(2z^3-5z)S^2}{36},

    :math:`\mathrm{VaR} = -(\mu + \sigma g(z_p))`. ES is the tail mean of the same
    transformed variable, :math:`\mathrm{ES} = -(\mu + \sigma\,\mathbb E[g(Z)\mid
    Z\le z_p])`, in closed form from the truncated normal moments
    :math:`\mathbb E[Z^j \mathbf 1\{Z\le c\}]` = :math:`\Phi(c)`, :math:`-\varphi(c)`,
    :math:`\Phi(c) - c\varphi(c)`, :math:`-(c^2+2)\varphi(c)` for ``j = 0..3``. It is
    the exact ES of :math:`\mu + \sigma g(Z)` when :math:`g` is increasing on the tail
    (moderate :math:`S, K`), and reduces to the Gaussian ES when :math:`S = K = 0`.

Covariance estimators (input ``(T, A)``, output ``(A, A)``)
-----------------------------------------------------------
* ``sample_covariance``: :math:`X_c^\top X_c/(T - \text{ddof})`.
* ``ewma_covariance`` (RiskMetrics): zero-mean, exponentially weighted

  .. math::  \Sigma = \frac{\sum_{t} \lambda^{T-1-t} r_t r_t^\top}{\sum_t \lambda^{T-1-t}},

  i.e. the recursion :math:`S_t = \lambda S_{t-1} + r_t r_t^\top`,
  :math:`N_t = \lambda N_{t-1} + 1`, :math:`\Sigma = S_T/N_T` from :math:`S_0 = 0`.
  Normalising by :math:`N_T = (1-\lambda^T)/(1-\lambda)` makes the weights sum to one
  in a finite sample; for long samples it equals the textbook recursion
  :math:`\Sigma_t = \lambda\Sigma_{t-1} + (1-\lambda) r_t r_t^\top`.
* ``ledoit_wolf``: shrinkage towards :math:`m I`, :math:`m = \operatorname{tr}S/A`,
  with :math:`S = X_c^\top X_c / T` (centred, ``ddof=0``):

  .. math::  d^2 = \lVert S - mI\rVert_F^2/A,\quad
             \bar b^2 = \frac{1}{T^2 A}\sum_t \lVert x_t x_t^\top - S\rVert_F^2
             = \frac{\sum_t \lVert x_t\rVert^4 - T\lVert S\rVert_F^2}{T^2 A},

  :math:`b^2 = \min(\bar b^2, d^2)`, shrinkage :math:`\delta = b^2/d^2` (0 if
  :math:`d^2 = 0` or :math:`A = 1`), :math:`\hat\Sigma = (1-\delta)S + \delta m I`.
  This matches ``sklearn.covariance.LedoitWolf``.

References
----------
P. Jorion, *Value at Risk*, 3rd ed. (2007). C. Acerbi & D. Tasche, "On the coherence of
expected shortfall", *J. Banking & Finance* 26 (2002) 1487-1503. E. A. Cornish &
R. A. Fisher, "Moments and cumulants in the specification of distributions",
*Rev. Int. Statist. Inst.* 5 (1938) 307-320. J.P. Morgan/Reuters, *RiskMetrics
Technical Document*, 4th ed. (1996). O. Ledoit & M. Wolf, "A well-conditioned
estimator for large-dimensional covariance matrices", *J. Multivariate Anal.* 88
(2004) 365-411.
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
    check_int,
    check_level,
    check_real,
    check_series,
    is_negligible,
    scalar_or_array,
)
from lipipy.numerics.special import norm_cdf, norm_ppf
from lipipy.numerics.stats import covariance

__all__ = [
    "LedoitWolfResult",
    "ewma_covariance",
    "expected_shortfall",
    "ledoit_wolf",
    "sample_covariance",
    "value_at_risk",
]

_METHODS = ("historical", "gaussian", "cornish_fisher")
_JORION = "P. Jorion, Value at Risk, 3rd ed. (2007)"
_ACERBI = "C. Acerbi & D. Tasche, J. Banking & Finance 26 (2002) 1487-1503"
_CF = "E. A. Cornish & R. A. Fisher, Rev. Int. Statist. Inst. 5 (1938) 307-320"
_VAR_COMPLEXITY = Complexity(
    time="O(T A log T)",
    space="O(T A)",
    notes="historical sorts each column; gaussian and cornish_fisher are O(T A)",
)
_SQRT2PI = float(np.sqrt(2.0 * np.pi))


def _check_method(method: Any) -> str:
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    return str(method)


def _phi(x: np.ndarray | float) -> np.ndarray:
    return np.exp(-0.5 * np.square(x)) / _SQRT2PI


def _historical_k(T: int, p: float) -> tuple[int, float]:
    """1-based order-statistic index k = ceil(T p) and the tail mass T p."""
    tp = T * p
    nearest = round(tp)
    if abs(tp - nearest) <= 1e-9 * max(1.0, tp):
        tp = float(nearest)
    k = max(1, int(np.ceil(tp)))
    return k, tp


def _moments(r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean, sd (ddof=1), skewness and excess kurtosis (biased ratios) per column."""
    if r.shape[0] < 2:
        raise ValueError("parametric methods need at least 2 observations")
    mu = r.mean(axis=0)
    rc = r - mu
    m2 = np.mean(rc**2, axis=0)
    sd = np.sqrt(m2 * r.shape[0] / (r.shape[0] - 1))
    flat = is_negligible(np.sqrt(m2), np.max(np.abs(r), axis=0))
    safe_m2 = np.where(flat, 1.0, m2)
    skew = np.where(flat, 0.0, np.mean(rc**3, axis=0) / safe_m2**1.5)
    kurt = np.where(flat, 0.0, np.mean(rc**4, axis=0) / safe_m2**2 - 3.0)
    sd = np.where(flat, 0.0, sd)
    return mu, sd, skew, kurt


def _cornish_fisher_z(z: float, skew: Any, kurt: Any) -> Any:
    """Cornish-Fisher expansion g(z) of the standardised quantile."""
    return (
        z
        + (z**2 - 1.0) * skew / 6.0
        + (z**3 - 3.0 * z) * kurt / 24.0
        - (2.0 * z**3 - 5.0 * z) * skew**2 / 36.0
    )


def _cornish_fisher_tail_mean(c: float, skew: Any, kurt: Any) -> Any:
    """E[g(Z) | Z <= c] for Z ~ N(0, 1), from the truncated moments of Z."""
    Phi = float(norm_cdf(c))
    phi = float(_phi(c))
    m1 = -phi / Phi
    m2 = (Phi - c * phi) / Phi
    m3 = -(c * c + 2.0) * phi / Phi
    return (
        m1
        + (m2 - 1.0) * skew / 6.0
        + (m3 - 3.0 * m1) * kurt / 24.0
        - (2.0 * m3 - 5.0 * m1) * skew**2 / 36.0
    )


def _risk(returns: ArrayLike, level: Any, method: Any, es: bool) -> Any:
    r = check_series(returns)
    lv = check_level(level)
    method = _check_method(method)
    p = 1.0 - lv
    if method == "historical":
        T = r.shape[0]
        k, tp = _historical_k(T, p)
        srt = np.sort(r, axis=0)
        q = srt[k - 1]
        if not es:
            return scalar_or_array(-q)
        tail = srt[: k - 1].sum(axis=0) + (tp - (k - 1)) * q
        return scalar_or_array(-tail / tp)
    mu, sd, skew, kurt = _moments(r)
    z = float(norm_ppf(p))
    if method == "gaussian":
        x = float(-_phi(z) / p) if es else z
    else:
        x = _cornish_fisher_tail_mean(z, skew, kurt) if es else _cornish_fisher_z(z, skew, kurt)
    return scalar_or_array(-(mu + sd * x))


@register(
    domain="finance",
    tasks=("risk", "value-at-risk"),
    complexity=_VAR_COMPLEXITY,
    references=(_JORION, _CF),
)
def value_at_risk(
    returns: ArrayLike, level: float = 0.95, method: str = "historical"
) -> float | np.ndarray:
    r"""One-period value at risk, reported as a positive loss.

    :math:`\mathrm{VaR} = -q_p(R)` with :math:`p = 1 - \text{level}`; see the module
    docstring for the quantile convention of each method.

    Parameters
    ----------
    returns : array_like of shape (T, ) or (T, A)
        Per-period returns.
    level : float, default=0.95
        Confidence level in (0, 1).
    method : {"historical", "gaussian", "cornish_fisher"}, default="historical"
        Lower empirical quantile, normal quantile, or Cornish-Fisher corrected
        normal quantile. Parametric methods need ``T >= 2``.

    Returns
    -------
    float or ndarray of shape (A, )

    Complexity
    ----------
    Time O(T A log T) for ``"historical"`` (sort), O(T A) otherwise; space O(T A).

    References
    ----------
    P. Jorion, *Value at Risk*, 3rd ed.; Cornish & Fisher (1938).

    Examples
    --------
    >>> r = [-0.05, -0.02, 0.0, 0.01, 0.03, -0.01, 0.02, 0.04, -0.03, 0.01]
    >>> value_at_risk(r, level=0.8)  # T p = 2: second-worst return
    0.03
    >>> round(value_at_risk(r, level=0.95, method="gaussian"), 6)
    0.045873
    """
    return _risk(returns, level, method, es=False)


@register(
    domain="finance",
    tasks=("risk", "expected-shortfall"),
    complexity=_VAR_COMPLEXITY,
    references=(_ACERBI, _JORION, _CF),
)
def expected_shortfall(
    returns: ArrayLike, level: float = 0.95, method: str = "historical"
) -> float | np.ndarray:
    r"""Expected shortfall (conditional VaR), the mean loss beyond the VaR quantile.

    :math:`\mathrm{ES} = -\mathbb E[R \mid R \le q_p]`, :math:`p = 1 - \text{level}`;
    historical uses the Acerbi-Tasche estimator, gaussian and cornish_fisher the
    closed forms in the module docstring. Always :math:`\ge` the VaR of the same
    method (for cornish_fisher, while :math:`g` is increasing on the tail).

    Parameters
    ----------
    returns : array_like of shape (T, ) or (T, A)
        Per-period returns.
    level : float, default=0.95
        Confidence level in (0, 1).
    method : {"historical", "gaussian", "cornish_fisher"}, default="historical"
        Estimation method.

    Returns
    -------
    float or ndarray of shape (A, )

    Complexity
    ----------
    Time O(T A log T) for ``"historical"`` (sort), O(T A) otherwise; space O(T A).

    References
    ----------
    Acerbi & Tasche (2002); P. Jorion, *Value at Risk*, 3rd ed.

    Examples
    --------
    >>> r = [-0.05, -0.02, 0.0, 0.01, 0.03, -0.01, 0.02, 0.04, -0.03, 0.01]
    >>> round(expected_shortfall(r, level=0.8), 12)  # mean of the two worst
    0.04
    """
    return _risk(returns, level, method, es=True)


# ---------------------------------------------------------------------------
# covariance estimators
# ---------------------------------------------------------------------------
def _check_panel(returns: ArrayLike, min_rows: int) -> np.ndarray:
    r = check_series(returns, min_rows=min_rows)
    if r.ndim != 2:
        raise ValueError(f"returns must have shape (T, A), got {r.shape}")
    return r


@register(
    domain="finance",
    tasks=("risk", "covariance-estimation"),
    complexity=Complexity(time="O(T A^2)", space="O(T A)"),
    references=("Chan, Golub & LeVeque, The American Statistician 37 (1983) 242-247",),
)
def sample_covariance(returns: ArrayLike, ddof: int = 1) -> np.ndarray:
    r"""Sample covariance matrix :math:`X_c^\top X_c / (T - \text{ddof})` of the columns.

    Thin wrapper over :func:`lipipy.numerics.covariance` with finance input checks.

    Parameters
    ----------
    returns : array_like of shape (T, A)
        Per-period returns, ``T > ddof``.
    ddof : int, default=1
        Delta degrees of freedom.

    Returns
    -------
    ndarray of shape (A, A)

    Complexity
    ----------
    Time O(T A^2), extra space O(T A).

    References
    ----------
    Chan, Golub & LeVeque (1983).

    Examples
    --------
    >>> sample_covariance([[0.01, 0.02], [0.03, 0.06]])
    array([[0.0002, 0.0004],
           [0.0004, 0.0008]])
    """
    ddof = check_int(ddof, "ddof", 0)
    r = _check_panel(returns, ddof + 1)
    return covariance(r, ddof=ddof)


@register(
    domain="finance",
    tasks=("risk", "covariance-estimation"),
    complexity=Complexity(time="O(T A^2)", space="O(T A)"),
    references=("J.P. Morgan/Reuters, RiskMetrics Technical Document, 4th ed. (1996)",),
)
def ewma_covariance(returns: ArrayLike, lam: float = 0.94) -> np.ndarray:
    r"""RiskMetrics exponentially weighted (zero-mean) covariance matrix.

    :math:`\Sigma = \sum_t w_t r_t r_t^\top` with weights
    :math:`w_t \propto \lambda^{T-1-t}` normalised to sum to one (most recent
    observation weighted most); see the module docstring for the recursion.

    Parameters
    ----------
    returns : array_like of shape (T, A)
        Per-period returns, oldest first.
    lam : float, default=0.94
        Decay factor in (0, 1); 0.94 is the RiskMetrics daily value.

    Returns
    -------
    ndarray of shape (A, A)

    Complexity
    ----------
    Time O(T A^2), extra space O(T A).

    References
    ----------
    J.P. Morgan/Reuters, *RiskMetrics Technical Document*, 4th ed. (1996), §5.2.

    Examples
    --------
    >>> ewma_covariance([[1.0, 0.0], [0.0, 1.0]], lam=0.5)  # weights 1/3, 2/3
    array([[0.33333333, 0.        ],
           [0.        , 0.66666667]])
    """
    r = _check_panel(returns, 1)
    lam = check_real(lam, "lam")
    if not 0.0 < lam < 1.0:
        raise ValueError(f"lam must be in (0, 1), got {lam!r}")
    T = r.shape[0]
    w = np.exp(np.arange(T - 1, -1, -1, dtype=np.float64) * np.log(lam))
    w /= w.sum()
    S = (r * w[:, None]).T @ r
    return (S + S.T) / 2.0


@dataclass(frozen=True)
class LedoitWolfResult(DictMixin):
    """Ledoit-Wolf shrunk covariance and the shrinkage intensity in [0, 1]."""

    covariance: np.ndarray
    shrinkage: float


@register(
    domain="finance",
    tasks=("risk", "covariance-estimation"),
    complexity=Complexity(time="O(T A^2)", space="O(T A + A^2)"),
    references=("O. Ledoit & M. Wolf, J. Multivariate Anal. 88 (2004) 365-411",),
)
def ledoit_wolf(returns: ArrayLike) -> LedoitWolfResult:
    r"""Ledoit-Wolf shrinkage of the sample covariance towards a scaled identity.

    See the module docstring for the formulas; results match
    ``sklearn.covariance.LedoitWolf`` (``assume_centered=False``). The estimate is
    positive definite whenever the shrinkage is positive, even for ``T < A``.

    Parameters
    ----------
    returns : array_like of shape (T, A)
        Per-period returns, ``T >= 2``.

    Returns
    -------
    LedoitWolfResult
        ``covariance`` of shape (A, A) and ``shrinkage`` :math:`\delta`.

    Complexity
    ----------
    Time O(T A^2), extra space O(T A + A^2).

    References
    ----------
    O. Ledoit & M. Wolf, *J. Multivariate Anal.* 88 (2004) 365-411.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.default_rng(0).normal(size=(20, 5))
    >>> res = ledoit_wolf(X)
    >>> 0.0 <= res.shrinkage <= 1.0, res.covariance.shape
    (True, (5, 5))
    """
    r = _check_panel(returns, 2)
    T, A = r.shape
    X = r - r.mean(axis=0)
    S = (X.T @ X) / T
    S = (S + S.T) / 2.0
    m = float(np.trace(S)) / A
    if A == 1:
        return LedoitWolfResult(covariance=S, shrinkage=0.0)
    target = m * np.eye(A)
    d2 = float(np.sum((S - target) ** 2)) / A
    row_sq = np.sum(X**2, axis=1)
    b2_bar = (float(row_sq @ row_sq) - T * float(np.sum(S**2))) / (T * T * A)
    b2 = min(b2_bar, d2)
    shrinkage = 0.0 if d2 == 0.0 else max(b2, 0.0) / d2
    return LedoitWolfResult(
        covariance=(1.0 - shrinkage) * S + shrinkage * target, shrinkage=shrinkage
    )
