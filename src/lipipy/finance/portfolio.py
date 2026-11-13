r"""Portfolio construction: minimum variance, tangency, efficient frontier, risk parity,
Black-Litterman.

Conventions
-----------
* ``cov`` is an ``(A, A)`` covariance matrix and ``mu`` an ``(A,)`` vector of expected
  returns, in the **same periodicity** (both daily, or both annual); ``risk_free``
  too. Weights are fractions of capital and sum to one.
* **Singular covariance.** The closed forms need :math:`\Sigma^{-1}`; we do not use a
  pseudo-inverse (the optimal portfolio is then not unique). ``cov`` must be
  symmetric positive definite, checked with a Cholesky factorisation
  (:func:`lipipy.numerics.cholesky`) and a relative pivot test
  :math:`\min_j L_{jj}^2 > 10^{-12}\max_j \Sigma_{jj}`; otherwise ``ValueError``.
  The long-only minimum-variance solver needs only a positive *semi*-definite
  matrix (its minimum variance is unique, its weights may not be).

Formulas
--------
With :math:`\mathbf 1` the vector of ones, :math:`a = \mathbf 1^\top\Sigma^{-1}\mathbf 1`,
:math:`b = \mathbf 1^\top\Sigma^{-1}\mu`, :math:`c = \mu^\top\Sigma^{-1}\mu`,
:math:`D = ac - b^2`:

* global minimum variance: :math:`w = \Sigma^{-1}\mathbf 1 / a`;
* tangency (maximum Sharpe, unconstrained): :math:`w \propto \Sigma^{-1}(\mu - r_f\mathbf 1)`,
  normalised to sum to one. It maximises the Sharpe ratio only when
  :math:`\mathbf 1^\top\Sigma^{-1}(\mu - r_f\mathbf 1) > 0`, i.e. :math:`r_f` is below the
  minimum-variance return :math:`b/a`; otherwise ``ValueError``;
* efficient frontier (two-fund theorem): the minimum-variance portfolio with
  return :math:`m` is :math:`w(m) = \Sigma^{-1}(\lambda\mathbf 1 + \gamma\mu)` with
  :math:`\lambda = (c - bm)/D`, :math:`\gamma = (am - b)/D`, variance
  :math:`(am^2 - 2bm + c)/D`; :math:`w(m)` is affine in :math:`m`, so every frontier
  portfolio is a combination of two frontier funds;
* long-only minimum variance: :math:`\min w^\top\Sigma w` over the simplex
  :math:`\{w \ge 0, \mathbf 1^\top w = 1\}` by accelerated projected gradient
  (FISTA with adaptive restart) and step :math:`1/L`, :math:`L = 2\lambda_{\max}(\Sigma)`;
  the Euclidean projection onto the simplex is the sort-based algorithm of Duchi
  et al. (2008). Stops when the projected-gradient residual
  :math:`\lVert w - \Pi(w - \nabla f(w)/L)\rVert_\infty \le` ``tol`` (default
  :math:`10^{-12}`);
* risk parity (equal risk contributions :math:`RC_i = w_i(\Sigma w)_i/w^\top\Sigma w`):
  Spinu's strictly convex problem :math:`\min_{y>0} \tfrac12 y^\top\Sigma y -
  \sum_i b_i\ln y_i`, whose optimum satisfies :math:`y_i(\Sigma y)_i = b_i`; then
  :math:`w = y/\mathbf 1^\top y` has :math:`RC_i = b_i`. Solved with Newton's method
  (:func:`lipipy.numerics.minimize`) on :math:`\Sigma` scaled to unit mean variance;
* Black-Litterman: prior (equilibrium) returns :math:`\pi = \delta\Sigma w_{mkt}`;
  views :math:`P\mu = Q + \varepsilon`, :math:`\varepsilon\sim N(0, \Omega)`;

  .. math::
      \mu_{BL} = \pi + \tau\Sigma P^\top(P\tau\Sigma P^\top + \Omega)^{-1}(Q - P\pi),

      M = \tau\Sigma - \tau\Sigma P^\top(P\tau\Sigma P^\top + \Omega)^{-1}P\tau\Sigma,
      \qquad \Sigma_{BL} = \Sigma + M.

  The default :math:`\Omega = \operatorname{diag}(P\tau\Sigma P^\top)` follows He &
  Litterman. With no views, :math:`\mu_{BL} = \pi` and :math:`\Sigma_{BL} = (1+\tau)\Sigma`.

References
----------
H. Markowitz, "Portfolio selection", *J. Finance* 7 (1952) 77-91. R. C. Merton, "An
analytic derivation of the efficient portfolio frontier", *J. Financial Quant. Anal.*
7 (1972) 1851-1872. J. Duchi, S. Shalev-Shwartz, Y. Singer & T. Chandra, "Efficient
projections onto the l1-ball", *ICML* (2008). A. Beck & M. Teboulle, *SIAM J. Imaging
Sci.* 2 (2009) 183-202. B. O'Donoghue & E. Candès, *Found. Comput. Math.* 15 (2015)
715-732. F. Spinu, "An algorithm for computing risk parity weights", SSRN 2297383
(2013). F. Black & R. Litterman, *Financial Analysts J.* 48 (1992) 28-43. G. He &
R. Litterman, "The intuition behind Black-Litterman model portfolios", Goldman Sachs
(1999).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

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
from lipipy.numerics.linalg import LinAlgError, cholesky, cholesky_solve
from lipipy.numerics.optimize import minimize

__all__ = [
    "BlackLittermanResult",
    "EfficientFrontier",
    "black_litterman",
    "efficient_frontier",
    "max_sharpe_weights",
    "min_variance_weights",
    "risk_contributions",
    "risk_parity_weights",
]

_MARKOWITZ = "H. Markowitz, J. Finance 7 (1952) 77-91"
_MERTON = "R. C. Merton, J. Financial Quant. Anal. 7 (1972) 1851-1872"
_PIVOT_RTOL = 1e-12


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------
def _check_cov(cov: ArrayLike) -> np.ndarray:
    C = as_float_array(cov, "cov")
    if C.ndim != 2 or C.shape[0] != C.shape[1] or C.shape[0] == 0:
        raise ValueError(f"cov must be a non-empty square (A, A) matrix, got shape {C.shape}")
    check_finite(C, "cov")
    if np.any(np.abs(C - C.T) > 1e-10 * np.max(np.abs(C))):
        raise ValueError("cov must be symmetric")
    if np.any(np.diag(C) < 0.0):
        raise ValueError("cov has negative variances on its diagonal")
    return (C + C.T) / 2.0


def _spd_factor(C: np.ndarray) -> np.ndarray:
    """Cholesky factor of an SPD covariance, or a clear ValueError if (near) singular."""
    scale = float(np.max(np.diag(C)))
    if scale == 0.0:
        raise ValueError("cov is zero; the portfolio problem is undefined")
    try:
        L = cholesky(C)
    except LinAlgError:
        L = None
    if L is None or np.min(np.diag(L)) ** 2 <= _PIVOT_RTOL * scale:
        raise ValueError(
            "cov is singular or not positive definite; the optimal weights are not unique. "
            "Use a shrinkage estimator (e.g. lipipy.finance.ledoit_wolf) or remove redundant "
            "assets."
        )
    return L


def _check_vector(x: ArrayLike, A: int, name: str) -> np.ndarray:
    v = as_float_array(x, name)
    if v.shape != (A,):
        raise ValueError(f"{name} must have shape ({A},), got {v.shape}")
    check_finite(v, name)
    return v


# ---------------------------------------------------------------------------
# minimum variance
# ---------------------------------------------------------------------------
def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {w >= 0, sum w = 1} (Duchi et al. 2008, Fig. 1)."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    k = np.arange(1, v.size + 1)
    rho = int(np.flatnonzero(u - css / k > 0.0)[-1])
    theta = css[rho] / (rho + 1)
    return np.maximum(v - theta, 0.0)


def _min_variance_simplex(C: np.ndarray, tol: float, maxiter: int) -> np.ndarray:
    A = C.shape[0]
    lam_max = float(np.linalg.eigvalsh(C)[-1])
    if lam_max <= 0.0:
        return np.full(A, 1.0 / A)  # zero matrix: every portfolio has zero variance
    step = 1.0 / (2.0 * lam_max)
    w = np.full(A, 1.0 / A)
    z, t = w.copy(), 1.0
    for _ in range(maxiter):
        w_new = _project_simplex(z - step * 2.0 * (C @ z))
        # adaptive restart (O'Donoghue & Candes) when momentum points uphill
        if (z - w_new) @ (w_new - w) > 0.0:
            t = 1.0
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = w_new + ((t - 1.0) / t_new) * (w_new - w)
        w, t = w_new, t_new
        residual = np.max(np.abs(w - _project_simplex(w - step * 2.0 * (C @ w))))
        if residual <= tol:
            return w
    warnings.warn(
        f"long-only minimum variance did not reach tol={tol:g} in {maxiter} iterations "
        f"(residual {residual:.3g})",
        ConvergenceWarning,
        stacklevel=3,
    )
    return w


@register(
    domain="finance",
    tasks=("portfolio-optimization",),
    complexity=Complexity(
        time="O(A^3)",
        space="O(A^2)",
        notes="long_only: O(A^3 + T A^2) for T projected-gradient iterations",
    ),
    references=(
        _MARKOWITZ,
        "J. Duchi et al., Efficient projections onto the l1-ball, ICML (2008)",
        "A. Beck & M. Teboulle, SIAM J. Imaging Sci. 2 (2009) 183-202",
    ),
)
def min_variance_weights(
    cov: ArrayLike, long_only: bool = False, tol: float = 1e-12, maxiter: int = 100_000
) -> np.ndarray:
    r"""Global minimum-variance portfolio, optionally long-only.

    Unconstrained: :math:`w = \Sigma^{-1}\mathbf 1/\mathbf 1^\top\Sigma^{-1}\mathbf 1` by a
    Cholesky solve (``cov`` must be positive definite). Long-only: projected
    gradient on the simplex (module docstring), for positive semi-definite ``cov``.

    Parameters
    ----------
    cov : array_like of shape (A, A)
        Covariance matrix.
    long_only : bool, default=False
        Constrain weights to be non-negative.
    tol : float, default=1e-12
        Long-only stopping tolerance on the projected-gradient residual.
    maxiter : int, default=100000
        Long-only iteration limit; a :class:`~lipipy.core.ConvergenceWarning` is
        emitted if it is reached.

    Returns
    -------
    ndarray of shape (A, )
        Weights summing to one.

    Complexity
    ----------
    Time O(A^3), space O(A^2); long-only adds O(A^2) per iteration.

    References
    ----------
    Markowitz (1952); Duchi et al. (2008); Beck & Teboulle (2009).

    Examples
    --------
    >>> min_variance_weights([[0.04, 0.0], [0.0, 0.01]])
    array([0.2, 0.8])
    >>> w = min_variance_weights([[0.04, 0.03], [0.03, 0.0225]], long_only=True)
    >>> w.round(6)  # the unconstrained optimum shorts asset 0
    array([0., 1.])
    """
    C = _check_cov(cov)
    if long_only:
        tol = check_positive_number(tol, "tol")
        maxiter = check_int(maxiter, "maxiter", 1)
        if float(np.linalg.eigvalsh(C)[0]) < -1e-10 * float(np.max(np.diag(C))):
            raise ValueError("cov must be positive semi-definite")
        return _min_variance_simplex(C, tol, maxiter)
    L = _spd_factor(C)
    x = cholesky_solve(L, np.ones(C.shape[0]))
    return x / x.sum()


@register(
    domain="finance",
    tasks=("portfolio-optimization",),
    complexity=Complexity(time="O(A^3)", space="O(A^2)"),
    references=(_MARKOWITZ, _MERTON),
)
def max_sharpe_weights(mu: ArrayLike, cov: ArrayLike, risk_free: float = 0.0) -> np.ndarray:
    r"""Tangency (maximum Sharpe ratio) portfolio, unconstrained (shorting allowed).

    :math:`w = \Sigma^{-1}(\mu - r_f\mathbf 1) / \mathbf 1^\top\Sigma^{-1}(\mu - r_f\mathbf 1)`.

    Parameters
    ----------
    mu : array_like of shape (A, )
        Expected returns.
    cov : array_like of shape (A, A)
        Positive-definite covariance matrix (same periodicity as ``mu``).
    risk_free : float, default=0.0
        Risk-free rate (same periodicity).

    Returns
    -------
    ndarray of shape (A, )
        Weights summing to one.

    Raises
    ------
    ValueError
        If ``risk_free`` is not below the minimum-variance portfolio's return, where
        no portfolio of risky assets has a maximal positive Sharpe ratio.

    Complexity
    ----------
    Time O(A^3), extra space O(A^2).

    References
    ----------
    Merton (1972).

    Examples
    --------
    >>> max_sharpe_weights([0.1, 0.05], [[0.04, 0.0], [0.0, 0.01]])  # w ~ mu / var
    array([0.33333333, 0.66666667])
    """
    C = _check_cov(cov)
    m = _check_vector(mu, C.shape[0], "mu")
    rf = check_real(risk_free, "risk_free")
    L = _spd_factor(C)
    x = cholesky_solve(L, m - rf)
    total = float(x.sum())
    ones_inv = cholesky_solve(L, np.ones(C.shape[0]))
    gmv_return = float(m @ ones_inv) / float(ones_inv.sum())
    if not total > 1e-12 * float(np.max(np.abs(x))):
        raise ValueError(
            f"risk_free={rf!r} is not below the minimum-variance return {gmv_return!r}; "
            "the tangency portfolio does not maximise the Sharpe ratio"
        )
    return x / total


@dataclass(frozen=True)
class EfficientFrontier(DictMixin):
    """Points on the mean-variance efficient frontier.

    Attributes
    ----------
    returns : ndarray of shape (n_points,)
        Target expected returns, increasing.
    volatilities : ndarray of shape (n_points,)
        Minimum standard deviation for each target.
    weights : ndarray of shape (n_points, A)
        Frontier portfolios (rows sum to one).
    """

    returns: np.ndarray
    volatilities: np.ndarray
    weights: np.ndarray


@register(
    domain="finance",
    tasks=("portfolio-optimization",),
    complexity=Complexity(time="O(A^3 + n A)", space="O(A^2 + n A)", notes="n = n_points"),
    references=(_MARKOWITZ, _MERTON),
)
def efficient_frontier(
    mu: ArrayLike,
    cov: ArrayLike,
    n_points: int = 50,
    min_return: float | None = None,
    max_return: float | None = None,
) -> EfficientFrontier:
    r"""Efficient frontier in closed form (two-fund theorem), shorting allowed.

    See the module docstring for :math:`w(m)` and its variance.

    Parameters
    ----------
    mu : array_like of shape (A, )
        Expected returns, not all equal.
    cov : array_like of shape (A, A)
        Positive-definite covariance matrix.
    n_points : int, default=50
        Number of frontier points (>= 2).
    min_return : float, optional
        Lowest target return; default the minimum-variance return :math:`b/a`.
    max_return : float, optional
        Highest target return; default ``max(mu)`` (must exceed ``min_return``).

    Returns
    -------
    EfficientFrontier
        ``returns``, ``volatilities``, ``weights``.

    Complexity
    ----------
    Time O(A^3 + n A), extra space O(A^2 + n A) for n points.

    References
    ----------
    Merton (1972).

    Examples
    --------
    >>> ef = efficient_frontier([0.1, 0.05], [[0.04, 0.0], [0.0, 0.01]], n_points=3)
    >>> ef.returns.round(4), ef.weights.sum(axis=1).round(12)
    (array([0.06, 0.08, 0.1 ]), array([1., 1., 1.]))
    """
    C = _check_cov(cov)
    m = _check_vector(mu, C.shape[0], "mu")
    n_points = check_int(n_points, "n_points", 2)
    L = _spd_factor(C)
    inv_one = cholesky_solve(L, np.ones(C.shape[0]))
    inv_mu = cholesky_solve(L, m)
    a, b, c = float(inv_one.sum()), float(inv_mu.sum()), float(m @ inv_mu)
    D = a * c - b * b
    if not D > 1e-12 * a * c:
        raise ValueError("all expected returns are equal; the frontier is a single point")
    lo = b / a if min_return is None else check_real(min_return, "min_return")
    hi = float(m.max()) if max_return is None else check_real(max_return, "max_return")
    if not hi > lo:
        raise ValueError(f"max_return ({hi!r}) must exceed min_return ({lo!r})")
    targets = np.linspace(lo, hi, n_points)
    lam = (c - b * targets) / D
    gam = (a * targets - b) / D
    weights = np.outer(lam, inv_one) + np.outer(gam, inv_mu)
    var = (a * targets**2 - 2.0 * b * targets + c) / D
    return EfficientFrontier(
        returns=targets, volatilities=np.sqrt(np.maximum(var, 0.0)), weights=weights
    )


# ---------------------------------------------------------------------------
# risk parity
# ---------------------------------------------------------------------------
@register(
    domain="finance",
    tasks=("portfolio-analysis", "risk"),
    complexity=Complexity(time="O(A^2)", space="O(A)"),
    references=("E. Qian, Risk parity portfolios, PanAgora (2005)",),
)
def risk_contributions(weights: ArrayLike, cov: ArrayLike) -> np.ndarray:
    r"""Fractional risk contributions :math:`RC_i = w_i(\Sigma w)_i / w^\top\Sigma w`.

    They sum to one (Euler decomposition of the portfolio variance).

    Parameters
    ----------
    weights : array_like of shape (A, )
        Portfolio weights.
    cov : array_like of shape (A, A)
        Covariance matrix.

    Returns
    -------
    ndarray of shape (A, )

    Complexity
    ----------
    Time O(A^2), extra space O(A).

    References
    ----------
    E. Qian, "Risk parity portfolios", PanAgora (2005).

    Examples
    --------
    >>> risk_contributions([0.5, 0.5], [[0.04, 0.0], [0.0, 0.01]])
    array([0.8, 0.2])
    """
    C = _check_cov(cov)
    w = _check_vector(weights, C.shape[0], "weights")
    marginal = C @ w
    total = float(w @ marginal)
    if not total > 0.0:
        raise ValueError("portfolio variance is zero; risk contributions are undefined")
    return w * marginal / total


@register(
    domain="finance",
    tasks=("portfolio-optimization",),
    complexity=Complexity(time="O(T A^3)", space="O(A^2)", notes="T Newton iterations"),
    references=(
        "F. Spinu, An algorithm for computing risk parity weights, SSRN 2297383 (2013)",
        "Nocedal & Wright, Numerical Optimization, 2nd ed. (2006), §3.3",
    ),
)
def risk_parity_weights(
    cov: ArrayLike, budgets: ArrayLike | None = None, tol: float = 1e-12, maxiter: int = 200
) -> np.ndarray:
    r"""Risk parity (equal, or budgeted, risk contribution) long-only weights.

    Solves Spinu's convex problem with Newton's method (module docstring). At the
    returned weights :math:`|RC_i - b_i| \le 10^{-8}` (checked; a
    :class:`~lipipy.core.ConvergenceWarning` is emitted otherwise).

    Parameters
    ----------
    cov : array_like of shape (A, A)
        Positive-definite covariance matrix.
    budgets : array_like of shape (A, ), optional
        Positive risk budgets :math:`b` (normalised to sum to one); default equal.
    tol : float, default=1e-12
        Gradient tolerance passed to the Newton solver (on the unit-variance scale).
    maxiter : int, default=200
        Newton iteration limit.

    Returns
    -------
    ndarray of shape (A, )
        Positive weights summing to one.

    Complexity
    ----------
    Time O(T A^3) for T Newton iterations, extra space O(A^2).

    References
    ----------
    F. Spinu (2013).

    Examples
    --------
    >>> risk_parity_weights([[0.04, 0.0], [0.0, 0.01]]).round(12)  # w ~ 1 / sigma
    array([0.33333333, 0.66666667])
    """
    C = _check_cov(cov)
    A = C.shape[0]
    tol = check_positive_number(tol, "tol")
    maxiter = check_int(maxiter, "maxiter", 1)
    if budgets is None:
        b = np.full(A, 1.0 / A)
    else:
        b = _check_vector(budgets, A, "budgets")
        if np.any(b <= 0.0):
            raise ValueError("budgets must be strictly positive")
        b = b / b.sum()
    _spd_factor(C)
    Cs = C / float(np.mean(np.diag(C)))  # risk parity weights are scale invariant

    def fun(y: np.ndarray) -> float:
        if np.any(y <= 0.0):
            return np.inf  # outside the domain: rejected by the line search
        return float(0.5 * y @ Cs @ y - b @ np.log(y))

    def jac(y: np.ndarray) -> np.ndarray:
        return Cs @ y - b / y

    def hess(y: np.ndarray) -> np.ndarray:
        return Cs + np.diag(b / (y * y))

    y0 = np.sqrt(b) / np.sqrt(np.diag(Cs))  # exact when Cs is diagonal
    y0 *= np.sqrt(1.0 / float(y0 @ Cs @ y0))  # y^T C y = sum b at the optimum
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        res = minimize(fun, y0, jac, method="newton", hess=hess, tol=tol, maxiter=maxiter)
    w = res.x / res.x.sum()
    rc = w * (Cs @ w) / float(w @ Cs @ w)
    if np.any(w <= 0.0) or np.max(np.abs(rc - b)) > 1e-8:
        warnings.warn(
            f"risk parity did not converge: max |RC - budget| = {np.max(np.abs(rc - b)):.3g} "
            f"({res.message})",
            ConvergenceWarning,
            stacklevel=2,
        )
    return w


# ---------------------------------------------------------------------------
# Black-Litterman
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BlackLittermanResult(DictMixin):
    """Black-Litterman prior and posterior.

    Attributes
    ----------
    prior_returns : ndarray of shape (A,)
        Equilibrium returns :math:`\\pi = \\delta\\Sigma w_{mkt}`.
    posterior_returns : ndarray of shape (A,)
        :math:`\\mu_{BL}`.
    posterior_covariance : ndarray of shape (A, A)
        :math:`\\Sigma_{BL} = \\Sigma + M` (covariance of returns for portfolio
        construction).
    """

    prior_returns: np.ndarray
    posterior_returns: np.ndarray
    posterior_covariance: np.ndarray


@register(
    domain="finance",
    tasks=("portfolio-optimization", "bayesian"),
    complexity=Complexity(time="O(A^2 k + k^3)", space="O(A^2 + A k)", notes="k = number of views"),
    references=(
        "F. Black & R. Litterman, Financial Analysts J. 48 (1992) 28-43",
        "G. He & R. Litterman, The intuition behind Black-Litterman model portfolios (1999)",
    ),
)
def black_litterman(
    cov: ArrayLike,
    market_weights: ArrayLike,
    P: ArrayLike | None = None,
    Q: ArrayLike | None = None,
    omega: ArrayLike | None = None,
    tau: float = 0.05,
    risk_aversion: float = 2.5,
) -> BlackLittermanResult:
    r"""Black-Litterman posterior expected returns and covariance.

    See the module docstring for the formulas.

    Parameters
    ----------
    cov : array_like of shape (A, A)
        Covariance of returns :math:`\Sigma` (positive semi-definite).
    market_weights : array_like of shape (A, )
        Market-capitalisation weights :math:`w_{mkt}`.
    P : array_like of shape (k, A), optional
        View pick matrix (one view per row); None or ``k = 0`` means no views.
    Q : array_like of shape (k, )
        View returns.
    omega : array_like of shape (k, k), optional
        View uncertainty (symmetric positive definite); default
        :math:`\operatorname{diag}(P\tau\Sigma P^\top)`.
    tau : float, default=0.05
        Scale of the uncertainty of the prior mean (> 0).
    risk_aversion : float, default=2.5
        Market risk aversion :math:`\delta` (> 0).

    Returns
    -------
    BlackLittermanResult
        ``prior_returns``, ``posterior_returns``, ``posterior_covariance``.

    Complexity
    ----------
    Time O(A^2 k + k^3), extra space O(A^2 + A k) for k views.

    References
    ----------
    Black & Litterman (1992); He & Litterman (1999).

    Examples
    --------
    >>> cov = [[0.04, 0.01], [0.01, 0.09]]
    >>> res = black_litterman(cov, [0.5, 0.5], P=[[1.0, -1.0]], Q=[0.02])
    >>> res.prior_returns.round(4)
    array([0.0625, 0.125 ])
    >>> bool(res.posterior_returns[0] - res.posterior_returns[1] > -0.0625)  # tilted to view
    True
    """
    C = _check_cov(cov)
    A = C.shape[0]
    w = _check_vector(market_weights, A, "market_weights")
    tau = check_positive_number(tau, "tau")
    delta = check_positive_number(risk_aversion, "risk_aversion")
    pi = delta * (C @ w)
    if P is None:
        if Q is not None or omega is not None:
            raise ValueError("Q and omega need a pick matrix P")
        return BlackLittermanResult(pi, pi.copy(), (1.0 + tau) * C)
    Pm = as_float_array(P, "P")
    if Pm.ndim != 2 or Pm.shape[1] != A:
        raise ValueError(f"P must have shape (k, {A}), got {Pm.shape}")
    check_finite(Pm, "P")
    k = Pm.shape[0]
    if k == 0:
        return BlackLittermanResult(pi, pi.copy(), (1.0 + tau) * C)
    if Q is None:
        raise ValueError("Q is required when P is given")
    q = _check_vector(Q, k, "Q")
    tS = tau * C
    PtS = Pm @ tS  # (k, A)
    base = PtS @ Pm.T
    if omega is None:
        Om = np.diag(np.diag(base))
    else:
        Om = as_float_array(omega, "omega")
        if Om.shape != (k, k):
            raise ValueError(f"omega must have shape ({k}, {k}), got {Om.shape}")
        check_finite(Om, "omega")
    Mk = base + Om
    Mk = (Mk + Mk.T) / 2.0
    try:
        Lk = cholesky(Mk)
    except LinAlgError:
        raise ValueError(
            "P tau Sigma P^T + omega is not positive definite (check omega and redundant views)"
        ) from None
    gain = cholesky_solve(Lk, PtS).T  # tau Sigma P^T (P tau Sigma P^T + Omega)^{-1}, (A, k)
    mu_bl = pi + gain @ (q - Pm @ pi)
    M = tS - gain @ PtS
    post = C + (M + M.T) / 2.0
    return BlackLittermanResult(pi, mu_bl, post)
