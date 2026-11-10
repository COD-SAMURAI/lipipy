r"""Unconstrained minimisation and scalar root finding.

Line search (Armijo backtracking)
---------------------------------
Given a descent direction :math:`p` (:math:`\nabla f(x)^\top p < 0`), accept the
first step :math:`\alpha = \alpha_0, \rho\alpha_0, \rho^2\alpha_0, \dots` that gives
sufficient decrease

.. math::  f(x + \alpha p) \le f(x) + c_1 \alpha \nabla f(x)^\top p .

Descent methods
---------------
Every method iterates :math:`x_{k+1} = x_k + \alpha_k p_k` with an Armijo step and
stops when :math:`\lVert\nabla f(x_k)\rVert_\infty \le \text{tol}`. They differ in the
direction :math:`p_k`:

* **Gradient descent**: :math:`p_k = -g_k`. The trial step starts at twice the
  previous accepted step, so it adapts to the local curvature.
* **Newton**: :math:`p_k = -H_k^{-1} g_k` through a Cholesky solve of the Hessian;
  if :math:`H_k` is not positive definite the direction is not guaranteed to descend,
  so we fall back to :math:`p_k = -g_k`.
* **BFGS**: :math:`p_k = -B_k g_k` with the inverse-Hessian approximation updated from
  :math:`s_k = x_{k+1}-x_k`, :math:`y_k = g_{k+1}-g_k`, :math:`\rho_k = 1/y_k^\top s_k`:

  .. math::  B_{k+1} = (I - \rho_k s_k y_k^\top) B_k (I - \rho_k y_k s_k^\top)
             + \rho_k s_k s_k^\top .

  Before the first update :math:`B_0` is rescaled to :math:`(s^\top y / y^\top y) I`.
  The update is skipped when the curvature condition :math:`y^\top s > 0` fails
  (an Armijo step does not enforce it), which keeps :math:`B_k` positive definite.
* **L-BFGS**: the same update applied implicitly from the last :math:`m` pairs
  :math:`(s_i, y_i)` with the two-loop recursion, in :math:`O(md)` time and memory,
  starting from :math:`H^0 = \gamma I`, :math:`\gamma = s^\top y / y^\top y`.

Except for Newton (whose step is already scaled), the first trial step is
:math:`\min(1, 1/\lVert g_0\rVert_2)` so that a huge initial gradient does not cause
a wild first step.

Root finding
------------
``brentq`` keeps a bracket :math:`[a, b]` with :math:`f(a)f(b) < 0` and combines
bisection (guaranteed linear convergence) with secant / inverse quadratic
interpolation steps (superlinear) whenever they stay well inside the bracket
(Brent 1973; the step-acceptance rules follow SciPy's ``brentq``). ``newton`` uses
:math:`x \leftarrow x - f(x)/f'(x)`, or the secant method when no derivative is given.

References
----------
J. Nocedal & S. J. Wright, *Numerical Optimization*, 2nd ed. (2006): Alg. 3.1
(backtracking), §3.3 (Newton), §6.1 (BFGS), Alg. 7.4-7.5 (L-BFGS). R. P. Brent,
*Algorithms for Minimization without Derivatives* (1973), ch. 4.
"""

from __future__ import annotations

import warnings
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from lipipy.core.complexity import Complexity
from lipipy.core.errors import ConvergenceWarning
from lipipy.core.registry import register
from lipipy.numerics.linalg import LinAlgError, cholesky, cholesky_solve

__all__ = [
    "LineSearchResult",
    "OptimizeResult",
    "RootResult",
    "brentq",
    "line_search_armijo",
    "minimize",
    "minimize_bfgs",
    "minimize_gd",
    "minimize_lbfgs",
    "minimize_newton",
    "newton",
]

_NW = "Nocedal & Wright, Numerical Optimization, 2nd ed. (2006)"
_EXCL = "excludes the cost of evaluating fun / jac / hess"
_METHODS = ("gd", "newton", "bfgs", "lbfgs")


# ---------------------------------------------------------------------------
# result types
# ---------------------------------------------------------------------------
def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class OptimizeResult:
    """Outcome of :func:`minimize`.

    Attributes
    ----------
    x : ndarray
        Final point.
    fun : float
        Objective value at ``x``.
    grad : ndarray
        Gradient at ``x``.
    nit : int
        Iterations performed.
    nfev, ngev : int
        Number of objective and gradient evaluations.
    success : bool
        Whether the gradient tolerance was met.
    message : str
        Why the iteration stopped.
    """

    x: np.ndarray
    fun: float
    grad: np.ndarray
    nit: int
    nfev: int
    ngev: int
    success: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dictionary (arrays become lists)."""
        return {k: _jsonable(v) for k, v in asdict(self).items()}


@dataclass(frozen=True)
class LineSearchResult:
    """Outcome of :func:`line_search_armijo`: step ``alpha``, ``fun`` = f(x + alpha p)."""

    alpha: float
    fun: float
    nfev: int
    success: bool

    def to_dict(self) -> dict[str, Any]:
        return {k: _jsonable(v) for k, v in asdict(self).items()}


@dataclass(frozen=True)
class RootResult:
    """Outcome of :func:`brentq` and :func:`newton`. ``float(result)`` gives the root."""

    root: float
    fun: float
    nit: int
    nfev: int
    converged: bool
    message: str

    def __float__(self) -> float:
        return float(self.root)

    def to_dict(self) -> dict[str, Any]:
        return {k: _jsonable(v) for k, v in asdict(self).items()}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class _Counted:
    """Wrap a callable and count its evaluations."""

    def __init__(self, func: Callable[..., Any]) -> None:
        self.func = func
        self.calls = 0

    def __call__(self, x: Any) -> Any:
        self.calls += 1
        return self.func(x)


def _check_callable(func: Any, name: str) -> None:
    if not callable(func):
        raise TypeError(f"{name} must be callable, got {type(func).__name__}")


def _check_x0(x0: ArrayLike) -> np.ndarray:
    x = np.array(x0, dtype=np.float64, copy=True)
    if x.ndim == 0:
        x = x.reshape(1)
    if x.ndim != 1 or x.size == 0:
        raise ValueError(f"x0 must be a non-empty 1-D array, got shape {np.shape(x0)}")
    if not np.all(np.isfinite(x)):
        raise ValueError("x0 contains NaN or infinity")
    return x


def _check_options(tol: float, maxiter: int) -> None:
    if not (np.isfinite(tol) and tol > 0):
        raise ValueError(f"tol must be a positive number, got {tol!r}")
    if not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
        raise ValueError(f"maxiter must be a positive integer, got {maxiter!r}")


def _eval_f(f: _Counted, x: np.ndarray) -> float:
    return float(f(x))


def _eval_g(g: _Counted, x: np.ndarray) -> np.ndarray:
    out = np.asarray(g(x), dtype=np.float64)
    if out.shape != x.shape:
        raise ValueError(f"jac returned shape {out.shape}, expected {x.shape}")
    return out


def _armijo(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    fx: float,
    slope: float,
    p: np.ndarray,
    alpha0: float,
    c1: float,
    rho: float,
    maxiter: int,
) -> LineSearchResult:
    """Backtracking core without validation; ``slope`` = g^T p < 0."""
    alpha = alpha0
    for i in range(1, maxiter + 1):
        f_new = float(f(x + alpha * p))
        if f_new <= fx + c1 * alpha * slope:  # False for NaN
            return LineSearchResult(alpha=alpha, fun=f_new, nfev=i, success=True)
        alpha *= rho
    return LineSearchResult(alpha=0.0, fun=fx, nfev=maxiter, success=False)


# ---------------------------------------------------------------------------
# line search
# ---------------------------------------------------------------------------
@register(
    domain="numerics",
    tasks=("optimization", "line-search"),
    complexity=Complexity(time="O(T d)", space="O(d)", notes=f"T backtracking steps; {_EXCL}"),
    references=(_NW + ", Algorithm 3.1",),
)
def line_search_armijo(
    f: Callable[[np.ndarray], float],
    grad: ArrayLike | Callable[[np.ndarray], ArrayLike],
    x: ArrayLike,
    direction: ArrayLike,
    *,
    fx: float | None = None,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    rho: float = 0.5,
    maxiter: int = 60,
) -> LineSearchResult:
    r"""Backtracking line search for the Armijo sufficient-decrease condition.

    Tries :math:`\alpha = \alpha_0\rho^i` until
    :math:`f(x+\alpha p) \le f(x) + c_1\alpha\nabla f(x)^\top p`.

    Parameters
    ----------
    f : callable ``f(x) -> float``
        Objective.
    grad : array_like or callable
        The gradient at ``x``, or a function returning it.
    x : array_like of shape (d,)
        Current point.
    direction : array_like of shape (d,)
        Descent direction ``p``.
    fx : float, optional
        ``f(x)`` if already known (saves one evaluation).
    alpha0 : float, default=1.0
        Initial trial step.
    c1 : float, default=1e-4
        Sufficient-decrease constant in (0, 1).
    rho : float, default=0.5
        Backtracking factor in (0, 1).
    maxiter : int, default=60
        Maximum trial steps.

    Returns
    -------
    LineSearchResult
        ``alpha`` (0.0 on failure), ``fun`` at the accepted point, ``nfev`` (trial
        evaluations, not counting ``fx``) and ``success``.

    Raises
    ------
    ValueError
        If ``direction`` is not a descent direction.

    Complexity
    ----------
    Time O(T d) for T trial steps plus their function evaluations; space O(d).

    References
    ----------
    Nocedal & Wright, *Numerical Optimization*, Algorithm 3.1.

    Examples
    --------
    >>> import numpy as np
    >>> f = lambda x: float(x @ x)
    >>> x = np.array([1.0, 1.0])
    >>> res = line_search_armijo(f, 2 * x, x, -2 * x)
    >>> res.alpha, res.fun, res.success
    (0.5, 0.0, True)
    """
    _check_callable(f, "f")
    x = _check_x0(x)
    p = np.asarray(direction, dtype=np.float64)
    g = np.asarray(grad(x) if callable(grad) else grad, dtype=np.float64)
    if p.shape != x.shape or g.shape != x.shape:
        raise ValueError(f"x, grad and direction must share shape {x.shape}")
    if not (0 < c1 < 1 and 0 < rho < 1 and alpha0 > 0):
        raise ValueError("need 0 < c1 < 1, 0 < rho < 1 and alpha0 > 0")
    if not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
        raise ValueError(f"maxiter must be a positive integer, got {maxiter!r}")
    slope = float(g @ p)
    if not slope < 0:
        raise ValueError(f"direction is not a descent direction (grad . p = {slope:.3g} >= 0)")
    fx = float(f(x)) if fx is None else float(fx)
    return _armijo(f, x, fx, slope, p, alpha0, c1, rho, maxiter)


# ---------------------------------------------------------------------------
# minimize
# ---------------------------------------------------------------------------
def _minimize(
    method: str,
    fun: Callable[[np.ndarray], float],
    x0: ArrayLike,
    jac: Callable[[np.ndarray], ArrayLike],
    hess: Callable[[np.ndarray], ArrayLike] | None,
    tol: float,
    maxiter: int,
    m: int,
) -> OptimizeResult:
    _check_callable(fun, "fun")
    _check_callable(jac, "jac")
    if method == "newton":
        if hess is None:
            raise ValueError("method='newton' requires hess")
        _check_callable(hess, "hess")
    if method == "lbfgs" and (not isinstance(m, (int, np.integer)) or m < 1):
        raise ValueError(f"m must be a positive integer, got {m!r}")
    _check_options(tol, maxiter)
    x = _check_x0(x0)
    d = x.size
    f, g = _Counted(fun), _Counted(jac)

    fx = _eval_f(f, x)
    gx = _eval_g(g, x)
    if not (np.isfinite(fx) and np.all(np.isfinite(gx))):
        raise ValueError("fun or jac is not finite at x0")

    # BFGS inverse-Hessian approximation (only allocated for BFGS: it is d x d).
    B = np.eye(d) if method == "bfgs" else np.empty((0, 0))
    B_scaled = False
    memory: deque[tuple[np.ndarray, np.ndarray, float]] = deque(maxlen=m)  # L-BFGS pairs
    alpha_prev = None
    nit, success, message = 0, False, ""
    while True:
        if np.max(np.abs(gx)) <= tol:
            success, message = True, "converged: gradient infinity-norm <= tol"
            break
        if nit >= maxiter:
            message = f"maximum number of iterations ({maxiter}) reached"
            break

        # -- search direction ---------------------------------------------------
        newton_step = False
        if method == "gd":
            p = -gx
        elif method == "newton":
            H = np.asarray(hess(x), dtype=np.float64)
            if H.shape != (d, d):
                raise ValueError(f"hess returned shape {H.shape}, expected {(d, d)}")
            try:
                p = -cholesky_solve(cholesky((H + H.T) / 2.0), gx)
                newton_step = True
            except (LinAlgError, ValueError):  # not positive definite (or not finite)
                p = -gx
        elif method == "bfgs":
            p = -B @ gx
        else:
            p = -_two_loop(gx, memory)
        slope = float(gx @ p)
        if not slope < 0:  # lost descent through rounding: restart from steepest descent
            if method == "bfgs":
                B, B_scaled = np.eye(d), False
            memory.clear()
            p, slope = -gx, -float(gx @ gx)

        # -- step length ----------------------------------------------------------
        if newton_step:
            alpha0 = 1.0  # the Newton step has the right scale
        elif alpha_prev is None or method == "newton":
            alpha0 = min(1.0, 1.0 / float(np.linalg.norm(gx)))
        elif method == "gd":
            alpha0 = 2.0 * alpha_prev
        else:
            alpha0 = 1.0
        ls = _armijo(f, x, fx, slope, p, alpha0, 1e-4, 0.5, 60)
        if not ls.success:
            message = "line search failed to find a step with sufficient decrease"
            break
        if not ls.fun < fx:
            # The predicted decrease is below the rounding error of f: further steps
            # cannot make progress, so tol is too small for this problem's scale.
            message = "no decrease in f: rounding error limits the attainable gradient norm"
            break
        alpha_prev = ls.alpha
        s = ls.alpha * p
        x_new = x + s
        g_new = _eval_g(g, x_new)
        y = g_new - gx
        sy = float(s @ y)

        # -- curvature updates ----------------------------------------------------
        if sy > 1e-10 * float(np.linalg.norm(s) * np.linalg.norm(y)):
            if method == "bfgs":
                if not B_scaled:
                    B = (sy / float(y @ y)) * np.eye(d)
                    B_scaled = True
                By = B @ y
                B = (
                    B
                    + ((sy + y @ By) / sy**2) * np.outer(s, s)
                    - (np.outer(By, s) + np.outer(s, By)) / sy
                )
            elif method == "lbfgs":
                memory.append((s, y, 1.0 / sy))

        x, fx, gx = x_new, ls.fun, g_new
        nit += 1

    if not success:
        warnings.warn(
            f"minimize(method={method!r}) did not converge: {message}",
            ConvergenceWarning,
            stacklevel=3,
        )
    return OptimizeResult(
        x=x,
        fun=fx,
        grad=gx,
        nit=nit,
        nfev=f.calls,
        ngev=g.calls,
        success=success,
        message=message,
    )


def _two_loop(g: np.ndarray, memory: deque[tuple[np.ndarray, np.ndarray, float]]) -> np.ndarray:
    """L-BFGS two-loop recursion: returns H_k g without forming H_k (N&W Alg. 7.4)."""
    q = g.copy()
    alphas = []
    for s, y, rho in reversed(memory):
        a = rho * float(s @ q)
        alphas.append(a)
        q -= a * y
    if memory:
        s, y, _ = memory[-1]
        q *= float(s @ y) / float(y @ y)  # H^0 = gamma I
    for (s, y, rho), a in zip(memory, reversed(alphas), strict=True):
        b = rho * float(y @ q)
        q += (a - b) * s
    return q


@register(
    domain="numerics",
    tasks=("optimization",),
    complexity=Complexity(time="O(T d)", space="O(d)", notes=f"T iterations; {_EXCL}"),
    references=(_NW + ", §3.1-3.3",),
)
def minimize_gd(
    fun: Callable[[np.ndarray], float],
    x0: ArrayLike,
    jac: Callable[[np.ndarray], ArrayLike],
    tol: float = 1e-6,
    maxiter: int = 1000,
) -> OptimizeResult:
    r"""Gradient descent :math:`x \leftarrow x - \alpha\nabla f(x)` with Armijo steps.

    See :func:`minimize` for parameters and the returned :class:`OptimizeResult`.

    Complexity
    ----------
    Time O(T d) for T iterations (plus function evaluations), space O(d).

    References
    ----------
    Nocedal & Wright, *Numerical Optimization*, §3.1-3.3.

    Examples
    --------
    >>> import numpy as np
    >>> res = minimize_gd(lambda x: float(x @ x), [3.0, -4.0], lambda x: 2 * x)
    >>> res.success, np.allclose(res.x, 0.0, atol=1e-6)
    (True, True)
    """
    return _minimize("gd", fun, x0, jac, None, tol, maxiter, 1)


@register(
    domain="numerics",
    tasks=("optimization",),
    complexity=Complexity(time="O(T d^3)", space="O(d^2)", notes=f"Cholesky per step; {_EXCL}"),
    references=(_NW + ", §3.3-3.4",),
)
def minimize_newton(
    fun: Callable[[np.ndarray], float],
    x0: ArrayLike,
    jac: Callable[[np.ndarray], ArrayLike],
    hess: Callable[[np.ndarray], ArrayLike],
    tol: float = 1e-6,
    maxiter: int = 1000,
) -> OptimizeResult:
    r"""Newton's method :math:`p = -H^{-1}g` with Armijo steps.

    Falls back to the steepest-descent direction when the Hessian is not positive
    definite. ``hess(x)`` must return a (d, d) array. See :func:`minimize`.

    Complexity
    ----------
    Time O(T d^3) for T iterations (plus evaluations), space O(d^2).

    References
    ----------
    Nocedal & Wright, *Numerical Optimization*, §3.3-3.4.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[3.0, 1.0], [1.0, 2.0]]); b = np.array([1.0, 1.0])
    >>> res = minimize_newton(lambda x: 0.5 * x @ A @ x - b @ x, [0.0, 0.0],
    ...                       lambda x: A @ x - b, lambda x: A)
    >>> res.nit, np.allclose(res.x, np.linalg.solve(A, b))
    (1, True)
    """
    return _minimize("newton", fun, x0, jac, hess, tol, maxiter, 1)


@register(
    domain="numerics",
    tasks=("optimization",),
    complexity=Complexity(time="O(T d^2)", space="O(d^2)", notes=f"T iterations; {_EXCL}"),
    references=(_NW + ", §6.1",),
)
def minimize_bfgs(
    fun: Callable[[np.ndarray], float],
    x0: ArrayLike,
    jac: Callable[[np.ndarray], ArrayLike],
    tol: float = 1e-6,
    maxiter: int = 1000,
) -> OptimizeResult:
    r"""BFGS quasi-Newton method with a dense inverse-Hessian approximation.

    See :func:`minimize` for parameters and the returned :class:`OptimizeResult`.

    Complexity
    ----------
    Time O(T d^2) for T iterations (plus evaluations), space O(d^2).

    References
    ----------
    Nocedal & Wright, *Numerical Optimization*, §6.1.

    Examples
    --------
    >>> import numpy as np
    >>> res = minimize_bfgs(lambda x: float(np.sum((x - 1.0) ** 2)), np.zeros(3),
    ...                     lambda x: 2 * (x - 1.0))
    >>> res.success, np.allclose(res.x, 1.0)
    (True, True)
    """
    return _minimize("bfgs", fun, x0, jac, None, tol, maxiter, 1)


@register(
    domain="numerics",
    tasks=("optimization",),
    complexity=Complexity(
        time="O(T m d)", space="O(m d)", notes=f"m = stored correction pairs; {_EXCL}"
    ),
    references=(_NW + ", Algorithms 7.4-7.5", "Liu & Nocedal, Math. Program. 45 (1989) 503-528"),
)
def minimize_lbfgs(
    fun: Callable[[np.ndarray], float],
    x0: ArrayLike,
    jac: Callable[[np.ndarray], ArrayLike],
    tol: float = 1e-6,
    maxiter: int = 1000,
    m: int = 10,
) -> OptimizeResult:
    r"""Limited-memory BFGS (two-loop recursion over the last ``m`` pairs).

    See :func:`minimize` for parameters and the returned :class:`OptimizeResult`.
    ``m`` is the number of stored correction pairs (default 10).

    Complexity
    ----------
    Time O(T m d) for T iterations (plus evaluations), space O(m d).

    References
    ----------
    Nocedal & Wright, Algorithms 7.4-7.5; Liu & Nocedal (1989).

    Examples
    --------
    >>> import numpy as np
    >>> res = minimize_lbfgs(lambda x: float(np.sum((x - 1.0) ** 2)), np.zeros(50),
    ...                      lambda x: 2 * (x - 1.0), m=5)
    >>> res.success, np.allclose(res.x, 1.0)
    (True, True)
    """
    return _minimize("lbfgs", fun, x0, jac, None, tol, maxiter, m)


@register(
    domain="numerics",
    tasks=("optimization",),
    complexity=Complexity(
        time="O(T d^3)",
        space="O(d^2)",
        notes=f"worst case (newton); gd O(T d)/O(d), bfgs O(T d^2)/O(d^2), "
        f"lbfgs O(T m d)/O(m d); {_EXCL}",
    ),
    references=(_NW,),
)
def minimize(
    fun: Callable[[np.ndarray], float],
    x0: ArrayLike,
    jac: Callable[[np.ndarray], ArrayLike],
    method: str = "bfgs",
    *,
    hess: Callable[[np.ndarray], ArrayLike] | None = None,
    tol: float = 1e-6,
    maxiter: int = 1000,
    m: int = 10,
) -> OptimizeResult:
    r"""Minimise a smooth function of several variables.

    All methods use Armijo backtracking and stop when
    :math:`\lVert\nabla f\rVert_\infty \le` ``tol``. If they stop for another reason
    (``maxiter``, failed line search) a :class:`~lipipy.core.ConvergenceWarning` is
    emitted and ``success`` is False.

    Parameters
    ----------
    fun : callable ``fun(x) -> float``
        Objective function of a 1-D array ``x``.
    x0 : array_like of shape (d,)
        Starting point.
    jac : callable ``jac(x) -> ndarray of shape (d,)``
        Gradient of ``fun``.
    method : {"gd", "newton", "bfgs", "lbfgs"}, default="bfgs"
        Algorithm (see the module docstring).
    hess : callable ``hess(x) -> ndarray of shape (d, d)``, optional
        Hessian; required for ``"newton"``.
    tol : float, default=1e-6
        Gradient infinity-norm tolerance.
    maxiter : int, default=1000
        Maximum number of iterations.
    m : int, default=10
        Number of correction pairs kept by ``"lbfgs"``.

    Returns
    -------
    OptimizeResult
        ``x``, ``fun``, ``grad``, ``nit``, ``nfev``, ``ngev``, ``success``, ``message``.

    Complexity
    ----------
    Per iteration (excluding evaluations): gd O(d), lbfgs O(m d), bfgs O(d^2),
    newton O(d^3). Space: gd O(d), lbfgs O(m d), bfgs and newton O(d^2).

    References
    ----------
    Nocedal & Wright, *Numerical Optimization*, 2nd ed., chapters 3, 6 and 7.

    Examples
    --------
    >>> import numpy as np
    >>> def rosen(x):
    ...     return float((1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2)
    >>> def rosen_grad(x):
    ...     return np.array([-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2),
    ...                      200 * (x[1] - x[0] ** 2)])
    >>> res = minimize(rosen, [-1.2, 1.0], rosen_grad, method="bfgs", tol=1e-8)
    >>> res.success, np.round(res.x, 6)
    (True, array([1., 1.]))
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    if method != "newton" and hess is not None:
        raise ValueError(f"hess is only used by method='newton', not {method!r}")
    return _minimize(method, fun, x0, jac, hess, tol, maxiter, m)


# ---------------------------------------------------------------------------
# scalar root finding
# ---------------------------------------------------------------------------
_EPS = float(np.finfo(np.float64).eps)


@register(
    domain="numerics",
    tasks=("root-finding",),
    complexity=Complexity(time="O(T)", space="O(1)", notes="T iterations, one f call each"),
    references=("R. P. Brent, Algorithms for Minimization without Derivatives (1973), ch. 4",),
)
def brentq(
    f: Callable[[float], float],
    a: float,
    b: float,
    xtol: float = 2e-12,
    rtol: float = 4 * _EPS,
    maxiter: int = 100,
) -> RootResult:
    r"""Find a root of ``f`` in a sign-changing bracket ``[a, b]`` with Brent's method.

    Converges whenever ``f`` is continuous on the bracket; the result satisfies
    :math:`|x - x^*| \le` ``xtol + rtol * |x|`` (approximately).

    Parameters
    ----------
    f : callable ``f(x) -> float``
        Continuous scalar function.
    a, b : float
        Bracket ends with ``f(a)`` and ``f(b)`` of opposite signs.
    xtol : float, default=2e-12
        Absolute tolerance (> 0).
    rtol : float, default=4 * machine epsilon
        Relative tolerance (>= 4 eps).
    maxiter : int, default=100
        Maximum iterations.

    Returns
    -------
    RootResult
        ``root``, ``fun`` = f(root), ``nit``, ``nfev``, ``converged``, ``message``.

    Raises
    ------
    ValueError
        If ``f(a)`` and ``f(b)`` have the same sign or inputs are invalid.

    Complexity
    ----------
    Time O(T) for T iterations (plus T evaluations of ``f``), space O(1).

    References
    ----------
    R. P. Brent, *Algorithms for Minimization without Derivatives* (1973), ch. 4.

    Examples
    --------
    >>> res = brentq(lambda x: x**2 - 2.0, 0.0, 2.0)
    >>> round(res.root, 12), res.converged
    (1.414213562373, True)
    """
    _check_callable(f, "f")
    xpre, xcur = float(a), float(b)
    if not (np.isfinite(xpre) and np.isfinite(xcur)) or xpre == xcur:
        raise ValueError(f"a and b must be finite and distinct, got {a!r}, {b!r}")
    if not xtol > 0 or not rtol >= 4 * _EPS:
        raise ValueError(f"need xtol > 0 and rtol >= 4*eps, got xtol={xtol!r}, rtol={rtol!r}")
    if not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
        raise ValueError(f"maxiter must be a positive integer, got {maxiter!r}")
    fpre, fcur = float(f(xpre)), float(f(xcur))
    nfev = 2
    if not (np.isfinite(fpre) and np.isfinite(fcur)):
        raise ValueError("f(a) and f(b) must be finite")
    if fpre == 0.0:
        return RootResult(xpre, 0.0, 0, nfev, True, "f(a) == 0")
    if fcur == 0.0:
        return RootResult(xcur, 0.0, 0, nfev, True, "f(b) == 0")
    if np.signbit(fpre) == np.signbit(fcur):
        raise ValueError("f(a) and f(b) must have opposite signs")

    xblk = fblk = spre = scur = 0.0
    for it in range(1, maxiter + 1):
        if fpre != 0.0 and fcur != 0.0 and np.signbit(fpre) != np.signbit(fcur):
            xblk, fblk = xpre, fpre  # xblk is the other end of the bracket
            spre = scur = xcur - xpre
        if abs(fblk) < abs(fcur):  # make xcur the best estimate
            xpre, xcur, xblk = xcur, xblk, xcur
            fpre, fcur, fblk = fcur, fblk, fcur

        delta = (xtol + rtol * abs(xcur)) / 2.0
        sbis = (xblk - xcur) / 2.0
        if fcur == 0.0 or abs(sbis) < delta:
            return RootResult(xcur, fcur, it, nfev, True, "converged")

        if abs(spre) > delta and abs(fcur) < abs(fpre):
            if xpre == xblk:  # secant (linear interpolation)
                stry = -fcur * (xcur - xpre) / (fcur - fpre)
            else:  # inverse quadratic interpolation
                dpre = (fpre - fcur) / (xpre - xcur)
                dblk = (fblk - fcur) / (xblk - xcur)
                stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
            if 2.0 * abs(stry) < min(abs(spre), 3.0 * abs(sbis) - delta):
                spre, scur = scur, stry  # accept the interpolation step
            else:
                spre = scur = sbis  # bisect
        else:
            spre = scur = sbis  # bisect

        xpre, fpre = xcur, fcur
        xcur += scur if abs(scur) > delta else (delta if sbis > 0 else -delta)
        fcur = float(f(xcur))
        nfev += 1

    msg = f"maximum number of iterations ({maxiter}) reached"
    warnings.warn(f"brentq did not converge: {msg}", ConvergenceWarning, stacklevel=2)
    return RootResult(xcur, fcur, maxiter, nfev, False, msg)


@register(
    domain="numerics",
    tasks=("root-finding",),
    complexity=Complexity(time="O(T)", space="O(1)", notes="T iterations"),
    references=("Nocedal & Wright, Numerical Optimization, 2nd ed., §11.1",),
)
def newton(
    f: Callable[[float], float],
    x0: float,
    fprime: Callable[[float], float] | None = None,
    tol: float = 1.48e-8,
    maxiter: int = 50,
) -> RootResult:
    r"""Newton-Raphson root finding, or the secant method when ``fprime`` is None.

    Newton: :math:`x_{k+1} = x_k - f(x_k)/f'(x_k)` (quadratic convergence near a
    simple root). Secant: :math:`f'` is replaced by the slope through the last two
    iterates (order :math:`\approx 1.618`); the second starting point is
    :math:`x_0(1 + 10^{-4}) \pm 10^{-4}`. Stops when :math:`|x_{k+1}-x_k| \le` ``tol``.
    Unlike :func:`brentq`, convergence is only local.

    Parameters
    ----------
    f : callable ``f(x) -> float``
        Scalar function.
    x0 : float
        Starting point.
    fprime : callable, optional
        Derivative of ``f``; if omitted the secant method is used.
    tol : float, default=1.48e-8
        Step-size tolerance.
    maxiter : int, default=50
        Maximum iterations.

    Returns
    -------
    RootResult
        ``root``, ``fun``, ``nit``, ``nfev``, ``converged``, ``message``. A
        :class:`~lipipy.core.ConvergenceWarning` is emitted when not converged
        (iteration limit, or a zero derivative / flat secant).

    Complexity
    ----------
    Time O(T) for T iterations (plus evaluations), space O(1).

    References
    ----------
    Nocedal & Wright, *Numerical Optimization*, 2nd ed., §11.1.

    Examples
    --------
    >>> round(newton(lambda x: x**2 - 2.0, 1.0, lambda x: 2 * x).root, 12)
    1.414213562373
    >>> round(newton(lambda x: x**3 - 8.0, 3.0).root, 10)  # secant
    2.0
    """
    _check_callable(f, "f")
    if fprime is not None:
        _check_callable(fprime, "fprime")
    x = float(x0)
    if not np.isfinite(x):
        raise ValueError(f"x0 must be finite, got {x0!r}")
    if not tol > 0:
        raise ValueError(f"tol must be positive, got {tol!r}")
    if not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
        raise ValueError(f"maxiter must be a positive integer, got {maxiter!r}")

    nfev = it = 0
    message = f"maximum number of iterations ({maxiter}) reached"
    if fprime is not None:
        for it in range(1, maxiter + 1):
            fx = float(f(x))
            nfev += 1
            if fx == 0.0:
                return RootResult(x, fx, it, nfev, True, "f(x) == 0")
            dfx = float(fprime(x))
            if dfx == 0.0:
                message = "derivative is zero"
                break
            x_new = x - fx / dfx
            if abs(x_new - x) <= tol:
                fx_new = float(f(x_new))
                return RootResult(x_new, fx_new, it, nfev + 1, True, "converged")
            x = x_new
    else:
        p0 = x
        p1 = x * (1 + 1e-4)
        p1 += 1e-4 if p1 >= 0 else -1e-4
        q0, q1 = float(f(p0)), float(f(p1))
        nfev = 2
        if abs(q1) < abs(q0):
            p0, p1, q0, q1 = p1, p0, q1, q0
        x = p1
        for it in range(1, maxiter + 1):
            if q1 == q0:
                message = "secant slope is zero"
                x = (p0 + p1) / 2.0
                break
            if abs(q1) > abs(q0):
                x = (-q0 / q1 * p1 + p0) / (1 - q0 / q1)
            else:
                x = (-q1 / q0 * p0 + p1) / (1 - q1 / q0)
            if abs(x - p1) <= tol:
                return RootResult(x, float(f(x)), it, nfev + 1, True, "converged")
            p0, q0 = p1, q1
            p1, q1 = x, float(f(x))
            nfev += 1
    warnings.warn(f"newton did not converge: {message}", ConvergenceWarning, stacklevel=2)
    return RootResult(x, float(f(x)), it, nfev + 1, False, message)
