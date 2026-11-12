r"""Numerical integration (quadrature) of sampled data and of functions.

Sampled data
------------
With samples :math:`y_i = f(x_i)` and spacings :math:`h_i = x_{i+1} - x_i`:

* **Composite trapezoid**: :math:`\int f \approx \sum_i \tfrac{h_i}{2}(y_i + y_{i+1})`,
  exact for linear :math:`f`, error :math:`O(h^2)`.
* **Composite Simpson**: fit a parabola through each pair of intervals
  :math:`(x_{2j}, x_{2j+1}, x_{2j+2})`. For spacings :math:`h_0, h_1` the parabola
  integrates to

  .. math::
      \frac{h_0+h_1}{6}\Bigl[\bigl(2 - \tfrac{h_1}{h_0}\bigr)y_0
      + \tfrac{(h_0+h_1)^2}{h_0h_1}\,y_1 + \bigl(2 - \tfrac{h_0}{h_1}\bigr)y_2\Bigr],

  which reduces to :math:`\tfrac{h}{3}(y_0 + 4y_1 + y_2)` for equal spacing. Exact for
  cubics on a uniform grid (quadratics on any grid), error :math:`O(h^4)`. With an
  **even number of samples** (an odd number of intervals) the last interval is left
  over; as in ``scipy.integrate.simpson``
  (SciPy >= 1.11) Simpson's rule is applied to the first :math:`N-1` samples and
  the last interval :math:`[x_{N-2}, x_{N-1}]` is integrated exactly over the
  parabola through the last three samples (Cartwright's correction):

  .. math::
      \int_{x_{N-2}}^{x_{N-1}} \approx \alpha y_{N-1} + \beta y_{N-2} - \eta y_{N-3},\quad
      \alpha = \frac{2h_1^2 + 3h_0h_1}{6(h_0+h_1)},\;
      \beta = \frac{h_1^2 + 3h_0h_1}{6h_0},\;
      \eta = \frac{h_1^3}{6h_0(h_0+h_1)},

  with :math:`h_0, h_1` the last two spacings. This end piece is exact for quadratics
  (local error :math:`O(h^4)`, so the composite rule stays :math:`O(h^4)`).
  Two samples use the trapezoid rule.

Gauss-Legendre
--------------
The :math:`n`-point rule :math:`\int_{-1}^1 f \approx \sum_i w_i f(x_i)` is exact for
polynomials of degree :math:`\le 2n-1`. Golub-Welsch: the nodes are the eigenvalues
of the symmetric tridiagonal Jacobi matrix of the Legendre three-term recurrence,

.. math::  J_{k,k+1} = J_{k+1,k} = \beta_k = \frac{k}{\sqrt{4k^2-1}},\quad k=1..n-1,
           \qquad J_{kk} = 0,

and the weights are :math:`w_i = \mu_0\, v_{0i}^2 = 2 v_{0i}^2` with :math:`v_{\cdot i}`
the normalised eigenvectors. An interval :math:`[a, b]` maps by
:math:`x = \tfrac{b-a}{2}t + \tfrac{a+b}{2}`.

Adaptive Gauss-Kronrod (G7-K15)
-------------------------------
On each interval the 15-point Kronrod rule :math:`K_{15}` (exact to degree 22) reuses the
7 Gauss nodes; :math:`|K_{15} - G_7|` estimates the error. As in QUADPACK's ``QK15``
the raw estimate is rescaled,

.. math::
    E = I_{\mathrm{asc}} \min\Bigl(1, \bigl(200 |K_{15} - G_7| / I_{\mathrm{asc}}\bigr)^{3/2}\Bigr),
    \qquad I_{\mathrm{asc}} = \int |f - \bar f|,

and floored at :math:`50\,\varepsilon \int|f|` (roundoff). The global strategy (QUADPACK
``QAG``) keeps every subinterval in a max-heap by error, bisects the worst one, and
stops when :math:`\sum E \le \max(\text{epsabs}, \text{epsrel}\,|I|)` or the number of
subintervals reaches ``limit``. There is no Wynn epsilon extrapolation (``QAGS``), so
endpoint singularities converge by bisection alone; they need more subintervals than
SciPy's ``quad`` but the estimate stays reliable.

Monte Carlo
-----------
For :math:`U_j \sim \mathrm{Uniform}(\Omega)`, :math:`\Omega = \prod_k [a_k, b_k]`,
:math:`\hat I = |\Omega|\,\overline{f(U)}` is unbiased with standard error
:math:`|\Omega|\,s/\sqrt{n}` (:math:`s` the sample standard deviation). Samples are
drawn in batches and summarised with a streaming Welford/Chan update, so memory stays
:math:`O(Bd)` for batch size :math:`B`.

References
----------
G. H. Golub & J. H. Welsch, "Calculation of Gauss quadrature rules", *Math. Comp.* 23
(1969) 221-230. R. Piessens, E. de Doncker-Kapenga, C. W. Überhuber & D. K. Kahaner,
*QUADPACK* (Springer, 1983), routines QK15 and QAG. D. K. Cartwright, "Extension of
Simpson's rule by parabolic interpolation", *Math. Gazette* 101 (2017). P. J. Davis &
P. Rabinowitz, *Methods of Numerical Integration*, 2nd ed. (1984).
"""

from __future__ import annotations

import heapq
import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from lipipy.core.complexity import Complexity
from lipipy.core.errors import ConvergenceWarning
from lipipy.core.random import RandomState, check_random_state
from lipipy.core.registry import register
from lipipy.numerics._common import (
    as_finite_float,
    as_real,
    check_callable,
    check_positive_int,
    jsonable,
)
from lipipy.numerics.stats import RunningMoments

__all__ = [
    "MonteCarloResult",
    "QuadResult",
    "gauss_legendre",
    "gauss_legendre_nodes",
    "monte_carlo",
    "quad",
    "simpson",
    "trapezoid",
]

_EPS = float(np.finfo(np.float64).eps)
_TINY = float(np.finfo(np.float64).tiny)
_QUADPACK = "Piessens, de Doncker-Kapenga, Überhuber & Kahaner, QUADPACK (Springer, 1983)"
_GW = "Golub & Welsch, Calculation of Gauss quadrature rules, Math. Comp. 23 (1969) 221-230"
_DR = "Davis & Rabinowitz, Methods of Numerical Integration, 2nd ed. (1984)"

# Kronrod nodes (positive half, descending; xgk[1], xgk[3], xgk[5], xgk[7] are the
# Gauss nodes) and weights from QUADPACK's QK15.
_XGK = np.array(
    [
        0.991455371120812639206854697526329,
        0.949107912342758524526189684047851,
        0.864864423359769072789712788640926,
        0.741531185599394439863864773280788,
        0.586087235467691130294144845693013,
        0.405845151377397166906606412076961,
        0.207784955007898467600689403773245,
        0.000000000000000000000000000000000,
    ]
)
_WGK = np.array(
    [
        0.022935322010529224963732008058970,
        0.063092092629978553290700663189204,
        0.104790010322250183839876322541518,
        0.140653259715525918745189590510238,
        0.169004726639267902826583426598550,
        0.190350578064785409913256402421014,
        0.204432940075298892414161999234649,
        0.209482141084727828012999174891714,
    ]
)
_WG = np.array(
    [
        0.129484966168869693270611432679082,
        0.279705391489276667901467771423780,
        0.381830050505118944950369775488975,
        0.417959183673469387755102040816327,
    ]
)
# Full 15-node layout on [-1, 1]: -xgk[0..6], 0, +xgk[6..0].
_K15_NODES = np.concatenate([-_XGK[:7], [0.0], _XGK[6::-1]])
_K15_WEIGHTS = np.concatenate([_WGK[:7], [_WGK[7]], _WGK[6::-1]])
_G7_INDEX = np.array([1, 3, 5, 7, 9, 11, 13])
_G7_WEIGHTS = np.concatenate([_WG[:3], [_WG[3]], _WG[2::-1]])


# ---------------------------------------------------------------------------
# result types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QuadResult:
    """Outcome of :func:`quad`. ``float(result)`` gives the integral.

    Attributes
    ----------
    value : float
        Estimate of the integral.
    abserr : float
        Estimate of the absolute error (sum of the subinterval estimates).
    neval : int
        Number of integrand evaluations.
    nintervals : int
        Number of subintervals in the final partition.
    converged : bool
        Whether ``abserr <= max(epsabs, epsrel * |value|)``.
    message : str
        Why the iteration stopped.
    """

    value: float
    abserr: float
    neval: int
    nintervals: int
    converged: bool
    message: str

    def __post_init__(self) -> None:
        if self.neval < 0 or self.nintervals < 0:
            raise ValueError("neval and nintervals must be non-negative")
        if not self.abserr >= 0:
            raise ValueError("abserr must be non-negative")

    def __float__(self) -> float:
        return float(self.value)

    def to_dict(self) -> dict[str, Any]:
        return {k: jsonable(v) for k, v in asdict(self).items()}


@dataclass(frozen=True)
class MonteCarloResult:
    """Outcome of :func:`monte_carlo`: estimate ``value`` and its ``stderr`` from ``n`` samples."""

    value: float
    stderr: float
    n: int

    def __post_init__(self) -> None:
        if self.n < 2 or not self.stderr >= 0:
            raise ValueError("need n >= 2 and a non-negative stderr")

    def __float__(self) -> float:
        return float(self.value)

    def to_dict(self) -> dict[str, Any]:
        return {k: jsonable(v) for k, v in asdict(self).items()}


# ---------------------------------------------------------------------------
# sampled data
# ---------------------------------------------------------------------------
def _samples(y: ArrayLike, x: ArrayLike | None, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """Validate ``y`` and return it with the spacings ``h`` (from ``x`` or ``dx``)."""
    yy = as_real(y, "y", ndim=1)
    if x is None:
        step = as_finite_float(dx, "dx")
        return yy, np.full(max(yy.size - 1, 0), step)
    xx = as_real(x, "x", ndim=1)
    if xx.shape != yy.shape:
        raise ValueError(f"x and y must have the same length, got {xx.size} and {yy.size}")
    return yy, np.diff(xx)


@register(
    domain="numerics",
    tasks=("integration", "quadrature"),
    complexity=Complexity(time="O(n)", space="O(n)", notes="n samples"),
    references=(_DR + ", §2.1",),
)
def trapezoid(y: ArrayLike, x: ArrayLike | None = None, dx: float = 1.0) -> float:
    r"""Composite trapezoid rule :math:`\sum_i \tfrac{h_i}{2}(y_i + y_{i+1})` for samples ``y``.

    Parameters
    ----------
    y : array_like, shape (n,)
        Finite samples. Fewer than two samples integrate to 0.
    x : array_like, shape (n,), optional
        Sample points (any order; a decreasing ``x`` gives the negated integral).
    dx : float, default=1.0
        Uniform spacing, used when ``x`` is None.

    Returns
    -------
    float

    Complexity
    ----------
    Time O(n), extra space O(n) (the spacings).

    References
    ----------
    Davis & Rabinowitz (1984), §2.1.

    Examples
    --------
    >>> trapezoid([0.0, 1.0, 4.0, 9.0])
    9.5
    >>> trapezoid([1.0, 1.0], x=[0.0, 2.5])
    2.5
    """
    yy, h = _samples(y, x, dx)
    if yy.size < 2:
        return 0.0
    return float(np.sum(h * (yy[:-1] + yy[1:])) / 2.0)


def _simpson_pairs(y: np.ndarray, h: np.ndarray) -> float:
    """Simpson's rule over an even number of intervals ``h`` (``y`` has len(h) + 1 samples)."""
    h0, h1 = h[0::2], h[1::2]
    y0, y1, y2 = y[0:-2:2], y[1:-1:2], y[2::2]
    hsum, ratio = h0 + h1, h0 / h1
    terms = (
        hsum
        / 6.0
        * (y0 * (2.0 - 1.0 / ratio) + y1 * (hsum * hsum / (h0 * h1)) + y2 * (2.0 - ratio))
    )
    return float(np.sum(terms))


@register(
    domain="numerics",
    tasks=("integration", "quadrature"),
    complexity=Complexity(time="O(n)", space="O(n)", notes="n samples"),
    references=(_DR + ", §2.2", "Cartwright, Math. Gazette 101 (2017) (even-N correction)"),
)
def simpson(y: ArrayLike, x: ArrayLike | None = None, dx: float = 1.0) -> float:
    r"""Composite Simpson's rule for samples ``y`` (uniform or non-uniform spacing).

    Exact for quadratics on any grid, and for cubics on a uniform grid with an odd
    number of samples (the even-sample end correction is only quadratic). With an even
    number of samples the last interval is integrated over the parabola through the
    last three samples, exactly like ``scipy.integrate.simpson`` (SciPy >= 1.11);
    see the module docstring.

    Parameters
    ----------
    y : array_like, shape (n,)
        Finite samples. ``n < 2`` gives 0; ``n == 2`` uses the trapezoid rule.
    x : array_like, shape (n,), optional
        Strictly monotonic sample points.
    dx : float, default=1.0
        Uniform spacing (non-zero), used when ``x`` is None.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If ``x`` is not strictly monotonic (repeated points make the parabolas
        undefined) or inputs are invalid.

    Complexity
    ----------
    Time O(n), extra space O(n).

    References
    ----------
    Davis & Rabinowitz (1984), §2.2; Cartwright (2017).

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0.0, 2.0, 5)
    >>> simpson(x**3, x=x)  # exact for cubics: 2^4 / 4
    4.0
    """
    yy, h = _samples(y, x, dx)
    if h.size and not (np.all(h > 0) or np.all(h < 0)):
        raise ValueError("x must be strictly monotonic (and dx non-zero)")
    n = yy.size
    if n < 2:
        return 0.0
    if n == 2:
        return float(h[0] * (yy[0] + yy[1]) / 2.0)
    if n % 2 == 1:
        return _simpson_pairs(yy, h)
    result = _simpson_pairs(yy[:-1], h[:-1])
    h0, h1 = h[-2], h[-1]
    alpha = (2.0 * h1**2 + 3.0 * h0 * h1) / (6.0 * (h0 + h1))
    beta = (h1**2 + 3.0 * h0 * h1) / (6.0 * h0)
    eta = h1**3 / (6.0 * h0 * (h0 + h1))
    return float(result + alpha * yy[-1] + beta * yy[-2] - eta * yy[-3])


# ---------------------------------------------------------------------------
# Gauss-Legendre
# ---------------------------------------------------------------------------
@register(
    domain="numerics",
    tasks=("integration", "quadrature"),
    complexity=Complexity(
        time="O(n^3)", space="O(n^2)", notes="n nodes; dense symmetric eigensolver"
    ),
    references=(_GW,),
)
def gauss_legendre_nodes(n: int) -> tuple[np.ndarray, np.ndarray]:
    r"""Nodes and weights of the ``n``-point Gauss-Legendre rule on :math:`[-1, 1]` (Golub-Welsch).

    Parameters
    ----------
    n : int
        Number of nodes (>= 1).

    Returns
    -------
    nodes : ndarray, shape (n,)
        Ascending nodes in :math:`(-1, 1)`.
    weights : ndarray, shape (n,)
        Positive weights summing to 2.

    Notes
    -----
    Tested against ``numpy.polynomial.legendre.leggauss`` for 1 <= n <= 200: nodes
    agree to :math:`10^{-15}` absolute and weights to :math:`3\times10^{-15} n^2`
    relative (eigenvector-based weights lose a little relative accuracy at the
    ends of large rules).

    Complexity
    ----------
    Time O(n^3) and space O(n^2) for the dense eigendecomposition of the n x n Jacobi
    matrix (a tridiagonal eigensolver would make it O(n^2)).

    References
    ----------
    Golub & Welsch, *Math. Comp.* 23 (1969) 221-230.

    Examples
    --------
    >>> x, w = gauss_legendre_nodes(2)
    >>> [round(float(v), 12) for v in x], [round(float(v), 12) for v in w]
    ([-0.57735026919, 0.57735026919], [1.0, 1.0])
    """
    n = check_positive_int(n, "n")
    k = np.arange(1, n, dtype=np.float64)
    beta = k / np.sqrt(4.0 * k * k - 1.0)
    J = np.diag(beta, 1) + np.diag(beta, -1)
    nodes, vecs = np.linalg.eigh(J)
    weights = 2.0 * vecs[0] ** 2
    # The rule is symmetric; symmetrising removes the eigensolver's tiny asymmetry.
    nodes = 0.5 * (nodes - nodes[::-1])
    weights = 0.5 * (weights + weights[::-1])
    return nodes, weights


@register(
    domain="numerics",
    tasks=("integration", "quadrature"),
    complexity=Complexity(
        time="O(n^3)",
        space="O(n^2)",
        notes="n nodes; dominated by the node computation, plus n integrand evaluations",
    ),
    references=(_GW, _DR + ", §2.7"),
)
def gauss_legendre(f: Callable[[np.ndarray], ArrayLike], a: float, b: float, n: int = 5) -> float:
    r"""Integrate ``f`` over :math:`[a, b]` with the ``n``-point Gauss-Legendre rule.

    .. math::
        \int_a^b f(x)\,dx \approx \frac{b-a}{2}\sum_{i=1}^n w_i\,
        f\Bigl(\frac{b-a}{2}t_i + \frac{a+b}{2}\Bigr),

    exact (up to rounding) for polynomials of degree :math:`\le 2n - 1`.

    Parameters
    ----------
    f : callable
        Vectorised integrand: called once with the array of ``n`` nodes and must
        return an array of the same shape.
    a, b : float
        Finite limits (``b < a`` gives the negated integral).
    n : int, default=5
        Number of nodes (>= 1).

    Returns
    -------
    float

    Complexity
    ----------
    Time O(n^3) for the nodes (see :func:`gauss_legendre_nodes`) plus n evaluations of f.

    References
    ----------
    Golub & Welsch (1969); Davis & Rabinowitz (1984), §2.7.

    Examples
    --------
    >>> round(gauss_legendre(lambda x: x**5 - 2 * x**2, 0.0, 1.0, n=3), 14)  # 1/6 - 2/3
    -0.5
    """
    check_callable(f, "f")
    a = as_finite_float(a, "a")
    b = as_finite_float(b, "b")
    t, w = gauss_legendre_nodes(n)
    half, mid = 0.5 * (b - a), 0.5 * (a + b)
    fx = np.asarray(f(half * t + mid), dtype=np.float64)
    if fx.shape != t.shape:
        raise ValueError(f"f must return an array of shape {t.shape}, got {fx.shape}")
    if not np.all(np.isfinite(fx)):
        raise ValueError("f returned NaN or infinity at a quadrature node")
    return float(half * (w @ fx))


# ---------------------------------------------------------------------------
# adaptive Gauss-Kronrod
# ---------------------------------------------------------------------------
def _qk15(f: Callable[[float], float], a: float, b: float) -> tuple[float, float]:
    """QUADPACK QK15 on [a, b]: (K15 estimate, error estimate)."""
    centr, hlgth = 0.5 * (a + b), 0.5 * (b - a)
    fv = np.empty(15)
    for i, t in enumerate(_K15_NODES):
        value = f(float(centr + hlgth * t))
        try:
            fv[i] = value
        except (TypeError, ValueError):
            raise TypeError(f"f must return a real scalar, got {type(value).__name__}") from None
    if not np.all(np.isfinite(fv)):
        bad = float(centr + hlgth * _K15_NODES[~np.isfinite(fv)][0])
        raise ValueError(f"f returned NaN or infinity at x = {bad!r}")
    resk = float(_K15_WEIGHTS @ fv)
    resg = float(_G7_WEIGHTS @ fv[_G7_INDEX])
    reskh = 0.5 * resk  # mean of f on the interval (the weights sum to 2)
    absh = abs(hlgth)
    resabs = absh * float(_K15_WEIGHTS @ np.abs(fv))
    resasc = absh * float(_K15_WEIGHTS @ np.abs(fv - reskh))
    err = abs((resk - resg) * hlgth)
    if resasc != 0.0 and err != 0.0:
        err = resasc * min(1.0, (200.0 * err / resasc) ** 1.5)
    if resabs > _TINY / (50.0 * _EPS):
        err = max(50.0 * _EPS * resabs, err)
    return resk * hlgth, err


@register(
    domain="numerics",
    tasks=("integration", "quadrature"),
    complexity=Complexity(
        time="O(T log T)",
        space="O(T)",
        notes="T subintervals (<= limit), 15 evaluations of f each; heap operations log T",
    ),
    references=(_QUADPACK + ", QK15 and QAG",),
)
def quad(
    f: Callable[[float], float],
    a: float,
    b: float,
    epsabs: float = 1.49e-8,
    epsrel: float = 1.49e-8,
    limit: int = 50,
) -> QuadResult:
    r"""Adaptive Gauss-Kronrod (G7-K15) integration of a scalar function on a finite interval.

    Globally adaptive (QUADPACK ``QAG``): the subinterval with the largest error
    estimate is bisected until the total estimate drops below
    :math:`\max(\text{epsabs}, \text{epsrel}\,|I|)` or ``limit`` subintervals exist.

    The returned ``abserr`` is an estimate, not a proof. The tests check that it
    bounds the true error for smooth (:math:`e^x`, :math:`\sin`), sharply peaked
    (:math:`1/(10^{-4} + (x-0.3)^2)`) and endpoint-singular (:math:`x^{-1/2}`,
    :math:`\log x`, :math:`x^{-0.9}`) integrands over a grid of tolerances.

    Parameters
    ----------
    f : callable ``f(x: float) -> float``
        Integrand, finite at the interior Kronrod nodes (it is never evaluated at
        the endpoints, so integrable endpoint singularities are allowed).
    a, b : float
        Finite limits. ``b < a`` gives the negated integral, ``a == b`` gives 0.
    epsabs, epsrel : float, default=1.49e-8
        Absolute and relative tolerances (>= 0; need ``epsabs > 0`` or
        ``epsrel >= 50 * machine epsilon``).
    limit : int, default=50
        Maximum number of subintervals.

    Returns
    -------
    QuadResult
        ``value``, ``abserr``, ``neval``, ``nintervals``, ``converged``, ``message``.

    Warns
    -----
    ConvergenceWarning
        If the tolerance is not met (subdivision limit reached, or subintervals
        became too small to bisect).

    Complexity
    ----------
    Time O(T log T) for T subintervals (heap), plus 15 (2T - 1) evaluations of f;
    space O(T).

    References
    ----------
    Piessens et al., *QUADPACK* (1983), routines QK15 and QAG.

    Examples
    --------
    >>> import math
    >>> res = quad(math.exp, 0.0, 1.0)
    >>> round(res.value, 14), res.converged, res.abserr < 1e-12
    (1.71828182845905, True, True)
    """
    check_callable(f, "f")
    a = as_finite_float(a, "a")
    b = as_finite_float(b, "b")
    epsabs = as_finite_float(epsabs, "epsabs")
    epsrel = as_finite_float(epsrel, "epsrel")
    if epsabs < 0 or epsrel < 0:
        raise ValueError("epsabs and epsrel must be non-negative")
    if epsabs <= 0 and epsrel < 50 * _EPS:
        raise ValueError("need epsabs > 0 or epsrel >= 50 * machine epsilon")
    limit = check_positive_int(limit, "limit")
    if a == b:
        return QuadResult(0.0, 0.0, 0, 0, True, "empty interval")
    sign = 1.0
    if b < a:
        a, b, sign = b, a, -1.0

    value, err = _qk15(f, a, b)
    neval = 15
    heap: list[tuple[float, float, float, float, float]] = [(-err, a, b, value, err)]
    total, errsum = value, err
    message = "converged"
    while errsum > max(epsabs, epsrel * abs(total)):
        if len(heap) >= limit:
            message = f"subdivision limit ({limit}) reached"
            break
        _, lo, hi, v, e = heapq.heappop(heap)
        mid = 0.5 * (lo + hi)
        if not lo < mid < hi:
            heapq.heappush(heap, (-e, lo, hi, v, e))
            message = "subinterval too small to bisect (roundoff limit)"
            break
        v1, e1 = _qk15(f, lo, mid)
        v2, e2 = _qk15(f, mid, hi)
        neval += 30
        heapq.heappush(heap, (-e1, lo, mid, v1, e1))
        heapq.heappush(heap, (-e2, mid, hi, v2, e2))
        total += v1 + v2 - v
        errsum += e1 + e2 - e
    total = math.fsum(item[3] for item in heap)
    errsum = math.fsum(item[4] for item in heap)
    converged = errsum <= max(epsabs, epsrel * abs(total))
    if not converged:
        if message == "converged":
            message = "tolerance not met"
        warnings.warn(f"quad did not converge: {message}", ConvergenceWarning, stacklevel=2)
    return QuadResult(sign * total, errsum, neval, len(heap), converged, message)


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
@register(
    domain="numerics",
    tasks=("integration", "monte-carlo"),
    complexity=Complexity(
        time="O(P d)",
        space="O(B d)",
        notes="P samples in batches of B (plus P evaluations of f)",
    ),
    references=("Robert & Casella, Monte Carlo Statistical Methods, 2nd ed. (2004), §3.2",),
)
def monte_carlo(
    f: Callable[[np.ndarray], ArrayLike],
    bounds: Sequence[tuple[float, float]] | ArrayLike,
    n: int = 10_000,
    random_state: RandomState = None,
    batch_size: int = 65_536,
) -> MonteCarloResult:
    r"""Plain Monte Carlo integration over a box :math:`\prod_k [a_k, b_k]`.

    .. math::
        \hat I = |\Omega|\,\frac1n\sum_{j=1}^n f(U_j),\qquad
        \widehat{\mathrm{se}} = |\Omega|\,\frac{s}{\sqrt n},

    with :math:`s` the sample standard deviation of :math:`f(U_j)`. The error is
    :math:`O(n^{-1/2})` in any dimension.

    Parameters
    ----------
    f : callable
        Vectorised integrand: ``f(X)`` with ``X`` of shape ``(m, d)`` returns shape ``(m,)``.
    bounds : sequence of (low, high), shape (d, 2)
        Finite box limits with ``low < high``.
    n : int, default=10000
        Number of samples (>= 2).
    random_state : None, int, SeedSequence or Generator
        Seed for reproducibility.
    batch_size : int, default=65536
        Samples per call of ``f``; bounds memory. The result does not depend on it
        beyond rounding (the uniform stream is drawn in the same order).

    Returns
    -------
    MonteCarloResult
        ``value``, ``stderr``, ``n``.

    Complexity
    ----------
    Time O(P d) for P = n samples (plus P evaluations of f); extra space
    O(B d) for batch size B.

    References
    ----------
    Robert & Casella, *Monte Carlo Statistical Methods*, 2nd ed. (2004), §3.2.

    Examples
    --------
    >>> res = monte_carlo(lambda X: X[:, 0] * X[:, 1], [(0, 1), (0, 2)], n=20_000, random_state=0)
    >>> abs(res.value - 1.0) < 4 * res.stderr  # exact integral is 1
    True
    """
    check_callable(f, "f")
    box = as_real(bounds, "bounds")
    if box.ndim == 1 and box.size == 2:
        box = box.reshape(1, 2)
    if box.ndim != 2 or box.shape[1] != 2 or box.shape[0] == 0:
        raise ValueError(f"bounds must have shape (d, 2), got {box.shape}")
    low, high = box[:, 0], box[:, 1]
    if not np.all(low < high):
        raise ValueError("each bound must satisfy low < high")
    n = check_positive_int(n, "n", minimum=2)
    batch_size = check_positive_int(batch_size, "batch_size")
    rng = check_random_state(random_state)
    width = high - low
    volume = float(np.prod(width))
    d = box.shape[0]
    moments = RunningMoments()
    for start in range(0, n, batch_size):
        m = min(batch_size, n - start)
        X = low + width * rng.random((m, d))
        fx = np.asarray(f(X), dtype=np.float64)
        if fx.shape != (m,):
            raise ValueError(f"f must return shape ({m},), got {fx.shape}")
        if not np.all(np.isfinite(fx)):
            raise ValueError("f returned NaN or infinity")
        moments.update(fx)
    mean = float(moments.mean)
    sd = float(moments.std(ddof=1))
    return MonteCarloResult(volume * mean, volume * sd / math.sqrt(n), n)
