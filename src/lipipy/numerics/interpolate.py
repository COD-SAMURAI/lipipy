r"""One-dimensional interpolation: piecewise linear and cubic splines.

Piecewise linear
----------------
For :math:`x_i \le x \le x_{i+1}`,
:math:`p(x) = y_i + (y_{i+1} - y_i)\,\frac{x - x_i}{x_{i+1} - x_i}`; the interval is
found by binary search. Outside :math:`[x_0, x_{n-1}]` the end values are held
(``numpy.interp`` behaviour) unless ``left``/``right`` are given.

Cubic spline
------------
On each interval :math:`[x_i, x_{i+1}]`, :math:`h_i = x_{i+1} - x_i`, the spline is the
cubic Hermite polynomial matching :math:`y_i, y_{i+1}` and the unknown slopes
:math:`s_i, s_{i+1}`. With divided differences :math:`m_i = (y_{i+1} - y_i)/h_i`,
continuity of the second derivative at the interior knots gives, for
:math:`i = 1, \dots, n-2`,

.. math::  h_i\,s_{i-1} + 2(h_{i-1} + h_i)\,s_i + h_{i-1}\,s_{i+1}
           = 3(h_i\,m_{i-1} + h_{i-1}\,m_i),

and two boundary rows close the system:

* ``"natural"``: :math:`p''(x_0) = p''(x_{n-1}) = 0`, i.e. :math:`2s_0 + s_1 = 3m_0` and
  :math:`s_{n-2} + 2s_{n-1} = 3m_{n-2}`;
* ``"clamped"``: :math:`p'(x_0) = p'(x_{n-1}) = 0`, i.e. :math:`s_0 = s_{n-1} = 0`;
* ``"not-a-knot"``: :math:`p'''` is continuous at :math:`x_1` and :math:`x_{n-2}`, i.e.
  :math:`h_1 s_0 + (h_0+h_1) s_1 = \frac{(h_0 + 2(h_0+h_1)) h_1 m_0 + h_0^2 m_1}{h_0+h_1}`
  and the mirror image at the right end.

The system is tridiagonal and solved by the Thomas algorithm (Gaussian elimination
without pivoting) in :math:`O(n)`. The interior rows are strictly diagonally dominant;
the not-a-knot rows are not, but eliminating them first leaves positive pivots (see
:func:`_thomas`). The tests check that values and first and second derivatives agree
with SciPy (banded LU with pivoting) to :math:`10^{-12}` relative to
:math:`\max(1, \max|p^{(\nu)}|)` for spacing ratios up to :math:`10^3`; for much more
uneven knots the problem itself becomes ill-conditioned. As in SciPy, two points give
the straight line and three points with not-a-knot ends give the parabola through
them. The
coefficients on interval :math:`i` are then

.. math::
    p(x) = c_{0,i}\,u^3 + c_{1,i}\,u^2 + c_{2,i}\,u + c_{3,i},\quad u = x - x_i,\qquad
    c_{0,i} = \frac{s_i + s_{i+1} - 2m_i}{h_i^2},\;
    c_{1,i} = \frac{3m_i - 2s_i - s_{i+1}}{h_i},\;
    c_{2,i} = s_i,\; c_{3,i} = y_i,

the same layout as ``scipy.interpolate.CubicSpline.c``. Outside the knots the end
polynomials are extrapolated.

References
----------
C. de Boor, *A Practical Guide to Splines*, rev. ed. (Springer, 2001), ch. IV.
R. L. Burden & J. D. Faires, *Numerical Analysis*, 9th ed. (2011), §3.5 and Alg. 6.7
(Crout/Thomas factorisation of tridiagonal systems).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from lipipy.core.complexity import Complexity
from lipipy.core.registry import register
from lipipy.numerics._common import as_finite_float, as_real, jsonable

__all__ = ["CubicSpline", "interp_linear"]

_DE_BOOR = "de Boor, A Practical Guide to Splines, rev. ed. (2001), ch. IV"
_BC_TYPES = ("not-a-knot", "natural", "clamped")


def _knots(x: ArrayLike, y: ArrayLike, minimum: int) -> tuple[np.ndarray, np.ndarray]:
    xx = as_real(x, "x", ndim=1)
    yy = as_real(y, "y", ndim=1)
    if xx.shape != yy.shape:
        raise ValueError(f"x and y must have the same length, got {xx.size} and {yy.size}")
    if xx.size < minimum:
        raise ValueError(f"need at least {minimum} points, got {xx.size}")
    dx = np.diff(xx)
    if np.any(dx == 0):
        raise ValueError("x must not contain duplicate values")
    if np.any(dx < 0):
        raise ValueError("x must be sorted in increasing order")
    return xx, yy


@register(
    domain="numerics",
    tasks=("interpolation",),
    complexity=Complexity(
        time="O((n + k) log n)",
        space="O(k)",
        notes="n knots, k query points (binary search per query)",
    ),
    references=("Burden & Faires, Numerical Analysis, 9th ed. (2011), §3.1",),
)
def interp_linear(
    x_new: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    left: float | None = None,
    right: float | None = None,
) -> np.ndarray:
    r"""Piecewise-linear interpolation of the points :math:`(x_i, y_i)` at ``x_new``.

    Parameters
    ----------
    x_new : array_like
        Query points (any shape, finite).
    x : array_like, shape (n,)
        Strictly increasing knots (n >= 1).
    y : array_like, shape (n,)
        Values at the knots.
    left, right : float, optional
        Values returned for ``x_new < x[0]`` / ``x_new > x[-1]``; default ``y[0]`` /
        ``y[-1]`` (as ``numpy.interp``).

    Returns
    -------
    ndarray of float64, same shape as ``x_new``

    Raises
    ------
    ValueError
        If ``x`` is unsorted, has duplicates or lengths differ.

    Complexity
    ----------
    Time O(k log n) for k query points and n knots, extra space O(k).

    References
    ----------
    Burden & Faires, *Numerical Analysis*, 9th ed. (2011), §3.1.

    Examples
    --------
    >>> interp_linear([0.5, 1.5, 3.0], [0.0, 1.0, 2.0], [0.0, 10.0, 0.0])
    array([5., 5., 0.])
    """
    xx, yy = _knots(x, y, minimum=1)
    q = as_real(x_new, "x_new")
    lo = yy[0] if left is None else as_finite_float(left, "left")
    hi = yy[-1] if right is None else as_finite_float(right, "right")
    if xx.size == 1:
        out = np.full(q.shape, yy[0])
    else:
        i = np.clip(np.searchsorted(xx, q, side="right") - 1, 0, xx.size - 2)
        w = (q - xx[i]) / (xx[i + 1] - xx[i])
        out = yy[i] + (yy[i + 1] - yy[i]) * w
    out = np.where(q < xx[0], lo, np.where(q > xx[-1], hi, out))
    return out[()] if out.ndim == 0 else out


def _thomas(sub: np.ndarray, diag: np.ndarray, sup: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    r"""Solve a tridiagonal system by forward elimination and back substitution.

    Row ``i`` reads ``sub[i] s[i-1] + diag[i] s[i] + sup[i] s[i+1] = rhs[i]``
    (``sub[0]`` and ``sup[-1]`` are ignored). No pivoting: for the spline systems
    every eliminated pivot :math:`\tilde d_i = d_i - l_i u_{i-1}/\tilde d_{i-1}` is
    positive (interior rows are diagonally dominant, and the not-a-knot first row's
    elimination leaves :math:`\tilde d_1 = h_0 + h_1 > 0`).
    """
    n = diag.size
    c = np.empty(n)  # modified super-diagonal
    d = np.empty(n)  # modified right-hand side
    pivot = diag[0]
    if pivot == 0.0:
        raise np.linalg.LinAlgError("zero pivot in tridiagonal solve")
    c[0] = sup[0] / pivot if n > 1 else 0.0
    d[0] = rhs[0] / pivot
    for i in range(1, n):
        pivot = diag[i] - sub[i] * c[i - 1]
        if pivot == 0.0:
            raise np.linalg.LinAlgError("zero pivot in tridiagonal solve")
        c[i] = sup[i] / pivot if i < n - 1 else 0.0
        d[i] = (rhs[i] - sub[i] * d[i - 1]) / pivot
    s = np.empty(n)
    s[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        s[i] = d[i] - c[i] * s[i + 1]
    return s


@register(
    domain="numerics",
    tasks=("interpolation", "splines"),
    complexity=Complexity(
        time="O(n)",
        space="O(n)",
        notes="construction for n knots (Thomas algorithm); evaluation O(k log n) for k points",
    ),
    references=(_DE_BOOR, "Burden & Faires, Numerical Analysis, 9th ed. (2011), §3.5"),
)
class CubicSpline:
    r"""Cubic spline interpolant with not-a-knot, natural or clamped end conditions.

    A twice continuously differentiable piecewise cubic through every
    :math:`(x_i, y_i)`; see the module docstring for the equations.

    Parameters
    ----------
    x : array_like, shape (n,)
        Strictly increasing, finite knots (n >= 2).
    y : array_like, shape (n,)
        Finite values.
    bc_type : {"not-a-knot", "natural", "clamped"}, default="not-a-knot"
        End conditions: continuous third derivative at the second and second-last
        knots, zero second derivative, or zero first derivative at both ends.

    Attributes
    ----------
    x : ndarray, shape (n,)
        Knots.
    c : ndarray, shape (4, n - 1)
        Polynomial coefficients per interval, highest power first (SciPy's layout).

    Raises
    ------
    ValueError
        If ``x`` is unsorted or has duplicates, lengths differ, or values are not finite.

    Complexity
    ----------
    Construction: time O(n), extra space O(n). Evaluation of k points: time
    O(k log n) (binary search), space O(k).

    References
    ----------
    de Boor, *A Practical Guide to Splines* (2001), ch. IV; Burden & Faires (2011), §3.5.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0.0, 1.0, 2.0, 3.0])
    >>> s = CubicSpline(x, x**3)  # not-a-knot reproduces cubics exactly
    >>> [round(float(s(1.5, nu=k)), 12) for k in (0, 1, 2)]  # 1.5^3, 3 * 1.5^2, 6 * 1.5
    [3.375, 6.75, 9.0]
    """

    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        bc_type: Literal["not-a-knot", "natural", "clamped"] = "not-a-knot",
    ) -> None:
        if bc_type not in _BC_TYPES:
            raise ValueError(f"bc_type must be one of {_BC_TYPES}, got {bc_type!r}")
        xx, yy = _knots(x, y, minimum=2)
        self.bc_type = bc_type
        self.x = xx
        self.c = self._coefficients(xx, yy, bc_type)

    @staticmethod
    def _coefficients(x: np.ndarray, y: np.ndarray, bc_type: str) -> np.ndarray:
        n = x.size
        h = np.diff(x)
        m = np.diff(y) / h
        if n == 2 and bc_type == "not-a-knot":
            s = np.array([m[0], m[0]])  # straight line
        elif n == 3 and bc_type == "not-a-knot":
            # The parabola through the three points: its slopes at the knots.
            curv = (m[1] - m[0]) / (x[2] - x[0])  # second divided difference
            s = np.array([m[0] - curv * h[0], m[0] + curv * h[0], m[1] + curv * h[1]])
        else:
            sub, diag, sup, rhs = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
            sub[1:-1] = h[1:]
            diag[1:-1] = 2.0 * (h[:-1] + h[1:])
            sup[1:-1] = h[:-1]
            rhs[1:-1] = 3.0 * (h[1:] * m[:-1] + h[:-1] * m[1:])
            if bc_type == "natural":
                diag[0], sup[0], rhs[0] = 2.0, 1.0, 3.0 * m[0]
                sub[-1], diag[-1], rhs[-1] = 1.0, 2.0, 3.0 * m[-1]
            elif bc_type == "clamped":
                diag[0], rhs[0] = 1.0, 0.0
                diag[-1], rhs[-1] = 1.0, 0.0
            else:  # not-a-knot, n >= 4
                span0 = x[2] - x[0]
                diag[0], sup[0] = h[1], span0
                rhs[0] = ((h[0] + 2.0 * span0) * h[1] * m[0] + h[0] ** 2 * m[1]) / span0
                span1 = x[-1] - x[-3]
                sub[-1], diag[-1] = span1, h[-2]
                rhs[-1] = (h[-1] ** 2 * m[-2] + (2.0 * span1 + h[-1]) * h[-2] * m[-1]) / span1
            s = _thomas(sub, diag, sup, rhs)
        c = np.empty((4, n - 1))
        c[0] = (s[:-1] + s[1:] - 2.0 * m) / h**2
        c[1] = (3.0 * m - 2.0 * s[:-1] - s[1:]) / h
        c[2] = s[:-1]
        c[3] = y[:-1]
        return c

    def __call__(self, x: ArrayLike, nu: int = 0) -> np.ndarray:
        r"""Evaluate the spline (``nu=0``) or its ``nu``-th derivative (``nu`` = 1, 2).

        Points outside the knots use the end polynomials (extrapolation).

        Parameters
        ----------
        x : array_like
            Finite evaluation points, any shape.
        nu : int, default=0
            Derivative order: 0, 1 or 2.

        Returns
        -------
        ndarray of float64, same shape as ``x`` (a NumPy scalar for scalar ``x``).
        """
        if isinstance(nu, bool) or not isinstance(nu, (int, np.integer)) or nu not in (0, 1, 2):
            raise ValueError(f"nu must be 0, 1 or 2, got {nu!r}")
        q = as_real(x, "x")
        i = np.clip(np.searchsorted(self.x, q, side="right") - 1, 0, self.x.size - 2)
        u = q - self.x[i]
        c0, c1, c2, c3 = (self.c[k][i] for k in range(4))
        if nu == 0:
            out = ((c0 * u + c1) * u + c2) * u + c3
        elif nu == 1:
            out = (3.0 * c0 * u + 2.0 * c1) * u + c2
        else:
            out = 6.0 * c0 * u + 2.0 * c1
        return out[()] if out.ndim == 0 else out

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly description: knots, coefficients and the end condition."""
        return {"bc_type": self.bc_type, "x": jsonable(self.x), "c": jsonable(self.c)}

    def __repr__(self) -> str:
        return f"CubicSpline(n={self.x.size}, bc_type={self.bc_type!r})"
