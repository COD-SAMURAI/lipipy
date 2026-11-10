r"""Numerically stable special functions.

Log-sum-exp and softmax
-----------------------
Evaluating :math:`\log\sum_i e^{a_i}` directly overflows once any :math:`a_i > 709`.
Shifting by the maximum :math:`m = \max_i a_i` gives the identity

.. math::  \log\sum_i e^{a_i} = m + \log\sum_i e^{a_i - m},

where every exponent is :math:`\le 0` and the largest term is exactly 1, so the sum
neither overflows nor underflows to zero. ``softmax`` and ``log_softmax`` use the
same shift. Like SciPy, a slice whose maximum is not finite is shifted by 0, so
``-inf`` entries contribute :math:`e^{-\infty}=0` and an all-``-inf`` slice gives
:math:`\log 0 = -\infty`.

Sigmoid family
--------------
With :math:`z = e^{-|x|}\in(0, 1]` (never overflows):

.. math::
    \sigma(x) = \begin{cases} 1/(1+z) & x \ge 0 \\ z/(1+z) & x < 0 \end{cases},\qquad
    \operatorname{softplus}(x) = \log(1+e^x) = \max(x, 0) + \operatorname{log1p}(z),

and :math:`\log\sigma(x) = -\operatorname{softplus}(-x)`.

Error function
--------------
.. math::
    \operatorname{erf}(x) = \frac{2}{\sqrt\pi}\int_0^x e^{-t^2}\,dt, \qquad
    \operatorname{erfc}(x) = 1 - \operatorname{erf}(x).

We use W. J. Cody's rational Chebyshev (minimax) approximations on three intervals
of :math:`y = |x|`:

* :math:`y \le 0.46875`: :math:`\operatorname{erf}(x) \approx x\,R_1(x^2)`;
* :math:`0.46875 < y \le 4`: :math:`\operatorname{erfc}(y) \approx e^{-y^2} R_2(y)`;
* :math:`y > 4`: :math:`\operatorname{erfc}(y) \approx \frac{e^{-y^2}}{y}
  \left(\frac{1}{\sqrt\pi} - y^{-2} R_3(y^{-2})\right)`.

The factor :math:`e^{-y^2}` is computed as :math:`e^{-s^2} e^{-(y-s)(y+s)}` with
:math:`s = \lfloor 16y \rfloor/16`, so :math:`s^2` is exact and no accuracy is lost
in the exponent. Negative arguments use :math:`\operatorname{erfc}(-y) = 2 -
\operatorname{erfc}(y)`. The approximations reach about :math:`10^{-16}` relative
accuracy in double precision.

Normal distribution
-------------------
.. math::  \Phi(x) = \tfrac12 \operatorname{erfc}(-x/\sqrt2)

stays accurate in the lower tail because ``erfc`` is accurate for large arguments.
The quantile :math:`\Phi^{-1}(p)` starts from P. J. Acklam's rational approximation
(relative error :math:`\le 1.15\times10^{-9}`) and applies one Halley step on
:math:`f(x) = \Phi(x) - p`:

.. math::  u = \frac{f(x)}{\varphi(x)},\qquad x \leftarrow x - \frac{u}{1 + x u / 2},

which triples the number of correct digits. In the central region we evaluate
:math:`f` as :math:`\tfrac12\operatorname{erf}(x/\sqrt2) - (p - \tfrac12)` and in the
upper tail as :math:`(1 - p) - \tfrac12\operatorname{erfc}(x/\sqrt2)`; both
differences are exact in floating point, so the result keeps full relative accuracy
near :math:`p=\tfrac12` and near :math:`p = 1`.

References
----------
W. J. Cody, "Rational Chebyshev approximations for the error function",
*Math. Comp.* 23 (1969) 631-637; coefficients from Cody's SPECFUN routine CALERF
(netlib). P. J. Acklam, "An algorithm for computing the inverse normal cumulative
distribution function" (2003). For the log-sum-exp trick: Blanchard, Higham &
Higham, "Accurately computing the log-sum-exp and softmax functions",
*IMA J. Numer. Anal.* 41 (2021).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from lipipy.core.complexity import Complexity
from lipipy.core.registry import register

__all__ = [
    "erf",
    "erfc",
    "log_sigmoid",
    "log_softmax",
    "logsumexp",
    "norm_cdf",
    "norm_ppf",
    "sigmoid",
    "softmax",
    "softplus",
]

_LSE_REFS = (
    "Blanchard, Higham & Higham, Accurately computing the log-sum-exp and softmax "
    "functions, IMA J. Numer. Anal. 41 (2021) 2311-2330",
)
_CODY_REF = (
    "W. J. Cody, Rational Chebyshev approximations for the error function, "
    "Math. Comp. 23 (1969) 631-637 (coefficients from SPECFUN CALERF)",
)
_ELEMENTWISE = Complexity(time="O(n)", space="O(n)", notes="n = number of elements")


def _as_float(x: ArrayLike, name: str = "x") -> np.ndarray:
    """Convert to a float64 array, with a clear error for non-numeric input."""
    if isinstance(x, (str, bytes)):
        raise TypeError(f"{name} must be numeric, got {type(x).__name__}")
    if np.iscomplexobj(x):
        raise TypeError(f"{name} must be real-valued, got a complex array")
    try:
        return np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be convertible to a float64 array: {exc}") from None


def _scalar_or_array(out: np.ndarray) -> np.ndarray:
    """Return a NumPy scalar for 0-d results, the array otherwise."""
    return out[()] if out.ndim == 0 else out


def _check_axis(axis: int | tuple[int, ...] | None, ndim: int) -> None:
    if axis is None:
        axes: tuple = ()
    elif isinstance(axis, (int, np.integer)):
        axes = (axis,)
    elif isinstance(axis, tuple):
        axes = axis
    else:
        raise TypeError(f"axis must be an int, a tuple of ints or None, got {axis!r}")
    for ax in axes:
        if not isinstance(ax, (int, np.integer)):
            raise TypeError(f"axis must be an int, a tuple of ints or None, got {axis!r}")
        if not -ndim <= ax < max(ndim, 1):
            raise ValueError(f"axis {ax} is out of bounds for an array of dimension {ndim}")


# ---------------------------------------------------------------------------
# log-sum-exp, softmax
# ---------------------------------------------------------------------------
def _shift(a: np.ndarray, axis) -> np.ndarray:
    """Max over ``axis`` (keepdims) with non-finite maxima replaced by 0, as in SciPy."""
    a_max = np.max(a, axis=axis, keepdims=True)
    return np.where(np.isfinite(a_max), a_max, 0.0)


@register(
    domain="numerics",
    tasks=("special-functions",),
    complexity=Complexity(
        time="O(n)",
        space="O(n)",
        notes="n = number of elements; one shifted temporary the size of the input",
    ),
    references=_LSE_REFS,
)
def logsumexp(
    a: ArrayLike, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
) -> np.ndarray:
    r"""Compute :math:`\log\sum_i e^{a_i}` without overflow.

    Uses the max-shift :math:`m + \log\sum_i e^{a_i - m}`. ``-inf`` entries contribute
    nothing; a slice of all ``-inf`` gives ``-inf``; ``+inf`` gives ``+inf``; NaN
    propagates (SciPy's conventions).

    Parameters
    ----------
    a : array_like
        Input values.
    axis : int, tuple of int or None, default=None
        Axis or axes to reduce over; ``None`` reduces over all elements.
    keepdims : bool, default=False
        Keep the reduced axes with length 1.

    Returns
    -------
    ndarray or float64
        The log of the sum of exponentials.

    Complexity
    ----------
    Time O(n), extra space O(n) for n elements (the shifted temporary).

    References
    ----------
    Blanchard, Higham & Higham, *IMA J. Numer. Anal.* 41 (2021).

    Examples
    --------
    >>> import numpy as np
    >>> float(logsumexp([1000.0, 1000.0]))  # naive log(sum(exp)) overflows
    1000.6931471805599
    >>> logsumexp([[0.0, -np.inf], [-np.inf, -np.inf]], axis=1)
    array([  0., -inf])
    """
    a = _as_float(a, "a")
    if a.size == 0:
        raise ValueError("logsumexp of an empty array is undefined")
    _check_axis(axis, a.ndim)
    shift = _shift(a, axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(np.sum(np.exp(a - shift), axis=axis, keepdims=True)) + shift
    if not keepdims:
        out = np.squeeze(out, axis=axis) if axis is not None else out.reshape(())
    return _scalar_or_array(np.asarray(out))


@register(
    domain="numerics",
    tasks=("special-functions",),
    complexity=_ELEMENTWISE,
    references=_LSE_REFS,
)
def softmax(a: ArrayLike, axis: int | tuple[int, ...] | None = -1) -> np.ndarray:
    r"""Softmax :math:`e^{a_i} / \sum_j e^{a_j}` along ``axis``, computed stably.

    Parameters
    ----------
    a : array_like
        Input values (logits).
    axis : int, tuple of int or None, default=-1
        Axis or axes over which the output sums to one; ``None`` uses all elements.

    Returns
    -------
    ndarray
        Same shape as ``a``, entries in [0, 1].

    Complexity
    ----------
    Time O(n), extra space O(n) for n elements.

    References
    ----------
    Blanchard, Higham & Higham, *IMA J. Numer. Anal.* 41 (2021).

    Examples
    --------
    >>> softmax([1000.0, 1000.0, -np.inf])
    array([0.5, 0.5, 0. ])
    """
    a = _as_float(a, "a")
    if a.size == 0:
        raise ValueError("softmax of an empty array is undefined")
    _check_axis(axis, a.ndim)
    # Same as SciPy: shift by the raw maximum (an all -inf slice gives NaN).
    with np.errstate(invalid="ignore"):
        e = np.exp(a - np.max(a, axis=axis, keepdims=True))
    return _scalar_or_array(e / np.sum(e, axis=axis, keepdims=True))


@register(
    domain="numerics",
    tasks=("special-functions",),
    complexity=_ELEMENTWISE,
    references=_LSE_REFS,
)
def log_softmax(a: ArrayLike, axis: int | tuple[int, ...] | None = -1) -> np.ndarray:
    r"""Logarithm of the softmax, :math:`a_i - \log\sum_j e^{a_j}`, computed stably.

    More accurate than ``np.log(softmax(a))`` for very negative outputs, which would
    underflow to ``log(0) = -inf``.

    Parameters
    ----------
    a : array_like
        Input values (logits).
    axis : int, tuple of int or None, default=-1
        Axis or axes to normalise over; ``None`` uses all elements.

    Returns
    -------
    ndarray
        Same shape as ``a``.

    Complexity
    ----------
    Time O(n), extra space O(n) for n elements.

    References
    ----------
    Blanchard, Higham & Higham, *IMA J. Numer. Anal.* 41 (2021).

    Examples
    --------
    >>> log_softmax([0.0, -800.0])  # np.log(softmax(...)) would give -inf
    array([   0., -800.])
    """
    a = _as_float(a, "a")
    if a.size == 0:
        raise ValueError("log_softmax of an empty array is undefined")
    _check_axis(axis, a.ndim)
    shifted = a - _shift(a, axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return _scalar_or_array(out)


# ---------------------------------------------------------------------------
# sigmoid family
# ---------------------------------------------------------------------------
@register(domain="numerics", tasks=("special-functions",), complexity=_ELEMENTWISE)
def sigmoid(x: ArrayLike) -> np.ndarray:
    r"""Logistic sigmoid :math:`\sigma(x) = 1 / (1 + e^{-x})`, stable for large :math:`|x|`.

    Uses :math:`z = e^{-|x|}`: :math:`1/(1+z)` for :math:`x \ge 0` and :math:`z/(1+z)`
    otherwise, so no exponential overflows.

    Parameters
    ----------
    x : array_like
        Input values.

    Returns
    -------
    ndarray or float64
        Values in [0, 1], same shape as ``x``.

    Complexity
    ----------
    Time O(n), extra space O(n) for n elements.

    Examples
    --------
    >>> sigmoid([-1000.0, 0.0, 1000.0])
    array([0. , 0.5, 1. ])
    """
    x = _as_float(x)
    z = np.exp(-np.abs(x))
    return _scalar_or_array(np.where(x >= 0, 1.0 / (1.0 + z), z / (1.0 + z)))


@register(domain="numerics", tasks=("special-functions",), complexity=_ELEMENTWISE)
def softplus(x: ArrayLike) -> np.ndarray:
    r"""Softplus :math:`\log(1 + e^x)` without overflow.

    Computed as :math:`\max(x, 0) + \operatorname{log1p}(e^{-|x|})`.

    Parameters
    ----------
    x : array_like
        Input values.

    Returns
    -------
    ndarray or float64
        Non-negative values, same shape as ``x``.

    Complexity
    ----------
    Time O(n), extra space O(n) for n elements.

    Examples
    --------
    >>> softplus([-1000.0, 0.0, 1000.0])
    array([0.00000000e+00, 6.93147181e-01, 1.00000000e+03])
    """
    x = _as_float(x)
    return _scalar_or_array(np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x))))


@register(domain="numerics", tasks=("special-functions",), complexity=_ELEMENTWISE)
def log_sigmoid(x: ArrayLike) -> np.ndarray:
    r"""Log of the logistic sigmoid, :math:`\log\sigma(x) = -\operatorname{softplus}(-x)`.

    Parameters
    ----------
    x : array_like
        Input values.

    Returns
    -------
    ndarray or float64
        Non-positive values, same shape as ``x``.

    Complexity
    ----------
    Time O(n), extra space O(n) for n elements.

    Examples
    --------
    >>> float(log_sigmoid(-1000.0)), round(float(log_sigmoid(0.0)), 12)
    (-1000.0, -0.69314718056)
    """
    x = _as_float(x)
    return _scalar_or_array(-(np.maximum(-x, 0.0) + np.log1p(np.exp(-np.abs(x)))))


# ---------------------------------------------------------------------------
# erf / erfc (Cody, CALERF)
# ---------------------------------------------------------------------------
# |x| <= 0.46875: erf(x) = x * P(x^2) / Q(x^2)
_A = (3.16112374387056560e00, 1.13864154151050156e02, 3.77485237685302021e02,
      3.20937758913846947e03, 1.85777706184603153e-1)  # fmt: skip
_B = (2.36012909523441209e01, 2.44024637934444173e02, 1.28261652607737228e03,
      2.84423683343917062e03)  # fmt: skip
# 0.46875 < |x| <= 4: erfc(y) = exp(-y^2) * P(y) / Q(y)
_C = (5.64188496988670089e-1, 8.88314979438837594e00, 6.61191906371416295e01,
      2.98635138197400131e02, 8.81952221241769090e02, 1.71204761263407058e03,
      2.05107837782607147e03, 1.23033935479799725e03, 2.15311535474403846e-8)  # fmt: skip
_D = (1.57449261107098347e01, 1.17693950891312499e02, 5.37181101862009858e02,
      1.62138957456669019e03, 3.29079923573345963e03, 4.36261909014324716e03,
      3.43936767414372164e03, 1.23033935480374942e03)  # fmt: skip
# |x| > 4: erfc(y) = exp(-y^2) / y * (1/sqrt(pi) + z * P(z) / Q(z)),  z = 1/y^2
_P = (3.05326634961232344e-1, 3.60344899949804439e-1, 1.25781726111229246e-1,
      1.60837851487422766e-2, 6.58749161529837803e-4, 1.63153871373020978e-2)  # fmt: skip
_Q = (2.56852019228982242e00, 1.87295284992346725e00, 5.27905102951428412e-1,
      6.05183413124413191e-2, 2.33520497626869185e-3)  # fmt: skip
_THRESH = 0.46875
_XBIG = 26.543  # erfc(y) underflows to 0 for y >= XBIG
_INV_SQRT_PI = 5.6418958354775628695e-1


def _exp_neg_sq(y: np.ndarray) -> np.ndarray:
    """exp(-y^2) without the rounding error of forming y*y (Cody's splitting)."""
    s = np.trunc(y * 16.0) / 16.0
    return np.exp(-s * s) * np.exp(-(y - s) * (y + s))


def _erf_small(x: np.ndarray) -> np.ndarray:
    """erf(x) for |x| <= 0.46875."""
    ysq = x * x
    num, den = _A[4] * ysq, ysq
    for i in range(3):
        num = (num + _A[i]) * ysq
        den = (den + _B[i]) * ysq
    return x * (num + _A[3]) / (den + _B[3])


def _erfc_mid(y: np.ndarray) -> np.ndarray:
    """erfc(y) for 0.46875 < y <= 4."""
    num, den = _C[8] * y, y
    for i in range(7):
        num = (num + _C[i]) * y
        den = (den + _D[i]) * y
    return _exp_neg_sq(y) * (num + _C[7]) / (den + _D[7])


def _erfc_large(y: np.ndarray) -> np.ndarray:
    """erfc(y) for y > 4 (0 once it underflows)."""
    big = y >= _XBIG
    y = np.minimum(y, _XBIG)  # avoid overflow in y*y; those entries are set to 0 below
    z = 1.0 / (y * y)
    num, den = _P[5] * z, z
    for i in range(4):
        num = (num + _P[i]) * z
        den = (den + _Q[i]) * z
    r = (_INV_SQRT_PI - z * (num + _P[4]) / (den + _Q[4])) / y
    return np.where(big, 0.0, _exp_neg_sq(y) * r)


def _erfc_abs(y: np.ndarray) -> np.ndarray:
    """erfc(y) for y = |x| > 0.46875, by interval."""
    out = np.empty_like(y)
    mid = y <= 4.0
    out[mid] = _erfc_mid(y[mid])
    out[~mid] = _erfc_large(y[~mid])
    return out


@register(
    domain="numerics",
    tasks=("special-functions",),
    complexity=_ELEMENTWISE,
    references=_CODY_REF,
)
def erf(x: ArrayLike) -> np.ndarray:
    r"""Error function :math:`\operatorname{erf}(x) = \frac{2}{\sqrt\pi}\int_0^x e^{-t^2}dt`.

    Cody's rational Chebyshev approximations (see the module docstring); relative
    error about :math:`10^{-16}` versus ``scipy.special.erf``.

    Parameters
    ----------
    x : array_like
        Input values; NaN propagates, ``erf(±inf) = ±1``.

    Returns
    -------
    ndarray or float64
        Values in [-1, 1], same shape as ``x``.

    Complexity
    ----------
    Time O(n), extra space O(n) for n elements.

    References
    ----------
    W. J. Cody, *Math. Comp.* 23 (1969) 631-637.

    Examples
    --------
    >>> round(float(erf(0.5)), 15)
    0.520499877813047
    >>> erf([-np.inf, 0.0, np.inf])
    array([-1.,  0.,  1.])
    """
    x = _as_float(x)
    out = np.full(x.shape, np.nan)
    y = np.abs(x)
    small = y <= _THRESH
    big = y > _THRESH  # NaN is in neither mask
    out[small] = _erf_small(x[small])
    out[big] = np.copysign(1.0 - _erfc_abs(y[big]), x[big])
    return _scalar_or_array(out)


@register(
    domain="numerics",
    tasks=("special-functions",),
    complexity=_ELEMENTWISE,
    references=_CODY_REF,
)
def erfc(x: ArrayLike) -> np.ndarray:
    r"""Complementary error function :math:`\operatorname{erfc}(x) = 1 - \operatorname{erf}(x)`.

    Accurate in the right tail where ``1 - erf(x)`` would cancel to 0: full relative
    accuracy until it underflows near :math:`x \approx 26.5`.

    Parameters
    ----------
    x : array_like
        Input values; NaN propagates, ``erfc(-inf) = 2``, ``erfc(inf) = 0``.

    Returns
    -------
    ndarray or float64
        Values in [0, 2], same shape as ``x``.

    Complexity
    ----------
    Time O(n), extra space O(n) for n elements.

    References
    ----------
    W. J. Cody, *Math. Comp.* 23 (1969) 631-637.

    Examples
    --------
    >>> print(f"{float(erfc(10.0)):.12e}")  # 1 - erf(10) would be exactly 0
    2.088487583763e-45
    """
    x = _as_float(x)
    out = np.full(x.shape, np.nan)
    y = np.abs(x)
    small = y <= _THRESH
    big = y > _THRESH
    out[small] = 1.0 - _erf_small(x[small])
    tail = _erfc_abs(y[big])
    out[big] = np.where(x[big] < 0, 2.0 - tail, tail)
    return _scalar_or_array(out)


# ---------------------------------------------------------------------------
# Normal distribution
# ---------------------------------------------------------------------------
_SQRT2 = np.sqrt(2.0)
_SQRT2PI = np.sqrt(2.0 * np.pi)

# Acklam's coefficients: central region (a / b) and tails (c / d).
_AA = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
       1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)  # fmt: skip
_AB = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
       6.680131188771972e01, -1.328068155288572e01)  # fmt: skip
_AC = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
       -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)  # fmt: skip
_AD = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
       3.754408661907416e00)  # fmt: skip
_P_LOW = 0.02425
_TINY = float(np.finfo(np.float64).tiny)  # smallest normal double, ~2.2e-308


def _polyval(coefs: tuple[float, ...], t: np.ndarray) -> np.ndarray:
    """Horner evaluation, highest-degree coefficient first."""
    out = np.full_like(t, coefs[0])
    for c in coefs[1:]:
        out = out * t + c
    return out


def _acklam_tail(p: np.ndarray) -> np.ndarray:
    """Acklam's lower-tail approximation of the quantile for small p (> 0)."""
    t = np.sqrt(-2.0 * np.log(p))
    return _polyval(_AC, t) / (_polyval(_AD, t) * t + 1.0)


@register(
    domain="numerics",
    tasks=("special-functions", "statistics"),
    complexity=_ELEMENTWISE,
    references=_CODY_REF,
)
def norm_cdf(x: ArrayLike) -> np.ndarray:
    r"""Standard normal CDF :math:`\Phi(x) = \tfrac12\operatorname{erfc}(-x/\sqrt2)`.

    Using ``erfc`` keeps full relative accuracy deep in the lower tail
    (e.g. :math:`\Phi(-30) \approx 4.9\times10^{-198}`).

    Parameters
    ----------
    x : array_like
        Input values.

    Returns
    -------
    ndarray or float64
        Probabilities in [0, 1], same shape as ``x``.

    Complexity
    ----------
    Time O(n), extra space O(n) for n elements.

    References
    ----------
    W. J. Cody, *Math. Comp.* 23 (1969) 631-637.

    Examples
    --------
    >>> norm_cdf([0.0, 1.959963984540054])
    array([0.5  , 0.975])
    """
    x = _as_float(x)
    return _scalar_or_array(0.5 * np.asarray(erfc(-x / _SQRT2)))


@register(
    domain="numerics",
    tasks=("special-functions", "statistics"),
    complexity=_ELEMENTWISE,
    references=(
        "P. J. Acklam, An algorithm for computing the inverse normal cumulative "
        "distribution function (2003)",
        "Halley's method; e.g. Press et al., Numerical Recipes 3e, §9.4",
    ),
)
def norm_ppf(p: ArrayLike) -> np.ndarray:
    r"""Standard normal quantile :math:`\Phi^{-1}(p)` (inverse CDF, "probit").

    Acklam's rational approximation refined by one Halley step (module docstring).
    Relative error :math:`\lesssim 10^{-15}` for :math:`p\in[10^{-300}, 1-10^{-16}]`.
    For subnormal :math:`p < 2.2\times10^{-308}` (``np.finfo(float).tiny``) the
    Halley step is skipped: its residual would be computed from numbers with few
    significant bits and :math:`e^{x^2/2}` overflows, so there the result is Acklam's
    approximation alone, with relative error :math:`\le 1.2\times10^{-9}`.

    Parameters
    ----------
    p : array_like
        Probabilities. ``0 -> -inf``, ``1 -> +inf``; values outside [0, 1] and NaN
        give NaN.

    Returns
    -------
    ndarray or float64
        Quantiles, same shape as ``p``.

    Complexity
    ----------
    Time O(n), extra space O(n) for n elements.

    References
    ----------
    P. J. Acklam (2003); Halley's method.

    Examples
    --------
    >>> round(float(norm_ppf(0.975)), 12)
    1.95996398454
    >>> norm_ppf([0.0, 0.5, 1.0, 2.0])
    array([-inf,   0.,  inf,  nan])
    """
    p = _as_float(p, "p")
    out = np.full(p.shape, np.nan)
    out[p == 0.0] = -np.inf
    out[p == 1.0] = np.inf

    low = (p >= _TINY) & (p < _P_LOW)
    central = (p >= _P_LOW) & (p <= 1.0 - _P_LOW)
    high = (p > 1.0 - _P_LOW) & (p < 1.0)

    # Central region: f(x) = erf(x/sqrt2)/2 - (p - 1/2), both terms accurate near 0.
    q = p[central] - 0.5
    r = q * q
    x = _polyval(_AA, r) * q / (_polyval(_AB, r) * r + 1.0)
    out[central] = _halley(x, 0.5 * np.asarray(erf(x / _SQRT2)) - q)

    # Lower tail: f(x) = Phi(x) - p.
    x = _acklam_tail(p[low])
    out[low] = _halley(x, 0.5 * np.asarray(erfc(-x / _SQRT2)) - p[low])

    # Subnormal p: Acklam only (see docstring). The upper tail never gets here,
    # since 1 - p >= 2^-53 for any double p < 1.
    sub = (p > 0.0) & (p < _TINY)
    out[sub] = _acklam_tail(p[sub])

    # Upper tail by symmetry with s = 1 - p (exact for p > 1/2):
    # f(x) = Phi(x) - p = s - Q(x),  Q(x) = erfc(x/sqrt2)/2.
    s = 1.0 - p[high]
    x = -_acklam_tail(s)
    out[high] = _halley(x, s - 0.5 * np.asarray(erfc(x / _SQRT2)))
    return _scalar_or_array(out)


def _halley(x: np.ndarray, f: np.ndarray) -> np.ndarray:
    """One Halley step for Phi(x) = p given the residual f = Phi(x) - p."""
    u = f * _SQRT2PI * np.exp(0.5 * x * x)  # f / phi(x)
    return x - u / (1.0 + 0.5 * x * u)
