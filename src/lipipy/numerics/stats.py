r"""Streaming moments, covariance and correlation.

Running mean and variance
-------------------------
The textbook formula :math:`\operatorname{Var} = \overline{x^2} - \bar x^2` subtracts
two nearly equal numbers when the mean is large relative to the spread, and loses
all accuracy (data around :math:`10^9` with unit noise give garbage). Welford's
update keeps the count :math:`n`, the mean :math:`\bar x_n` and the sum of squared
deviations :math:`M_n = \sum_i (x_i - \bar x_n)^2` instead:

.. math::
    \delta = x - \bar x_{n-1},\quad \bar x_n = \bar x_{n-1} + \delta / n,\quad
    M_n = M_{n-1} + \delta\,(x - \bar x_n).

Two summaries :math:`(n_a, \bar x_a, M_a)` and :math:`(n_b, \bar x_b, M_b)` combine
exactly with the parallel formula of Chan, Golub & LeVeque:

.. math::
    n = n_a + n_b,\quad \delta = \bar x_b - \bar x_a,\quad
    \bar x = \bar x_a + \delta\,\frac{n_b}{n},\quad
    M = M_a + M_b + \delta^2\,\frac{n_a n_b}{n}.

Welford's update is the special case :math:`n_b = 1`. A batch is summarised with a
two-pass (centred) computation in fixed-size chunks and merged with this formula, so
an update costs O(n d) time and only O(d) extra memory regardless of batch size.
The variance is :math:`M_n / (n - \text{ddof})`.

All moments are accumulated for the shifted data :math:`x - K`, with :math:`K` the
first observation (the "shifted data" recommendation of Chan et al.). The variance
is shift-invariant, and shifting keeps the running means small: without it, data
around :math:`10^9` would carry mean rounding errors of order
:math:`\text{ulp}(10^9)\approx 10^{-7}` into :math:`\delta` in every merge.

Covariance and correlation
--------------------------
For data :math:`X\in\mathbb R^{n\times d}` (rows are samples), centred columns
:math:`X_c = X - \mathbf 1\bar x^\top` give

.. math::
    C = \frac{X_c^\top X_c}{n - \text{ddof}},\qquad
    R_{ij} = \frac{C_{ij}}{\sqrt{C_{ii} C_{jj}}}.

References
----------
B. P. Welford, "Note on a method for calculating corrected sums of squares and
products", *Technometrics* 4 (1962) 419-420. T. F. Chan, G. H. Golub & R. J.
LeVeque, "Algorithms for computing the sample variance: analysis and
recommendations", *The American Statistician* 37 (1983) 242-247.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from lipipy.core.complexity import Complexity
from lipipy.core.registry import register
from lipipy.core.validation import check_array

__all__ = ["RunningMoments", "correlation", "covariance"]

_WELFORD_REFS = (
    "B. P. Welford, Technometrics 4 (1962) 419-420",
    "Chan, Golub & LeVeque, The American Statistician 37 (1983) 242-247",
)
#: Rows per chunk when summarising a batch; bounds extra memory at O(chunk * d).
_CHUNK = 4096


@register(
    domain="numerics",
    tasks=("statistics", "streaming"),
    complexity=Complexity(
        time="O(n d)",
        space="O(d)",
        notes="per update of n rows with d columns; batches are processed in fixed-size chunks",
    ),
    references=_WELFORD_REFS,
)
class RunningMoments:
    r"""Streaming mean and variance (Welford / Chan et al.), per column.

    Feed data with :meth:`update`; combine summaries computed on separate shards
    with :meth:`merge`. The result is as accurate as a two-pass computation, even
    when the data have a huge offset.

    Input shapes: a scalar is one observation; a 1-D array of length ``n`` is ``n``
    scalar observations; a 2-D array of shape ``(n, d)`` is ``n`` observations of
    ``d`` columns. The first update fixes the feature shape (``()`` or ``(d,)``).

    Attributes
    ----------
    count : int
        Number of observations seen.
    mean : ndarray or float64
        Running mean, shape ``()`` or ``(d,)``.

    Complexity
    ----------
    :meth:`update` with ``n`` rows: time O(n d), extra space O(d) (chunks of 4096 rows,
    for float64 input; other dtypes are converted, which copies the batch).
    :meth:`merge`: time and space O(d).

    References
    ----------
    Welford (1962); Chan, Golub & LeVeque (1983).

    Examples
    --------
    >>> import numpy as np
    >>> rm = RunningMoments()
    >>> rm.update([1.0, 2.0])
    >>> rm.update([3.0, 4.0])
    >>> rm.count, float(rm.mean), float(rm.variance()), float(rm.variance(ddof=1))
    (4, 2.5, 1.25, 1.6666666666666667)
    >>> other = RunningMoments()
    >>> other.update(np.array([5.0]))
    >>> float(rm.merge(other).mean)
    3.0
    """

    def __init__(self) -> None:
        self._n = 0
        self._shape: tuple[int, ...] | None = None
        self._shift = np.zeros(())  # K: the first observation; moments are of x - K
        self._mean = np.zeros(())  # mean of x - K
        self._m2 = np.zeros(())

    # -- updates ----------------------------------------------------------------
    def update(self, batch: ArrayLike) -> None:
        """Add observations (scalar, 1-D of scalars, or 2-D ``(n, d)``).

        Raises
        ------
        ValueError
            If the batch contains NaN/inf, has more than 2 dimensions, or its
            feature shape differs from earlier updates.
        """
        x = np.asarray(batch, dtype=np.float64)
        if x.ndim > 2:
            raise ValueError(f"batch must be 0-, 1- or 2-dimensional, got {x.ndim}D")
        rows = x.reshape(1) if x.ndim == 0 else x
        shape = rows.shape[1:]
        if rows.shape[0] == 0:
            return
        if self._shape is None:
            if not np.all(np.isfinite(rows[0])):
                raise ValueError("batch contains NaN or infinity")
            self._shape = shape
            self._shift = rows[0].copy()
            self._mean = np.zeros(shape)
            self._m2 = np.zeros(shape)
        elif shape != self._shape:
            raise ValueError(f"batch has feature shape {shape}, expected {self._shape}")
        for start in range(0, rows.shape[0], _CHUNK):
            chunk = rows[start : start + _CHUNK]
            if not np.all(np.isfinite(chunk)):
                raise ValueError("batch contains NaN or infinity")
            shifted = chunk - self._shift
            mean_b = shifted.mean(axis=0)
            shifted -= mean_b
            m2_b = (
                np.einsum("ij,ij->j", shifted, shifted) if shifted.ndim == 2 else shifted @ shifted
            )
            self._combine(chunk.shape[0], mean_b, m2_b)

    def merge(self, other: RunningMoments) -> RunningMoments:
        """Combine with another summary in place (Chan et al.) and return ``self``.

        The result equals the summary of the concatenated data.
        """
        if not isinstance(other, RunningMoments):
            raise TypeError(f"can only merge RunningMoments, got {type(other).__name__}")
        if other._n == 0:
            return self
        if self._shape is None:
            self._shape = other._shape
            self._shift = other._shift.copy()
            self._mean = np.zeros_like(other._mean)
            self._m2 = np.zeros_like(other._m2)
        elif other._shape != self._shape:
            raise ValueError(f"cannot merge feature shape {other._shape} into {self._shape}")
        # Express the other mean relative to our shift K before combining.
        self._combine(other._n, other._mean + (other._shift - self._shift), other._m2)
        return self

    def _combine(self, n_b: int, mean_b: np.ndarray, m2_b: np.ndarray) -> None:
        n_a = self._n
        n = n_a + n_b
        delta = mean_b - self._mean
        self._mean = self._mean + delta * (n_b / n)
        self._m2 = self._m2 + m2_b + delta**2 * (n_a * n_b / n)
        self._n = n

    # -- results ----------------------------------------------------------------
    @property
    def count(self) -> int:
        """Number of observations seen."""
        return self._n

    @property
    def mean(self) -> np.ndarray:
        """Running mean (shape ``()`` or ``(d,)``)."""
        self._require_data(0)
        return (self._shift + self._mean)[()]

    def variance(self, ddof: int = 0) -> np.ndarray:
        """Variance ``M / (count - ddof)``; ``ddof=1`` gives the unbiased estimate."""
        self._require_data(ddof)
        return (self._m2 / (self._n - ddof))[()]

    def std(self, ddof: int = 0) -> np.ndarray:
        """Standard deviation, the square root of :meth:`variance`."""
        return np.sqrt(self.variance(ddof))

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly summary (count, mean, population variance)."""
        if self._n == 0:
            return {"count": 0, "mean": None, "variance": None}
        return {
            "count": self._n,
            "mean": np.asarray(self.mean).tolist(),
            "variance": np.asarray(self.variance()).tolist(),
        }

    def _require_data(self, ddof: int) -> None:
        if not isinstance(ddof, (int, np.integer)) or ddof < 0:
            raise ValueError(f"ddof must be a non-negative integer, got {ddof!r}")
        if self._n == 0:
            raise ValueError("no observations yet; call update() first")
        if self._n <= ddof:
            raise ValueError(f"need more than ddof={ddof} observations, have {self._n}")

    def __repr__(self) -> str:
        return f"RunningMoments(count={self._n}, shape={self._shape})"


def _check_data(X: ArrayLike, ddof: int) -> np.ndarray:
    if not isinstance(ddof, (int, np.integer)) or ddof < 0:
        raise ValueError(f"ddof must be a non-negative integer, got {ddof!r}")
    X = check_array(X)
    if X.shape[0] <= ddof:
        raise ValueError(f"need more than ddof={ddof} samples, got {X.shape[0]}")
    return X


@register(
    domain="numerics",
    tasks=("statistics",),
    complexity=Complexity(time="O(n d^2)", space="O(n d)", notes="one centred copy of X"),
    references=("Chan, Golub & LeVeque, The American Statistician 37 (1983) 242-247",),
)
def covariance(X: ArrayLike, ddof: int = 1) -> np.ndarray:
    r"""Sample covariance matrix of the columns of ``X``.

    :math:`C = X_c^\top X_c / (n - \text{ddof})` with column-centred :math:`X_c`;
    equals ``np.cov(X, rowvar=False, ddof=ddof)``.

    Parameters
    ----------
    X : array_like of shape (n, d)
        Rows are observations, columns are variables.
    ddof : int, default=1
        Delta degrees of freedom (1 = unbiased).

    Returns
    -------
    ndarray of shape (d, d)
        Symmetric covariance matrix.

    Complexity
    ----------
    Time O(n d^2), extra space O(n d).

    References
    ----------
    Chan, Golub & LeVeque (1983) on centred (two-pass) formulas.

    Examples
    --------
    >>> covariance([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    array([[1., 2.],
           [2., 4.]])
    """
    X = _check_data(X, ddof)
    Xc = X - X.mean(axis=0)
    C = (Xc.T @ Xc) / (X.shape[0] - ddof)
    return (C + C.T) / 2.0  # exact symmetry


@register(
    domain="numerics",
    tasks=("statistics",),
    complexity=Complexity(time="O(n d^2)", space="O(n d)", notes="one centred copy of X"),
    references=("Chan, Golub & LeVeque, The American Statistician 37 (1983) 242-247",),
)
def correlation(X: ArrayLike) -> np.ndarray:
    r"""Pearson correlation matrix of the columns of ``X``.

    :math:`R_{ij} = C_{ij} / \sqrt{C_{ii} C_{jj}}`, clipped to [-1, 1]; equals
    ``np.corrcoef(X, rowvar=False)``. Constant columns give NaN rows/columns.

    Parameters
    ----------
    X : array_like of shape (n, d)
        Rows are observations (n >= 2), columns are variables.

    Returns
    -------
    ndarray of shape (d, d)
        Symmetric correlation matrix.

    Complexity
    ----------
    Time O(n d^2), extra space O(n d).

    References
    ----------
    Chan, Golub & LeVeque (1983).

    Examples
    --------
    >>> correlation([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
    array([[ 1., -1.],
           [-1.,  1.]])
    """
    C = covariance(X, ddof=1)
    s = np.sqrt(np.diag(C))
    with np.errstate(divide="ignore", invalid="ignore"):
        R = C / np.outer(s, s)  # outer(s, s) is exactly symmetric, so R is too
    return np.clip(R, -1.0, 1.0)
