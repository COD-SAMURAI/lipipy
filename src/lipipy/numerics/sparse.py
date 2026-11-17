r"""Compressed sparse row (CSR) matrices.

Storage
-------
An :math:`m\times n` matrix with ``nnz`` stored entries is kept in three arrays:

* ``data[k]`` - the value of the k-th stored entry,
* ``indices[k]`` - its column,
* ``indptr`` (length :math:`m + 1`) - row :math:`i` occupies positions
  ``indptr[i]:indptr[i+1]``.

Storage is :math:`O(\text{nnz} + m)` instead of :math:`O(mn)`. Matrices built here
are in *canonical form*: column indices are sorted within each row and duplicate
``(row, column)`` pairs have been summed (as SciPy's ``sum_duplicates``). Explicit
zeros produced by summing duplicates (e.g. ``+1`` and ``-1``) are kept as stored
entries, like SciPy, so ``nnz`` counts stored entries, not non-zero values.

Products
--------
For :math:`y = Ax`,

.. math::  y_i = \sum_{k=\text{indptr}_i}^{\text{indptr}_{i+1}-1} \text{data}_k\,
           x_{\text{indices}_k},

which touches every stored entry once: :math:`O(\text{nnz} + m)` time. The
implementation forms the products ``data * x[indices]`` and sums each row's segment
with ``np.add.reduceat`` (empty rows are skipped explicitly, since ``reduceat`` does
not handle empty segments). A dense right-hand side with ``k`` columns costs
:math:`O(k(\text{nnz} + m))`. The transpose is built with a stable sort of the
column indices, :math:`O(\text{nnz}\log\text{nnz} + n)`.

The stored arrays are read-only, so a :class:`CSRMatrix` is immutable; operations
return new matrices.

References
----------
Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM (2003), §3.4.
T. A. Davis, *Direct Methods for Sparse Linear Systems*, SIAM (2006), ch. 2.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from lipipy.core.complexity import Complexity
from lipipy.core.registry import register
from lipipy.numerics._common import as_real, jsonable

__all__ = ["CSRMatrix"]


def _readonly(a: np.ndarray) -> np.ndarray:
    a.flags.writeable = False
    return a


def _check_shape(shape: Any) -> tuple[int, int]:
    if not (isinstance(shape, tuple) and len(shape) == 2):
        raise TypeError(f"shape must be a tuple (m, n), got {shape!r}")
    for s in shape:
        if isinstance(s, (bool, np.bool_)) or not isinstance(s, (int, np.integer)) or s < 0:
            raise ValueError(f"shape entries must be non-negative integers, got {shape!r}")
    return int(shape[0]), int(shape[1])


def _as_index(x: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.size and not np.issubdtype(arr.dtype, np.integer):
        if not (np.issubdtype(arr.dtype, np.floating) and np.all(arr == np.round(arr))):
            raise TypeError(f"{name} must contain integers, got dtype {arr.dtype}")
    return arr.astype(np.int64)


@register(
    domain="numerics",
    tasks=("sparse-matrices", "linear-algebra"),
    complexity=Complexity(
        time="O(nnz + n)",
        space="O(nnz + n)",
        notes="matvec on an n-row matrix with nnz stored entries; matmat with k columns "
        "is O(k (nnz + n)); construction from COO and transpose sort: O(nnz log nnz + n)",
    ),
    references=(
        "Saad, Iterative Methods for Sparse Linear Systems, 2nd ed. (2003), §3.4",
        "Davis, Direct Methods for Sparse Linear Systems (2006), ch. 2",
    ),
)
class CSRMatrix:
    r"""Immutable sparse matrix in compressed sparse row format.

    Build one with :meth:`from_dense` or :meth:`from_coo`, or directly from CSR
    arrays (validated and brought to canonical form).

    Parameters
    ----------
    data : array_like of shape (nnz,)
        Stored values.
    indices : array_like of int, shape (nnz,)
        Column index of each stored value.
    indptr : array_like of int, shape (m + 1,)
        Row pointers, non-decreasing, ``indptr[0] == 0`` and ``indptr[-1] == nnz``.
    shape : tuple (m, n)
        Matrix shape.

    Attributes
    ----------
    data, indices, indptr : ndarray
        Canonical CSR arrays (read-only).
    shape : tuple of int
    nnz : int
        Number of stored entries.

    Complexity
    ----------
    ``A @ x``: time O(nnz + m); ``A @ X`` with k columns: O(k (nnz + m));
    ``transpose`` and construction: O(nnz log nnz + m + n). Storage O(nnz + m).

    References
    ----------
    Saad (2003), §3.4; Davis (2006), ch. 2.

    Examples
    --------
    >>> import numpy as np
    >>> A = CSRMatrix.from_coo([0, 0, 1, 0], [0, 2, 1, 0], [1.0, 2.0, 3.0, 4.0], (2, 3))
    >>> A.nnz, A.to_dense()
    (3, array([[5., 0., 2.],
           [0., 3., 0.]]))
    >>> A @ np.array([1.0, 1.0, 1.0])
    array([7., 3.])
    """

    __slots__ = ("_row", "data", "indices", "indptr", "shape")
    __array_ufunc__ = None  # ndarray @ CSRMatrix defers to __rmatmul__

    def __init__(
        self, data: ArrayLike, indices: ArrayLike, indptr: ArrayLike, shape: tuple[int, int]
    ) -> None:
        m, n = _check_shape(shape)
        vals = as_real(data, "data", ndim=1)
        cols = _as_index(indices, "indices")
        ptr = _as_index(indptr, "indptr")
        if ptr.shape != (m + 1,):
            raise ValueError(f"indptr must have length m + 1 = {m + 1}, got {ptr.size}")
        if ptr[0] != 0 or np.any(np.diff(ptr) < 0):
            raise ValueError("indptr must start at 0 and be non-decreasing")
        if ptr[-1] != vals.size or cols.size != vals.size:
            raise ValueError(
                f"data ({vals.size}), indices ({cols.size}) and indptr[-1] ({ptr[-1]}) disagree"
            )
        if cols.size and (cols.min() < 0 or cols.max() >= n):
            raise ValueError(f"column indices must lie in [0, {n})")
        rows = np.repeat(np.arange(m, dtype=np.int64), np.diff(ptr))
        self._set_canonical(rows, cols, vals, (m, n))

    # -- construction -------------------------------------------------------------------
    def _set_canonical(
        self, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, shape: tuple[int, int]
    ) -> None:
        """Sort by (row, col), sum duplicates, and store read-only arrays (inputs validated)."""
        m, n = shape
        order = np.lexsort((cols, rows))
        rows, cols, vals = rows[order], cols[order], vals[order]
        if rows.size:
            new = np.empty(rows.size, dtype=bool)
            new[0] = True
            new[1:] = (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])
            starts = np.flatnonzero(new)
            vals = np.add.reduceat(vals, starts)
            rows, cols = rows[starts], cols[starts]
        ptr = np.zeros(m + 1, dtype=np.int64)
        np.cumsum(np.bincount(rows, minlength=m), out=ptr[1:])
        self.data = _readonly(np.ascontiguousarray(vals, dtype=np.float64))
        self.indices = _readonly(np.ascontiguousarray(cols, dtype=np.int64))
        self.indptr = _readonly(ptr)
        self._row = _readonly(np.ascontiguousarray(rows, dtype=np.int64))
        self.shape = (m, n)

    @classmethod
    def _from_validated(
        cls, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, shape: tuple[int, int]
    ) -> CSRMatrix:
        obj = cls.__new__(cls)
        obj._set_canonical(rows, cols, vals, shape)
        return obj

    @classmethod
    def from_coo(
        cls, rows: ArrayLike, cols: ArrayLike, values: ArrayLike, shape: tuple[int, int]
    ) -> CSRMatrix:
        """Build from coordinate triplets ``(rows[k], cols[k], values[k])``; duplicates are summed.

        Examples
        --------
        >>> CSRMatrix.from_coo([1, 1], [0, 0], [2.0, 3.0], (2, 2)).to_dense()
        array([[0., 0.],
               [5., 0.]])
        """
        m, n = _check_shape(shape)
        r = _as_index(rows, "rows")
        c = _as_index(cols, "cols")
        v = as_real(values, "values", ndim=1)
        if not (r.size == c.size == v.size):
            raise ValueError(f"rows ({r.size}), cols ({c.size}), values ({v.size}) differ")
        if r.size and (r.min() < 0 or r.max() >= m or c.min() < 0 or c.max() >= n):
            raise ValueError(f"indices out of range for shape {(m, n)}")
        return cls._from_validated(r, c, v.copy(), (m, n))

    @classmethod
    def from_dense(cls, A: ArrayLike) -> CSRMatrix:
        """Store the non-zero entries of a dense 2-D array.

        Examples
        --------
        >>> CSRMatrix.from_dense([[0.0, 1.0], [2.0, 0.0]]).nnz
        2
        """
        A = as_real(A, "A", ndim=2)
        r, c = np.nonzero(A)
        return cls._from_validated(r.astype(np.int64), c.astype(np.int64), A[r, c], A.shape)

    # -- properties ---------------------------------------------------------------------
    @property
    def nnz(self) -> int:
        """Number of stored entries."""
        return int(self.data.size)

    @property
    def T(self) -> CSRMatrix:
        return self.transpose()

    def __repr__(self) -> str:
        return f"CSRMatrix(shape={self.shape}, nnz={self.nnz})"

    def to_dense(self) -> np.ndarray:
        """Dense ndarray copy, O(m n) memory."""
        out = np.zeros(self.shape)
        out[self._row, self.indices] = self.data  # canonical: no duplicate positions
        return out

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly CSR arrays."""
        return {
            "shape": list(self.shape),
            "data": jsonable(self.data),
            "indices": jsonable(self.indices),
            "indptr": jsonable(self.indptr),
        }

    def diagonal(self) -> np.ndarray:
        """Main diagonal (length ``min(m, n)``), O(nnz)."""
        k = min(self.shape)
        out = np.zeros(k)
        on = (self._row == self.indices) & (self._row < k)
        out[self._row[on]] = self.data[on]
        return out

    def transpose(self) -> CSRMatrix:
        """The transpose as a new CSR matrix."""
        m, n = self.shape
        return CSRMatrix._from_validated(self.indices, self._row, self.data.copy(), (n, m))

    # -- products -------------------------------------------------------------------------
    def _matmul_dense(self, x: np.ndarray) -> np.ndarray:
        m = self.shape[0]
        out_shape = (m,) + x.shape[1:]
        out = np.zeros(out_shape)
        if self.nnz == 0:
            return out
        prod = self.data.reshape((-1,) + (1,) * (x.ndim - 1)) * x[self.indices]
        nonempty = np.flatnonzero(np.diff(self.indptr) > 0)
        out[nonempty] = np.add.reduceat(prod, self.indptr[nonempty], axis=0)
        return out

    def __matmul__(self, other: Any) -> np.ndarray:
        if isinstance(other, CSRMatrix):
            raise TypeError("sparse @ sparse is not supported; convert one side with to_dense()")
        x = as_real(other, "x")
        if x.ndim not in (1, 2):
            raise ValueError(f"x must be 1-D or 2-D, got {x.ndim}D")
        if x.shape[0] != self.shape[1]:
            raise ValueError(f"shape mismatch: {self.shape} @ {x.shape}")
        return self._matmul_dense(x)

    def __rmatmul__(self, other: Any) -> np.ndarray:
        x = as_real(other, "x")
        if x.ndim not in (1, 2):
            raise ValueError(f"x must be 1-D or 2-D, got {x.ndim}D")
        if x.shape[-1] != self.shape[0]:
            raise ValueError(f"shape mismatch: {x.shape} @ {self.shape}")
        return self.transpose()._matmul_dense(x.T).T

    def matvec(self, x: ArrayLike) -> np.ndarray:
        """``A @ x`` for a 1-D ``x``."""
        x = as_real(x, "x", ndim=1)
        return self @ x

    # -- scalar operations ----------------------------------------------------------------
    def _scaled(self, factor: float) -> CSRMatrix:
        out = CSRMatrix.__new__(CSRMatrix)
        out.data = _readonly(self.data * factor)
        out.indices, out.indptr, out._row, out.shape = (
            self.indices,
            self.indptr,
            self._row,
            self.shape,
        )
        return out

    @staticmethod
    def _scalar(value: Any) -> float | None:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
            return None
        if np.iscomplexobj(value):
            return None
        if not np.isfinite(value):
            raise ValueError(f"scalar must be finite, got {value!r}")
        return float(value)

    def __mul__(self, other: Any) -> CSRMatrix:
        c = self._scalar(other)
        return NotImplemented if c is None else self._scaled(c)

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> CSRMatrix:
        c = self._scalar(other)
        if c is None:
            return NotImplemented
        if c == 0.0:
            raise ZeroDivisionError("division of a sparse matrix by zero")
        return self._scaled(1.0 / c)

    def __neg__(self) -> CSRMatrix:
        return self._scaled(-1.0)
