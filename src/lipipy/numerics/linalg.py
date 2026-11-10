r"""Dense direct linear algebra: triangular solves, LU, Cholesky, Householder QR.

Triangular systems
------------------
For lower-triangular :math:`L`, forward substitution solves :math:`Lx = b` row by row,

.. math::  x_i = \Bigl(b_i - \sum_{j<i} L_{ij} x_j\Bigr) / L_{ii},

and back substitution does the same from the last row for upper-triangular systems.

LU with partial pivoting
------------------------
Gaussian elimination with row exchanges produces :math:`PA = LU` with a permutation
:math:`P`, unit lower-triangular :math:`L` and upper-triangular :math:`U`. At step
:math:`k` the row with the largest :math:`|a_{ik}|`, :math:`i\ge k`, is swapped into
the pivot position, which bounds the multipliers :math:`|l_{ik}|\le 1`; then

.. math::  l_{ik} = a_{ik}/a_{kk},\qquad a_{ij} \leftarrow a_{ij} - l_{ik} a_{kj}
           \quad (i, j > k).

Then :math:`Ax=b` becomes :math:`Ly = Pb`, :math:`Ux = y`, and
:math:`\det A = (-1)^{s}\prod_k u_{kk}` with :math:`s` row swaps.

Cholesky
--------
A symmetric positive-definite matrix factors as :math:`A = LL^\top` with

.. math::  l_{jj} = \sqrt{a_{jj} - \textstyle\sum_{k<j} l_{jk}^2},\qquad
           l_{ij} = \bigl(a_{ij} - \textstyle\sum_{k<j} l_{ik} l_{jk}\bigr)/l_{jj}
           \quad (i>j).

A non-positive value under the square root proves :math:`A` is not positive definite.

Householder QR
--------------
For column :math:`j`, let :math:`x` be the entries on and below the diagonal. The
reflector :math:`H = I - 2vv^\top` with :math:`v \propto x - \alpha e_1`,
:math:`\alpha = -\operatorname{sign}(x_1)\lVert x\rVert`, maps :math:`x` to
:math:`\alpha e_1` (the sign avoids cancellation in :math:`x_1 - \alpha`). After
:math:`k=\min(n,d)` reflections :math:`H_k\cdots H_1 A = R` and
:math:`Q = H_1\cdots H_k`. **Sign convention:** :math:`R_{jj} = -\operatorname{sign}
(x_1)\lVert x\rVert`, the LAPACK ``geqrf`` convention, so results match
``numpy.linalg.qr`` up to rounding; when :math:`x` is already a multiple of
:math:`e_1` no reflection is applied (as in LAPACK). Least squares
:math:`\min\lVert Ax-b\rVert_2` then reduces to :math:`R_1 x = (Q^\top b)_{1:d}`,
which avoids squaring the condition number as the normal equations do.

References
----------
G. H. Golub & C. F. Van Loan, *Matrix Computations*, 4th ed. (2013): §3.1
(triangular systems), §3.2-3.4 (LU, pivoting), §4.2 (Cholesky), §5.1-5.3
(Householder QR, least squares). L. N. Trefethen & D. Bau, *Numerical Linear
Algebra* (1997), Lectures 10, 11, 20-23.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from lipipy.core.complexity import Complexity
from lipipy.core.registry import register
from lipipy.core.validation import check_array

__all__ = [
    "cholesky",
    "cholesky_solve",
    "det",
    "lstsq_qr",
    "lu",
    "lu_solve",
    "qr",
    "solve_triangular",
]

LinAlgError = np.linalg.LinAlgError
_GVL = "Golub & Van Loan, Matrix Computations, 4th ed. (2013)"
_TB = "Trefethen & Bau, Numerical Linear Algebra (1997)"


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------
def _square(A: ArrayLike, name: str = "A") -> np.ndarray:
    A = check_array(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"{name} must be square, got shape {A.shape}")
    return A


def _rhs(b: ArrayLike, d: int) -> np.ndarray:
    b = check_array(b, ensure_2d=False)
    if b.ndim not in (1, 2):
        raise ValueError(f"b must be 1-D or 2-D, got {b.ndim}D")
    if b.shape[0] != d:
        raise ValueError(f"b has {b.shape[0]} rows, expected {d}")
    return b


# ---------------------------------------------------------------------------
# triangular solve
# ---------------------------------------------------------------------------
def _solve_tri(T: np.ndarray, b: np.ndarray, lower: bool, unit: bool) -> np.ndarray:
    """Substitution without validation (internal)."""
    d = T.shape[0]
    x = np.array(b, dtype=np.float64, copy=True)
    rows = range(d) if lower else range(d - 1, -1, -1)
    for i in rows:
        known = slice(0, i) if lower else slice(i + 1, d)
        x[i] -= T[i, known] @ x[known]
        if not unit:
            x[i] /= T[i, i]
    return x


@register(
    domain="numerics",
    tasks=("linear-algebra", "linear-systems"),
    complexity=Complexity(
        time="O(d^2 n)", space="O(1)", notes="n right-hand sides (n = 1 for a vector)"
    ),
    references=(_GVL + ", §3.1",),
)
def solve_triangular(
    T: ArrayLike, b: ArrayLike, lower: bool = True, unit_diagonal: bool = False
) -> np.ndarray:
    r"""Solve :math:`Tx = b` for triangular :math:`T` by forward or back substitution.

    Only the relevant triangle of ``T`` is read.

    Parameters
    ----------
    T : array_like of shape (d, d)
        Triangular matrix.
    b : array_like of shape (d,) or (d, n)
        Right-hand side(s).
    lower : bool, default=True
        ``True`` for lower-triangular (forward substitution), ``False`` for upper.
    unit_diagonal : bool, default=False
        Assume ones on the diagonal (the diagonal of ``T`` is not read).

    Returns
    -------
    ndarray, same shape as ``b``
        The solution ``x``.

    Raises
    ------
    numpy.linalg.LinAlgError
        If a diagonal entry is zero (singular matrix).

    Complexity
    ----------
    Time O(d^2 n) for n right-hand sides; no extra memory beyond the output.

    References
    ----------
    Golub & Van Loan, *Matrix Computations*, §3.1.

    Examples
    --------
    >>> solve_triangular([[2.0, 0.0], [1.0, 1.0]], [2.0, 3.0])
    array([1., 2.])
    >>> solve_triangular([[2.0, 1.0], [0.0, 1.0]], [4.0, 2.0], lower=False)
    array([1., 2.])
    """
    T = _square(T, "T")
    b = _rhs(b, T.shape[0])
    if not unit_diagonal:
        zero = np.flatnonzero(np.diag(T) == 0.0)
        if zero.size:
            raise LinAlgError(f"singular triangular matrix: zero on the diagonal at {zero[0]}")
    return _solve_tri(T, b, lower, unit_diagonal)


# ---------------------------------------------------------------------------
# LU
# ---------------------------------------------------------------------------
def _lu_factor(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Compact LU of a validated square matrix: (LU, perm, n_swaps) with A[perm] = L U."""
    d = A.shape[0]
    LU = A.copy()
    perm = np.arange(d)
    swaps = 0
    for k in range(d):
        p = k + int(np.argmax(np.abs(LU[k:, k])))
        if p != k:
            LU[[k, p]] = LU[[p, k]]
            perm[[k, p]] = perm[[p, k]]
            swaps += 1
        pivot = LU[k, k]
        if pivot == 0.0:
            continue  # column already zero below the diagonal: singular, U[k, k] = 0
        LU[k + 1 :, k] /= pivot
        LU[k + 1 :, k + 1 :] -= np.outer(LU[k + 1 :, k], LU[k, k + 1 :])
    return LU, perm, swaps


@register(
    domain="numerics",
    tasks=("linear-algebra", "matrix-factorization"),
    complexity=Complexity(time="O(d^3)", space="O(d^2)", notes="(2/3) d^3 flops"),
    references=(_GVL + ", §3.2-3.4",),
)
def lu(A: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""LU factorisation with partial pivoting, :math:`PA = LU`.

    Singular matrices are factored too (``U`` then has a zero on its diagonal).

    Parameters
    ----------
    A : array_like of shape (d, d)
        Square matrix.

    Returns
    -------
    P : ndarray of shape (d, d)
        Permutation matrix.
    L : ndarray of shape (d, d)
        Unit lower-triangular factor, :math:`|L_{ij}| \le 1`.
    U : ndarray of shape (d, d)
        Upper-triangular factor.

    Notes
    -----
    ``scipy.linalg.lu`` returns ``p`` with ``A = p @ L @ U``, i.e. ``p = P.T``.

    Complexity
    ----------
    Time O(d^3), extra space O(d^2).

    References
    ----------
    Golub & Van Loan, *Matrix Computations*, §3.2-3.4.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> P, L, U = lu(A)
    >>> np.allclose(P @ A, L @ U)
    True
    >>> U
    array([[3.        , 4.        ],
           [0.        , 0.66666667]])
    """
    LU, perm, _ = _lu_factor(_square(A))
    d = LU.shape[0]
    P = np.eye(d)[perm]
    L = np.tril(LU, -1) + np.eye(d)
    U = np.triu(LU)
    return P, L, U


@register(
    domain="numerics",
    tasks=("linear-algebra", "linear-systems"),
    complexity=Complexity(
        time="O(d^2 n)", space="O(d n)", notes="n right-hand sides; factorisation not included"
    ),
    references=(_GVL + ", §3.2",),
)
def lu_solve(factors: tuple[ArrayLike, ArrayLike, ArrayLike], b: ArrayLike) -> np.ndarray:
    r"""Solve :math:`Ax = b` given ``(P, L, U) = lu(A)``.

    Solves :math:`Ly = Pb` then :math:`Ux = y`.

    Parameters
    ----------
    factors : tuple (P, L, U)
        Output of :func:`lu`.
    b : array_like of shape (d,) or (d, n)
        Right-hand side(s).

    Returns
    -------
    ndarray, same shape as ``b``

    Raises
    ------
    numpy.linalg.LinAlgError
        If ``U`` has a zero on the diagonal (``A`` is singular).

    Complexity
    ----------
    Time O(d^2 n) (plus O(d^2 n) for the permutation product), extra space O(d n).

    References
    ----------
    Golub & Van Loan, *Matrix Computations*, §3.2.

    Examples
    --------
    >>> A = [[4.0, 3.0], [6.0, 3.0]]
    >>> lu_solve(lu(A), [10.0, 12.0])
    array([1., 2.])
    """
    if not isinstance(factors, tuple) or len(factors) != 3:
        raise TypeError("factors must be the (P, L, U) tuple returned by lu()")
    P, L, U = (_square(M, name) for M, name in zip(factors, "PLU", strict=True))
    if not (P.shape == L.shape == U.shape):
        raise ValueError("P, L and U must have the same shape")
    b = _rhs(b, U.shape[0])
    zero = np.flatnonzero(np.diag(U) == 0.0)
    if zero.size:
        raise LinAlgError(f"singular matrix: U[{zero[0]}, {zero[0]}] = 0")
    y = _solve_tri(L, P @ b, lower=True, unit=True)
    return _solve_tri(U, y, lower=False, unit=False)


@register(
    domain="numerics",
    tasks=("linear-algebra",),
    complexity=Complexity(time="O(d^3)", space="O(d^2)"),
    references=(_GVL + ", §3.2",),
)
def det(A: ArrayLike) -> float:
    r"""Determinant via LU: :math:`\det A = (-1)^{s}\prod_k U_{kk}` for ``s`` row swaps.

    Parameters
    ----------
    A : array_like of shape (d, d)
        Square matrix.

    Returns
    -------
    float
        The determinant (0.0 for an exactly singular matrix). May overflow for
        large matrices; compute ``log|det|`` from the diagonal of ``U`` if needed.

    Complexity
    ----------
    Time O(d^3), extra space O(d^2).

    References
    ----------
    Golub & Van Loan, *Matrix Computations*, §3.2.

    Examples
    --------
    >>> det([[1.0, 2.0], [3.0, 4.0]])
    -2.0
    """
    LU, _, swaps = _lu_factor(_square(A))
    return float((-1.0) ** swaps * np.prod(np.diag(LU)))


# ---------------------------------------------------------------------------
# Cholesky
# ---------------------------------------------------------------------------
@register(
    domain="numerics",
    tasks=("linear-algebra", "matrix-factorization"),
    complexity=Complexity(time="O(d^3)", space="O(d^2)", notes="(1/3) d^3 flops"),
    references=(_GVL + ", §4.2",),
)
def cholesky(A: ArrayLike) -> np.ndarray:
    r"""Cholesky factor :math:`L` (lower-triangular) with :math:`A = LL^\top`.

    Parameters
    ----------
    A : array_like of shape (d, d)
        Symmetric positive-definite matrix.

    Returns
    -------
    ndarray of shape (d, d)
        Lower-triangular ``L`` with positive diagonal.

    Raises
    ------
    numpy.linalg.LinAlgError
        If ``A`` is not symmetric or not positive definite.

    Complexity
    ----------
    Time O(d^3), extra space O(d^2).

    References
    ----------
    Golub & Van Loan, *Matrix Computations*, §4.2.

    Examples
    --------
    >>> cholesky([[4.0, 2.0], [2.0, 2.0]])
    array([[2., 0.],
           [1., 1.]])
    """
    A = _square(A)
    scale = np.max(np.abs(A))
    if np.any(np.abs(A - A.T) > 1e-10 * scale):
        raise LinAlgError("matrix is not symmetric, so it is not symmetric positive definite")
    d = A.shape[0]
    L = np.zeros_like(A)
    for j in range(d):
        pivot = A[j, j] - L[j, :j] @ L[j, :j]
        if not pivot > 0.0:  # also catches NaN
            raise LinAlgError(
                f"matrix is not positive definite (pivot {pivot:.3g} at index {j}); "
                "Cholesky needs a symmetric positive-definite matrix"
            )
        L[j, j] = np.sqrt(pivot)
        L[j + 1 :, j] = (A[j + 1 :, j] - L[j + 1 :, :j] @ L[j, :j]) / L[j, j]
    return L


@register(
    domain="numerics",
    tasks=("linear-algebra", "linear-systems"),
    complexity=Complexity(
        time="O(d^2 n)", space="O(d n)", notes="n right-hand sides; factorisation not included"
    ),
    references=(_GVL + ", §4.2",),
)
def cholesky_solve(L: ArrayLike, b: ArrayLike) -> np.ndarray:
    r"""Solve :math:`Ax = b` given the Cholesky factor ``L`` of :math:`A = LL^\top`.

    Solves :math:`Ly = b` then :math:`L^\top x = y`.

    Parameters
    ----------
    L : array_like of shape (d, d)
        Lower-triangular factor from :func:`cholesky`.
    b : array_like of shape (d,) or (d, n)
        Right-hand side(s).

    Returns
    -------
    ndarray, same shape as ``b``

    Complexity
    ----------
    Time O(d^2 n), extra space O(d n).

    References
    ----------
    Golub & Van Loan, *Matrix Computations*, §4.2.

    Examples
    --------
    >>> A = [[4.0, 2.0], [2.0, 2.0]]
    >>> cholesky_solve(cholesky(A), [8.0, 6.0])
    array([1., 2.])
    """
    L = _square(L, "L")
    b = _rhs(b, L.shape[0])
    zero = np.flatnonzero(np.diag(L) == 0.0)
    if zero.size:
        raise LinAlgError(f"singular factor: L[{zero[0]}, {zero[0]}] = 0")
    y = _solve_tri(L, b, lower=True, unit=False)
    return _solve_tri(L.T, y, lower=False, unit=False)


# ---------------------------------------------------------------------------
# Householder QR
# ---------------------------------------------------------------------------
def _householder(A: np.ndarray) -> tuple[np.ndarray, list[np.ndarray | None]]:
    """Reduce A (n x d) to upper-triangular R in place; return R and unit vectors v_j."""
    R = A.copy()
    n, d = R.shape
    vs: list[np.ndarray | None] = []
    for j in range(min(n, d)):
        x = R[j:, j]
        tail = np.linalg.norm(x[1:])
        if tail == 0.0:  # already a multiple of e_1: H = I (LAPACK's tau = 0)
            vs.append(None)
            continue
        alpha = -np.copysign(np.hypot(x[0], tail), x[0])
        v = x.copy()
        v[0] -= alpha
        v /= np.linalg.norm(v)
        R[j:, j:] -= 2.0 * np.outer(v, v @ R[j:, j:])
        R[j + 1 :, j] = 0.0  # exact zeros below the diagonal
        vs.append(v)
    return R, vs


def _apply_q(vs: list[np.ndarray | None], M: np.ndarray, transpose: bool) -> np.ndarray:
    """Compute Q^T M (transpose=True) or Q M from the stored reflectors."""
    M = np.array(M, dtype=np.float64, copy=True)
    order = range(len(vs)) if transpose else range(len(vs) - 1, -1, -1)
    for j in order:
        v = vs[j]
        if v is not None:
            M[j:] -= 2.0 * np.outer(v, v @ M[j:]) if M.ndim == 2 else 2.0 * v * (v @ M[j:])
    return M


@register(
    domain="numerics",
    tasks=("linear-algebra", "matrix-factorization"),
    complexity=Complexity(
        time="O(n d^2)",
        space="O(n d)",
        notes="A is n x d with n >= d; mode='complete' forms Q in O(n^2 d) time",
    ),
    references=(_GVL + ", §5.1-5.2", _TB + ", Lecture 10"),
)
def qr(A: ArrayLike, mode: str = "reduced") -> tuple[np.ndarray, np.ndarray]:
    r"""QR factorisation :math:`A = QR` by Householder reflections.

    Parameters
    ----------
    A : array_like of shape (n, d)
        Matrix to factor.
    mode : {"reduced", "complete"}, default="reduced"
        ``"reduced"``: ``Q`` is (n, k), ``R`` is (k, d) with ``k = min(n, d)``.
        ``"complete"``: ``Q`` is (n, n) orthogonal, ``R`` is (n, d).

    Returns
    -------
    Q : ndarray
        Matrix with orthonormal columns.
    R : ndarray
        Upper-triangular (trapezoidal) matrix. Sign convention:
        :math:`R_{jj} = -\operatorname{sign}(x_1)\lVert x\rVert` as in LAPACK, so the
        diagonal can be negative and ``Q, R`` agree with ``numpy.linalg.qr``.

    Complexity
    ----------
    Time O(n d^2), extra space O(n d) for ``n >= d``.

    References
    ----------
    Golub & Van Loan, *Matrix Computations*, §5.1-5.2; Trefethen & Bau, Lecture 10.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[3.0, 1.0], [4.0, 2.0]])
    >>> Q, R = qr(A)
    >>> np.allclose(Q @ R, A), np.allclose(Q.T @ Q, np.eye(2))
    (True, True)
    >>> R
    array([[-5. , -2.2],
           [ 0. ,  0.4]])
    """
    if mode not in ("reduced", "complete"):
        raise ValueError(f"mode must be 'reduced' or 'complete', got {mode!r}")
    A = check_array(A)
    n, d = A.shape
    R, vs = _householder(A)
    k = min(n, d)
    cols = k if mode == "reduced" else n
    Q = _apply_q(vs, np.eye(n, cols), transpose=False)
    return Q, (R[:k] if mode == "reduced" else R)


@register(
    domain="numerics",
    tasks=("linear-algebra", "least-squares", "regression"),
    complexity=Complexity(time="O(n d^2)", space="O(n d)", notes="A is n x d, n >= d"),
    references=(_GVL + ", §5.3", _TB + ", Lecture 11"),
)
def lstsq_qr(A: ArrayLike, b: ArrayLike) -> np.ndarray:
    r"""Least-squares solution of :math:`\min_x \lVert Ax - b\rVert_2` via Householder QR.

    Applies the reflectors to ``b`` (``Q`` is never formed) and back-substitutes
    :math:`R_1 x = (Q^\top b)_{1:d}`.

    Parameters
    ----------
    A : array_like of shape (n, d)
        Design matrix with ``n >= d`` and full column rank.
    b : array_like of shape (n,) or (n, m)
        Right-hand side(s).

    Returns
    -------
    ndarray of shape (d,) or (d, m)

    Raises
    ------
    numpy.linalg.LinAlgError
        If ``A`` is (numerically) rank deficient; use an SVD-based solver then.

    Complexity
    ----------
    Time O(n d^2), extra space O(n d).

    References
    ----------
    Golub & Van Loan, *Matrix Computations*, §5.3; Trefethen & Bau, Lecture 11.

    Examples
    --------
    >>> A = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]
    >>> lstsq_qr(A, [1.0, 3.0, 5.0])  # fits b = 1 + 2 t exactly
    array([1., 2.])
    """
    A = check_array(A)
    n, d = A.shape
    if n < d:
        raise ValueError(f"lstsq_qr needs at least as many rows as columns, got shape {A.shape}")
    b = _rhs(b, n)
    R, vs = _householder(A)
    diag = np.abs(np.diag(R))
    if diag.min() <= np.finfo(np.float64).eps * max(n, d) * diag.max():
        raise LinAlgError("matrix is rank deficient; least squares via QR needs full column rank")
    qtb = _apply_q(vs, b, transpose=True)
    return _solve_tri(R[:d], qtb[:d], lower=False, unit=False)
