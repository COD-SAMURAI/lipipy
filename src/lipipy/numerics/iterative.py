r"""Krylov iterative solvers for :math:`Ax = b`: conjugate gradient and restarted GMRES.

Both solvers only need products :math:`v \mapsto Av`, so ``A`` may be a dense array,
a :class:`~lipipy.numerics.sparse.CSRMatrix` or a callable ``matvec(v)``.

Stopping rule (both solvers)
----------------------------
A solve is reported ``converged`` **if and only if** the true residual of the
returned :math:`x` satisfies

.. math::  \lVert b - Ax\rVert_2 \le \max(\text{tol}\,\lVert b\rVert_2,\ \text{atol}).

The cheap residual estimates of the recurrences can drift from the true residual in
floating point, so whenever an estimate passes the test the true residual is
recomputed; if it does not pass, the iteration continues from the true residual.

Conjugate gradient
------------------
For symmetric positive-definite :math:`A`, CG minimises the :math:`A`-norm of the
error over the Krylov space :math:`x_0 + \mathcal K_k(A, r_0)`. With a symmetric
positive-definite preconditioner :math:`M \approx A^{-1}` (applied as
:math:`z = Mr`):

.. math::
    \alpha_k = \frac{r_k^\top z_k}{p_k^\top A p_k},\quad
    x_{k+1} = x_k + \alpha_k p_k,\quad r_{k+1} = r_k - \alpha_k A p_k,\\
    z_{k+1} = M r_{k+1},\quad \beta_k = \frac{r_{k+1}^\top z_{k+1}}{r_k^\top z_k},\quad
    p_{k+1} = z_{k+1} + \beta_k p_k .

In exact arithmetic CG terminates in at most as many iterations as :math:`A` has
distinct eigenvalues (so at most :math:`n`), and
:math:`\lVert e_k\rVert_A \le 2\bigl(\tfrac{\sqrt\kappa - 1}{\sqrt\kappa + 1}\bigr)^k
\lVert e_0\rVert_A`. ``M="jacobi"`` uses :math:`M = \operatorname{diag}(A)^{-1}`.
A non-positive curvature :math:`p^\top A p \le 0` proves :math:`A` is not positive
definite; the solve then stops with ``converged=False``.

GMRES(r)
--------
For general non-singular :math:`A`, each cycle builds an orthonormal Krylov basis
:math:`V_{j+1}` by the Arnoldi process (modified Gram-Schmidt),
:math:`AV_j = V_{j+1}\bar H_j`, and picks :math:`x = x_0 + V_j y` minimising
:math:`\lVert \beta e_1 - \bar H_j y\rVert_2`. Givens rotations keep
:math:`\bar H_j` triangular, which makes the residual norm available after every
step as :math:`|g_{j+1}|`. After ``restart`` steps the basis is discarded and a new
cycle starts from the current :math:`x` (memory :math:`O(rn)`). Restarting can
stagnate on hard problems; then the solve stops at ``maxiter`` with
``converged=False``.

References
----------
Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM (2003):
Alg. 6.18 (CG), Alg. 9.1 (PCG), Alg. 6.9-6.11 (GMRES with Givens rotations).
Y. Saad & M. H. Schultz, "GMRES: a generalized minimal residual algorithm for
solving nonsymmetric linear systems", *SIAM J. Sci. Stat. Comput.* 7 (1986) 856-869.
M. R. Hestenes & E. Stiefel, "Methods of conjugate gradients for solving linear
systems", *J. Res. Nat. Bur. Standards* 49 (1952) 409-436.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from lipipy.core.complexity import Complexity
from lipipy.core.errors import ConvergenceWarning
from lipipy.core.registry import register
from lipipy.numerics._common import as_real, check_positive_int, jsonable
from lipipy.numerics.sparse import CSRMatrix

__all__ = ["IterativeResult", "conjugate_gradient", "gmres"]

Operator = np.ndarray | CSRMatrix | Callable[[np.ndarray], Any]


@dataclass(frozen=True)
class IterativeResult:
    """Outcome of :func:`conjugate_gradient` and :func:`gmres`.

    Attributes
    ----------
    x : ndarray
        Final iterate.
    residual_norm : float
        True residual :math:`\\lVert b - Ax\\rVert_2` of ``x``.
    residual_norms : ndarray
        Residual norm history: the initial residual, then one entry per iteration
        (recurrence estimates; entries where the true residual was recomputed hold
        the true value). The last entry equals ``residual_norm``.
    iterations : int
        Iterations performed (matrix-vector products after the initial residual).
    converged : bool
        Whether ``residual_norm <= max(tol * ||b||, atol)``.
    message : str
        Why the iteration stopped.
    """

    x: np.ndarray
    residual_norm: float
    residual_norms: np.ndarray
    iterations: int
    converged: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dictionary (arrays become lists)."""
        return {
            "x": jsonable(self.x),
            "residual_norm": self.residual_norm,
            "residual_norms": jsonable(self.residual_norms),
            "iterations": self.iterations,
            "converged": self.converged,
            "message": self.message,
        }


# ---------------------------------------------------------------------------
# operator handling
# ---------------------------------------------------------------------------
def _operator(A: Any, n: int, name: str = "A") -> Callable[[np.ndarray], np.ndarray]:
    """Turn a dense array, CSRMatrix or callable into a validated matvec of size n."""
    if isinstance(A, CSRMatrix):
        if A.shape != (n, n):
            raise ValueError(f"{name} has shape {A.shape}, expected {(n, n)}")
        return A._matmul_dense
    if callable(A):

        def matvec(v: np.ndarray) -> np.ndarray:
            out = np.asarray(A(v), dtype=np.float64)
            if out.shape != (n,):
                raise ValueError(f"{name}(v) returned shape {out.shape}, expected {(n,)}")
            return out

        return matvec
    M = as_real(A, name, ndim=2)
    if M.shape != (n, n):
        raise ValueError(f"{name} has shape {M.shape}, expected {(n, n)}")
    return lambda v: M @ v


def _diagonal(A: Any, n: int) -> np.ndarray:
    if isinstance(A, CSRMatrix):
        return A.diagonal()
    if callable(A):
        raise ValueError("M='jacobi' needs the diagonal of A; pass A as an array or CSRMatrix")
    return np.diag(as_real(A, "A", ndim=2)).copy()


def _setup(
    A: Any, b: ArrayLike, x0: ArrayLike | None, tol: float, atol: float, maxiter: int | None
) -> tuple[Callable[[np.ndarray], np.ndarray], np.ndarray, np.ndarray, float, int]:
    b = as_real(b, "b", ndim=1)
    n = b.size
    if n == 0:
        raise ValueError("b must not be empty")
    if isinstance(tol, (bool, np.bool_)) or not (np.isfinite(tol) and tol >= 0):
        raise ValueError(f"tol must be a finite number >= 0, got {tol!r}")
    if isinstance(atol, (bool, np.bool_)) or not (np.isfinite(atol) and atol >= 0):
        raise ValueError(f"atol must be a finite number >= 0, got {atol!r}")
    if tol == 0 and atol == 0:
        raise ValueError("tol and atol cannot both be 0")
    iters = 10 * n if maxiter is None else check_positive_int(maxiter, "maxiter")
    matvec = _operator(A, n)
    x = np.zeros(n) if x0 is None else as_real(x0, "x0", ndim=1).copy()
    if x.shape != (n,):
        raise ValueError(f"x0 has shape {x.shape}, expected {(n,)}")
    threshold = max(float(tol) * float(np.linalg.norm(b)), float(atol))
    return matvec, b, x, threshold, iters


def _finish(
    name: str,
    x: np.ndarray,
    res: float,
    history: list[float],
    it: int,
    converged: bool,
    message: str,
) -> IterativeResult:
    hist = np.array(history)
    hist[-1] = res
    hist.flags.writeable = False
    if not converged:
        warnings.warn(f"{name} did not converge: {message}", ConvergenceWarning, stacklevel=3)
    return IterativeResult(
        x=x,
        residual_norm=res,
        residual_norms=hist,
        iterations=it,
        converged=converged,
        message=message,
    )


# ---------------------------------------------------------------------------
# conjugate gradient
# ---------------------------------------------------------------------------
@register(
    domain="numerics",
    tasks=("linear-systems", "iterative-solvers", "sparse-matrices"),
    complexity=Complexity(
        time="O(T (nnz + n))",
        space="O(n)",
        notes="T iterations, one matvec each; nnz = stored entries of A (n^2 if dense); "
        "at most n iterations in exact arithmetic",
    ),
    references=(
        "Hestenes & Stiefel, J. Res. Nat. Bur. Standards 49 (1952) 409-436",
        "Saad, Iterative Methods for Sparse Linear Systems, 2nd ed. (2003), Alg. 6.18, 9.1",
    ),
)
def conjugate_gradient(
    A: Operator,
    b: ArrayLike,
    x0: ArrayLike | None = None,
    tol: float = 1e-5,
    maxiter: int | None = None,
    M: Operator | str | None = None,
    atol: float = 0.0,
) -> IterativeResult:
    r"""Solve :math:`Ax = b` for symmetric positive-definite :math:`A` by (preconditioned) CG.

    Parameters
    ----------
    A : ndarray (n, n), CSRMatrix or callable
        SPD matrix, or ``matvec(v) -> A v``.
    b : array_like of shape (n,)
        Right-hand side.
    x0 : array_like of shape (n,), optional
        Initial guess (zeros by default).
    tol : float, default=1e-5
        Relative tolerance: converged when :math:`\lVert b-Ax\rVert_2 \le
        \max(\text{tol}\lVert b\rVert_2, \text{atol})` (module docstring).
    maxiter : int, optional
        Maximum iterations (default ``10 n``).
    M : {None, "jacobi"} or ndarray, CSRMatrix or callable, optional
        Preconditioner applied as :math:`z = Mr` (an approximation of
        :math:`A^{-1}`, symmetric positive definite). ``"jacobi"`` uses
        :math:`\operatorname{diag}(A)^{-1}`.
    atol : float, default=0.0
        Absolute tolerance.

    Returns
    -------
    IterativeResult
        ``x``, ``residual_norm``, ``residual_norms``, ``iterations``, ``converged``,
        ``message``. A :class:`~lipipy.core.ConvergenceWarning` is emitted when not
        converged.

    Complexity
    ----------
    Time O(T (nnz + n)) for T iterations; extra space O(n).

    References
    ----------
    Hestenes & Stiefel (1952); Saad (2003), Alg. 6.18 and 9.1.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[4.0, 1.0], [1.0, 3.0]])
    >>> res = conjugate_gradient(A, [1.0, 2.0], tol=1e-12)
    >>> res.converged, res.iterations, np.allclose(res.x, np.linalg.solve(A, [1.0, 2.0]))
    (True, 2, True)
    """
    matvec, b, x, threshold, maxiter = _setup(A, b, x0, tol, atol, maxiter)
    n = b.size
    if M is None:
        precond: Callable[[np.ndarray], np.ndarray] = lambda r: r  # noqa: E731
    elif isinstance(M, str):
        if M != "jacobi":
            raise ValueError(f"M must be None, 'jacobi', an array or a callable, got {M!r}")
        diag = _diagonal(A, n)
        if np.any(diag <= 0):
            raise ValueError("M='jacobi' needs a positive diagonal (A must be SPD)")
        inv_diag = 1.0 / diag
        precond = lambda r: inv_diag * r  # noqa: E731
    else:
        precond = _operator(M, n, "M")

    r = b - matvec(x)
    res = float(np.linalg.norm(r))
    history = [res]
    if res <= threshold:
        return _finish(
            "conjugate_gradient", x, res, history, 0, True, "initial guess is a solution"
        )

    z = precond(r)
    rz = float(r @ z)
    p = z.copy()
    it, converged = 0, False
    message = f"maximum number of iterations ({maxiter}) reached"
    while it < maxiter:
        if not rz > 0:
            message = "preconditioner is not positive definite (r.Mr <= 0)"
            break
        Ap = matvec(p)
        pAp = float(p @ Ap)
        if not pAp > 0:
            message = "matrix is not positive definite (p.Ap <= 0)"
            break
        alpha = rz / pAp
        x += alpha * p
        r -= alpha * Ap
        it += 1
        res = float(np.linalg.norm(r))
        history.append(res)
        if res <= threshold:
            r = b - matvec(x)  # confirm with the true residual
            res = float(np.linalg.norm(r))
            history[-1] = res
            if res <= threshold:
                converged, message = True, "converged: ||b - Ax|| <= max(tol ||b||, atol)"
                break
            z = precond(r)  # restart from the true residual
            rz = float(r @ z)
            p = z.copy()
            continue
        z = precond(r)
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    if not converged:
        res = float(np.linalg.norm(b - matvec(x)))
    return _finish("conjugate_gradient", x, res, history, it, converged, message)


# ---------------------------------------------------------------------------
# GMRES
# ---------------------------------------------------------------------------
@register(
    domain="numerics",
    tasks=("linear-systems", "iterative-solvers", "sparse-matrices"),
    complexity=Complexity(
        time="O(T (nnz + r n))",
        space="O(r n)",
        notes="T total iterations, r = restart length; nnz = stored entries of A",
    ),
    references=(
        "Saad & Schultz, SIAM J. Sci. Stat. Comput. 7 (1986) 856-869",
        "Saad, Iterative Methods for Sparse Linear Systems, 2nd ed. (2003), Alg. 6.9-6.11",
    ),
)
def gmres(
    A: Operator,
    b: ArrayLike,
    x0: ArrayLike | None = None,
    restart: int = 20,
    tol: float = 1e-5,
    maxiter: int | None = None,
    atol: float = 0.0,
) -> IterativeResult:
    r"""Solve :math:`Ax = b` for general non-singular :math:`A` by restarted GMRES.

    Parameters
    ----------
    A : ndarray (n, n), CSRMatrix or callable
        Matrix, or ``matvec(v) -> A v``.
    b : array_like of shape (n,)
        Right-hand side.
    x0 : array_like of shape (n,), optional
        Initial guess (zeros by default).
    restart : int, default=20
        Krylov basis size per cycle (memory ``O(restart * n)``).
    tol : float, default=1e-5
        Relative tolerance (same criterion as :func:`conjugate_gradient`).
    maxiter : int, optional
        Maximum total number of Arnoldi steps over all cycles (default ``10 n``).
    atol : float, default=0.0
        Absolute tolerance.

    Returns
    -------
    IterativeResult
        As for :func:`conjugate_gradient`; ``iterations`` counts Arnoldi steps.

    Complexity
    ----------
    Time O(T (nnz + r n)) for T steps with restart r; extra space O(r n).

    References
    ----------
    Saad & Schultz (1986); Saad (2003), Alg. 6.9-6.11.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[3.0, 2.0], [-1.0, 1.0]])
    >>> res = gmres(A, [1.0, 0.0], tol=1e-12)
    >>> res.converged, res.iterations, np.allclose(res.x, [0.2, 0.2])
    (True, 2, True)
    """
    matvec, b, x, threshold, maxiter = _setup(A, b, x0, tol, atol, maxiter)
    restart = check_positive_int(restart, "restart")
    n = b.size

    r = b - matvec(x)
    res = float(np.linalg.norm(r))
    history = [res]
    if res <= threshold:
        return _finish("gmres", x, res, history, 0, True, "initial guess is a solution")

    it, converged = 0, False
    message = f"maximum number of iterations ({maxiter}) reached"
    while it < maxiter:
        m = min(restart, maxiter - it)
        V = np.zeros((m + 1, n))
        H = np.zeros((m + 1, m))
        cs, sn = np.zeros(m), np.zeros(m)
        g = np.zeros(m + 1)
        g[0] = res
        V[0] = r / res
        k = 0  # basis vectors used in this cycle
        singular = False
        for j in range(m):
            w = matvec(V[j])
            for i in range(j + 1):  # modified Gram-Schmidt
                H[i, j] = w @ V[i]
                w -= H[i, j] * V[i]
            h_next = float(np.linalg.norm(w))
            H[j + 1, j] = h_next
            for i in range(j):  # apply earlier rotations to the new column
                hi, hi1 = H[i, j], H[i + 1, j]
                H[i, j] = cs[i] * hi + sn[i] * hi1
                H[i + 1, j] = -sn[i] * hi + cs[i] * hi1
            denom = float(np.hypot(H[j, j], H[j + 1, j]))
            if denom == 0.0:
                singular = True
                break
            cs[j], sn[j] = H[j, j] / denom, H[j + 1, j] / denom
            H[j, j], H[j + 1, j] = denom, 0.0
            g[j + 1] = -sn[j] * g[j]
            g[j] = cs[j] * g[j]
            it += 1
            k = j + 1
            history.append(abs(float(g[j + 1])))
            if abs(g[j + 1]) <= threshold or h_next == 0.0 or it >= maxiter:
                break  # converged estimate, lucky breakdown, or out of iterations
            V[j + 1] = w / h_next
        if k > 0:
            y = np.zeros(k)  # back substitution on the triangular H[:k, :k]
            for i in range(k - 1, -1, -1):
                y[i] = (g[i] - H[i, i + 1 : k] @ y[i + 1 :]) / H[i, i]
            x += V[:k].T @ y
        r = b - matvec(x)
        res = float(np.linalg.norm(r))
        if k > 0:
            history[-1] = res
        if res <= threshold:
            converged, message = True, "converged: ||b - Ax|| <= max(tol ||b||, atol)"
            break
        if singular or res == 0.0:
            message = "breakdown: A is singular on the Krylov subspace"
            break
    return _finish("gmres", x, res, history, it, converged, message)
