"""lipipy.numerics.iterative: CG and GMRES vs direct solves and SciPy."""

import numpy as np
import pytest

from lipipy.core import ConvergenceWarning
from lipipy.numerics.iterative import conjugate_gradient, gmres
from lipipy.numerics.sparse import CSRMatrix


def poisson1d(n):
    i = np.arange(n)
    rows = np.concatenate([i, i[1:], i[:-1]])
    cols = np.concatenate([i, i[1:] - 1, i[:-1] + 1])
    vals = np.concatenate([np.full(n, 2.0), -np.ones(n - 1), -np.ones(n - 1)])
    return CSRMatrix.from_coo(rows, cols, vals, (n, n))


def poisson2d(k):
    """5-point Laplacian on a k x k grid (n = k^2)."""
    T = poisson1d(k).to_dense()
    eye = np.eye(k)
    return CSRMatrix.from_dense(np.kron(eye, T) + np.kron(T, eye))


def diag_dominant(rng, n, symmetric=False):
    A = rng.normal(size=(n, n))
    if symmetric:
        A = (A + A.T) / 2
    A += np.diag(np.sum(np.abs(A), axis=1) + 1.0)
    return A


def check_criterion(res, A, b, tol, atol=0.0):
    """The documented rule: converged iff ||b - A x|| <= max(tol ||b||, atol), exactly.

    ``A`` must be the operator passed to the solver (CSR and dense products round
    differently), so the recomputed residual is bit-identical.
    """
    true = float(np.linalg.norm(b - A @ res.x))
    assert res.residual_norm == true
    assert res.residual_norms[-1] == true
    assert res.converged == (true <= max(tol * np.linalg.norm(b), atol))


SYSTEMS = {
    "poisson1d_50": lambda rng: poisson1d(50),
    "poisson2d_10x10": lambda rng: poisson2d(10),
    "dd_spd_dense": lambda rng: diag_dominant(rng, 60, symmetric=True),
    "dd_spd_csr": lambda rng: CSRMatrix.from_dense(diag_dominant(rng, 40, symmetric=True)),
}


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("name", sorted(SYSTEMS))
@pytest.mark.parametrize("tol", [1e-6, 1e-10])
@pytest.mark.parametrize("precond", [None, "jacobi"])
def test_cg_matches_direct_solve_and_scipy(name, tol, precond):
    rng = np.random.default_rng(0)
    A = SYSTEMS[name](rng)
    D = A.to_dense() if isinstance(A, CSRMatrix) else A
    b = rng.normal(size=D.shape[0])
    res = conjugate_gradient(A, b, tol=tol, M=precond)
    assert res.converged
    check_criterion(res, A, b, tol)
    x_ref = np.linalg.solve(D, b)
    cond = np.linalg.cond(D)
    assert np.linalg.norm(res.x - x_ref) <= 2 * cond * tol * np.linalg.norm(x_ref)
    sla = pytest.importorskip("scipy.sparse.linalg")
    M = None if precond is None else np.diag(1 / np.diag(D))
    x_sp, info = sla.cg(D, b, rtol=tol, atol=0.0, M=M, maxiter=10 * len(b))
    assert info == 0
    assert np.linalg.norm(res.x - x_sp) <= 4 * cond * tol * np.linalg.norm(x_ref)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("k", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("seed", range(5))
def test_cg_terminates_within_number_of_distinct_eigenvalues(k, seed):
    rng = np.random.default_rng(seed)
    n = 80
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eigs = rng.choice(np.linspace(1.0, 4.0, k), size=n)
    eigs[:k] = np.linspace(1.0, 4.0, k)  # all k values present
    A = Q @ np.diag(eigs) @ Q.T
    A = (A + A.T) / 2
    b = rng.normal(size=n)
    res = conjugate_gradient(A, b, tol=1e-10)
    assert res.converged and res.iterations <= k


@pytest.mark.filterwarnings("error")
def test_cg_on_1d_poisson_converges_within_n_iterations():
    n = 30
    A = poisson1d(n)
    b = np.random.default_rng(1).normal(size=n)
    res = conjugate_gradient(A, b, tol=1e-12)
    assert res.converged and res.iterations <= n
    assert res.residual_norms.shape == (res.iterations + 1,)


@pytest.mark.filterwarnings("ignore::lipipy.core.ConvergenceWarning")
@pytest.mark.parametrize("seed", range(40))
def test_residual_criterion_exact_many_seeds(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 60))
    A = diag_dominant(rng, n, symmetric=True)
    b = rng.normal(size=n)
    tol = float(10.0 ** rng.uniform(-12, -2))
    atol = float(rng.choice([0.0, 1e-3]))
    maxiter = int(rng.integers(1, 2 * n))
    for solve in (
        lambda: conjugate_gradient(A, b, tol=tol, atol=atol, maxiter=maxiter),
        lambda: gmres(A, b, tol=tol, atol=atol, maxiter=maxiter, restart=7),
    ):
        check_criterion(solve(), A, b, tol, atol)


GMRES_SYSTEMS = {
    "dd_nonsym_dense": lambda rng: diag_dominant(rng, 60),
    "dd_nonsym_csr": lambda rng: CSRMatrix.from_dense(diag_dominant(rng, 50)),
    "poisson2d": lambda rng: poisson2d(8),
    "convection": lambda rng: CSRMatrix.from_dense(
        poisson1d(60).to_dense() + 0.4 * (np.eye(60, k=1) - np.eye(60, k=-1))
    ),
}


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("name", sorted(GMRES_SYSTEMS))
@pytest.mark.parametrize("restart", [5, 20, 200])
def test_gmres_matches_direct_solve_and_scipy(name, restart):
    rng = np.random.default_rng(3)
    A = GMRES_SYSTEMS[name](rng)
    D = A.to_dense() if isinstance(A, CSRMatrix) else A
    b = rng.normal(size=D.shape[0])
    tol = 1e-10
    res = gmres(A, b, restart=restart, tol=tol, maxiter=20 * len(b))
    assert res.converged
    check_criterion(res, A, b, tol)
    x_ref = np.linalg.solve(D, b)
    cond = np.linalg.cond(D)
    assert np.linalg.norm(res.x - x_ref) <= 2 * cond * tol * np.linalg.norm(x_ref)
    if restart >= len(b):
        assert res.iterations <= len(b)  # full GMRES: at most n steps
    sla = pytest.importorskip("scipy.sparse.linalg")
    x_sp, info = sla.gmres(D, b, rtol=tol, atol=0.0, restart=restart, maxiter=20 * len(b))
    assert info == 0
    assert np.linalg.norm(res.x - x_sp) <= 4 * cond * tol * np.linalg.norm(x_ref)


@pytest.mark.filterwarnings("error")
def test_callable_operator_and_x0():
    rng = np.random.default_rng(5)
    A = diag_dominant(rng, 20, symmetric=True)
    b = rng.normal(size=20)
    x_star = np.linalg.solve(A, b)
    for solver in (conjugate_gradient, gmres):
        res = solver(lambda v: A @ v, b, tol=1e-12)
        np.testing.assert_allclose(res.x, x_star, rtol=1e-9, atol=1e-9)
        warm = solver(A, b, x0=x_star, tol=1e-12)
        assert warm.iterations <= 1
    exact = conjugate_gradient(A, A @ np.ones(20), x0=np.ones(20), tol=1e-3)
    assert exact.iterations == 0 and exact.converged
    zero = gmres(A, np.zeros(20), atol=1e-14)
    assert zero.converged and zero.iterations == 0 and np.all(zero.x == 0)


def test_cg_user_preconditioner_reduces_iterations():
    rng = np.random.default_rng(7)
    d = np.geomspace(1.0, 1e4, 100)
    A = np.diag(d) + 0.1 * np.eye(100, k=1) + 0.1 * np.eye(100, k=-1)
    b = rng.normal(size=100)
    plain = conjugate_gradient(A, b, tol=1e-10, maxiter=5000)
    pre = conjugate_gradient(A, b, tol=1e-10, M=lambda r: r / d)
    jac = conjugate_gradient(A, b, tol=1e-10, M="jacobi")
    assert pre.converged and jac.converged
    assert pre.iterations < plain.iterations / 3
    np.testing.assert_allclose(pre.x, jac.x, rtol=1e-6)


def test_not_converged_warns():
    A = poisson1d(100)
    b = np.ones(100)
    with pytest.warns(ConvergenceWarning, match="maximum"):
        res = conjugate_gradient(A, b, tol=1e-12, maxiter=5)
    assert not res.converged and res.iterations == 5
    with pytest.warns(ConvergenceWarning, match="maximum"):
        res = gmres(A, b, tol=1e-12, restart=3, maxiter=7)
    assert not res.converged and res.iterations == 7


def test_cg_rejects_indefinite_matrix():
    A = np.diag([1.0, -1.0, 2.0])
    with pytest.warns(ConvergenceWarning, match="positive definite"):
        res = conjugate_gradient(A, [1.0, 1.0, 1.0])
    assert not res.converged


def test_gmres_singular_breakdown():
    A = np.array([[1.0, 0.0], [0.0, 0.0]])
    with pytest.warns(ConvergenceWarning):
        res = gmres(A, [0.0, 1.0], tol=1e-10)
    assert not res.converged


def test_input_validation():
    A = np.eye(3)
    with pytest.raises(ValueError, match="shape"):
        conjugate_gradient(np.eye(4), np.ones(3))
    with pytest.raises(ValueError, match="shape"):
        gmres(CSRMatrix.from_dense(np.eye(4)), np.ones(3))
    with pytest.raises(ValueError, match="x0"):
        conjugate_gradient(A, np.ones(3), x0=np.ones(2))
    with pytest.raises(ValueError, match="tol"):
        gmres(A, np.ones(3), tol=-1.0)
    with pytest.raises(ValueError, match="both"):
        gmres(A, np.ones(3), tol=0.0)
    with pytest.raises(ValueError, match="maxiter"):
        conjugate_gradient(A, np.ones(3), maxiter=0)
    with pytest.raises(ValueError, match="restart"):
        gmres(A, np.ones(3), restart=0)
    with pytest.raises(ValueError, match="jacobi"):
        conjugate_gradient(lambda v: v, np.ones(3), M="jacobi")
    with pytest.raises(ValueError, match="positive diagonal"):
        conjugate_gradient(-np.eye(3), np.ones(3), M="jacobi")
    with pytest.raises(ValueError, match="M must"):
        conjugate_gradient(A, np.ones(3), M="ilu")
    with pytest.raises(ValueError, match="returned shape"):
        conjugate_gradient(lambda v: np.ones(2), np.ones(3))
    with pytest.raises(ValueError, match="NaN"):
        gmres(A, [1.0, np.nan, 0.0])
    res = conjugate_gradient(A, np.ones(3))
    d = res.to_dict()
    assert d["converged"] is True and d["x"] == [1.0, 1.0, 1.0]
    with pytest.raises(ValueError):
        res.residual_norms[0] = 0.0
