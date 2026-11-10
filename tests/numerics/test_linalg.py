"""lipipy.numerics.linalg: parity with NumPy / SciPy on well-conditioned matrices."""

import numpy as np
import pytest

from lipipy.bench import estimate_complexity
from lipipy.numerics import (
    cholesky,
    cholesky_solve,
    det,
    lstsq_qr,
    lu,
    lu_solve,
    qr,
    solve_triangular,
)

SIZES = [1, 2, 5, 20, 60]
EPS = np.finfo(np.float64).eps
LinAlgError = np.linalg.LinAlgError


def well_conditioned(rng, d):
    """Random matrix with singular values in [1, 10] (condition number <= 10)."""
    U, _ = np.linalg.qr(rng.normal(size=(d, d)))
    V, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return U @ np.diag(rng.uniform(1.0, 10.0, size=d)) @ V.T


def spd(rng, d):
    U, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return U @ np.diag(rng.uniform(1.0, 10.0, size=d)) @ U.T


def tol(A):
    """Forward-error tolerance for a backward-stable solver: c * d * cond(A) * eps."""
    return 50 * A.shape[0] * np.linalg.cond(A) * EPS


def rhs_shapes(d):
    return [(d,), (d, 3)]


# -- triangular -----------------------------------------------------------------
@pytest.mark.parametrize("d", SIZES)
@pytest.mark.parametrize("lower", [True, False])
@pytest.mark.parametrize("unit", [False, True])
def test_solve_triangular_matches_scipy(rng, d, lower, unit):
    sla = pytest.importorskip("scipy.linalg")
    T = rng.normal(size=(d, d)) + 4 * np.eye(d)
    T = np.tril(T) if lower else np.triu(T)
    for shape in rhs_shapes(d):
        b = rng.normal(size=shape)
        ours = solve_triangular(T, b, lower=lower, unit_diagonal=unit)
        ref = sla.solve_triangular(T, b, lower=lower, unit_diagonal=unit)
        assert ours.shape == b.shape
        np.testing.assert_allclose(ours, ref, rtol=1e-12, atol=1e-12)


def test_solve_triangular_errors():
    with pytest.raises(LinAlgError, match="singular"):
        solve_triangular([[1.0, 0.0], [1.0, 0.0]], [1.0, 1.0])
    with pytest.raises(ValueError, match="square"):
        solve_triangular(np.ones((2, 3)), [1.0, 1.0])
    with pytest.raises(ValueError, match="rows"):
        solve_triangular(np.eye(2), [1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="NaN"):
        solve_triangular(np.eye(2), [np.nan, 1.0])


# -- LU -----------------------------------------------------------------------------
@pytest.mark.parametrize("d", SIZES)
def test_lu_factors_match_scipy(rng, d):
    sla = pytest.importorskip("scipy.linalg")
    A = well_conditioned(rng, d)
    P, L, U = lu(A)
    np.testing.assert_allclose(P @ A, L @ U, atol=1e-13 * np.abs(A).max() * d)
    np.testing.assert_array_equal(L, np.tril(L))
    np.testing.assert_array_equal(np.diag(L), 1.0)
    np.testing.assert_array_equal(U, np.triu(U))
    assert np.all(np.abs(L) <= 1.0)  # partial pivoting bounds the multipliers
    np.testing.assert_array_equal(P @ P.T, np.eye(d))
    p_ref, L_ref, U_ref = sla.lu(A)
    np.testing.assert_array_equal(P, p_ref.T)
    np.testing.assert_allclose(L, L_ref, atol=1e-12)
    np.testing.assert_allclose(U, U_ref, atol=1e-12 * np.abs(A).max())


@pytest.mark.parametrize("d", SIZES)
def test_lu_solve_and_det_match_numpy(rng, d):
    A = well_conditioned(rng, d)
    factors = lu(A)
    for shape in rhs_shapes(d):
        b = rng.normal(size=shape)
        x = lu_solve(factors, b)
        ref = np.linalg.solve(A, b)
        np.testing.assert_allclose(x, ref, rtol=tol(A), atol=tol(A) * np.abs(ref).max())
    assert det(A) == pytest.approx(np.linalg.det(A), rel=tol(A))


def test_lu_singular_matrix():
    A = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 0.0, 1.0]])
    P, L, U = lu(A)
    np.testing.assert_allclose(P @ A, L @ U, atol=1e-14)
    assert det(A) == 0.0
    with pytest.raises(LinAlgError, match="singular"):
        lu_solve((P, L, U), [1.0, 2.0, 3.0])
    assert det(np.zeros((2, 2))) == 0.0


def test_lu_errors():
    with pytest.raises(ValueError, match="square"):
        lu(np.ones((2, 3)))
    with pytest.raises(TypeError, match="lu"):
        lu_solve(np.eye(2), [1.0, 1.0])
    with pytest.raises(ValueError, match="same shape"):
        lu_solve((np.eye(2), np.eye(2), np.eye(3)), [1.0, 1.0])


def test_det_permutation_sign():
    assert det([[0.0, 1.0], [1.0, 0.0]]) == -1.0
    assert det(np.eye(4)[[1, 2, 3, 0]]) == -1.0  # 4-cycle: odd permutation
    assert det([[2.0]]) == 2.0


# -- Cholesky ------------------------------------------------------------------
@pytest.mark.parametrize("d", SIZES)
def test_cholesky_matches_numpy(rng, d):
    A = spd(rng, d)
    L = cholesky(A)
    np.testing.assert_allclose(L, np.linalg.cholesky(A), atol=1e-12 * np.sqrt(np.abs(A).max()))
    np.testing.assert_allclose(L @ L.T, A, atol=1e-13 * d * np.abs(A).max())
    for shape in rhs_shapes(d):
        b = rng.normal(size=shape)
        ref = np.linalg.solve(A, b)
        np.testing.assert_allclose(
            cholesky_solve(L, b), ref, rtol=tol(A), atol=tol(A) * np.abs(ref).max()
        )


def test_cholesky_rejects_non_spd():
    with pytest.raises(LinAlgError, match="positive definite"):
        cholesky([[1.0, 2.0], [2.0, 1.0]])  # indefinite
    with pytest.raises(LinAlgError, match="positive definite"):
        cholesky([[1.0, 1.0], [1.0, 1.0]])  # singular (semi-definite)
    with pytest.raises(LinAlgError, match="symmetric"):
        cholesky([[2.0, 1.0], [0.0, 2.0]])
    with pytest.raises(LinAlgError, match="singular"):
        cholesky_solve(np.zeros((2, 2)), [1.0, 1.0])


# -- QR ------------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(1, 1), (5, 5), (20, 7), (7, 20), (60, 30), (3, 1)])
@pytest.mark.parametrize("mode", ["reduced", "complete"])
def test_qr_matches_numpy(rng, shape, mode):
    A = rng.normal(size=shape)
    n, d = shape
    k = min(n, d)
    Q, R = qr(A, mode=mode)
    Q_ref, R_ref = np.linalg.qr(A, mode=mode)
    assert Q.shape == Q_ref.shape and R.shape == R_ref.shape
    np.testing.assert_allclose(Q @ R, A, atol=1e-13 * max(shape))
    np.testing.assert_allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-13 * max(shape))
    np.testing.assert_array_equal(R, np.triu(R))
    # Same sign convention as LAPACK geqrf: factors agree, not only up to signs.
    np.testing.assert_allclose(R, R_ref, atol=1e-12 * max(shape))
    np.testing.assert_allclose(Q[:, :k], Q_ref[:, :k], atol=1e-12 * max(shape))


def test_qr_sign_convention_documented():
    _, R = qr(np.array([[3.0], [4.0]]))
    assert R[0, 0] == -5.0  # -sign(x_1) * ||x||
    _, R = qr(np.array([[-3.0], [4.0]]))
    assert R[0, 0] == 5.0
    _, R = qr(np.array([[-2.0, 1.0], [0.0, 1.0]]))  # already triangular: no reflection
    np.testing.assert_array_equal(R, [[-2.0, 1.0], [0.0, 1.0]])


def test_qr_errors():
    with pytest.raises(ValueError, match="mode"):
        qr(np.eye(2), mode="economic")
    with pytest.raises(ValueError, match="2D"):
        qr([1.0, 2.0])


@pytest.mark.parametrize("shape", [(5, 5), (40, 6), (200, 30)])
def test_lstsq_qr_matches_numpy(rng, shape):
    A = rng.normal(size=shape)
    for bshape in [(shape[0],), (shape[0], 2)]:
        b = rng.normal(size=bshape)
        ref, *_ = np.linalg.lstsq(A, b, rcond=None)
        np.testing.assert_allclose(lstsq_qr(A, b), ref, rtol=1e-11, atol=1e-11)


def test_lstsq_qr_errors():
    with pytest.raises(LinAlgError, match="rank deficient"):
        lstsq_qr([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="rows"):
        lstsq_qr(np.ones((2, 3)), [1.0, 2.0])


def test_lu_memory_is_quadratic_in_d():
    """Declared space O(d^2): one working copy of A."""
    est = estimate_complexity(
        lu, lambda d, rng: rng.normal(size=(d, d)), sizes=(100, 200, 400), repeats=1
    )
    assert est.space_exponent == pytest.approx(2.0, abs=0.2)


@pytest.mark.complexity
def test_lu_time_is_at_most_cubic():
    est = estimate_complexity(
        lu, lambda d, rng: rng.normal(size=(d, d)), sizes=(100, 200, 400, 800), repeats=2
    )
    assert est.time_exponent < 3.3
