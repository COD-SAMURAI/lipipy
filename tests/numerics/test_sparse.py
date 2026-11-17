"""lipipy.numerics.sparse: CSRMatrix against scipy.sparse and dense NumPy."""

import numpy as np
import pytest

from lipipy.bench import estimate_complexity
from lipipy.numerics.sparse import CSRMatrix


def random_coo(rng, m, n, nnz, duplicates=True):
    rows = rng.integers(0, m, size=nnz)
    cols = rng.integers(0, n, size=nnz)
    if duplicates and nnz > 4:
        rows[:4], cols[:4] = (
            rows[4:8] if nnz >= 8 else rows[:4],
            cols[4:8] if nnz >= 8 else cols[:4],
        )
    vals = rng.normal(size=nnz)
    return rows, cols, vals


SHAPES = [(1, 1), (5, 7), (40, 30), (30, 40), (100, 100)]


@pytest.mark.parametrize(("m", "n"), SHAPES)
@pytest.mark.parametrize("seed", range(5))
def test_from_coo_matches_scipy_including_duplicates(m, n, seed):
    sps = pytest.importorskip("scipy.sparse")
    rng = np.random.default_rng(seed)
    nnz = int(rng.integers(0, 3 * max(m, n)))
    rows, cols, vals = random_coo(rng, m, n, nnz)
    A = CSRMatrix.from_coo(rows, cols, vals, (m, n))
    ref = sps.csr_matrix((vals, (rows, cols)), shape=(m, n))
    ref.sum_duplicates()
    ref.sort_indices()
    np.testing.assert_array_equal(A.indptr, ref.indptr)
    np.testing.assert_array_equal(A.indices, ref.indices)
    np.testing.assert_allclose(A.data, ref.data, rtol=1e-15, atol=1e-15)
    assert A.nnz == ref.nnz
    np.testing.assert_allclose(A.to_dense(), ref.toarray(), rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(A.diagonal(), ref.diagonal(), rtol=0, atol=1e-15)
    x = rng.normal(size=n)
    X = rng.normal(size=(n, 3))
    np.testing.assert_allclose(A @ x, ref @ x, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(A @ X, ref @ X, rtol=1e-13, atol=1e-13)
    y = rng.normal(size=m)
    np.testing.assert_allclose(y @ A, ref.T @ y, rtol=1e-13, atol=1e-13)
    T = A.T
    assert T.shape == (n, m)
    np.testing.assert_array_equal(T.to_dense(), A.to_dense().T)
    np.testing.assert_array_equal(T.indptr, ref.T.tocsr().indptr)


def test_empty_rows_and_empty_matrix():
    A = CSRMatrix.from_coo([0, 3, 3], [1, 0, 2], [1.0, 2.0, 3.0], (5, 3))
    np.testing.assert_array_equal(A.indptr, [0, 1, 1, 1, 3, 3])
    np.testing.assert_array_equal(A @ np.ones(3), [1.0, 0.0, 0.0, 5.0, 0.0])
    Z = CSRMatrix.from_coo([], [], [], (3, 4))
    assert Z.nnz == 0
    np.testing.assert_array_equal(Z @ np.ones(4), np.zeros(3))
    np.testing.assert_array_equal(Z @ np.ones((4, 2)), np.zeros((3, 2)))
    np.testing.assert_array_equal(Z.T.to_dense(), np.zeros((4, 3)))
    E = CSRMatrix.from_dense(np.zeros((0, 3)))
    assert E.shape == (0, 3) and (E @ np.ones(3)).shape == (0,)


def test_from_dense_roundtrip_and_explicit_zero_from_duplicates(rng):
    D = rng.normal(size=(6, 5)) * (rng.random((6, 5)) < 0.3)
    A = CSRMatrix.from_dense(D)
    assert A.nnz == np.count_nonzero(D)
    np.testing.assert_array_equal(A.to_dense(), D)
    # +1 and -1 at the same position sum to a stored zero (as in SciPy).
    B = CSRMatrix.from_coo([0, 0], [1, 1], [1.0, -1.0], (2, 2))
    assert B.nnz == 1 and B.data[0] == 0.0


def test_constructor_canonicalizes_and_validates():
    A = CSRMatrix([2.0, 1.0, 5.0], [2, 0, 2], [0, 3, 3], (2, 3))  # unsorted + duplicate
    np.testing.assert_array_equal(A.indices, [0, 2])
    np.testing.assert_array_equal(A.data, [1.0, 7.0])
    np.testing.assert_array_equal(A.indptr, [0, 2, 2])
    with pytest.raises(ValueError, match="indptr"):
        CSRMatrix([1.0], [0], [0, 2], (1, 1))
    with pytest.raises(ValueError, match="non-decreasing"):
        CSRMatrix([1.0, 1.0], [0, 0], [0, 2, 1, 2], (3, 1))
    with pytest.raises(ValueError, match="column"):
        CSRMatrix([1.0], [3], [0, 1], (1, 2))
    with pytest.raises(ValueError, match="NaN"):
        CSRMatrix([np.nan], [0], [0, 1], (1, 1))
    with pytest.raises(TypeError, match="shape"):
        CSRMatrix([1.0], [0], [0, 1], [1, 1])
    with pytest.raises(ValueError, match="out of range"):
        CSRMatrix.from_coo([2], [0], [1.0], (2, 2))
    with pytest.raises(ValueError, match="differ"):
        CSRMatrix.from_coo([0, 1], [0], [1.0], (2, 2))
    with pytest.raises(TypeError, match="integers"):
        CSRMatrix.from_coo([0.5], [0], [1.0], (2, 2))


def test_products_validate_shapes():
    A = CSRMatrix.from_dense(np.eye(3))
    with pytest.raises(ValueError, match="mismatch"):
        A @ np.ones(4)
    with pytest.raises(ValueError, match="mismatch"):
        np.ones(4) @ A
    with pytest.raises(TypeError, match="sparse @ sparse"):
        A @ A
    with pytest.raises(ValueError, match="1-D or 2-D"):
        A @ np.ones((3, 1, 1))
    with pytest.raises(ValueError, match="NaN"):
        A @ np.array([np.nan, 0.0, 0.0])


def test_scalar_ops_and_immutability(rng):
    D = rng.normal(size=(4, 4)) * (rng.random((4, 4)) < 0.5)
    A = CSRMatrix.from_dense(D)
    np.testing.assert_array_equal((2.0 * A).to_dense(), 2.0 * D)
    np.testing.assert_array_equal((A * 3).to_dense(), 3.0 * D)
    np.testing.assert_allclose((A / 4.0).to_dense(), D / 4.0, rtol=1e-15)
    np.testing.assert_array_equal((-A).to_dense(), -D)
    np.testing.assert_array_equal(A.to_dense(), D)  # unchanged
    with pytest.raises(ZeroDivisionError):
        A / 0
    with pytest.raises(ValueError, match="finite"):
        A * np.inf
    with pytest.raises(TypeError):
        A * "x"
    for arr in (A.data, A.indices, A.indptr):
        with pytest.raises(ValueError):
            arr[0] = 0
    src = np.array([1.0, 2.0])
    B = CSRMatrix.from_coo([0, 1], [0, 1], src, (2, 2))
    src[0] = 99.0
    assert B.data[0] == 1.0
    assert B.to_dict()["shape"] == [2, 2] and "nnz=2" in repr(B)
    np.testing.assert_array_equal(B.matvec([1.0, 1.0]), [1.0, 2.0])


def test_matvec_memory_linear_in_nnz():
    """Extra memory of A @ x is O(nnz + m): products and row sums."""
    sizes = (100_000, 200_000, 400_000, 800_000)

    def make(nnz, rng):
        m = nnz // 10
        A = CSRMatrix.from_coo(rng.integers(0, m, nnz), rng.integers(0, m, nnz),
                               rng.normal(size=nnz), (m, m))  # fmt: skip
        return A, rng.normal(size=m)

    est = estimate_complexity(lambda A, x: A @ x, make, sizes=sizes, repeats=1)
    assert est.space_exponent == pytest.approx(1.0, abs=0.15)


@pytest.mark.complexity
def test_matvec_time_linear_in_nnz():
    """Fixed number of rows, growing nnz: time O(nnz) (compute-bound, repeated runs)."""
    m = 2_000

    def make(nnz, rng):
        A = CSRMatrix.from_coo(rng.integers(0, m, nnz), rng.integers(0, m, nnz),
                               rng.normal(size=nnz), (m, m))  # fmt: skip
        return A, rng.normal(size=m)

    est = estimate_complexity(
        lambda A, x: [A @ x for _ in range(5)], make, sizes=(200_000, 400_000, 800_000, 1_600_000)
    )
    assert est.time_exponent == pytest.approx(1.0, abs=0.3)
