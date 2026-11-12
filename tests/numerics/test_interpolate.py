"""lipipy.numerics.interpolate vs numpy.interp and scipy.interpolate.CubicSpline."""

import numpy as np
import pytest

from lipipy.bench import estimate_complexity
from lipipy.numerics import CubicSpline, interp_linear
from lipipy.numerics.interpolate import _thomas

BCS = ["not-a-knot", "natural", "clamped"]


def make_knots(rng, n, ratio):
    h = np.exp(rng.uniform(0.0, np.log(ratio), n - 1))
    return np.concatenate([[0.0], np.cumsum(h)]) + rng.normal()


# ---------------------------------------------------------------------------
# linear
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 3, 10, 200])
def test_interp_linear_matches_numpy(n):
    rng = np.random.default_rng(n)
    x = np.sort(rng.uniform(-5, 5, n))
    y = rng.normal(size=n)
    q = rng.uniform(-7, 7, 500)
    q[:n] = x  # exact knots included
    np.testing.assert_allclose(interp_linear(q, x, y), np.interp(q, x, y), rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(
        interp_linear(q, x, y, left=-9.0, right=9.0),
        np.interp(q, x, y, left=-9.0, right=9.0),
        rtol=1e-15,
        atol=1e-15,
    )


def test_interp_linear_shapes_and_exactness():
    x, y = [0.0, 1.0, 3.0], [1.0, 3.0, -1.0]
    assert interp_linear(2.0, x, y) == 1.0
    assert np.ndim(interp_linear(2.0, x, y)) == 0
    assert interp_linear(np.zeros((2, 3)), x, y).shape == (2, 3)
    q = np.linspace(0, 3, 31)
    np.testing.assert_allclose(interp_linear(q, x, [2 * v + 1 for v in x]), 2 * q + 1, atol=1e-14)


@pytest.mark.parametrize(
    "x, y",
    [
        ([0.0, 2.0, 1.0], [1.0, 2.0, 3.0]),  # unsorted
        ([0.0, 1.0, 1.0], [1.0, 2.0, 3.0]),  # duplicate
        ([0.0, 1.0], [1.0, 2.0, 3.0]),  # length mismatch
        ([], []),  # empty
        ([0.0, np.nan], [1.0, 2.0]),
        ([[0.0, 1.0]], [[1.0, 2.0]]),
    ],
)
def test_interp_linear_rejects_bad_knots(x, y):
    with pytest.raises(ValueError):
        interp_linear([0.5], x, y)


# ---------------------------------------------------------------------------
# cubic spline
# ---------------------------------------------------------------------------
@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("bc", BCS)
@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 10, 50, 300])
@pytest.mark.parametrize("ratio", [1.0, 10.0, 1e3])
def test_cubic_spline_matches_scipy(bc, n, ratio):
    si = pytest.importorskip("scipy.interpolate")
    for seed in range(10):
        rng = np.random.default_rng(1000 * n + seed)
        x = make_knots(rng, n, ratio)
        y = rng.normal(size=n)
        ours = CubicSpline(x, y, bc)
        ref = si.CubicSpline(x, y, bc_type=bc)
        assert ours.c.shape == ref.c.shape == (4, n - 1)
        q = np.concatenate([x, np.linspace(x[0] - 0.5, x[-1] + 0.5, 400)])
        for nu in range(3):
            got, want = ours(q, nu), ref(q, nu)
            assert np.max(np.abs(got - want)) <= 1e-12 * max(1.0, np.max(np.abs(want)))


@pytest.mark.parametrize("bc", BCS)
def test_cubic_spline_interpolates_and_is_c2(bc):
    rng = np.random.default_rng(1)
    x = make_knots(rng, 12, 5.0)
    y = rng.normal(size=12)
    s = CubicSpline(x, y, bc)
    np.testing.assert_allclose(s(x), y, atol=1e-13)
    for nu in range(3):  # continuity at interior knots of p, p', p''
        left, right = s(x[1:-1] - 1e-9, nu), s(x[1:-1] + 1e-9, nu)
        np.testing.assert_allclose(left, right, atol=1e-5)
    if bc == "natural":
        np.testing.assert_allclose(s(x[[0, -1]], 2), 0.0, atol=1e-12)
    if bc == "clamped":
        np.testing.assert_allclose(s(x[[0, -1]], 1), 0.0, atol=1e-12)
    if bc == "not-a-knot" and x.size >= 4:
        third = 6 * s.c[0]  # p''' per interval
        assert third[0] == pytest.approx(third[1], rel=1e-9, abs=1e-9)
        assert third[-1] == pytest.approx(third[-2], rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("n", [4, 5, 9, 40])
def test_not_a_knot_reproduces_any_cubic(n):
    rng = np.random.default_rng(n)
    x = make_knots(rng, n, 20.0)
    a, b, c, d = rng.normal(size=4)
    p = lambda t: ((a * t + b) * t + c) * t + d  # noqa: E731
    s = CubicSpline(x, p(x))
    q = np.linspace(x[0] - 1, x[-1] + 1, 301)
    scale = 1 + np.max(np.abs(p(q)))
    np.testing.assert_allclose(s(q), p(q), atol=1e-11 * scale)
    np.testing.assert_allclose(s(q, 1), (3 * a * q + 2 * b) * q + c, atol=1e-10 * scale)
    np.testing.assert_allclose(s(q, 2), 6 * a * q + 2 * b, atol=1e-9 * scale)


@pytest.mark.parametrize("n", [2, 3, 6, 25])
def test_clamped_reproduces_cubics_with_zero_end_slopes(n):
    # p(x) = A (2u^3 - 3u^2) + B with u = (x - x0)/L has p'(x0) = p'(x_{n-1}) = 0.
    rng = np.random.default_rng(n)
    x = make_knots(rng, n, 10.0)
    L, A, B = x[-1] - x[0], rng.normal(), rng.normal()
    p = lambda t: A * (2 * ((t - x[0]) / L) ** 3 - 3 * ((t - x[0]) / L) ** 2) + B  # noqa: E731
    s = CubicSpline(x, p(x), "clamped")
    q = np.linspace(x[0], x[-1], 101)
    np.testing.assert_allclose(s(q), p(q), atol=1e-12 * (1 + abs(A) + abs(B)))


def test_natural_reproduces_linear_and_small_cases():
    x = np.array([0.0, 1.0, 2.5, 4.0])
    s = CubicSpline(x, 3 * x - 1, "natural")
    q = np.linspace(-1, 5, 50)
    np.testing.assert_allclose(s(q), 3 * q - 1, atol=1e-13)
    two = CubicSpline([0.0, 2.0], [1.0, 5.0])  # straight line
    np.testing.assert_allclose(two(q), 1 + 2 * q, atol=1e-13)
    three = CubicSpline([0.0, 1.0, 3.0], [0.0, 1.0, 9.0])  # parabola x^2
    np.testing.assert_allclose(three(q), q**2, atol=1e-13)
    assert np.ndim(three(1.5)) == 0


@pytest.mark.parametrize("bc", ["not-a-knot", "natural"])
def test_spline_convergence_order(bc):
    # not-a-knot: O(h^4); natural with p''(end) != 0 drops to O(h^2) near the ends.
    f = lambda t: np.exp(t) * np.cos(2 * t)  # noqa: E731
    errs, hs = [], []
    for n in (81, 161, 321, 641):
        x = np.linspace(0, 2, n)
        s = CubicSpline(x, f(x), bc)
        q = np.linspace(0, 2, 2001)
        errs.append(np.max(np.abs(s(q) - f(q))))
        hs.append(x[1] - x[0])
    slopes = np.diff(np.log(errs)) / np.diff(np.log(hs))
    np.testing.assert_allclose(slopes, 4.0 if bc == "not-a-knot" else 2.0, atol=0.15)


def test_thomas_matches_dense_solve():
    rng = np.random.default_rng(0)
    n = 30
    sub, sup = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    diag = 3 + rng.uniform(0, 1, n)
    rhs = rng.normal(size=n)
    A = np.diag(diag) + np.diag(sub[1:], -1) + np.diag(sup[:-1], 1)
    np.testing.assert_allclose(_thomas(sub, diag, sup, rhs), np.linalg.solve(A, rhs), rtol=1e-13)
    with pytest.raises(np.linalg.LinAlgError):
        _thomas(np.zeros(2), np.zeros(2), np.zeros(2), np.ones(2))


@pytest.mark.parametrize(
    "x, y, bc, exc",
    [
        ([0.0, 2.0, 1.0], [1.0, 2.0, 3.0], "natural", ValueError),
        ([0.0, 1.0, 1.0], [1.0, 2.0, 3.0], "not-a-knot", ValueError),
        ([0.0], [1.0], "natural", ValueError),
        ([0.0, 1.0], [1.0], "natural", ValueError),
        ([0.0, 1.0], [1.0, np.inf], "natural", ValueError),
        ([0.0, 1.0], [1.0, 2.0], "periodic", ValueError),
        ([0.0, 1.0], [1j, 2.0], "natural", TypeError),
    ],
)
def test_cubic_spline_rejects_bad_input(x, y, bc, exc):
    with pytest.raises(exc):
        CubicSpline(x, y, bc)


def test_call_errors_and_dict():
    s = CubicSpline([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], "natural")
    for nu in (3, -1, 1.0, True):
        with pytest.raises(ValueError):
            s([0.5], nu)
    with pytest.raises(ValueError):
        s([np.nan])
    d = s.to_dict()
    assert d["bc_type"] == "natural" and len(d["c"]) == 4
    assert "CubicSpline" in repr(s)


def test_inputs_not_mutated():
    x, y = np.array([0.0, 1.0, 2.0, 4.0]), np.array([1.0, -1.0, 2.0, 0.0])
    xk, yk = x.copy(), y.copy()
    CubicSpline(x, y)(x)
    interp_linear(x, x, y)
    np.testing.assert_array_equal(x, xk)
    np.testing.assert_array_equal(y, yk)


def test_spline_construction_memory_is_linear():
    def make(n, rng):
        return np.cumsum(rng.uniform(0.5, 1.0, n)), rng.normal(size=n)

    est = estimate_complexity(CubicSpline, make, sizes=(20_000, 40_000, 80_000), repeats=1)
    assert est.space_exponent == pytest.approx(1.0, abs=0.15)


@pytest.mark.complexity
def test_spline_construction_time_is_linear():
    def make(n, rng):
        return np.cumsum(rng.uniform(0.5, 1.0, n)), rng.normal(size=n)

    # The Thomas sweep is a Python loop: compute-bound, so the exponent is reliable.
    est = estimate_complexity(CubicSpline, make, sizes=(20_000, 40_000, 80_000, 160_000))
    assert est.time_exponent < 1.25


def test_registered():
    import lipipy

    assert lipipy.registry.get("numerics.CubicSpline").complexity.time == "O(n)"
    assert lipipy.registry.get("numerics.interp_linear").domain == "numerics"
