"""lipipy.numerics.integrate vs scipy.integrate and exact integrals."""

import math
import warnings

import numpy as np
import pytest

from lipipy.bench import estimate_complexity
from lipipy.core import ConvergenceWarning
from lipipy.numerics import (
    MonteCarloResult,
    QuadResult,
    gauss_legendre,
    gauss_legendre_nodes,
    monte_carlo,
    quad,
    simpson,
    trapezoid,
)


# ---------------------------------------------------------------------------
# trapezoid / simpson
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 6, 10, 11, 101])
@pytest.mark.parametrize("uniform", [True, False])
def test_trapezoid_and_simpson_match_scipy(n, uniform):
    si = pytest.importorskip("scipy.integrate")
    rng = np.random.default_rng(n)
    for _ in range(5):
        y = rng.normal(size=n)
        if uniform:
            dx = float(rng.uniform(0.1, 2.0))
            assert trapezoid(y, dx=dx) == pytest.approx(
                si.trapezoid(y, dx=dx), rel=1e-13, abs=1e-14
            )
            if n >= 1:
                ref = si.simpson(y, dx=dx)
                assert simpson(y, dx=dx) == pytest.approx(ref, rel=1e-12, abs=1e-13)
        else:
            x = np.cumsum(rng.uniform(0.05, 1.0, size=n))
            assert trapezoid(y, x) == pytest.approx(si.trapezoid(y, x=x), rel=1e-13, abs=1e-14)
            if n >= 1:
                ref = si.simpson(y, x=x)
                assert simpson(y, x) == pytest.approx(ref, rel=1e-12, abs=1e-13)


def test_simpson_decreasing_x_matches_scipy():
    si = pytest.importorskip("scipy.integrate")
    x = np.linspace(3.0, 0.0, 8)
    y = np.exp(x)
    assert simpson(y, x) == pytest.approx(si.simpson(y, x=x), rel=1e-13)
    assert simpson(y, x) < 0


@pytest.mark.parametrize("n", [3, 4, 5, 8, 9, 20, 21])
def test_simpson_exactness(n):
    # Uniform grid with an odd number of samples: exact for cubics.
    x = np.linspace(-1.0, 2.0, n)
    p = lambda t: 2 * t**3 - t**2 + 3 * t - 5  # noqa: E731
    P = lambda t: t**4 / 2 - t**3 / 3 + 1.5 * t**2 - 5 * t  # noqa: E731
    if n % 2 == 1:
        assert simpson(p(x), x) == pytest.approx(P(2.0) - P(-1.0), rel=1e-13)
    # Any grid and any sample count (incl. the even-N end correction): exact for quadratics.
    q = lambda t: 3 * t**2 - t + 1  # noqa: E731
    Q = lambda t: t**3 - t**2 / 2 + t  # noqa: E731
    xr = np.sort(np.random.default_rng(n).uniform(-1, 2, size=n))
    for grid in (x, xr):
        assert simpson(q(grid), grid) == pytest.approx(Q(grid[-1]) - Q(grid[0]), rel=1e-12)


def test_trapezoid_exact_for_linear_and_converges_quadratically():
    x = np.sort(np.random.default_rng(1).uniform(0, 1, 30))
    assert trapezoid(3 * x + 1, x) == pytest.approx(1.5 * (x[-1] ** 2 - x[0] ** 2) + x[-1] - x[0])
    errs = []
    for n in (11, 21, 41, 81):
        t = np.linspace(0, 1, n)
        errs.append(abs(trapezoid(np.exp(t), t) - (math.e - 1)))
    slopes = np.diff(np.log(errs)) / np.diff(np.log([10, 20, 40, 80]))
    np.testing.assert_allclose(slopes, -2.0, atol=0.05)


@pytest.mark.parametrize("parity", [0, 1])
def test_simpson_converges_at_fourth_order_for_odd_and_even_n(parity):
    errs, hs = [], []
    for m in (20, 40, 80, 160):
        n = m + parity  # parity 1: odd sample count; parity 0: even (end correction)
        t = np.linspace(0, 2, n)
        errs.append(abs(simpson(np.exp(t) * np.cos(3 * t), t) - _EXP_COS))
        hs.append(t[1] - t[0])
    slopes = np.diff(np.log(errs)) / np.diff(np.log(hs))
    np.testing.assert_allclose(slopes, 4.0, atol=0.1)


_EXP_COS = (math.exp(2) * (math.cos(6) + 3 * math.sin(6)) - 1) / 10  # int_0^2 e^t cos 3t dt


def test_sampled_rules_small_and_bad_inputs():
    assert trapezoid([]) == 0.0 and trapezoid([5.0]) == 0.0
    assert simpson([]) == 0.0 and simpson([5.0]) == 0.0
    assert simpson([1.0, 3.0], dx=2.0) == 4.0
    with pytest.raises(ValueError):
        trapezoid([1.0, 2.0], x=[0.0])
    with pytest.raises(ValueError):
        trapezoid([[1.0, 2.0]])
    with pytest.raises(ValueError):
        trapezoid([1.0, np.nan])
    with pytest.raises(TypeError):
        trapezoid([1j, 2.0])
    with pytest.raises(ValueError):
        simpson([1.0, 2.0, 3.0], x=[0.0, 1.0, 1.0])  # duplicate
    with pytest.raises(ValueError):
        simpson([1.0, 2.0, 3.0], x=[0.0, 2.0, 1.0])  # unsorted
    with pytest.raises(ValueError):
        simpson([1.0, 2.0, 3.0], dx=0.0)
    with pytest.raises(ValueError):
        simpson([1.0, 2.0], dx=np.inf)


# ---------------------------------------------------------------------------
# Gauss-Legendre
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", list(range(1, 31)) + [50, 64, 100, 150, 200])
def test_gauss_legendre_nodes_match_numpy(n):
    x, w = gauss_legendre_nodes(n)
    X, W = np.polynomial.legendre.leggauss(n)
    np.testing.assert_allclose(x, X, rtol=0, atol=1e-15)
    np.testing.assert_allclose(w, W, rtol=3e-15 * n * n, atol=0)
    assert np.all(np.diff(x) > 0) and np.all(w > 0)
    assert math.fsum(w) == pytest.approx(2.0, rel=1e-14)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 12, 20])
def test_gauss_legendre_exact_up_to_degree_2n_minus_1(n):
    rng = np.random.default_rng(n)
    for _ in range(5):
        a, b = sorted(rng.uniform(-3, 3, size=2))
        coef = rng.normal(size=2 * n)  # degree 2n - 1
        poly = np.polynomial.Polynomial(coef)
        exact = poly.integ()(b) - poly.integ()(a)
        scale = np.polynomial.Polynomial(np.abs(coef)).integ()(max(abs(a), abs(b))) * 2 + 1
        assert abs(gauss_legendre(poly, a, b, n) - exact) <= 1e-13 * scale
    # Degree 2n is no longer exact: the error matches the Gauss remainder formula
    # E = 2^(2n+1) (n!)^4 / ((2n+1) ((2n)!)^2) for f = x^(2n).
    t2n = lambda x: x ** (2 * n)  # noqa: E731
    remainder = 2 ** (2 * n + 1) * math.factorial(n) ** 4
    remainder /= (2 * n + 1) * math.factorial(2 * n) ** 2
    err = 2.0 / (2 * n + 1) - gauss_legendre(t2n, -1.0, 1.0, n)
    assert err == pytest.approx(remainder, rel=1e-6 if remainder > 1e-9 else 1e-2)


def test_gauss_legendre_converges_on_smooth_functions():
    assert gauss_legendre(np.exp, 0.0, 1.0, 10) == pytest.approx(math.e - 1, rel=1e-15)
    assert gauss_legendre(np.sin, 0.0, math.pi, 12) == pytest.approx(2.0, rel=1e-14)
    assert gauss_legendre(np.exp, 1.0, 0.0, 10) == pytest.approx(1 - math.e, rel=1e-15)


def test_gauss_legendre_errors():
    with pytest.raises(ValueError):
        gauss_legendre_nodes(0)
    with pytest.raises(TypeError):
        gauss_legendre_nodes(2.0)
    with pytest.raises(TypeError):
        gauss_legendre(3.0, 0, 1)
    with pytest.raises(ValueError):
        gauss_legendre(lambda x: 1.0, 0, 1, 3)  # not vectorised
    with pytest.raises(ValueError):
        gauss_legendre(lambda x: np.where(x == 0, np.inf, x), -1, 1, 3)  # f(0) = inf
    with pytest.raises(ValueError):
        gauss_legendre(np.exp, 0, np.inf)


# ---------------------------------------------------------------------------
# quad
# ---------------------------------------------------------------------------
def test_kronrod_rule_is_exact_to_degree_22_and_gauss_part_to_13():
    from lipipy.numerics.integrate import _G7_INDEX, _G7_WEIGHTS, _K15_NODES, _K15_WEIGHTS

    for deg in range(23):
        exact = 0.0 if deg % 2 else 2.0 / (deg + 1)
        assert _K15_WEIGHTS @ _K15_NODES**deg == pytest.approx(exact, abs=1e-15)
        if deg <= 13:
            assert _G7_WEIGHTS @ _K15_NODES[_G7_INDEX] ** deg == pytest.approx(exact, abs=1e-15)
    x7, w7 = gauss_legendre_nodes(7)
    np.testing.assert_allclose(_K15_NODES[_G7_INDEX], x7, atol=1e-15)
    np.testing.assert_allclose(_G7_WEIGHTS, w7, atol=1e-15)


EPS_PEAK = 1e-4
QUAD_CASES = {
    "exp": (math.exp, 0.0, 1.0, math.e - 1, 50),
    "sin": (math.sin, 0.0, 10.0, 1 - math.cos(10.0), 50),
    "peak": (
        lambda x: 1.0 / (EPS_PEAK + (x - 0.3) ** 2),
        0.0,
        1.0,
        (math.atan(0.7 / 1e-2) + math.atan(0.3 / 1e-2)) / 1e-2,
        50,
    ),
    "inv_sqrt": (lambda x: x**-0.5, 0.0, 1.0, 2.0, 200),
    "log": (math.log, 0.0, 1.0, -1.0, 50),
    "x^-0.9": (lambda x: x**-0.9, 0.0, 1.0, 10.0, 400),
}


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("case", list(QUAD_CASES))
@pytest.mark.parametrize("tol", [1e-4, 1e-6, 1e-8, 1e-10, 1e-12])
def test_quad_abserr_bounds_true_error(case, tol):
    f, a, b, exact, limit = QUAD_CASES[case]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)  # x^-0.9 cannot reach 1e-8
        res = quad(f, a, b, epsabs=tol, epsrel=tol, limit=limit)
    true_err = abs(res.value - exact)
    assert true_err <= res.abserr, (true_err, res)
    if res.converged:
        assert true_err <= max(tol, tol * abs(exact))
        assert res.abserr <= max(tol, tol * abs(res.value))
    else:
        assert case == "x^-0.9" and tol <= 1e-8
    assert res.neval == 15 * (2 * res.nintervals - 1)


def test_quad_matches_scipy():
    si = pytest.importorskip("scipy.integrate")
    funcs = [
        (lambda x: math.exp(-x * x), -2.0, 3.0),
        (lambda x: math.cos(5 * x) * x**2, 0.0, 4.0),
        (lambda x: 1 / (1 + x * x), -10.0, 10.0),
        (lambda x: abs(x - 0.4), 0.0, 1.0),
        (lambda x: math.log(x), 0.0, 2.0),
    ]
    for f, a, b in funcs:
        ours = quad(f, a, b, epsabs=1e-12, epsrel=1e-12, limit=200)
        ref, _ = si.quad(f, a, b, epsabs=1e-12, epsrel=1e-12, limit=200)
        assert ours.converged
        assert ours.value == pytest.approx(ref, rel=1e-11, abs=1e-12)


def test_quad_limit_warns_and_reports():
    with pytest.warns(ConvergenceWarning, match="subdivision limit"):
        res = quad(lambda x: x**-0.9, 0.0, 1.0, epsabs=1e-10, epsrel=0.0, limit=10)
    assert not res.converged and res.nintervals == 10
    assert abs(res.value - 10.0) <= res.abserr


def test_quad_orientation_and_empty():
    r1 = quad(math.exp, 0.0, 1.0)
    r2 = quad(math.exp, 1.0, 0.0)
    assert r2.value == -r1.value and r2.abserr == r1.abserr
    r0 = quad(math.exp, 2.0, 2.0)
    assert r0.value == 0.0 and r0.converged and r0.neval == 0
    assert float(r1) == r1.value
    d = r1.to_dict()
    assert set(d) == {"value", "abserr", "neval", "nintervals", "converged", "message"}


def test_quad_errors():
    with pytest.raises(ValueError):
        quad(math.exp, 0.0, np.inf)
    with pytest.raises(ValueError):
        quad(math.exp, 0.0, 1.0, epsabs=0.0, epsrel=1e-20)
    with pytest.raises(ValueError):
        quad(math.exp, 0.0, 1.0, epsabs=-1.0)
    with pytest.raises(ValueError):
        quad(math.exp, 0.0, 1.0, limit=0)
    with pytest.raises(TypeError):
        quad("exp", 0.0, 1.0)
    with pytest.raises(ValueError):
        quad(lambda x: 1 / (x - 0.5) if x != 0.5 else math.inf, 0.0, 1.0)  # f(0.5) = inf
    with pytest.raises(TypeError):
        quad(lambda x: "a", 0.0, 1.0)
    with pytest.raises(ValueError):
        QuadResult(1.0, -1.0, 1, 1, True, "")


def test_quad_time_and_memory_scale_with_subintervals():
    # The number of subintervals T is the size variable: force T = limit on x^-0.9.
    def run(limit):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            return quad(lambda x: x**-0.9, 0.0, 1.0, epsabs=1e-14, epsrel=0.0, limit=limit)

    est = estimate_complexity(run, lambda n, rng: n, sizes=(200, 400, 800, 1600), repeats=1)
    assert est.space_exponent == pytest.approx(1.0, abs=0.2)


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
def test_monte_carlo_unbiased_with_honest_stderr():
    # integral of sum(x^2) over [0, 1]^3 = 1, Var f = 3 * (1/5 - 1/9) = 4/15.
    f = lambda X: np.sum(X**2, axis=1)  # noqa: E731
    n = 4000
    z = []
    for seed in range(200):
        res = monte_carlo(f, [(0, 1)] * 3, n=n, random_state=seed)
        assert res.stderr == pytest.approx(math.sqrt(4 / 15 / n), rel=0.1)
        z.append((res.value - 1.0) / res.stderr)
    z = np.array(z)
    assert abs(z.mean()) < 4 / math.sqrt(200)  # unbiased
    assert 0.85 < z.std() < 1.15  # stderr calibrated
    assert np.mean(np.abs(z) < 1.96) > 0.9


def test_monte_carlo_reproducible_and_batch_independent():
    f = lambda X: np.exp(-X[:, 0]) * np.cos(X[:, 1])  # noqa: E731
    a = monte_carlo(f, [(-1, 2), (0, 3)], n=10_001, random_state=7)
    b = monte_carlo(f, [(-1, 2), (0, 3)], n=10_001, random_state=7, batch_size=999)
    assert a.value == pytest.approx(b.value, rel=1e-13)
    assert a.stderr == pytest.approx(b.stderr, rel=1e-12)
    assert monte_carlo(f, [(-1, 2), (0, 3)], n=10_001, random_state=7) == a
    exact = (math.exp(1) - math.exp(-2)) * math.sin(3)
    assert abs(a.value - exact) < 5 * a.stderr
    assert isinstance(a, MonteCarloResult) and a.to_dict()["n"] == 10_001


def test_monte_carlo_one_dim_and_constant():
    res = monte_carlo(lambda X: np.full(X.shape[0], 2.0), (1.0, 4.0), n=10, random_state=0)
    assert res.value == pytest.approx(6.0) and res.stderr == 0.0


def test_monte_carlo_errors():
    f = lambda X: X[:, 0]  # noqa: E731
    with pytest.raises(ValueError):
        monte_carlo(f, [(1, 0)])
    with pytest.raises(ValueError):
        monte_carlo(f, [(0, np.inf)])
    with pytest.raises(ValueError):
        monte_carlo(f, [(0, 1, 2)])
    with pytest.raises(ValueError):
        monte_carlo(f, [(0, 1)], n=1)
    with pytest.raises(ValueError):
        monte_carlo(lambda X: X, [(0, 1), (0, 1)], n=10)  # wrong output shape
    with pytest.raises(ValueError):
        monte_carlo(lambda X: np.full(X.shape[0], np.nan), [(0, 1)], n=10)
    with pytest.raises(TypeError):
        monte_carlo(f, [(0, 1)], random_state="seed")


def test_monte_carlo_memory_independent_of_n():
    f = lambda X: X[:, 0] * X[:, 1]  # noqa: E731

    def run(n):
        return monte_carlo(f, [(0, 1), (0, 1)], n=n, random_state=0, batch_size=4096)

    est = estimate_complexity(run, lambda n, rng: n, sizes=(50_000, 100_000, 200_000), repeats=1)
    assert est.space_exponent < 0.2


def test_registered():
    import lipipy

    for name in ("trapezoid", "simpson", "gauss_legendre", "gauss_legendre_nodes", "quad"):
        assert lipipy.registry.get(f"numerics.{name}").domain == "numerics"
    assert lipipy.registry.get("numerics.monte_carlo").complexity.space == "O(B d)"
