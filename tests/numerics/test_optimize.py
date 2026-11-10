"""lipipy.numerics.optimize: minimisers and root finders vs closed forms and SciPy."""

import warnings

import numpy as np
import pytest

from lipipy.core import ConvergenceWarning
from lipipy.numerics import (
    OptimizeResult,
    brentq,
    line_search_armijo,
    minimize,
    minimize_bfgs,
    minimize_gd,
    minimize_lbfgs,
    minimize_newton,
    newton,
)


def rosen(x):
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def rosen_grad(x):
    g = np.zeros_like(x)
    g[:-1] = -400.0 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2.0 * (1.0 - x[:-1])
    g[1:] += 200.0 * (x[1:] - x[:-1] ** 2)
    return g


def rosen_hess(x):
    d = x.size
    H = np.zeros((d, d))
    i = np.arange(d - 1)
    H[i, i] += 1200.0 * x[:-1] ** 2 - 400.0 * x[1:] + 2.0
    H[i + 1, i + 1] += 200.0
    H[i, i + 1] = H[i + 1, i] = -400.0 * x[:-1]
    return H


def quadratic(rng, d, cond=50.0):
    U, _ = np.linalg.qr(rng.normal(size=(d, d)))
    A = U @ np.diag(np.geomspace(1.0, cond, d)) @ U.T
    b = rng.normal(size=d)
    return A, b


def least_squares(rng, d, cond=20.0):
    """f(x) = |Mx - c|^2 / 2 with f* = 0, so f keeps relative precision near x* = M^-1 c."""
    U, _ = np.linalg.qr(rng.normal(size=(d, d)))
    V, _ = np.linalg.qr(rng.normal(size=(d, d)))
    M = U @ np.diag(np.sqrt(np.geomspace(1.0, cond, d))) @ V.T
    c = rng.normal(size=d)
    return (
        lambda x: 0.5 * float((M @ x - c) @ (M @ x - c)),
        lambda x: M.T @ (M @ x - c),
        lambda x: M.T @ M,
        np.linalg.solve(M, c),
    )


def test_rosenbrock_derivatives_match_scipy(rng):
    so = pytest.importorskip("scipy.optimize")
    x = rng.normal(size=6)
    assert rosen(x) == pytest.approx(so.rosen(x), rel=1e-14)
    np.testing.assert_allclose(rosen_grad(x), so.rosen_der(x), rtol=1e-13)
    np.testing.assert_allclose(rosen_hess(x), so.rosen_hess(x), rtol=1e-13)


STARTS = [(-1.2, 1.0), (2.0, 2.0), (-1.0, -1.0), (0.0, 0.0), (1.5, -0.5)]


@pytest.mark.parametrize("x0", STARTS)
@pytest.mark.parametrize("method", ["bfgs", "lbfgs", "newton"])
def test_rosenbrock_converges_to_one(method, x0):
    hess = rosen_hess if method == "newton" else None
    res = minimize(rosen, x0, rosen_grad, method, hess=hess, tol=1e-8)
    assert res.success, res.message
    np.testing.assert_allclose(res.x, [1.0, 1.0], atol=1e-6)
    assert res.fun < 1e-12
    assert np.max(np.abs(res.grad)) <= 1e-8
    assert res.nfev >= res.nit + 1 and res.ngev == res.nit + 1


@pytest.mark.parametrize("x0", STARTS)
@pytest.mark.parametrize(("method", "ref_method"), [("bfgs", "BFGS"), ("lbfgs", "L-BFGS-B")])
def test_rosenbrock_iterations_comparable_to_scipy(method, ref_method, x0):
    so = pytest.importorskip("scipy.optimize")
    ours = minimize(rosen, x0, rosen_grad, method, tol=1e-8)
    ref = so.minimize(rosen, x0, jac=rosen_grad, method=ref_method)
    np.testing.assert_allclose(ours.x, ref.x, atol=1e-4)
    # Armijo-only line search vs SciPy's Wolfe search: same ballpark, not identical.
    assert ours.nit <= 2.5 * ref.nit + 5


def test_rosenbrock_higher_dimension():
    x0 = np.tile([-1.2, 1.0], 5)
    for method in ("bfgs", "lbfgs"):
        res = minimize(rosen, x0, rosen_grad, method, tol=1e-8)
        assert res.success
        np.testing.assert_allclose(res.x, 1.0, atol=1e-6)


@pytest.mark.parametrize("method", ["gd", "newton", "bfgs", "lbfgs"])
@pytest.mark.parametrize("d", [1, 5, 30])
def test_convex_quadratic_matches_closed_form(rng, method, d):
    f, g, h, x_star = least_squares(rng, d)
    hess = h if method == "newton" else None
    res = minimize(f, np.zeros(d), g, method, hess=hess, tol=1e-10, maxiter=5000)
    assert res.success, res.message
    # Hessian eigenvalues are >= 1, so |x - x*|_2 <= |g|_2 <= sqrt(d) * tol.
    np.testing.assert_allclose(res.x, x_star, rtol=0, atol=np.sqrt(d) * 1e-10)
    if method == "newton":
        assert res.nit == 1  # Newton is exact on a quadratic


def test_lbfgs_quadratic_parity_with_scipy(rng):
    so = pytest.importorskip("scipy.optimize")
    f, g, _, x_star = least_squares(rng, 40, cond=100.0)
    ours = minimize_lbfgs(f, np.zeros(40), g, tol=1e-9, m=10)
    ref = so.minimize(f, np.zeros(40), jac=g, method="L-BFGS-B", options={"gtol": 1e-9})
    assert ours.success
    np.testing.assert_allclose(ours.x, x_star, atol=1e-8)
    # SciPy also stops on a relative f-change test, so it may stop a little earlier.
    np.testing.assert_allclose(ours.x, ref.x, atol=1e-4)
    assert ours.nit <= 2 * ref.nit + 5


def test_newton_falls_back_on_indefinite_hessian():
    # f(x, y) = x^4/4 - x^2/2 + y^2: Hessian is indefinite at the start (x = 0.1).
    f = lambda z: z[0] ** 4 / 4 - z[0] ** 2 / 2 + z[1] ** 2  # noqa: E731
    g = lambda z: np.array([z[0] ** 3 - z[0], 2 * z[1]])  # noqa: E731
    h = lambda z: np.diag([3 * z[0] ** 2 - 1, 2.0])  # noqa: E731
    res = minimize(f, [0.1, 1.0], g, "newton", hess=h, tol=1e-8)
    assert res.success
    np.testing.assert_allclose(np.abs(res.x), [1.0, 0.0], atol=1e-8)
    assert res.fun < f(np.array([0.1, 1.0]))


def test_method_wrappers_equal_minimize(rng):
    A, b = quadratic(rng, 4)
    f = lambda x: 0.5 * x @ A @ x - b @ x  # noqa: E731
    g = lambda x: A @ x - b  # noqa: E731
    x0 = np.ones(4)
    for func, method, extra in [
        (minimize_gd, "gd", {}),
        (minimize_bfgs, "bfgs", {}),
        (minimize_lbfgs, "lbfgs", {"m": 3}),
    ]:
        a = func(f, x0, g, **extra)
        b_ = minimize(f, x0, g, method, **extra)
        np.testing.assert_array_equal(a.x, b_.x)
    a = minimize_newton(f, x0, g, lambda x: A)
    np.testing.assert_array_equal(a.x, minimize(f, x0, g, "newton", hess=lambda x: A).x)


def test_not_converged_warns_and_reports():
    with pytest.warns(ConvergenceWarning, match="maximum number of iterations"):
        res = minimize(rosen, [-1.2, 1.0], rosen_grad, "gd", maxiter=20)
    assert not res.success
    assert res.nit == 20
    assert "maximum" in res.message


def test_tolerance_below_rounding_level_stops_early(rng):
    A, b = quadratic(rng, 5)
    with pytest.warns(ConvergenceWarning, match="rounding"):
        res = minimize(
            lambda x: 1e6 + 0.5 * x @ A @ x - b @ x, np.zeros(5), lambda x: A @ x - b, tol=1e-14
        )
    assert not res.success and res.nit < 100


def test_line_search_failure_reported():
    # Gradient that points the wrong way: no step decreases f.
    # f is undefined (NaN) everywhere except the start, so no step is acceptable.
    with pytest.warns(ConvergenceWarning, match="line search"):
        res = minimize(lambda x: 1.0 if x[0] == 1e-20 else np.nan, [1e-20], lambda x: np.ones(1))
    assert not res.success and res.nit == 0


def test_converged_run_emits_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = minimize(lambda x: float(x @ x), [3.0, 4.0], lambda x: 2 * x)
    assert res.success


def test_result_to_dict_is_json_friendly():
    import json

    res = minimize(lambda x: float(x @ x), [3.0, 4.0], lambda x: 2 * x)
    assert isinstance(res, OptimizeResult)
    d = res.to_dict()
    assert set(d) == {"x", "fun", "grad", "nit", "nfev", "ngev", "success", "message"}
    json.dumps(d)
    with pytest.raises(AttributeError):
        res.nit = 0  # frozen


def test_minimize_input_errors():
    f, g = (lambda x: float(x @ x)), (lambda x: 2 * x)
    with pytest.raises(ValueError, match="method"):
        minimize(f, [1.0], g, "nelder-mead")
    with pytest.raises(ValueError, match="hess"):
        minimize(f, [1.0], g, "newton")
    with pytest.raises(ValueError, match="hess"):
        minimize(f, [1.0], g, "bfgs", hess=lambda x: np.eye(1))
    with pytest.raises(TypeError, match="jac"):
        minimize(f, [1.0], None)
    with pytest.raises(ValueError, match="x0"):
        minimize(f, [[1.0, 2.0]], g)
    with pytest.raises(ValueError, match="NaN"):
        minimize(f, [np.nan], g)
    with pytest.raises(ValueError, match="tol"):
        minimize(f, [1.0], g, tol=0.0)
    with pytest.raises(ValueError, match="maxiter"):
        minimize(f, [1.0], g, maxiter=0)
    with pytest.raises(ValueError, match="m must"):
        minimize(f, [1.0], g, "lbfgs", m=0)
    with pytest.raises(ValueError, match="jac returned shape"):
        minimize(f, [1.0, 2.0], lambda x: np.zeros(3))
    with pytest.raises(ValueError, match="not finite"):
        minimize(lambda x: np.inf, [1.0], g)
    with pytest.raises(ValueError, match="hess returned shape"):
        minimize(f, [1.0, 2.0], g, "newton", hess=lambda x: np.eye(3))


# -- line search -----------------------------------------------------------------
def test_line_search_armijo_condition():
    f = lambda x: float(x @ x)  # noqa: E731
    x = np.array([1.0, -2.0])
    g = 2 * x
    res = line_search_armijo(f, g, x, -g, c1=1e-4, rho=0.5)
    assert res.success
    assert res.fun <= f(x) + 1e-4 * res.alpha * float(g @ -g)
    assert res.alpha == 0.5 and res.fun == 0.0
    # Callable gradient and a precomputed f(x) give the same result.
    res2 = line_search_armijo(f, lambda z: 2 * z, x, -g, fx=f(x))
    assert res2 == res


def test_line_search_errors_and_failure():
    f = lambda x: float(x @ x)  # noqa: E731
    x = np.array([1.0])
    with pytest.raises(ValueError, match="descent"):
        line_search_armijo(f, 2 * x, x, 2 * x)
    with pytest.raises(ValueError, match="c1"):
        line_search_armijo(f, 2 * x, x, -2 * x, c1=1.5)
    with pytest.raises(ValueError, match="shape"):
        line_search_armijo(f, [1.0, 2.0], x, -2 * x)
    res = line_search_armijo(lambda z: np.nan, 2 * x, x, -2 * x, fx=1.0, maxiter=5)
    assert not res.success and res.nfev == 5 and res.alpha == 0.0
    assert res.to_dict()["success"] is False


# -- root finding ----------------------------------------------------------------
ROOT_CASES = [
    (lambda x: x**2 - 2.0, 0.0, 2.0),
    (lambda x: np.cos(x) - x, 0.0, 1.0),
    (lambda x: x**3 - 2 * x - 5, 2.0, 3.0),
    (lambda x: np.exp(x) - 10.0, -5.0, 5.0),
    (np.sin, 3.0, 4.0),
    (lambda x: x**9 - 1e-3, 0.0, 2.0),
    (lambda x: np.arctan(x - 0.3), -1e3, 1e3),
]


@pytest.mark.parametrize(("f", "a", "b"), ROOT_CASES)
def test_brentq_matches_scipy(f, a, b):
    so = pytest.importorskip("scipy.optimize")
    res = brentq(f, a, b)
    ref, info = so.brentq(f, a, b, full_output=True)
    assert res.converged
    assert res.root == pytest.approx(ref, abs=4e-12)
    assert res.nit == info.iterations  # same step-acceptance rules as SciPy
    assert res.nfev == info.function_calls
    assert float(res) == res.root


def test_brentq_endpoints_and_errors():
    assert brentq(lambda x: x, 0.0, 1.0).root == 0.0
    assert brentq(lambda x: x - 1.0, 0.0, 1.0).root == 1.0
    with pytest.raises(ValueError, match="opposite signs"):
        brentq(lambda x: x**2 + 1, -1.0, 1.0)
    with pytest.raises(ValueError, match="distinct"):
        brentq(lambda x: x, 1.0, 1.0)
    with pytest.raises(ValueError, match="xtol"):
        brentq(lambda x: x, -1.0, 1.0, xtol=0.0)
    with pytest.raises(ValueError, match="finite"):
        brentq(lambda x: np.nan, -1.0, 1.0)
    with pytest.warns(ConvergenceWarning):
        res = brentq(lambda x: np.cos(x) - x, 0.0, 1.0, maxiter=2)
    assert not res.converged and res.nit == 2


@pytest.mark.parametrize(("f", "fp", "x0"), [
    (lambda x: x**2 - 2.0, lambda x: 2 * x, 1.0),
    (lambda x: np.cos(x) - x, lambda x: -np.sin(x) - 1, 1.0),
    (lambda x: x**3 - 2 * x - 5, lambda x: 3 * x**2 - 2, 2.0),
    (lambda x: np.exp(x) - 10.0, np.exp, 0.0),
])  # fmt: skip
def test_newton_and_secant_match_scipy(f, fp, x0):
    so = pytest.importorskip("scipy.optimize")
    for deriv in (fp, None):
        res = newton(f, x0, fprime=deriv)
        ref = so.newton(f, x0, fprime=deriv)
        assert res.converged
        assert res.root == pytest.approx(ref, rel=1e-12, abs=1e-12)
        assert abs(res.fun) < 1e-10


def test_newton_failure_modes():
    with pytest.warns(ConvergenceWarning, match="derivative is zero"):
        res = newton(lambda x: x**2 + 1, 0.0, lambda x: 2 * x)
    assert not res.converged
    with pytest.warns(ConvergenceWarning, match="maximum"):
        res = newton(lambda x: x**2 + 1, 0.5, lambda x: 2 * x, maxiter=5)
    assert not res.converged and res.nit == 5
    with pytest.warns(ConvergenceWarning, match="secant slope"):
        res = newton(lambda x: 1.0, 1.0)
    assert not res.converged
    assert newton(lambda x: x - 3.0, 3.0, lambda x: 1.0).root == 3.0
    with pytest.raises(ValueError, match="x0"):
        newton(lambda x: x, np.inf)
    with pytest.raises(TypeError, match="fprime"):
        newton(lambda x: x, 1.0, fprime=1.0)


def test_lbfgs_memory_is_linear_in_d():
    """Declared space O(m d): no d x d matrix is formed."""
    from lipipy.bench import estimate_complexity

    def run(x0):
        minimize_lbfgs(lambda x: float(x @ x), x0, lambda x: 2 * x, m=5, maxiter=5)

    est = estimate_complexity(
        run, lambda d, rng: rng.normal(size=d), sizes=(20_000, 40_000, 80_000), repeats=1
    )
    assert est.space_exponent == pytest.approx(1.0, abs=0.2)
