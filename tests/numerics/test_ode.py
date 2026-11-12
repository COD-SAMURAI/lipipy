"""lipipy.numerics.ode vs exact solutions and scipy.integrate.solve_ivp."""

import numpy as np
import pytest

from lipipy.numerics import ODEResult, solve_ode
from lipipy.numerics.ode import _A, _B, _B_HAT, _C, _P


def osc(t, y):
    return np.array([y[1], -y[0]])


PROBLEMS = {
    "oscillator": (osc, [1.0, 0.0], 10.0, lambda t: np.array([np.cos(t), -np.sin(t)])),
    "decay": (lambda t, y: -y, [1.0], 5.0, lambda t: np.exp(-t)[None]),
    "growth": (lambda t, y: y, [1.0], 3.0, lambda t: np.exp(t)[None]),
    "gaussian": (lambda t, y: -2 * t * y, [1.0], 3.0, lambda t: np.exp(-t * t)[None]),
    "logistic": (lambda t, y: y * (1 - y), [0.1], 10.0, lambda t: (1 / (1 + 9 * np.exp(-t)))[None]),
}


# ---------------------------------------------------------------------------
# tableau sanity
# ---------------------------------------------------------------------------
def test_dormand_prince_tableau_order_conditions():
    for s in range(1, 6):
        assert _A[s].sum() == pytest.approx(_C[s], abs=1e-15)  # row-sum condition
    b7 = np.append(_B, 0.0)
    c7 = np.append(_C, 1.0)
    for k in range(5):  # sum b_i c_i^k = 1/(k+1) up to order 5
        assert b7 @ c7**k == pytest.approx(1 / (k + 1), abs=1e-15)
        if k < 4:
            assert _B_HAT @ c7**k == pytest.approx(1 / (k + 1), abs=1e-15)
    # dense output reproduces the step end: sum of each row of P = b
    np.testing.assert_allclose(_P.sum(axis=1), b7, atol=1e-15)


# ---------------------------------------------------------------------------
# rk45
# ---------------------------------------------------------------------------
@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("name", list(PROBLEMS))
def test_rk45_global_error_within_tolerance(name):
    f, y0, T, exact = PROBLEMS[name]
    te = np.linspace(0, T, 101)
    for rtol in (1e-3, 1e-4, 1e-6, 1e-8, 1e-10):
        for atol in (rtol, rtol * 1e-3):
            res = solve_ode(f, (0, T), y0, rtol=rtol, atol=atol, t_eval=te)
            assert res.success, res.message
            E = exact(te)
            err = np.max(np.abs(res.y - E))
            assert err <= 10 * (atol + rtol * np.max(np.abs(E))), (rtol, atol, err)


@pytest.mark.parametrize("name", list(PROBLEMS))
@pytest.mark.parametrize("rtol", [1e-3, 1e-6, 1e-9])
def test_rk45_matches_scipy_solve_ivp(name, rtol):
    ivp = pytest.importorskip("scipy.integrate")
    f, y0, T, _ = PROBLEMS[name]
    atol = rtol * 1e-3
    te = np.linspace(0, T, 37)
    ours = solve_ode(f, (0, T), y0, rtol=rtol, atol=atol, t_eval=te)
    ref = ivp.solve_ivp(f, (0, T), y0, method="RK45", rtol=rtol, atol=atol, t_eval=te)
    scale = atol + rtol * np.abs(ref.y)
    assert np.all(np.abs(ours.y - ref.y) <= scale)
    # Same step-size rules, so the same work (up to rounding in the error norm).
    assert abs(ours.nfev - ref.nfev) <= 0.1 * ref.nfev
    steps = solve_ode(f, (0, T), y0, rtol=rtol, atol=atol)
    ref_steps = ivp.solve_ivp(f, (0, T), y0, method="RK45", rtol=rtol, atol=atol)
    assert abs(steps.t.size - ref_steps.t.size) <= 0.1 * ref_steps.t.size + 1


def test_rk45_dense_output_matches_step_values():
    f, y0, T, exact = PROBLEMS["oscillator"]
    steps = solve_ode(f, (0, T), y0, rtol=1e-8, atol=1e-10)
    dense = solve_ode(f, (0, T), y0, rtol=1e-8, atol=1e-10, t_eval=steps.t)
    np.testing.assert_allclose(dense.y, steps.y, rtol=0, atol=1e-14)
    assert dense.nfev == steps.nfev  # t_eval never shortens a step


def test_rk45_backwards_and_first_step_and_max_step():
    f, y0, T, exact = PROBLEMS["decay"]
    back = solve_ode(f, (T, 0.0), exact(T)[:, None][:, 0], rtol=1e-9, atol=1e-12)
    assert back.success and back.t[-1] == 0.0 and np.all(np.diff(back.t) < 0)
    assert back.y[0, -1] == pytest.approx(1.0, rel=1e-7)
    capped = solve_ode(f, (0, T), y0, max_step=0.01)
    assert np.max(np.diff(capped.t)) <= 0.01 + 1e-15
    first = solve_ode(f, (0, T), y0, h=1e-4)
    assert first.t[1] == pytest.approx(1e-4)


def test_rk45_stiff_problem_is_slow_but_correct():
    stiff = lambda t, y: -1000.0 * (y - np.cos(t))  # noqa: E731
    res = solve_ode(stiff, (0, 1), [0.0], rtol=1e-6, atol=1e-9)
    assert res.success
    assert res.nfev > 1000  # stability-limited steps of about 3.3e-3
    assert res.y[0, -1] == pytest.approx(np.cos(1.0), abs=1e-3)


def test_rk45_step_size_underflow_reports_failure():
    blowup = lambda t, y: y**2  # noqa: E731  # y = 1/(1 - t) blows up at t = 1
    res = solve_ode(blowup, (0, 2), [1.0])
    assert not res.success
    assert res.t[-1] < 1.001 and res.y.shape == (1, res.t.size)


# ---------------------------------------------------------------------------
# fixed-step methods: measured convergence order
# ---------------------------------------------------------------------------
def _order(method, hs, problem="oscillator", T=2.0):
    f, y0, _, exact = PROBLEMS[problem]
    errs = []
    for h in hs:
        res = solve_ode(f, (0, T), y0, method=method, h=h, rtol=1e-13, atol=1e-15)
        assert res.success and res.t[-1] == T
        errs.append(np.max(np.abs(res.y[:, -1] - exact(T))))
    return np.diff(np.log(errs)) / np.diff(np.log(hs))


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("problem", ["oscillator", "growth", "logistic"])
def test_rk4_is_fourth_order(problem):
    slopes = _order("rk4", [0.1, 0.05, 0.025, 0.0125], problem)
    np.testing.assert_allclose(slopes, 4.0, atol=0.1)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("problem", ["oscillator", "growth", "logistic"])
def test_implicit_euler_is_first_order(problem):
    slopes = _order("implicit_euler", [0.02, 0.01, 0.005, 0.0025], problem)
    np.testing.assert_allclose(slopes, 1.0, atol=0.05)


def test_fixed_step_grid_lands_on_t1():
    res = solve_ode(osc, (0.0, 1.0), [1.0, 0.0], method="rk4", h=0.3)
    np.testing.assert_allclose(res.t, [0.0, 0.25, 0.5, 0.75, 1.0])
    assert res.nfev == 4 * 4  # 4 stages per step (k1 is f at the previous grid point)
    res = solve_ode(osc, (0.0, 1.0), [1.0, 0.0], method="rk4", h=0.1)
    assert res.t.size == 11 and res.t[-1] == 1.0


def test_rk4_hermite_t_eval_is_fourth_order():
    f, y0, _, exact = PROBLEMS["oscillator"]
    te = np.linspace(0, 2, 23)[1:-1]  # never on the grid
    errs = []
    hs = [0.2, 0.1, 0.05, 0.025]
    for h in hs:
        res = solve_ode(f, (0, 2), y0, method="rk4", h=h, t_eval=te)
        np.testing.assert_array_equal(res.t, te)
        errs.append(np.max(np.abs(res.y - exact(te))))
    slopes = np.diff(np.log(errs)) / np.diff(np.log(hs))
    assert np.all(slopes > 3.7)


def test_implicit_euler_t_eval_is_linear_interpolation():
    f, y0, _, exact = PROBLEMS["decay"]
    full = solve_ode(f, (0, 1), y0, method="implicit_euler", h=0.5)
    res = solve_ode(f, (0, 1), y0, method="implicit_euler", h=0.5, t_eval=[0.0, 0.25, 1.0])
    np.testing.assert_allclose(res.y[0], [1.0, (1 + full.y[0, 1]) / 2, full.y[0, 2]])
    # backward Euler on y' = -y: y_{n+1} = y_n / (1 + h)
    np.testing.assert_allclose(full.y[0], [1.0, 1 / 1.5, 1 / 1.5**2], rtol=1e-12)


# ---------------------------------------------------------------------------
# stiffness
# ---------------------------------------------------------------------------
def stiff(t, y):
    return -1000.0 * (y - np.cos(t))


def stiff_exact(t):
    # y' = -1000 (y - cos t), y(0) = 0
    k = 1000.0
    return k * (k * np.cos(t) + np.sin(t)) / (k * k + 1) - k * k / (k * k + 1) * np.exp(-k * t)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("h", [0.01, 0.05, 0.1, 0.5])
def test_implicit_euler_stable_on_stiff_problem_where_rk4_blows_up(h):
    ie = solve_ode(stiff, (0, 5), [0.0], method="implicit_euler", h=h)
    assert ie.success
    # Bounded, and tracks the slow solution cos(t) once the transient has decayed.
    assert np.all(np.abs(ie.y) <= 1.0 + 1e-12)
    late = ie.t >= 1.0
    assert np.max(np.abs(ie.y[0, late] - stiff_exact(ie.t[late]))) <= 2 * h
    rk = solve_ode(stiff, (0, 5), [0.0], method="rk4", h=h)  # h * 1000 >> 2.78
    assert (not rk.success and "non-finite" in rk.message) or np.max(np.abs(rk.y)) > 1e50


@pytest.mark.parametrize("use_jac", [False, True, "matrix"])
def test_implicit_euler_jacobian_options_agree(use_jac):
    A = np.array([[-1000.0, 1.0], [0.0, -0.5]])
    f = lambda t, y: A @ y  # noqa: E731
    jac = {False: None, True: (lambda t, y: A), "matrix": A}[use_jac]
    res = solve_ode(f, (0, 1), [1.0, 1.0], method="implicit_euler", h=0.1, jac=jac)
    ref = [1.0, 1.0]
    for _ in range(10):  # linear problem: (I - hA) y_{n+1} = y_n
        ref = np.linalg.solve(np.eye(2) - 0.1 * A, ref)
    np.testing.assert_allclose(res.y[:, -1], ref, rtol=1e-6, atol=1e-12)
    if use_jac is False:
        assert res.njev > 0 and res.nfev > res.njev * 2  # FD columns are counted in nfev
    else:
        assert res.njev > 0


def robertson(t, y):
    return np.array(
        [
            -0.04 * y[0] + 1e4 * y[1] * y[2],
            0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] ** 2,
            3e7 * y[1] ** 2,
        ]
    )


@pytest.mark.filterwarnings("error")
def test_implicit_euler_nonlinear_stiff_robertson():
    ivp = pytest.importorskip("scipy.integrate")
    ref = ivp.solve_ivp(robertson, (0, 40), [1, 0, 0], method="Radau", rtol=1e-10, atol=1e-14)
    errs = []
    for h in (0.1, 0.05, 0.025):
        res = solve_ode(robertson, (0, 40), [1.0, 0.0, 0.0], method="implicit_euler", h=h)
        assert res.success
        # Linear invariants are conserved exactly by implicit Euler (up to Newton/rounding).
        assert np.max(np.abs(res.y.sum(axis=0) - 1.0)) <= 1e-12
        errs.append(np.max(np.abs(res.y[:, -1] - ref.y[:, -1]) / np.abs(ref.y[:, -1])))
        assert errs[-1] <= 0.02 * h
    slopes = np.diff(np.log(errs)) / np.log(0.5)
    np.testing.assert_allclose(slopes, 1.0, atol=0.05)
    explicit = solve_ode(robertson, (0, 40), [1.0, 0.0, 0.0], method="rk4", h=0.1)
    assert not explicit.success


# ---------------------------------------------------------------------------
# API and edge cases
# ---------------------------------------------------------------------------
def test_scalar_y0_and_result_object():
    res = solve_ode(lambda t, y: -2 * y, (0, 1), 3.0)
    assert res.y.shape == (1, res.t.size)
    assert res.y[0, -1] == pytest.approx(3 * np.exp(-2), rel=1e-5)
    d = res.to_dict()
    assert set(d) == {"t", "y", "nfev", "njev", "success", "message"}
    assert isinstance(res, ODEResult)
    with pytest.raises(ValueError):
        ODEResult(np.zeros(3), np.zeros((1, 2)), 0, 0, True, "")


def test_empty_interval():
    res = solve_ode(osc, (1.0, 1.0), [1.0, 2.0], t_eval=[1.0, 1.0])
    assert res.success and res.y.shape == (2, 2)
    np.testing.assert_array_equal(res.y[:, 0], [1.0, 2.0])


def test_t_eval_endpoints_and_order():
    f, y0, T, exact = PROBLEMS["decay"]
    for method, h in (("rk45", None), ("rk4", 0.1), ("implicit_euler", 0.1)):
        res = solve_ode(f, (0, 1), y0, method=method, h=h, t_eval=[0.0, 0.0, 0.3, 1.0])
        assert res.t.tolist() == [0.0, 0.0, 0.3, 1.0]
        assert res.y[0, 0] == 1.0 and res.y[0, 1] == 1.0
        back = solve_ode(f, (1, 0), [np.exp(-1)], method=method, h=h, t_eval=[1.0, 0.5, 0.0])
        assert back.y[0, -1] == pytest.approx(1.0, rel=0.06)


def test_inputs_not_mutated():
    y0 = np.array([1.0, 0.0])
    te = np.array([0.5, 1.0])
    solve_ode(osc, (0, 1), y0, t_eval=te)
    np.testing.assert_array_equal(y0, [1.0, 0.0])
    np.testing.assert_array_equal(te, [0.5, 1.0])


def test_errors():
    with pytest.raises(ValueError, match="method"):
        solve_ode(osc, (0, 1), [1.0, 0.0], method="euler")
    with pytest.raises(ValueError, match="needs h"):
        solve_ode(osc, (0, 1), [1.0, 0.0], method="rk4")
    with pytest.raises(ValueError):
        solve_ode(osc, (0, 1), [1.0, 0.0], method="rk4", h=0.5, max_step=0.1)
    with pytest.raises(ValueError):
        solve_ode(osc, (0, 1), [1.0, 0.0], h=-0.1)
    with pytest.raises(ValueError):
        solve_ode(osc, (0, np.inf), [1.0, 0.0])
    with pytest.raises(ValueError):
        solve_ode(osc, (0, 1, 2), [1.0, 0.0])
    with pytest.raises(ValueError):
        solve_ode(osc, (0, 1), [[1.0, 0.0]])
    with pytest.raises(ValueError):
        solve_ode(osc, (0, 1), [])
    with pytest.raises(ValueError):
        solve_ode(osc, (0, 1), [np.nan, 0.0])
    with pytest.raises(ValueError):
        solve_ode(osc, (0, 1), [1.0, 0.0], rtol=-1.0)
    with pytest.raises(ValueError):
        solve_ode(osc, (0, 1), [1.0, 0.0], atol=0.0)
    with pytest.raises(ValueError):
        solve_ode(osc, (0, 1), [1.0, 0.0], max_step=0.0)
    with pytest.raises(ValueError):
        solve_ode(osc, (0, 1), [1.0, 0.0], t_eval=[0.5, 2.0])
    with pytest.raises(ValueError):
        solve_ode(osc, (0, 1), [1.0, 0.0], t_eval=[0.5, 0.2])
    with pytest.raises(ValueError):
        solve_ode(osc, (0, 1), [1.0, 0.0], method="implicit_euler", h=0.1, jac=np.eye(3))
    with pytest.raises(ValueError):
        solve_ode(lambda t, y: np.zeros(3), (0, 1), [1.0, 0.0])
    with pytest.raises(TypeError):
        solve_ode("f", (0, 1), [1.0])


def test_nonfinite_fun_reports_failure():
    res = solve_ode(lambda t, y: np.array([np.nan]) if t > 0.5 else -y, (0, 1), [1.0])
    assert not res.success and "non-finite" in res.message
    assert np.all(res.t <= 0.5 + 1e-12)


def test_singular_newton_matrix_reports_failure():
    # f = y / h makes I - h J singular for implicit Euler with step h
    res = solve_ode(lambda t, y: 10.0 * y, (0, 1), [1.0], method="implicit_euler", h=0.1)
    assert not res.success and "singular" in res.message


def test_registered():
    import lipipy

    info = lipipy.registry.get("numerics.solve_ode")
    assert "ode" in info.tasks and info.complexity.space == "O(T d)"


def test_memory_linear_in_steps():
    from lipipy.bench import estimate_complexity

    def run(n_steps):
        return solve_ode(osc, (0.0, 1.0), [1.0, 0.0], method="rk4", h=1.0 / n_steps)

    est = estimate_complexity(run, lambda n, rng: n, sizes=(4000, 8000, 16000), repeats=1)
    assert est.space_exponent == pytest.approx(1.0, abs=0.15)
