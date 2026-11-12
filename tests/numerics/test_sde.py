"""lipipy.numerics.sde on geometric Brownian motion (exact solution known)."""

import numpy as np
import pytest

from lipipy.bench import estimate_complexity
from lipipy.numerics import SDEResult, euler_maruyama, milstein

MU, SIG, X0, T = 0.5, 0.8, 1.0, 1.0


def drift(t, x):
    return MU * x


def diffusion(t, x):
    return SIG * x


def diffusion_dx(t, x):
    return SIG


def fine_paths(seed, n_paths=4000, n_fine=2**10):
    rng = np.random.default_rng(seed)
    dW = rng.normal(0.0, np.sqrt(T / n_fine), size=(n_fine, n_paths))
    exact = X0 * np.exp((MU - SIG**2 / 2) * T + SIG * dW.sum(axis=0))
    return dW, exact


def strong_slope(method, seed):
    dW_fine, exact = fine_paths(seed)
    n_fine, n_paths = dW_fine.shape
    dts, errs = [], []
    for k in (2**4, 2**5, 2**6, 2**7, 2**8):
        dW = dW_fine.reshape(k, n_fine // k, n_paths).sum(axis=1)  # same Brownian path
        if method == "em":
            res = euler_maruyama(drift, diffusion, X0, (0, T), k, n_paths, dW=dW)
        else:
            res = milstein(drift, diffusion, diffusion_dx, X0, (0, T), k, n_paths, dW=dW)
        dts.append(T / k)
        errs.append(np.mean(np.abs(res.terminal - exact)))
    return np.polyfit(np.log(dts), np.log(errs), 1)[0]


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("seed", range(5))
def test_euler_maruyama_strong_order_one_half(seed):
    assert strong_slope("em", seed) == pytest.approx(0.5, abs=0.1)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("seed", range(5))
def test_milstein_strong_order_one(seed):
    assert strong_slope("mil", seed) == pytest.approx(1.0, abs=0.1)


def test_milstein_beats_euler_maruyama_on_same_paths():
    dW_fine, exact = fine_paths(11)
    dW = dW_fine.reshape(64, 16, -1).sum(axis=1)
    em = euler_maruyama(drift, diffusion, X0, (0, T), 64, dW.shape[1], dW=dW)
    mil = milstein(drift, diffusion, diffusion_dx, X0, (0, T), 64, dW.shape[1], dW=dW)
    assert np.mean(np.abs(mil.terminal - exact)) < 0.3 * np.mean(np.abs(em.terminal - exact))


@pytest.mark.parametrize("method", ["em", "mil"])
def test_weak_error_of_mean_is_small(method):
    # For GBM each step multiplies the mean by exactly (1 + mu dt) under both schemes
    # (E dW = 0, E dW^2 = dt), so E[X_N] = X0 (1 + mu dt)^N: a weak error of order dt.
    n_paths, n_steps = 200_000, 50
    if method == "em":
        res = euler_maruyama(drift, diffusion, X0, (0, T), n_steps, n_paths, random_state=3)
    else:
        res = milstein(drift, diffusion, diffusion_dx, X0, (0, T), n_steps, n_paths, random_state=3)
    se = res.terminal.std(ddof=1) / np.sqrt(n_paths)
    scheme_mean = X0 * (1 + MU * T / n_steps) ** n_steps
    assert abs(res.terminal.mean() - scheme_mean) <= 4 * se
    assert abs(res.terminal.mean() - X0 * np.exp(MU * T)) <= 4 * se + 0.01
    var_exact = X0**2 * np.exp(2 * MU * T) * (np.exp(SIG**2 * T) - 1)
    assert res.terminal.var() == pytest.approx(var_exact, rel=0.05)


def test_weak_order_one_on_gbm_mean():
    # The exact scheme mean X0 (1 + mu dt)^N converges to X0 e^{mu T} at order 1 in dt.
    ns = np.array([16, 32, 64, 128, 256])
    bias = X0 * np.exp(MU * T) - X0 * (1 + MU * T / ns) ** ns
    slope = np.polyfit(np.log(T / ns), np.log(bias), 1)[0]
    assert slope == pytest.approx(1.0, abs=0.02)


def test_reproducible_with_seed_and_return_paths_consistent():
    kw = dict(n_steps=20, n_paths=50, random_state=42)
    a = euler_maruyama(drift, diffusion, X0, (0, T), **kw)
    b = euler_maruyama(drift, diffusion, X0, (0, T), **kw)
    np.testing.assert_array_equal(a.paths, b.paths)
    c = euler_maruyama(drift, diffusion, X0, (0, T), return_paths=False, **kw)
    assert c.paths is None
    np.testing.assert_array_equal(c.terminal, a.terminal)
    np.testing.assert_array_equal(a.paths[-1], a.terminal)
    assert a.paths[0].tolist() == [X0] * 50
    d = euler_maruyama(drift, diffusion, X0, (0, T), n_steps=20, n_paths=50, random_state=43)
    assert not np.array_equal(d.terminal, a.terminal)
    m1 = milstein(drift, diffusion, diffusion_dx, X0, (0, T), **kw)
    m2 = milstein(drift, diffusion, diffusion_dx, X0, (0, T), **kw)
    np.testing.assert_array_equal(m1.paths, m2.paths)


def test_time_grid_and_shapes():
    res = euler_maruyama(drift, diffusion, X0, (0.5, 2.5), n_steps=8, n_paths=3, random_state=0)
    np.testing.assert_allclose(res.t, np.linspace(0.5, 2.5, 9))
    assert res.t[-1] == 2.5
    assert res.paths.shape == (9, 3) and res.terminal.shape == (3,)
    assert isinstance(res, SDEResult)
    assert set(res.to_dict()) == {"t", "paths", "terminal"}


def test_zero_noise_reduces_to_explicit_euler():
    res = euler_maruyama(lambda t, x: -x, lambda t, x: 0.0, 1.0, (0, 1), 10, 2, random_state=0)
    np.testing.assert_allclose(res.terminal, (1 - 0.1) ** 10, rtol=1e-14)


def test_diagonal_noise_vector_state_matches_componentwise_scalar_runs():
    rng = np.random.default_rng(5)
    n_steps, n_paths = 32, 100
    dW = rng.normal(0, np.sqrt(1 / n_steps), size=(n_steps, n_paths, 2))
    mus, sigs = np.array([0.1, -0.3]), np.array([0.2, 0.5])
    vec = milstein(
        lambda t, x: mus * x,
        lambda t, x: sigs * x,
        lambda t, x: sigs,
        [1.0, 2.0],
        (0, 1),
        n_steps,
        n_paths,
        dW=dW,
    )
    assert vec.paths.shape == (n_steps + 1, n_paths, 2)
    for i in range(2):
        scalar = milstein(
            lambda t, x, i=i: mus[i] * x,
            lambda t, x, i=i: sigs[i] * x,
            lambda t, x, i=i: sigs[i],
            [1.0, 2.0][i],
            (0, 1),
            n_steps,
            n_paths,
            dW=dW[:, :, i],
        )
        np.testing.assert_allclose(vec.paths[:, :, i], scalar.paths, rtol=1e-15)


def test_ornstein_uhlenbeck_additive_noise_moments():
    # dX = -theta X dt + s dW: Milstein == EM for additive noise; stationary var s^2/(2 theta).
    theta, s = 2.0, 0.5
    em = euler_maruyama(
        lambda t, x: -theta * x, lambda t, x: s, 1.0, (0, 5), 500, 50_000, random_state=1
    )
    var_exact = s**2 / (2 * theta) * (1 - np.exp(-2 * theta * 5))
    assert em.terminal.mean() == pytest.approx(np.exp(-theta * 5), abs=0.01)
    assert em.terminal.var() == pytest.approx(var_exact, rel=0.03)


def test_memory_is_O_paths_without_returned_paths():
    def run(n_steps):
        return euler_maruyama(
            drift, diffusion, X0, (0, T), n_steps, 4096, random_state=0, return_paths=False
        )

    est = estimate_complexity(run, lambda n, rng: n, sizes=(50, 100, 200, 400), repeats=1)
    assert est.space_exponent < 0.15  # independent of n_steps

    def run_paths(n_paths):
        return euler_maruyama(
            drift, diffusion, X0, (0, T), 50, n_paths, random_state=0, return_paths=False
        )

    est = estimate_complexity(
        run_paths, lambda n, rng: n, sizes=(10_000, 20_000, 40_000), repeats=1
    )
    assert est.space_exponent == pytest.approx(1.0, abs=0.15)


def test_memory_is_O_paths_times_steps_with_paths():
    def run(n_steps):
        return euler_maruyama(drift, diffusion, X0, (0, T), n_steps, 2048, random_state=0)

    est = estimate_complexity(run, lambda n, rng: n, sizes=(50, 100, 200, 400), repeats=1)
    assert est.space_exponent == pytest.approx(1.0, abs=0.15)


def test_errors():
    with pytest.raises(ValueError):
        euler_maruyama(drift, diffusion, X0, (1, 0), 10)
    with pytest.raises(ValueError):
        euler_maruyama(drift, diffusion, X0, (0, 1), 0)
    with pytest.raises(TypeError):
        euler_maruyama(drift, diffusion, X0, (0, 1), 10.0)
    with pytest.raises(ValueError):
        euler_maruyama(drift, diffusion, X0, (0, 1), 10, n_paths=0)
    with pytest.raises(ValueError):
        euler_maruyama(drift, diffusion, [[1.0]], (0, 1), 10)
    with pytest.raises(ValueError):
        euler_maruyama(drift, diffusion, np.nan, (0, 1), 10)
    with pytest.raises(ValueError):
        euler_maruyama(drift, diffusion, X0, (0, 1), 10, n_paths=3, dW=np.zeros((10, 2)))
    with pytest.raises(ValueError):
        euler_maruyama(drift, diffusion, X0, (0, 1), 2, dW=[[np.inf], [0.0]])
    with pytest.raises(ValueError):
        euler_maruyama(lambda t, x: np.zeros(7), diffusion, X0, (0, 1), 5, n_paths=3)
    with pytest.raises(TypeError):
        euler_maruyama("a", diffusion, X0, (0, 1), 5)
    with pytest.raises(TypeError):
        euler_maruyama(drift, diffusion, X0, (0, 1), 5, return_paths="no")
    with pytest.raises(TypeError):
        milstein(drift, diffusion, None, X0, (0, 1), 5)
    with pytest.raises(TypeError):
        euler_maruyama(drift, diffusion, X0, (0, 1), 5, random_state=1.5)
    with pytest.raises(FloatingPointError):
        euler_maruyama(lambda t, x: x**4, diffusion, 10.0, (0, 10), 10, random_state=0)


def test_dw_input_not_mutated():
    dW = np.random.default_rng(0).normal(size=(5, 3)) * 0.1
    keep = dW.copy()
    milstein(drift, diffusion, diffusion_dx, X0, (0, 1), 5, 3, dW=dW)
    np.testing.assert_array_equal(dW, keep)


def test_registered():
    import lipipy

    for name in ("euler_maruyama", "milstein"):
        info = lipipy.registry.get(f"numerics.{name}")
        assert info.complexity.time == "O(T P d)"
