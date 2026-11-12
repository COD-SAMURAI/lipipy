"""Tests for lipipy.finance.portfolio: closed forms, KKT conditions and properties."""

import itertools

import numpy as np
import pytest

from lipipy.core.errors import ConvergenceWarning
from lipipy.finance import (
    black_litterman,
    efficient_frontier,
    max_sharpe_weights,
    min_variance_weights,
    risk_contributions,
    risk_parity_weights,
)

pytestmark = pytest.mark.filterwarnings("error")

SEEDS = range(12)


def _cov(seed, A=5, cond=None):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(4 * A, A)) @ rng.normal(size=(A, A))
    C = np.cov(X, rowvar=False) * 1e-4
    if cond is not None:  # prescribe the condition number
        vals, vecs = np.linalg.eigh(C)
        vals = np.geomspace(1.0, 1.0 / cond, A) * 1e-4
        C = (vecs * vals) @ vecs.T
        C = (C + C.T) / 2
    return C


def _mu(seed, A=5):
    return np.random.default_rng(seed + 1000).normal(0.0005, 0.0003, size=A)


def _brute_long_only(C):
    """Exact long-only minimum variance by enumerating active sets (small A)."""
    A = C.shape[0]
    best, best_w = np.inf, None
    for k in range(1, A + 1):
        for S in itertools.combinations(range(A), k):
            idx = list(S)
            x = np.linalg.solve(C[np.ix_(idx, idx)], np.ones(k))
            w_s = x / x.sum()
            if np.all(w_s >= -1e-14):
                w = np.zeros(A)
                w[idx] = np.maximum(w_s, 0.0)
                v = w @ C @ w
                if v < best:
                    best, best_w = v, w
    return best, best_w


def _project_bisect(v):
    """Independent simplex projection: bisection on theta in sum(max(v - theta, 0)) = 1."""
    lo, hi = v.min() - 1.0, v.max()
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if np.maximum(v - mid, 0).sum() > 1.0:
            lo = mid
        else:
            hi = mid
    return np.maximum(v - 0.5 * (lo + hi), 0.0)


# ---------------------------------------------------------------------------
# minimum variance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_min_variance_closed_form_and_kkt(seed):
    C = _cov(seed, A=6)
    w = min_variance_weights(C)
    x = np.linalg.solve(C, np.ones(6))
    np.testing.assert_allclose(w, x / x.sum(), rtol=1e-10, atol=1e-13)
    assert w.sum() == pytest.approx(1.0, abs=1e-14)
    grad = 2 * C @ w  # KKT: gradient parallel to the constraint normal (all equal)
    np.testing.assert_allclose(grad, grad.mean(), rtol=1e-9)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("A", [3, 5, 7])
def test_long_only_matches_bruteforce_active_set(seed, A):
    C = _cov(seed, A=A)
    w = min_variance_weights(C, long_only=True)
    best, best_w = _brute_long_only(C)
    assert np.all(w >= 0) and w.sum() == pytest.approx(1.0, abs=1e-14)
    assert w @ C @ w == pytest.approx(best, rel=1e-10)
    np.testing.assert_allclose(w, best_w, atol=1e-7)  # C positive definite: unique optimum
    # KKT for the simplex: g_i = nu on the support, g_i >= nu off it
    g = 2 * C @ w
    support = w > 1e-9
    nu = g[support].mean()
    np.testing.assert_allclose(g[support], nu, rtol=1e-6)
    assert np.all(g[~support] >= nu * (1 - 1e-6))
    # the documented stopping rule: projected-gradient residual <= tol (1e-12)
    step = 1.0 / (2.0 * np.linalg.eigvalsh(C)[-1])
    assert np.max(np.abs(w - _project_bisect(w - step * g))) <= 1e-12


@pytest.mark.parametrize("seed", SEEDS)
def test_simplex_projection_matches_bisection(seed):
    from lipipy.finance.portfolio import _project_simplex

    v = np.random.default_rng(seed).normal(size=9) * 3
    np.testing.assert_allclose(_project_simplex(v), _project_bisect(v), atol=1e-14)
    np.testing.assert_allclose(_project_simplex(np.full(4, 0.25)), 0.25, atol=1e-16)


@pytest.mark.parametrize("seed", SEEDS)
def test_long_only_equals_unconstrained_when_interior(seed):
    rng = np.random.default_rng(seed)
    vols = rng.uniform(0.1, 0.3, size=5)
    C = np.diag(vols**2)  # unconstrained optimum ~ 1/var > 0
    np.testing.assert_allclose(
        min_variance_weights(C, long_only=True), min_variance_weights(C), atol=1e-10
    )


def test_long_only_singular_covariance_unique_variance_kkt():
    """Duplicated asset: the optimum is not unique, compare only its variance (lesson e)."""
    C = _cov(3, A=4)
    C5 = np.zeros((5, 5))
    C5[:4, :4] = C
    C5[4, :4] = C5[:4, 4] = C[0]
    C5[4, 4] = C[0, 0]  # asset 4 is a copy of asset 0
    w = min_variance_weights(C5, long_only=True)
    best, best_w = _brute_long_only(C)
    assert w @ C5 @ w == pytest.approx(best, rel=1e-9)
    merged = np.append(w[:4], 0.0)
    merged[0] += w[4]
    np.testing.assert_allclose(merged[:4], best_w, atol=1e-6)  # the combined position is unique
    with pytest.raises(ValueError, match="singular"):
        min_variance_weights(C5)


def test_min_variance_singular_and_invalid_inputs():
    rng = np.random.default_rng(0)
    rank_deficient = np.cov(rng.normal(size=(3, 6)), rowvar=False)  # T < A
    with pytest.raises(ValueError, match="singular"):
        min_variance_weights(rank_deficient)
    with pytest.raises(ValueError, match="singular"):
        min_variance_weights(np.array([[1.0, 1.0], [1.0, 1.0]]))
    with pytest.raises(ValueError, match="zero"):
        min_variance_weights(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="symmetric"):
        min_variance_weights([[1.0, 0.5], [0.0, 1.0]])
    with pytest.raises(ValueError, match="square"):
        min_variance_weights(np.ones((2, 3)))
    with pytest.raises(ValueError, match="NaN"):
        min_variance_weights([[1.0, np.nan], [np.nan, 1.0]])
    with pytest.raises(ValueError, match="semi-definite"):
        min_variance_weights([[1.0, 2.0], [2.0, 1.0]], long_only=True)
    with pytest.raises(ValueError, match="negative"):
        min_variance_weights([[-1.0, 0.0], [0.0, 1.0]])
    # zero matrix is fine long-only: every portfolio has zero variance
    np.testing.assert_allclose(min_variance_weights(np.zeros((3, 3)), long_only=True), 1 / 3)
    assert min_variance_weights([[0.04]]).tolist() == [1.0]


def test_long_only_convergence_warning():
    C = _cov(1, A=6, cond=1e6)
    with pytest.warns(ConvergenceWarning):
        w = min_variance_weights(C, long_only=True, maxiter=3)
    assert w.sum() == pytest.approx(1.0)


def test_cvxpy_parity_long_only():
    cp = pytest.importorskip("cvxpy")
    for seed in range(5):
        C = _cov(seed, A=8)
        w = cp.Variable(8)
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(C))), [w >= 0, cp.sum(w) == 1])
        prob.solve()
        ours = min_variance_weights(C, long_only=True)
        assert ours @ C @ ours <= prob.value * (1 + 1e-7)
        np.testing.assert_allclose(ours, w.value, atol=1e-4)


# ---------------------------------------------------------------------------
# tangency and frontier
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_max_sharpe_closed_form_and_optimality(seed):
    C, mu = _cov(seed), _mu(seed)
    rf = 0.0001
    x = np.linalg.solve(C, mu - rf)
    if x.sum() <= 0:
        with pytest.raises(ValueError, match="minimum-variance"):
            max_sharpe_weights(mu, C, rf)
        return
    w = max_sharpe_weights(mu, C, rf)
    np.testing.assert_allclose(w, x / x.sum(), rtol=1e-9, atol=1e-12)
    sharpe = (w @ mu - rf) / np.sqrt(w @ C @ w)
    assert sharpe == pytest.approx(np.sqrt((mu - rf) @ x), rel=1e-10)  # maximal Sharpe
    rng = np.random.default_rng(seed)
    for _ in range(200):
        v = w + rng.normal(scale=0.2, size=w.size)
        v /= v.sum()
        assert (v @ mu - rf) / np.sqrt(v @ C @ v) <= sharpe * (1 + 1e-12)


def test_max_sharpe_errors():
    C = np.diag([0.04, 0.01])
    with pytest.raises(ValueError, match="minimum-variance"):
        max_sharpe_weights([0.05, 0.05], C, risk_free=0.06)
    with pytest.raises(ValueError, match="minimum-variance"):
        max_sharpe_weights([0.05, 0.05], C, risk_free=0.05)
    with pytest.raises(ValueError, match="shape"):
        max_sharpe_weights([0.05], C)
    with pytest.raises(ValueError, match="singular"):
        max_sharpe_weights([0.05, 0.06], [[1.0, 1.0], [1.0, 1.0]])


@pytest.mark.parametrize("seed", SEEDS)
def test_efficient_frontier_solves_kkt_system(seed):
    C, mu = _cov(seed), _mu(seed)
    A = C.shape[0]
    ef = efficient_frontier(mu, C, n_points=15)
    K = np.zeros((A + 2, A + 2))
    K[:A, :A] = 2 * C
    K[:A, A] = K[A, :A] = 1.0
    K[:A, A + 1] = K[A + 1, :A] = mu
    for m, s, w in zip(ef.returns, ef.volatilities, ef.weights, strict=True):
        sol = np.linalg.solve(K, np.concatenate([np.zeros(A), [1.0, m]]))
        np.testing.assert_allclose(w, sol[:A], rtol=1e-8, atol=1e-10)
        assert w.sum() == pytest.approx(1.0, abs=1e-12)
        assert w @ mu == pytest.approx(m, rel=1e-10, abs=1e-15)
        assert s == pytest.approx(np.sqrt(w @ C @ w), rel=1e-9)
    # first point is the global minimum-variance portfolio
    np.testing.assert_allclose(ef.weights[0], min_variance_weights(C), rtol=1e-8, atol=1e-11)
    # two-fund theorem: w(m) is affine in m
    t = (ef.returns - ef.returns[0]) / (ef.returns[-1] - ef.returns[0])
    affine = np.outer(1 - t, ef.weights[0]) + np.outer(t, ef.weights[-1])
    np.testing.assert_allclose(ef.weights, affine, rtol=1e-9, atol=1e-11)
    assert np.all(np.diff(ef.volatilities) >= 0)  # upper branch: risk rises with return


def test_tangency_lies_on_frontier_and_custom_range():
    C, mu = _cov(2), _mu(2)
    w = max_sharpe_weights(mu, C, 0.0) if np.linalg.solve(C, mu).sum() > 0 else None
    if w is not None:
        ef = efficient_frontier(mu, C, n_points=2, min_return=w @ mu, max_return=w @ mu + 1e-4)
        np.testing.assert_allclose(ef.weights[0], w, rtol=1e-8, atol=1e-11)
    ef = efficient_frontier(mu, C, n_points=4, min_return=-0.01, max_return=0.01)
    np.testing.assert_allclose(ef.returns, np.linspace(-0.01, 0.01, 4))
    assert set(ef.to_dict()) == {"returns", "volatilities", "weights"}


def test_efficient_frontier_errors():
    C = np.diag([0.04, 0.01])
    with pytest.raises(ValueError, match="equal"):
        efficient_frontier([0.05, 0.05], C)
    with pytest.raises(ValueError, match="exceed"):
        efficient_frontier([0.1, 0.05], C, min_return=0.2)
    with pytest.raises(ValueError, match="n_points"):
        efficient_frontier([0.1, 0.05], C, n_points=1)


# ---------------------------------------------------------------------------
# risk parity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("A,cond", [(3, None), (10, None), (25, None), (8, 1e6)])
def test_risk_parity_equal_contributions(seed, A, cond):
    C = _cov(seed, A=A, cond=cond)
    w = risk_parity_weights(C)
    assert np.all(w > 0) and w.sum() == pytest.approx(1.0, abs=1e-14)
    rc = risk_contributions(w, C)
    np.testing.assert_allclose(rc, 1.0 / A, rtol=0, atol=1e-8)


@pytest.mark.parametrize("seed", SEEDS)
def test_risk_parity_budgets_and_scale_invariance(seed):
    C = _cov(seed, A=6)
    b = np.random.default_rng(seed).uniform(0.5, 3.0, size=6)
    w = risk_parity_weights(C, budgets=b)
    np.testing.assert_allclose(risk_contributions(w, C), b / b.sum(), atol=1e-8)
    np.testing.assert_allclose(risk_parity_weights(C * 1e6), risk_parity_weights(C), atol=1e-10)


@pytest.mark.parametrize("rho", [-0.9, -0.3, 0.0, 0.5, 0.95])
def test_risk_parity_two_assets_inverse_volatility(rho):
    s = np.array([0.3, 0.1])
    C = np.outer(s, s) * np.array([[1.0, rho], [rho, 1.0]])
    np.testing.assert_allclose(risk_parity_weights(C), (1 / s) / (1 / s).sum(), atol=1e-12)


def test_risk_parity_errors():
    with pytest.raises(ValueError, match="singular"):
        risk_parity_weights([[1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="positive"):
        risk_parity_weights(np.eye(2), budgets=[1.0, 0.0])
    with pytest.raises(ValueError, match="zero"):
        risk_contributions([0.0, 0.0], np.eye(2))


# ---------------------------------------------------------------------------
# Black-Litterman
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_black_litterman_no_views_is_prior(seed):
    C = _cov(seed)
    wm = np.random.default_rng(seed).dirichlet(np.ones(5))
    for res in (black_litterman(C, wm), black_litterman(C, wm, P=np.zeros((0, 5)), Q=[])):
        np.testing.assert_allclose(res.prior_returns, 2.5 * C @ wm, rtol=1e-14)
        np.testing.assert_array_equal(res.posterior_returns, res.prior_returns)
        np.testing.assert_allclose(res.posterior_covariance, 1.05 * C, rtol=1e-14)
    # the market portfolio is optimal for the prior: Sigma^{-1} pi / delta = w_mkt
    np.testing.assert_allclose(np.linalg.solve(C, res.prior_returns) / 2.5, wm, rtol=1e-8)


@pytest.mark.parametrize("seed", SEEDS)
def test_black_litterman_matches_precision_form(seed):
    C = _cov(seed)
    rng = np.random.default_rng(seed)
    wm = rng.dirichlet(np.ones(5))
    P = rng.normal(size=(2, 5))
    Q = rng.normal(0.001, 0.001, size=2)
    Om = np.diag(rng.uniform(1e-7, 1e-6, size=2))
    tau = 0.025
    res = black_litterman(C, wm, P, Q, Om, tau=tau, risk_aversion=3.0)
    pi = 3.0 * C @ wm
    prec = np.linalg.inv(tau * C) + P.T @ np.linalg.inv(Om) @ P
    M = np.linalg.inv(prec)
    mu = M @ (np.linalg.inv(tau * C) @ pi + P.T @ np.linalg.inv(Om) @ Q)
    np.testing.assert_allclose(res.posterior_returns, mu, rtol=1e-7, atol=1e-12)
    np.testing.assert_allclose(res.posterior_covariance, C + M, rtol=1e-8, atol=1e-14)


def test_black_litterman_confident_view_is_honoured_and_default_omega():
    C = _cov(4)
    wm = np.full(5, 0.2)
    P = np.array([[1.0, -1.0, 0.0, 0.0, 0.0]])
    res = black_litterman(C, wm, P, [0.01], omega=[[1e-16]])
    assert P @ res.posterior_returns == pytest.approx([0.01], rel=1e-6)
    default = black_litterman(C, wm, P, [0.01], tau=0.05)
    explicit = black_litterman(C, wm, P, [0.01], omega=P @ (0.05 * C) @ P.T, tau=0.05)
    np.testing.assert_allclose(default.posterior_returns, explicit.posterior_returns)
    # He-Litterman default: posterior view return is halfway between prior and view
    prior_view = (P @ default.prior_returns)[0]
    assert (P @ default.posterior_returns)[0] == pytest.approx((prior_view + 0.01) / 2, 1e-10)


def test_black_litterman_errors():
    C = np.eye(2) * 0.01
    with pytest.raises(ValueError, match="Q is required"):
        black_litterman(C, [0.5, 0.5], P=[[1.0, 0.0]])
    with pytest.raises(ValueError, match="pick matrix"):
        black_litterman(C, [0.5, 0.5], Q=[0.1])
    with pytest.raises(ValueError, match=r"P must have shape"):
        black_litterman(C, [0.5, 0.5], P=[[1.0, 0.0, 0.0]], Q=[0.1])
    with pytest.raises(ValueError, match="omega"):
        black_litterman(C, [0.5, 0.5], P=[[1.0, 0.0]], Q=[0.1], omega=np.eye(2))
    with pytest.raises(ValueError, match="positive definite"):
        black_litterman(C, [0.5, 0.5], P=[[1.0, 0.0]], Q=[0.1], omega=[[-1.0]])
    with pytest.raises(ValueError, match="tau"):
        black_litterman(C, [0.5, 0.5], tau=0.0)
