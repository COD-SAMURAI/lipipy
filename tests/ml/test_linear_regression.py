"""LinearRegression: correctness tests plus parity with scikit-learn."""

import numpy as np
import pytest

from lipipy.ml import LinearRegression


def make_data(rng, n=200, d=5, noise=0.1):
    X = rng.normal(size=(n, d)) * rng.uniform(0.5, 10, size=d) + rng.normal(size=d)
    w = rng.normal(size=d)
    y = X @ w + 3.0 + noise * rng.normal(size=n)
    return X, y, w


def test_recovers_exact_line():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = 2 * X[:, 0] + 1
    m = LinearRegression().fit(X, y)
    np.testing.assert_allclose(m.coef_, [2.0])
    assert m.intercept_ == pytest.approx(1.0)
    assert m.score(X, y) == pytest.approx(1.0)


@pytest.mark.parametrize("solver", ["lstsq", "gd"])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_parity_with_sklearn(rng, solver, fit_intercept):
    sk_linear = pytest.importorskip("sklearn.linear_model")
    X, y, _ = make_data(rng)
    ours = LinearRegression(fit_intercept=fit_intercept, solver=solver).fit(X, y)
    ref = sk_linear.LinearRegression(fit_intercept=fit_intercept).fit(X, y)
    tol = 1e-8 if solver == "lstsq" else 1e-4
    np.testing.assert_allclose(ours.coef_, ref.coef_, rtol=tol, atol=tol)
    np.testing.assert_allclose(ours.intercept_, ref.intercept_, rtol=tol, atol=tol)
    np.testing.assert_allclose(ours.predict(X), ref.predict(X), rtol=tol, atol=tol)


def test_rank_deficient_matches_sklearn(rng):
    sk_linear = pytest.importorskip("sklearn.linear_model")
    X, y, _ = make_data(rng, d=3)
    X = np.hstack([X, X[:, :1] * 2.0])  # duplicate column -> singular X^T X
    ours = LinearRegression().fit(X, y)
    ref = sk_linear.LinearRegression().fit(X, y)
    np.testing.assert_allclose(ours.predict(X), ref.predict(X), atol=1e-8)


def test_gd_loss_decreases(rng):
    X, y, _ = make_data(rng)
    m = LinearRegression(solver="gd").fit(X, y)
    assert m.loss_history_[-1] < m.loss_history_[0]
    assert np.all(np.diff(m.loss_history_) <= 1e-12)


def test_bad_inputs():
    with pytest.raises(ValueError, match="solver"):
        LinearRegression(solver="magic").fit([[1.0], [2.0]], [1.0, 2.0])
    m = LinearRegression().fit([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="features"):
        m.predict([[1.0]])
