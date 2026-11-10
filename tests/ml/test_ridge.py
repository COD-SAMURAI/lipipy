"""Ridge: parity with scikit-learn, analytic checks and error paths."""

import numpy as np
import pytest

from lipipy.core import NotFittedError, registry
from lipipy.ml import LinearRegression, Ridge


def make_data(rng, n=120, d=6):
    X = rng.normal(size=(n, d)) * rng.uniform(0.5, 5, size=d) + rng.normal(size=d)
    y = X @ rng.normal(size=d) + 2.0 + 0.3 * rng.normal(size=n)
    return X, y


@pytest.mark.parametrize("alpha", [0.0, 1e-3, 0.5, 1.0, 10.0, 1e4])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_parity_with_sklearn(rng, alpha, fit_intercept):
    sk_linear = pytest.importorskip("sklearn.linear_model")
    X, y = make_data(rng)
    ours = Ridge(alpha=alpha, fit_intercept=fit_intercept).fit(X, y)
    ref = sk_linear.Ridge(alpha=alpha, fit_intercept=fit_intercept).fit(X, y)
    np.testing.assert_allclose(ours.coef_, ref.coef_, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(ours.intercept_, ref.intercept_, rtol=1e-8, atol=1e-10)
    X_new = rng.normal(size=(30, X.shape[1]))
    np.testing.assert_allclose(ours.predict(X_new), ref.predict(X_new), rtol=1e-8, atol=1e-10)


def test_wide_data_parity(rng):
    """d > n: X^T X is singular but X^T X + alpha I is not."""
    sk_linear = pytest.importorskip("sklearn.linear_model")
    X, y = make_data(rng, n=10, d=25)
    ours = Ridge(alpha=0.7).fit(X, y)
    ref = sk_linear.Ridge(alpha=0.7).fit(X, y)
    np.testing.assert_allclose(ours.coef_, ref.coef_, rtol=1e-8, atol=1e-10)


def test_alpha_zero_equals_ols(rng):
    X, y = make_data(rng)
    ridge = Ridge(alpha=0.0).fit(X, y)
    ols = LinearRegression().fit(X, y)
    np.testing.assert_allclose(ridge.coef_, ols.coef_, rtol=1e-9)
    assert ridge.intercept_ == pytest.approx(ols.intercept_, rel=1e-9)


def test_closed_form_single_feature():
    # Centred x = (-1, 0, 1), y = x: w = sum(x y) / (sum(x^2) + alpha) = 2 / (2 + alpha)
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 1.0, 2.0])
    for alpha in (0.0, 1.0, 3.0):
        m = Ridge(alpha=alpha).fit(X, y)
        w = 2.0 / (2.0 + alpha)
        assert m.coef_[0] == pytest.approx(w)
        assert m.intercept_ == pytest.approx(1.0 - w)


def test_large_alpha_shrinks_to_mean(rng):
    X, y = make_data(rng)
    m = Ridge(alpha=1e12).fit(X, y)
    assert np.max(np.abs(m.coef_)) < 1e-6
    assert m.intercept_ == pytest.approx(y.mean(), rel=1e-6)


def test_alpha_zero_singular_raises(rng):
    X, y = make_data(rng, d=3)
    X = np.hstack([X, X[:, :1] * 2.0])
    with pytest.raises(ValueError, match="singular"):
        Ridge(alpha=0.0).fit(X, y)
    Ridge(alpha=0.1).fit(X, y)  # fine with regularisation


def test_alpha_zero_constant_column_singular_after_centring(rng):
    X, y = make_data(rng, d=2)
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    with pytest.raises(ValueError, match="after centring"):
        Ridge(alpha=0.0).fit(X, y)
    Ridge(alpha=0.0, fit_intercept=False).fit(X, y)  # full rank without centring


def test_bad_inputs(rng):
    X, y = make_data(rng)
    with pytest.raises(ValueError, match="alpha"):
        Ridge(alpha=-1.0).fit(X, y)
    with pytest.raises(ValueError, match="alpha"):
        Ridge(alpha=np.inf).fit(X, y)
    with pytest.raises(TypeError, match="alpha"):
        Ridge(alpha="big").fit(X, y)
    with pytest.raises(NotFittedError):
        Ridge().predict(X)
    m = Ridge().fit(X, y)
    with pytest.raises(ValueError, match="features"):
        m.predict(X[:, :2])
    with pytest.raises(ValueError, match="samples"):
        Ridge().fit(X, y[:-1])


def test_score_and_registry(rng):
    X, y = make_data(rng)
    assert Ridge(alpha=1e-6).fit(X, y).score(X, y) > 0.9
    info = registry.get("ml.Ridge")
    assert info.obj is Ridge and "regression" in info.tasks
