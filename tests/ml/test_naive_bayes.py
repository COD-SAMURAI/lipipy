"""GaussianNB: parity with scikit-learn, log-space stability, error paths."""

import numpy as np
import pytest

from lipipy.core import NotFittedError
from lipipy.ml import GaussianNB


def make_data(rng, n=200, d=4, k=3):
    means = rng.normal(scale=2.0, size=(k, d))
    scales = rng.uniform(0.5, 2.0, size=(k, d))
    y = rng.integers(k, size=n)
    return means[y] + scales[y] * rng.normal(size=(n, d)), y


def _assert_matches(ours, ref, X):
    np.testing.assert_array_equal(ours.classes_, ref.classes_)
    np.testing.assert_allclose(ours.theta_, ref.theta_, rtol=1e-12)
    np.testing.assert_allclose(ours.var_, ref.var_, rtol=1e-10)
    np.testing.assert_allclose(ours.class_prior_, ref.class_prior_, rtol=1e-12)
    assert ours.epsilon_ == pytest.approx(ref.epsilon_, rel=1e-12)
    np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X), atol=1e-10)
    np.testing.assert_allclose(
        ours.predict_log_proba(X), ref.predict_log_proba(X), rtol=1e-10, atol=1e-10
    )
    np.testing.assert_array_equal(ours.predict(X), ref.predict(X))


@pytest.mark.parametrize("var_smoothing", [1e-9, 1e-3, 0.5])
def test_parity_with_sklearn(rng, var_smoothing):
    sk_nb = pytest.importorskip("sklearn.naive_bayes")
    X, y = make_data(rng)
    X_test = rng.normal(scale=3.0, size=(50, X.shape[1]))
    ours = GaussianNB(var_smoothing=var_smoothing).fit(X, y)
    ref = sk_nb.GaussianNB(var_smoothing=var_smoothing).fit(X, y)
    _assert_matches(ours, ref, X_test)
    np.testing.assert_allclose(ours.class_count_, ref.class_count_)


@pytest.mark.filterwarnings("ignore:divide by zero encountered in log:RuntimeWarning")  # sklearn
@pytest.mark.parametrize("priors", [[0.2, 0.3, 0.5], [0.0, 0.5, 0.5], [0.98, 0.01, 0.01]])
def test_parity_with_user_priors(rng, priors):
    sk_nb = pytest.importorskip("sklearn.naive_bayes")
    X, y = make_data(rng)
    X_test = rng.normal(scale=3.0, size=(40, X.shape[1]))
    ours = GaussianNB(priors=priors).fit(X, y)
    ref = sk_nb.GaussianNB(priors=priors).fit(X, y)
    _assert_matches(ours, ref, X_test)


def test_parity_numerically_extreme(rng):
    """Far-away queries and many features: densities underflow, log space does not."""
    sk_nb = pytest.importorskip("sklearn.naive_bayes")
    X, y = make_data(rng, n=300, d=200, k=4)
    X_test = rng.normal(scale=50.0, size=(30, 200))
    ours = GaussianNB().fit(X, y)
    ref = sk_nb.GaussianNB().fit(X, y)
    jll = ours._joint_log_likelihood(X_test)
    assert np.all(np.exp(jll) == 0.0)  # naive probability products would all be 0
    proba = ours.predict_proba(X_test)
    assert np.all(np.isfinite(proba)) and np.allclose(proba.sum(axis=1), 1.0)
    _assert_matches(ours, ref, X_test)


def test_string_labels_and_score(rng):
    X, y = make_data(rng)
    names = np.array(["a", "b", "c"])[y]
    nb = GaussianNB().fit(X, names)
    assert set(nb.predict(X)) <= {"a", "b", "c"}
    assert nb.score(X, names) > 0.6


def test_constant_feature_is_smoothed(rng):
    X, y = make_data(rng)
    X[:, 0] = 3.0
    nb = GaussianNB().fit(X, y)
    assert np.all(nb.var_[:, 0] == pytest.approx(nb.epsilon_))
    assert np.all(np.isfinite(nb.predict_log_proba(X)))


def test_bad_inputs(rng):
    X, y = make_data(rng)
    with pytest.raises(ValueError, match="priors"):
        GaussianNB(priors=[0.5, 0.5]).fit(X, y)
    with pytest.raises(ValueError, match="sum to 1"):
        GaussianNB(priors=[0.5, 0.5, 0.5]).fit(X, y)
    with pytest.raises(ValueError, match="non-negative"):
        GaussianNB(priors=[1.5, -0.25, -0.25]).fit(X, y)
    with pytest.raises(ValueError, match="var_smoothing"):
        GaussianNB(var_smoothing=-1.0).fit(X, y)
    with pytest.raises(ValueError, match="zero-variance"):
        GaussianNB().fit(np.ones((4, 2)), [0, 0, 1, 1])
    with pytest.raises(NotFittedError):
        GaussianNB().predict(X)
    nb = GaussianNB().fit(X, y)
    with pytest.raises(ValueError, match="features"):
        nb.predict_proba(X[:, :2])
