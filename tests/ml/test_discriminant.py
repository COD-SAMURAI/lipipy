"""LDA and QDA: parity with sklearn.discriminant_analysis, edge cases."""

import numpy as np
import pytest

from lipipy.core import NotFittedError
from lipipy.ml import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

sk_da = pytest.importorskip("sklearn.discriminant_analysis")


def make_data(seed, n=120, d=4, k=3):
    rng = np.random.default_rng(seed)
    y = rng.integers(k, size=n)
    X = rng.normal(size=(n, d)) * rng.uniform(0.5, 3.0, size=d)
    X += y[:, None] * rng.normal(size=d)
    X_test = rng.normal(scale=2.0, size=(40, d)) + rng.normal(size=d)
    return X, y, X_test


def align_signs(ours, ref):
    """Flip each column of ``ours`` to match ``ref`` (directions are defined up to sign)."""
    signs = np.sign(np.sum(ours * ref, axis=0))
    signs[signs == 0] = 1.0
    return ours * signs


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("k", [2, 3, 5])
def test_lda_parity(seed, k):
    X, y, X_test = make_data(seed, k=k)
    ours = LinearDiscriminantAnalysis().fit(X, y)
    ref = sk_da.LinearDiscriminantAnalysis(solver="svd").fit(X, y)
    np.testing.assert_allclose(ours.predict_proba(X_test), ref.predict_proba(X_test), atol=1e-10)
    np.testing.assert_allclose(
        ours.predict_log_proba(X_test), ref.predict_log_proba(X_test), rtol=1e-9, atol=1e-9
    )
    np.testing.assert_array_equal(ours.predict(X_test), ref.predict(X_test))
    np.testing.assert_allclose(ours.coef_, ref.coef_, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(ours.intercept_, ref.intercept_, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(
        ours.explained_variance_ratio_, ref.explained_variance_ratio_, rtol=1e-10
    )
    Z, Z_ref = ours.transform(X_test), ref.transform(X_test)
    np.testing.assert_allclose(align_signs(Z, Z_ref), Z_ref, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("seed", range(5))
def test_lda_priors_and_n_components(seed):
    X, y, X_test = make_data(seed, k=4)
    priors = [0.1, 0.2, 0.3, 0.4]
    ours = LinearDiscriminantAnalysis(priors=priors, n_components=2).fit(X, y)
    ref = sk_da.LinearDiscriminantAnalysis(priors=priors, n_components=2).fit(X, y)
    np.testing.assert_allclose(ours.predict_proba(X_test), ref.predict_proba(X_test), atol=1e-10)
    assert ours.transform(X_test).shape == (40, 2)
    assert ours.explained_variance_ratio_.shape == (2,)


@pytest.mark.parametrize("seed", range(5))
def test_lda_fewer_samples_than_features(seed):
    """n <= d: the within-class scatter is singular; the SVD solver drops null directions."""
    X, y, X_test = make_data(seed, n=8, d=12, k=2)
    ours = LinearDiscriminantAnalysis().fit(X, y)
    ref = sk_da.LinearDiscriminantAnalysis().fit(X, y)
    np.testing.assert_allclose(ours.predict_proba(X_test), ref.predict_proba(X_test), atol=1e-10)


def test_lda_projection_separates_and_whitens(rng):
    X, y, _ = make_data(0, n=400, k=3)
    lda = LinearDiscriminantAnalysis().fit(X, y)
    Z = lda.transform(X)
    within = np.vstack([Z[y == c] - Z[y == c].mean(axis=0) for c in range(3)])
    # the whitened within-class covariance (ML estimate) is the identity
    np.testing.assert_allclose(within.T @ within / len(Z), np.eye(2), atol=1e-10)
    assert lda.explained_variance_ratio_[0] >= lda.explained_variance_ratio_[1]


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("reg_param", [0.0, 0.05, 0.5])
def test_qda_parity(seed, reg_param):
    X, y, X_test = make_data(seed, k=2 + seed % 3)
    ours = QuadraticDiscriminantAnalysis(reg_param=reg_param).fit(X, y)
    ref = sk_da.QuadraticDiscriminantAnalysis(reg_param=reg_param).fit(X, y)
    np.testing.assert_allclose(ours.predict_proba(X_test), ref.predict_proba(X_test), atol=1e-10)
    np.testing.assert_allclose(
        ours.decision_function(X_test), ref.decision_function(X_test), rtol=1e-9, atol=1e-9
    )
    np.testing.assert_array_equal(ours.predict(X_test), ref.predict(X_test))
    for a, b in zip(ours.scalings_, ref.scalings_, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-10)


def test_qda_user_priors_including_zero():
    X, y, X_test = make_data(1)
    priors = [0.0, 0.3, 0.7]
    ours = QuadraticDiscriminantAnalysis(priors=priors).fit(X, y)
    with np.errstate(divide="ignore"):
        ref = sk_da.QuadraticDiscriminantAnalysis(priors=priors).fit(X, y)
        expected = ref.predict_proba(X_test)
    np.testing.assert_allclose(ours.predict_proba(X_test), expected, atol=1e-10)
    assert np.all(ours.predict(X_test) != 0)


@pytest.mark.filterwarnings("error")
def test_qda_extreme_points_stay_finite():
    X, y, _ = make_data(2)
    qda = QuadraticDiscriminantAnalysis().fit(X, y)
    far = np.full((3, 4), 1e6)
    p = qda.predict_proba(far)
    assert np.all(np.isfinite(p)) and np.allclose(p.sum(axis=1), 1.0)


def test_qda_singular_classes_raise():
    X, y, _ = make_data(0, d=3)
    with pytest.raises(ValueError, match="single sample"):
        QuadraticDiscriminantAnalysis().fit(np.vstack([X, [[9.0, 9.0, 9.0]]]), np.append(y, 7))
    small = np.vstack([X, np.random.default_rng(0).normal(size=(3, 3))])
    with pytest.raises(ValueError, match="singular"):  # class with n_c = d samples
        QuadraticDiscriminantAnalysis().fit(small, np.append(y, [7, 7, 7]))
    Xc = X.copy()
    Xc[y == 0, 1] = 2.0  # constant feature within class 0
    with pytest.raises(ValueError, match="singular"):
        QuadraticDiscriminantAnalysis().fit(Xc, y)
    QuadraticDiscriminantAnalysis(reg_param=0.1).fit(Xc, y)  # regularisation fixes it


def test_failed_fit_is_atomic():
    X, y, _ = make_data(0)
    qda = QuadraticDiscriminantAnalysis().fit(X, y)
    means = qda.means_.copy()
    Xc = X.copy()
    Xc[y == 0, 1] = 2.0
    with pytest.raises(ValueError):
        qda.fit(Xc, y)
    np.testing.assert_array_equal(qda.means_, means)
    lda = LinearDiscriminantAnalysis()
    with pytest.raises(ValueError):
        lda.fit(X, y[:-1])
    assert not hasattr(lda, "coef_")


@pytest.mark.parametrize(
    ("estimator", "match"),
    [
        (LinearDiscriminantAnalysis(solver="lsqr"), "solver"),
        (LinearDiscriminantAnalysis(n_components=5), "n_components"),
        (LinearDiscriminantAnalysis(priors=[0.5, 0.6, 0.1]), "priors"),
        (LinearDiscriminantAnalysis(priors=[0.5, 0.5]), "priors"),
        (LinearDiscriminantAnalysis(tol=-1.0), "tol"),
        (QuadraticDiscriminantAnalysis(reg_param=1.5), "reg_param"),
        (QuadraticDiscriminantAnalysis(priors=[1.2, -0.1, -0.1]), "priors"),
    ],
)
def test_bad_params(estimator, match):
    X, y, _ = make_data(0)
    with pytest.raises(ValueError, match=match):
        estimator.fit(X, y)


def test_one_class_and_unfitted():
    X, _, _ = make_data(0)
    with pytest.raises(ValueError, match="2 classes"):
        LinearDiscriminantAnalysis().fit(X, np.zeros(len(X)))
    with pytest.raises(NotFittedError):
        LinearDiscriminantAnalysis().transform(X)
    with pytest.raises(NotFittedError):
        QuadraticDiscriminantAnalysis().predict(X)
    lda = LinearDiscriminantAnalysis().fit(*make_data(0)[:2])
    with pytest.raises(ValueError, match="features"):
        lda.predict(X[:, :2])
