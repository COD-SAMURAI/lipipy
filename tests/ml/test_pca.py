"""PCA: parity with scikit-learn (including signs), reconstruction, edge cases."""

import numpy as np
import pytest

from lipipy.core import NotFittedError
from lipipy.ml import PCA


def make_data(rng, n=80, d=6):
    A = rng.normal(size=(d, d)) * np.linspace(3.0, 0.1, d)
    return rng.normal(size=(n, d)) @ A + rng.normal(size=d)


def _sklearn_pca():
    sklearn = pytest.importorskip("sklearn")
    major, minor = (int(p) for p in sklearn.__version__.split(".")[:2])
    if (major, minor) < (1, 5):
        pytest.skip("sklearn < 1.5 uses a different svd_flip sign convention in PCA")
    from sklearn.decomposition import PCA as SkPCA  # noqa: N811

    return SkPCA


@pytest.mark.parametrize("n_components", [None, 1, 3, 6, 0.5, 0.9, 0.999])
@pytest.mark.parametrize("whiten", [False, True])
@pytest.mark.parametrize("shape", [(80, 6), (5, 9)])
def test_parity_with_sklearn(rng, n_components, whiten, shape):
    SkPCA = _sklearn_pca()
    n, d = shape
    if isinstance(n_components, int) and n_components > min(n, d):
        pytest.skip("more components than min(n, d)")
    X = make_data(rng, n=n, d=d)
    ours = PCA(n_components=n_components, whiten=whiten).fit(X)
    ref = SkPCA(n_components=n_components, whiten=whiten, svd_solver="full").fit(X)
    assert ours.n_components_ == ref.n_components_
    for attr in (
        "components_",
        "explained_variance_",
        "explained_variance_ratio_",
        "singular_values_",
        "mean_",
    ):
        np.testing.assert_allclose(
            getattr(ours, attr), getattr(ref, attr), rtol=1e-8, atol=1e-10, err_msg=attr
        )
    assert ours.noise_variance_ == pytest.approx(ref.noise_variance_, rel=1e-8, abs=1e-12)
    X_new = make_data(rng, n=10, d=d)
    Z = ours.transform(X_new)
    np.testing.assert_allclose(Z, ref.transform(X_new), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(ours.inverse_transform(Z), ref.inverse_transform(Z), atol=1e-8)
    np.testing.assert_allclose(ours.fit_transform(X), ref.fit_transform(X), atol=1e-8)


def test_sign_convention_largest_entry_positive(rng):
    pca = PCA().fit(make_data(rng))
    rows = np.arange(pca.n_components_)
    biggest = np.argmax(np.abs(pca.components_), axis=1)
    assert np.all(pca.components_[rows, biggest] > 0)
    np.testing.assert_allclose(pca.components_ @ pca.components_.T, np.eye(6), atol=1e-12)


@pytest.mark.parametrize("whiten", [False, True])
def test_full_reconstruction_is_exact(rng, whiten):
    X = make_data(rng)
    pca = PCA(whiten=whiten).fit(X)
    np.testing.assert_allclose(pca.inverse_transform(pca.transform(X)), X, atol=1e-10)


def test_explained_variance_matches_covariance_eigenvalues(rng):
    X = make_data(rng)
    pca = PCA().fit(X)
    eig = np.sort(np.linalg.eigvalsh(np.cov(X, rowvar=False)))[::-1]
    np.testing.assert_allclose(pca.explained_variance_, eig, rtol=1e-10)
    assert pca.explained_variance_ratio_.sum() == pytest.approx(1.0)


def test_whitened_output_has_unit_variance(rng):
    X = make_data(rng, n=500)
    Z = PCA(n_components=3, whiten=True).fit_transform(X)
    np.testing.assert_allclose(np.var(Z, axis=0, ddof=1), 1.0, rtol=1e-10)


def test_constant_data_gives_zero_variance(rng):
    X = np.ones((5, 3))
    pca = PCA().fit(X)
    np.testing.assert_allclose(pca.explained_variance_, 0.0, atol=1e-30)
    np.testing.assert_allclose(pca.transform(X), 0.0, atol=1e-30)


@pytest.mark.parametrize(
    ("n_components", "error", "match"),
    [
        (7, ValueError, "between 0"),
        (-1, ValueError, "between 0"),
        (1.5, ValueError, r"\(0, 1\)"),
        (0.0, ValueError, r"\(0, 1\)"),
        ("all", TypeError, "n_components"),
        (True, TypeError, "n_components"),
    ],
)
def test_bad_n_components(rng, n_components, error, match):
    with pytest.raises(error, match=match):
        PCA(n_components=n_components).fit(make_data(rng))


def test_bad_inputs(rng):
    X = make_data(rng)
    with pytest.raises(ValueError, match="2 samples"):
        PCA().fit(X[:1])
    with pytest.raises(NotFittedError):
        PCA().transform(X)
    with pytest.raises(NotFittedError):
        PCA().inverse_transform(X)
    pca = PCA(n_components=2).fit(X)
    with pytest.raises(ValueError, match="features"):
        pca.transform(X[:, :3])
    with pytest.raises(ValueError, match="components"):
        pca.inverse_transform(np.ones((2, 3)))
