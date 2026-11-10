"""KMeans: parity with scikit-learn (explicit init), k-means++ quality, edge cases."""

import warnings

import numpy as np
import pytest

from lipipy.bench import estimate_complexity
from lipipy.core import ConvergenceWarning, NotFittedError
from lipipy.ml import KMeans, adjusted_rand_score


def make_blobs(rng, n=300, d=2, k=4, spread=1.0, scale=10.0):
    centres = rng.uniform(-scale, scale, size=(k, d))
    y = np.arange(n) % k
    return centres[y] + spread * rng.normal(size=(n, d)), y, centres


@pytest.mark.parametrize("k", [2, 3, 5])
@pytest.mark.parametrize("spread", [1.0, 4.0])
def test_parity_with_sklearn_explicit_init(rng, k, spread):
    sk_cluster = pytest.importorskip("sklearn.cluster")
    X, _, _ = make_blobs(rng, n=400, d=3, k=4, spread=spread)
    init = X[rng.choice(len(X), size=k, replace=False)].copy()
    ours = KMeans(n_clusters=k, init=init, n_init=1).fit(X)
    ref = sk_cluster.KMeans(n_clusters=k, init=init, n_init=1, algorithm="lloyd").fit(X)
    np.testing.assert_allclose(ours.cluster_centers_, ref.cluster_centers_, rtol=1e-8, atol=1e-8)
    np.testing.assert_array_equal(ours.labels_, ref.labels_)
    assert ours.inertia_ == pytest.approx(ref.inertia_, rel=1e-8)
    assert ours.n_iter_ == ref.n_iter_
    X_new = rng.normal(scale=8.0, size=(50, 3))
    np.testing.assert_array_equal(ours.predict(X_new), ref.predict(X_new))
    np.testing.assert_allclose(ours.transform(X_new), ref.transform(X_new), rtol=1e-8)


def test_parity_when_max_iter_stops_early(rng):
    sk_cluster = pytest.importorskip("sklearn.cluster")
    X, _, _ = make_blobs(rng, n=500, k=6, spread=5.0)
    init = X[:6].copy()
    ours = KMeans(n_clusters=6, init=init, max_iter=2).fit(X)
    ref = sk_cluster.KMeans(n_clusters=6, init=init, n_init=1, max_iter=2, algorithm="lloyd")
    ref.fit(X)
    np.testing.assert_allclose(ours.cluster_centers_, ref.cluster_centers_, rtol=1e-8, atol=1e-8)
    np.testing.assert_array_equal(ours.labels_, ref.labels_)
    assert ours.inertia_ == pytest.approx(ref.inertia_, rel=1e-8)


def test_kmeans_plus_plus_recovers_separated_blobs(rng):
    X, y, _ = make_blobs(rng, n=400, d=2, k=5, spread=0.3, scale=20.0)
    km = KMeans(n_clusters=5, random_state=0).fit(X)
    assert adjusted_rand_score(y, km.labels_) == 1.0
    assert km.inertia_ == pytest.approx(np.sum((X - km.cluster_centers_[km.labels_]) ** 2))


def test_seeded_runs_are_reproducible(rng):
    X, _, _ = make_blobs(rng, spread=3.0)
    a = KMeans(n_clusters=4, n_init=3, random_state=7).fit(X)
    b = KMeans(n_clusters=4, n_init=3, random_state=7).fit(X)
    np.testing.assert_array_equal(a.cluster_centers_, b.cluster_centers_)
    np.testing.assert_array_equal(a.labels_, b.labels_)
    gen = KMeans(n_clusters=4, random_state=np.random.default_rng(7)).fit(X)
    assert gen.inertia_ > 0


def test_more_inits_never_worse(rng):
    X, _, _ = make_blobs(rng, k=6, spread=2.5)
    one = KMeans(n_clusters=6, n_init=1, random_state=3).fit(X)
    many = KMeans(n_clusters=6, n_init=10, random_state=3).fit(X)
    assert many.inertia_ <= one.inertia_ + 1e-9


def test_fit_predict_transform_consistency(rng):
    X, _, _ = make_blobs(rng)
    km = KMeans(n_clusters=4, random_state=0)
    labels = km.fit_predict(X)
    np.testing.assert_array_equal(labels, km.labels_)
    np.testing.assert_array_equal(km.predict(X), labels)
    D = km.fit_transform(X)
    assert D.shape == (len(X), 4)
    np.testing.assert_array_equal(np.argmin(D, axis=1), km.labels_)


def test_empty_cluster_is_reseeded_from_farthest_point():
    sk_cluster = pytest.importorskip("sklearn.cluster")
    # The third initial centre is far from every point, so it is empty after the
    # first assignment and gets re-seeded with the point farthest from its centre.
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [10.0, 10.0], [10.0, 11.0], [30.0, 0.0]])
    init = np.array([[0.0, 0.0], [10.0, 10.0], [100.0, 100.0]])
    ours = KMeans(n_clusters=3, init=init).fit(X)
    ref = sk_cluster.KMeans(n_clusters=3, init=init, n_init=1, algorithm="lloyd").fit(X)
    np.testing.assert_array_equal(ours.labels_, ref.labels_)
    np.testing.assert_allclose(ours.cluster_centers_, ref.cluster_centers_, atol=1e-10)
    assert sorted(np.bincount(ours.labels_).tolist()) == [1, 2, 3]
    assert ours.labels_[5] != ours.labels_[0]  # the outlier got its own cluster


def test_fewer_distinct_points_than_clusters_warns():
    X = np.array([[1.0, 1.0]] * 5 + [[2.0, 2.0]] * 5)
    with pytest.warns(ConvergenceWarning, match="distinct clusters"):
        km = KMeans(n_clusters=3, random_state=0).fit(X)
    assert km.inertia_ == pytest.approx(0.0)


def test_single_cluster_is_the_mean(rng):
    X, _, _ = make_blobs(rng)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        km = KMeans(n_clusters=1, random_state=0).fit(X)
    np.testing.assert_allclose(km.cluster_centers_[0], X.mean(axis=0))


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"n_clusters": 0}, "n_clusters"),
        ({"n_clusters": 1000}, "larger than"),
        ({"n_init": 0}, "n_init"),
        ({"max_iter": 0}, "max_iter"),
        ({"tol": -1.0}, "tol"),
        ({"init": "random"}, "init"),
        ({"init": np.zeros((3, 2))}, "shape"),
    ],
)
def test_bad_params(rng, params, match):
    X, _, _ = make_blobs(rng)
    params = {"n_clusters": 2, **params}
    with pytest.raises(ValueError, match=match):
        KMeans(**params).fit(X)


def test_unfitted_and_feature_mismatch(rng):
    X, _, _ = make_blobs(rng)
    with pytest.raises(NotFittedError):
        KMeans().predict(X)
    with pytest.raises(NotFittedError):
        KMeans().transform(X)
    km = KMeans(n_clusters=2, random_state=0).fit(X)
    with pytest.raises(ValueError, match="features"):
        km.predict(np.ones((2, 5)))


def test_memory_is_linear_in_n():
    """Declared extra space O(n (k + d)): no n x n temporaries."""
    k, d = 4, 3

    def make(n, rng):
        return rng.normal(size=(n, d))

    def fit(X):
        return KMeans(n_clusters=k, init=X[:k].copy(), max_iter=5).fit(X)

    est = estimate_complexity(fit, make, sizes=(4000, 8000, 16000, 32000), repeats=1)
    assert est.space_exponent < 1.2, est.summary()


@pytest.mark.complexity
def test_time_is_linear_in_n():
    def make(n, rng):
        return rng.normal(size=(n, 3))

    def fit(X):
        return KMeans(n_clusters=4, init=X[:4].copy(), max_iter=10, tol=0.0).fit(X)

    est = estimate_complexity(fit, make, sizes=(5000, 10000, 20000, 40000), repeats=3)
    assert est.time_exponent < 1.3, est.summary()
