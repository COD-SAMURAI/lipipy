"""k-nearest neighbours: parity with scikit-learn, tie rules, chunked memory."""

import numpy as np
import pytest

from lipipy.bench import estimate_complexity
from lipipy.core import NotFittedError
from lipipy.ml import KNeighborsClassifier, KNeighborsRegressor


def make_classification(rng, n=150, d=4, k=3):
    centres = rng.normal(scale=2.0, size=(k, d))
    y = rng.integers(k, size=n)
    X = centres[y] + rng.normal(size=(n, d))
    return X, y


@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
@pytest.mark.parametrize("n_neighbors", [1, 4, 7])
def test_classifier_parity(rng, weights, metric, n_neighbors):
    sk = pytest.importorskip("sklearn.neighbors")
    X, y = make_classification(rng)
    X_test = rng.normal(scale=2.0, size=(60, X.shape[1]))  # continuous: no distance ties
    kwargs = dict(n_neighbors=n_neighbors, weights=weights, metric=metric)
    ours = KNeighborsClassifier(**kwargs, chunk_size=7).fit(X, y)
    ref = sk.KNeighborsClassifier(**kwargs, algorithm="brute").fit(X, y)
    np.testing.assert_array_equal(ours.predict(X_test), ref.predict(X_test))
    np.testing.assert_allclose(ours.predict_proba(X_test), ref.predict_proba(X_test), atol=1e-12)
    d_ours, i_ours = ours.kneighbors(X_test)
    d_ref, i_ref = ref.kneighbors(X_test)
    np.testing.assert_array_equal(i_ours, i_ref)
    np.testing.assert_allclose(d_ours, d_ref, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
def test_regressor_parity(rng, weights, metric):
    sk = pytest.importorskip("sklearn.neighbors")
    X = rng.normal(size=(200, 3))
    y = np.sin(X[:, 0]) + X[:, 1] ** 2
    X_test = rng.normal(size=(50, 3))
    kwargs = dict(n_neighbors=5, weights=weights, metric=metric)
    ours = KNeighborsRegressor(**kwargs, chunk_size=16).fit(X, y)
    ref = sk.KNeighborsRegressor(**kwargs, algorithm="brute").fit(X, y)
    np.testing.assert_allclose(ours.predict(X_test), ref.predict(X_test), rtol=1e-12)


def test_string_labels_and_score(rng):
    X, y = make_classification(rng)
    names = np.array(["cat", "dog", "emu"])[y]
    clf = KNeighborsClassifier(n_neighbors=1).fit(X, names)
    assert clf.classes_.tolist() == ["cat", "dog", "emu"]
    assert clf.score(X, names) == 1.0  # each point is its own nearest neighbour


def test_distance_weights_exact_match_takes_all_weight():
    sk = pytest.importorskip("sklearn.neighbors")
    # Integer data: distances are exact, so the query equal to X[2] has distance 0.
    X = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0], [5.0, 5.0], [6.0, 5.0]])
    y = np.array([0, 0, 1, 2, 2])
    yr = np.array([1.0, 2.0, 10.0, 20.0, 100.0])
    query = np.array([[5.0, 5.0], [0.0, 1.0]])
    clf = KNeighborsClassifier(n_neighbors=4, weights="distance").fit(X, y)
    ref = sk.KNeighborsClassifier(n_neighbors=4, weights="distance", algorithm="brute").fit(X, y)
    proba = clf.predict_proba(query)
    # Two exact matches (labels 1 and 2) share the weight; the rest get none.
    np.testing.assert_allclose(proba[0], [0.0, 0.5, 0.5])
    np.testing.assert_allclose(proba, ref.predict_proba(query))
    np.testing.assert_array_equal(clf.predict(query), ref.predict(query))

    reg = KNeighborsRegressor(n_neighbors=4, weights="distance").fit(X, yr)
    sk_reg = sk.KNeighborsRegressor(n_neighbors=4, weights="distance", algorithm="brute")
    np.testing.assert_allclose(reg.predict(query), sk_reg.fit(X, yr).predict(query))
    assert reg.predict(query)[0] == pytest.approx(15.0)  # mean of the two exact matches


def test_distance_ties_keep_lowest_training_index():
    """Documented rule: among equidistant candidates the lowest index wins (like sklearn)."""
    sk = pytest.importorskip("sklearn.neighbors")
    # Query at 0: training points at -1 and +1 (indices 0..3) are all at distance 1.
    X = np.array([[1.0], [-1.0], [1.0], [-1.0], [3.0]])
    y = np.array([0, 1, 1, 0, 1])
    clf = KNeighborsClassifier(n_neighbors=2).fit(X, y)
    dist, idx = clf.kneighbors([[0.0]])
    assert idx.tolist() == [[0, 1]] and dist.tolist() == [[1.0, 1.0]]
    # Vote is 1-1; the class that sorts first wins.
    assert clf.predict([[0.0]]).tolist() == [0]
    ref = sk.KNeighborsClassifier(n_neighbors=2, algorithm="brute").fit(X, y)
    assert ref.predict([[0.0]]).tolist() == [0]
    assert sorted(ref.kneighbors([[0.0]])[1][0].tolist()) == [0, 1]
    np.testing.assert_allclose(clf.predict_proba([[0.0]]), ref.predict_proba([[0.0]]))


def test_chunk_size_does_not_change_results(rng):
    X, y = make_classification(rng)
    X_test = rng.normal(size=(37, X.shape[1]))
    base = KNeighborsClassifier(n_neighbors=5).fit(X, y).predict_proba(X_test)
    for chunk in (1, 5, 36, 37, 1000):
        out = KNeighborsClassifier(n_neighbors=5, chunk_size=chunk).fit(X, y)
        np.testing.assert_array_equal(out.predict_proba(X_test), base)


def test_k_equals_n_and_kneighbors_override(rng):
    X, y = make_classification(rng, n=10)
    clf = KNeighborsClassifier(n_neighbors=10).fit(X, y)
    np.testing.assert_allclose(clf.predict_proba(X[:1])[0], np.bincount(y, minlength=3) / 10)
    idx = clf.kneighbors(X[:2], n_neighbors=1, return_distance=False)
    assert idx.ravel().tolist() == [0, 1]


def test_fit_copies_training_data(rng):
    X, y = make_classification(rng)
    clf = KNeighborsClassifier(n_neighbors=1).fit(X, y)
    before = clf.predict(X)
    X[:] = 0.0
    np.testing.assert_array_equal(clf.predict(clf._fit_X), before)


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"n_neighbors": 0}, "n_neighbors"),
        ({"n_neighbors": 2.5}, "n_neighbors"),
        ({"n_neighbors": 500}, "larger than"),
        ({"weights": "gaussian"}, "weights"),
        ({"metric": "cosine"}, "metric"),
        ({"chunk_size": 0}, "chunk_size"),
    ],
)
def test_bad_params(rng, params, match):
    X, y = make_classification(rng)
    with pytest.raises(ValueError, match=match):
        KNeighborsClassifier(**params).fit(X, y)


def test_bad_inputs(rng):
    X, y = make_classification(rng)
    with pytest.raises(NotFittedError):
        KNeighborsClassifier().predict(X)
    with pytest.raises(NotFittedError):
        KNeighborsRegressor().predict(X)
    clf = KNeighborsClassifier().fit(X, y)
    with pytest.raises(ValueError, match="features"):
        clf.predict(X[:, :2])
    with pytest.raises(ValueError, match="n_neighbors"):
        clf.kneighbors(X, n_neighbors=0)
    with pytest.raises(ValueError, match="samples"):
        KNeighborsClassifier().fit(X, y[:-1])
    with pytest.raises(ValueError, match="1D"):
        KNeighborsClassifier().fit(X, y[:, None])
    with pytest.raises(ValueError, match="NaN"):
        KNeighborsRegressor().fit(X, np.full(len(X), np.nan))


def _knn_memory(chunk_size, sizes=(500, 1000, 2000, 4000), n_train=4000):
    d = 3
    train_rng = np.random.default_rng(1)
    X_train = train_rng.normal(size=(n_train, d))
    y_train = train_rng.integers(3, size=n_train)
    clf = KNeighborsClassifier(n_neighbors=5, chunk_size=chunk_size).fit(X_train, y_train)
    return estimate_complexity(
        clf.predict_proba, lambda n, rng: rng.normal(size=(n, d)), sizes=sizes, repeats=1
    )


def test_memory_is_constant_in_n_test_for_fixed_chunk():
    """Extra memory is O(chunk_size * n_train): it must not grow with n_test."""
    est = _knn_memory(chunk_size=64)
    assert est.space_exponent < 0.25, est.summary()
    # Bounded by a handful of (chunk x n_train) blocks plus the small outputs.
    assert max(est.peak_bytes) < 10 * 64 * 4000 * 8 + 4000 * 200, est.peak_bytes


def test_memory_without_chunking_grows_with_n_test():
    """Control: one chunk covering all queries makes memory O(n_test * n_train)."""
    est = _knn_memory(chunk_size=10**9, sizes=(250, 500, 1000, 2000), n_train=500)
    assert est.space_exponent > 0.8, est.summary()


@pytest.mark.complexity
def test_predict_time_is_linear_in_n_test():
    est = _knn_memory(chunk_size=64, sizes=(1000, 2000, 4000, 8000))
    assert est.time_exponent < 1.3, est.summary()
