"""DBSCAN: parity with sklearn, the documented border rule, chunked memory."""

import numpy as np
import pytest

from lipipy.bench import estimate_complexity
from lipipy.ml import DBSCAN, adjusted_rand_score


def make_data(seed, n=300):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    X[: n // 3] += 5.0
    X[n // 3 : n // 2] *= 0.3
    return X


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("seed", range(15))
@pytest.mark.parametrize(("eps", "min_samples"), [(0.3, 5), (0.5, 4), (0.8, 10), (1.5, 2)])
@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
def test_parity_with_sklearn(seed, eps, min_samples, metric):
    sk_cluster = pytest.importorskip("sklearn.cluster")
    X = make_data(seed)
    ours = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, chunk_size=37).fit(X)
    ref = sk_cluster.DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(X)
    np.testing.assert_array_equal(ours.core_sample_indices_, ref.core_sample_indices_)
    np.testing.assert_array_equal(ours.labels_ == -1, ref.labels_ == -1)  # same noise
    assert adjusted_rand_score(ours.labels_, ref.labels_) == 1.0
    # our border rule is sklearn's processing order, so labels agree exactly
    np.testing.assert_array_equal(ours.labels_, ref.labels_)


def brute_force(X, eps, min_samples):
    """Core points and clusters from the definitions (dense matrix, BFS)."""
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    near = D <= eps
    core = near.sum(1) >= min_samples
    labels = np.full(len(X), -1)
    cluster = 0
    for i in np.flatnonzero(core):
        if labels[i] != -1:
            continue
        stack = [i]
        labels[i] = cluster
        while stack:
            j = stack.pop()
            for nb in np.flatnonzero(near[j] & core):
                if labels[nb] == -1:
                    labels[nb] = cluster
                    stack.append(nb)
        cluster += 1
    for i in np.flatnonzero(~core):  # border: smallest-numbered touching cluster
        touching = labels[near[i] & core]
        if touching.size:
            labels[i] = touching.min()
    return np.flatnonzero(core), labels


@pytest.mark.parametrize("seed", range(10))
def test_matches_brute_force_definition(seed):
    X = make_data(seed, n=120)
    core, labels = brute_force(X, 0.5, 4)
    model = DBSCAN(eps=0.5, min_samples=4, chunk_size=7).fit(X)
    np.testing.assert_array_equal(model.core_sample_indices_, core)
    np.testing.assert_array_equal(model.labels_, labels)


def test_border_point_between_two_clusters_joins_the_lower_numbered():
    # two dense groups; point 8 is within eps of a core point of each but is not core
    X = np.array([[0.0], [0.1], [0.2], [0.3], [2.0], [2.1], [2.2], [2.3], [1.15]])
    model = DBSCAN(eps=0.9, min_samples=4).fit(X)
    assert model.core_sample_indices_.tolist() == list(range(8))
    assert model.labels_.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 0]
    # reversing the data renumbers the clusters, and the rule follows the numbering
    rev = DBSCAN(eps=0.9, min_samples=4).fit(X[::-1])
    assert rev.labels_.tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1]


def test_chunk_size_invariance():
    X = make_data(1)
    ref = DBSCAN(eps=0.5, min_samples=5).fit(X)
    for chunk in (1, 13, 299, 300, 10_000):
        m = DBSCAN(eps=0.5, min_samples=5, chunk_size=chunk).fit(X)
        np.testing.assert_array_equal(m.labels_, ref.labels_)


def test_all_noise_single_cluster_and_duplicates():
    X = np.arange(10, dtype=float)[:, None] * 10
    noise = DBSCAN(eps=1.0, min_samples=2).fit(X)
    assert noise.labels_.tolist() == [-1] * 10 and noise.core_sample_indices_.size == 0
    assert noise.components_.shape == (0, 1)
    one = DBSCAN(eps=100.0, min_samples=1).fit(X)
    assert one.labels_.tolist() == [0] * 10
    dup = DBSCAN(eps=1e-9, min_samples=3).fit(np.zeros((5, 2)))
    assert dup.labels_.tolist() == [0] * 5
    assert DBSCAN(min_samples=1).fit_predict([[0.0], [9.0]]).tolist() == [0, 1]


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"eps": 0.0}, "eps"),
        ({"min_samples": 0}, "min_samples"),
        ({"metric": "cosine"}, "metric"),
        ({"chunk_size": 0}, "chunk_size"),
    ],
)
def test_bad_params(params, match):
    model = DBSCAN(**params)
    with pytest.raises(ValueError, match=match):
        model.fit(make_data(0))
    assert not hasattr(model, "labels_")


def test_memory_is_linear_in_n_for_fixed_chunk():
    """Extra memory O(B n): doubling n with a fixed chunk roughly doubles the peak."""

    def fit(X):
        return DBSCAN(eps=0.3, min_samples=5, chunk_size=64).fit(X)

    est = estimate_complexity(
        fit, lambda n, rng: rng.normal(size=(n, 2)), sizes=(2000, 4000, 8000), repeats=1
    )
    assert est.space_exponent < 1.25, est.summary()


@pytest.mark.complexity
def test_time_is_quadratic_in_n():
    def fit(X):
        return DBSCAN(eps=0.1, min_samples=5, chunk_size=256).fit(X)

    est = estimate_complexity(
        fit, lambda n, rng: rng.normal(size=(n, 2)), sizes=(2000, 4000, 8000), repeats=3
    )
    assert 1.6 < est.time_exponent < 2.4, est.summary()
