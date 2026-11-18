"""AgglomerativeClustering: SciPy linkage parity, sklearn labels (ARI), edge cases.

Merge order is compared only where it is unique (all merge heights distinct);
labels are compared with the adjusted Rand index, which ignores label permutations.
"""

import numpy as np
import pytest

import lipipy.ml.cluster.hierarchical as hierarchical
from lipipy.bench import estimate_complexity
from lipipy.core import NotFittedError, ResourceLimitError
from lipipy.ml import AgglomerativeClustering, adjusted_rand_score

hierarchy = pytest.importorskip("scipy.cluster.hierarchy")
LINKAGES = ["ward", "complete", "average", "single"]


def make_data(seed, n=90, d=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    X[: n // 3] += 5.0
    X[n // 3 : 2 * n // 3] -= 5.0
    return X


def heights_distinct(heights, rel=1e-9):
    gaps = np.diff(np.sort(heights))
    return bool(np.all(gaps > rel * max(1.0, heights.max())))


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("linkage", LINKAGES)
def test_linkage_matrix_matches_scipy(seed, linkage):
    X = make_data(seed)
    Z = AgglomerativeClustering(n_clusters=3, linkage=linkage).fit(X).linkage_matrix()
    Z_ref = hierarchy.linkage(X, method=linkage)
    np.testing.assert_allclose(Z[:, 2], Z_ref[:, 2], rtol=1e-12, atol=0)
    np.testing.assert_array_equal(Z[:, 3], Z_ref[:, 3])
    assert np.all(np.diff(Z[:, 2]) >= 0)
    if heights_distinct(Z_ref[:, 2]):
        # Unique merge order. Heights from SciPy's scalar C code can differ from ours
        # in the last bit, which can reorder merges whose heights agree to ~1e-15.
        np.testing.assert_array_equal(Z[:, :2], Z_ref[:, :2])


def test_distinct_heights_case_is_exercised():
    count = sum(
        heights_distinct(hierarchy.linkage(make_data(s), method=m)[:, 2])
        for s in range(12)
        for m in LINKAGES
    )
    assert count >= 24


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("linkage", LINKAGES)
@pytest.mark.parametrize("n_clusters", [1, 2, 3, 7])
def test_labels_match_sklearn(seed, linkage, n_clusters):
    sk_cluster = pytest.importorskip("sklearn.cluster")
    X = make_data(seed)
    ours = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit(X)
    ref = sk_cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit(X)
    assert adjusted_rand_score(ours.labels_, ref.labels_) == 1.0
    assert ours.n_clusters_ == len(np.unique(ours.labels_)) == n_clusters
    assert ours.labels_[0] == 0  # numbered by smallest member


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("linkage", LINKAGES)
def test_distance_threshold_matches_sklearn(seed, linkage):
    sk_cluster = pytest.importorskip("sklearn.cluster")
    X = make_data(seed)
    t = float(np.median(hierarchy.linkage(X, method=linkage)[:, 2])) * 3
    ours = AgglomerativeClustering(n_clusters=None, distance_threshold=t, linkage=linkage).fit(X)
    ref = sk_cluster.AgglomerativeClustering(
        n_clusters=None, distance_threshold=t, linkage=linkage
    ).fit(X)
    assert ours.n_clusters_ == ref.n_clusters_
    assert adjusted_rand_score(ours.labels_, ref.labels_) == 1.0
    np.testing.assert_allclose(ours.distances_, ref.distances_, rtol=1e-12)


@pytest.mark.parametrize("linkage", LINKAGES)
def test_linkage_matrix_is_valid_for_scipy_tools(linkage):
    X = make_data(4)
    model = AgglomerativeClustering(n_clusters=4, linkage=linkage).fit(X)
    Z = model.linkage_matrix()
    assert hierarchy.is_valid_linkage(Z)
    flat = hierarchy.fcluster(Z, 4, criterion="maxclust")
    assert adjusted_rand_score(flat, model.labels_) == 1.0
    Z[0, 2] = -1.0  # a copy: the model is unaffected
    assert model.linkage_matrix()[0, 2] >= 0


def test_brute_force_single_and_complete_heights():
    """Recompute linkage heights from the definitions on a tiny problem."""
    X = make_data(7, n=12)
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    for linkage, agg in (("single", np.min), ("complete", np.max), ("average", np.mean)):
        model = AgglomerativeClustering(n_clusters=1, linkage=linkage).fit(X)
        members = {i: [i] for i in range(12)}
        for i, (a, b) in enumerate(model.children_):
            height = agg(D[np.ix_(members[a], members[b])])
            assert model.distances_[i] == pytest.approx(height, rel=1e-12)
            members[12 + i] = members[a] + members[b]


def test_duplicates_and_tiny_inputs():
    X = np.array([[1.0, 1.0]] * 4 + [[5.0, 5.0]] * 3)
    model = AgglomerativeClustering(n_clusters=2).fit(X)
    assert model.labels_.tolist() == [0, 0, 0, 0, 1, 1, 1]
    assert np.count_nonzero(model.distances_ == 0) == 5
    one = AgglomerativeClustering(n_clusters=1).fit([[3.0]])
    assert one.labels_.tolist() == [0] and one.children_.shape == (0, 2)
    assert AgglomerativeClustering(n_clusters=3).fit_predict(X[:3]).tolist() == [0, 1, 2]


def test_resource_guard_raises_before_allocating(monkeypatch):
    monkeypatch.setattr(hierarchical, "MAX_SAMPLES", 50)
    X = make_data(0, n=51)
    model = AgglomerativeClustering()
    with pytest.raises(ResourceLimitError, match="MAX_SAMPLES"):
        model.fit(X)
    assert not hasattr(model, "labels_")
    AgglomerativeClustering().fit(X[:50])


def test_default_limit_is_documented():
    assert hierarchical.MAX_SAMPLES == 10_000
    assert "10 000" in hierarchical.__doc__


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"linkage": "centroid"}, "linkage"),
        ({"n_clusters": None}, "exactly one"),
        ({"n_clusters": 2, "distance_threshold": 1.0}, "exactly one"),
        ({"n_clusters": 0}, "n_clusters"),
        ({"n_clusters": 100}, "n_clusters"),
        ({"n_clusters": None, "distance_threshold": -1.0}, "distance_threshold"),
    ],
)
def test_bad_params(params, match):
    with pytest.raises(ValueError, match=match):
        AgglomerativeClustering(**params).fit(make_data(0))


def test_unfitted():
    with pytest.raises(NotFittedError):
        AgglomerativeClustering().linkage_matrix()


def test_memory_is_quadratic_as_declared():
    est = estimate_complexity(
        lambda X: AgglomerativeClustering(linkage="average").fit(X),
        lambda n, rng: rng.normal(size=(n, 2)),
        sizes=(250, 500, 1000),
        repeats=1,
    )
    assert 1.7 < est.space_exponent < 2.3, est.summary()
