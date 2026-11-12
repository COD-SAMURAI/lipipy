"""CART trees: exact scikit-learn parity without ties, optimality, edge cases, errors."""

import numpy as np
import pytest

from lipipy.bench import estimate_complexity
from lipipy.core import NotFittedError
from lipipy.ml import DecisionTreeClassifier, DecisionTreeRegressor, Tree
from tests.ml._tree_helpers import check_splits_are_optimal, impurity, node_samples


def f32(X):
    """scikit-learn trees work in float32: give both libraries the same numbers."""
    return np.asarray(X, dtype=np.float32).astype(np.float64)


def make_data(seed, n=300, d=5):
    rng = np.random.default_rng(seed)
    X = f32(rng.normal(size=(n, d)))
    y_reg = np.sin(2 * X[:, 0]) + X[:, 1] ** 2 + 0.3 * rng.normal(size=n)
    y_clf = (X[:, 0] + X[:, 1] * X[:, 2] + 0.5 * rng.normal(size=n) > 0).astype(int)
    y_clf = y_clf + (X[:, 3] > 1)
    return X, y_reg, y_clf


def fit_pair(criterion, X, y, **params):
    sk_tree = pytest.importorskip("sklearn.tree")
    if criterion == "squared_error":
        ours = DecisionTreeRegressor(**params).fit(X, y)
        ref = sk_tree.DecisionTreeRegressor(random_state=0, **params).fit(X, y)
    else:
        ours = DecisionTreeClassifier(criterion=criterion, **params).fit(X, y)
        ref = sk_tree.DecisionTreeClassifier(criterion=criterion, random_state=0, **params)
        ref.fit(X, y)
    return ours, ref


def assert_same_tree(ours, ref, X_test):
    t, r = ours.tree_, ref.tree_
    assert t.node_count == r.node_count
    assert ours.get_depth() == ref.get_depth()
    assert ours.get_n_leaves() == ref.get_n_leaves()
    np.testing.assert_array_equal(t.children_left, r.children_left)
    np.testing.assert_array_equal(t.children_right, r.children_right)
    np.testing.assert_array_equal(t.feature, r.feature)
    np.testing.assert_array_equal(t.threshold, r.threshold)
    np.testing.assert_array_equal(t.n_node_samples, r.n_node_samples)
    np.testing.assert_allclose(t.impurity, r.impurity, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ours.feature_importances_, ref.feature_importances_, atol=1e-12)
    np.testing.assert_array_equal(ours.apply(X_test), ref.apply(X_test.astype(np.float32)))
    if isinstance(ours, DecisionTreeRegressor):
        np.testing.assert_allclose(t.value, r.value[:, 0, 0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(ours.predict(X_test), ref.predict(X_test), atol=1e-12)
    else:
        np.testing.assert_allclose(t.value, r.value[:, 0, :], atol=1e-12)
        proba_ref = ref.predict_proba(X_test)
        np.testing.assert_allclose(ours.predict_proba(X_test), proba_ref, atol=1e-12)
        np.testing.assert_array_equal(ours.predict(X_test), ref.predict(X_test))


DEPTH_LIMITED = [
    {"max_depth": 2},
    {"max_depth": 4, "min_samples_leaf": 5},
    {"max_depth": 4, "min_samples_leaf": 10},
    {"max_depth": 6, "min_samples_split": 40},
    {"max_depth": 5, "min_impurity_decrease": 0.01},
    {"max_depth": 4, "min_samples_leaf": 0.03, "min_samples_split": 0.1},
]
# Fully grown classification trees always end in tied small nodes; regression ones
# (continuous targets) do not once leaves hold a few samples.
FULLY_GROWN = [{"min_samples_leaf": 5}, {"min_samples_leaf": 10}, {"min_samples_split": 30}]
CASES = [(c, p) for c in ("squared_error", "gini", "entropy") for p in DEPTH_LIMITED] + [
    ("squared_error", p) for p in FULLY_GROWN
]


@pytest.mark.parametrize("criterion, params", CASES)
def test_parity_with_sklearn_without_ties(criterion, params):
    """Identical trees whenever no node has two splits of equal gain.

    Ties (common in small classification nodes, where the gain depends only on
    counts) make the split choice non-unique (lesson e): those trees are checked by
    the optimality test below instead. Every split must be optimal either way.
    """
    compared = 0
    for seed in range(10):
        X, y_reg, y_clf = make_data(seed)
        y = y_reg if criterion == "squared_error" else y_clf
        ours, ref = fit_pair(criterion, X, y, **params)
        msl = params.get("min_samples_leaf", 1)
        msl = int(np.ceil(msl * len(y))) if isinstance(msl, float) else msl
        if check_splits_are_optimal(ours.tree_, X, y, criterion, msl):
            X_test = f32(np.random.default_rng(100 + seed).normal(size=(200, X.shape[1])))
            assert_same_tree(ours, ref, X_test)
            compared += 1
    assert compared >= 2, "too few tie-free trees to compare: choose other data"


@pytest.mark.parametrize("criterion", ["squared_error", "gini", "entropy"])
def test_fully_grown_tree_is_optimal_and_fits_training_data(criterion):
    """With ties (lesson e): every split maximises the gain, leaves are pure."""
    for seed in range(4):
        X, y_reg, y_clf = make_data(seed, n=150)
        y = y_reg if criterion == "squared_error" else y_clf
        ours, ref = fit_pair(criterion, X, y)
        check_splits_are_optimal(ours.tree_, X, y, criterion)
        assert np.all(ours.tree_.impurity[ours.tree_.children_left == -1] <= 1e-12)
        if criterion == "squared_error":
            np.testing.assert_allclose(ours.predict(X), y, atol=1e-12)
        else:
            assert ours.score(X, y) == ref.score(X, y) == 1.0


def test_tied_feature_values_and_ties_in_gain():
    """Integer features: many equal values and many tied gains; splits stay optimal."""
    rng = np.random.default_rng(5)
    X = rng.integers(0, 4, size=(200, 4)).astype(float)
    y = (X[:, 0] + X[:, 1] + rng.integers(0, 2, size=200) > 3).astype(int)
    for criterion in ("gini", "entropy"):
        clf = DecisionTreeClassifier(criterion=criterion, min_samples_leaf=3).fit(X, y)
        check_splits_are_optimal(clf.tree_, X, y, criterion, 3)
        # thresholds lie strictly between distinct observed values
        internal = clf.tree_.children_left != -1
        assert np.all(clf.tree_.threshold[internal] % 1 == 0.5)


def test_documented_tie_break_rule():
    """Equal gains: lowest feature index, then smallest threshold."""
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])  # identical columns
    clf = DecisionTreeClassifier(max_depth=1).fit(X, [0, 0, 1, 1])
    assert clf.tree_.feature[0] == 0
    X2 = np.array([[0.0], [1.0], [2.0]])
    reg = DecisionTreeRegressor(max_depth=1).fit(X2, [1.0, 4.0, 1.0])  # both cuts: SSE 4.5
    assert reg.tree_.threshold[0] == 0.5


def test_regression_is_numerically_stable_for_large_offsets():
    """Centred prefix sums: a 1e9 offset in y changes neither the tree nor the impurities."""
    X, y_reg, _ = make_data(3, n=200)
    a = DecisionTreeRegressor(min_samples_leaf=10).fit(X, y_reg)
    b = DecisionTreeRegressor(min_samples_leaf=10).fit(X, y_reg + 1e9)
    np.testing.assert_array_equal(a.tree_.feature, b.tree_.feature)
    np.testing.assert_array_equal(a.tree_.threshold, b.tree_.threshold)
    np.testing.assert_allclose(a.tree_.impurity, b.tree_.impurity, rtol=1e-6, atol=1e-6)


@pytest.mark.filterwarnings("error")
def test_extreme_feature_values():
    X = np.array([[-1.7e308], [-1e300], [1e300], [1.7e308]])
    clf = DecisionTreeClassifier().fit(X, [0, 1, 0, 1])
    thr = clf.tree_.threshold[clf.tree_.children_left != -1]
    assert np.all(np.isfinite(thr))
    assert clf.score(X, [0, 1, 0, 1]) == 1.0
    # adjacent floats: the midpoint rounds to b, so the lower value is used
    a = 1.0
    b = np.nextafter(1.0, 2.0)
    reg = DecisionTreeRegressor().fit([[a], [b]], [0.0, 1.0])
    assert reg.tree_.threshold[0] == a
    assert reg.predict([[a], [b]]).tolist() == [0.0, 1.0]


def test_constant_features_single_class_duplicates_and_tiny_inputs():
    X = np.ones((10, 3))
    reg = DecisionTreeRegressor().fit(X, np.arange(10.0))
    assert reg.tree_.node_count == 1 and reg.get_depth() == 0
    np.testing.assert_allclose(reg.predict(X), 4.5)
    np.testing.assert_array_equal(reg.feature_importances_, np.zeros(3))

    X = np.random.default_rng(0).normal(size=(10, 2))
    clf = DecisionTreeClassifier().fit(X, ["a"] * 10)
    assert clf.get_n_leaves() == 1 and clf.predict(X).tolist() == ["a"] * 10
    np.testing.assert_array_equal(clf.predict_proba(X), np.ones((10, 1)))

    # duplicate rows with conflicting labels cannot be separated: impure leaf
    clf = DecisionTreeClassifier().fit([[0.0], [0.0], [1.0]], [0, 1, 1])
    assert clf.get_n_leaves() == 2
    np.testing.assert_allclose(clf.predict_proba([[0.0]]), [[0.5, 0.5]])

    one = DecisionTreeRegressor().fit([[1.0, 2.0]], [3.0])
    assert one.tree_.node_count == 1 and one.predict([[0.0, 0.0]]).tolist() == [3.0]


def test_n_less_than_d_and_one_sample_per_leaf():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(8, 50))
    y = rng.normal(size=8)
    reg = DecisionTreeRegressor().fit(X, y)
    assert reg.get_n_leaves() == 8
    np.testing.assert_array_equal(reg.tree_.n_node_samples[reg.tree_.children_left == -1], 1)
    check_splits_are_optimal(reg.tree_, X, y, "squared_error")


def test_impurities_and_values_match_definitions():
    X, y_reg, y_clf = make_data(7, n=200)
    for criterion in ("gini", "entropy"):
        clf = DecisionTreeClassifier(criterion=criterion, max_depth=4).fit(X, y_clf)
        for node, idx in enumerate(node_samples(clf.tree_, X)):
            expected = impurity(y_clf[idx], criterion)
            assert clf.tree_.impurity[node] == pytest.approx(expected, abs=1e-12)
            counts = np.bincount(y_clf[idx], minlength=clf.n_classes_)
            np.testing.assert_allclose(clf.tree_.value[node], counts / len(idx), atol=1e-15)
    reg = DecisionTreeRegressor(max_depth=4).fit(X, y_reg)
    for node, idx in enumerate(node_samples(reg.tree_, X)):
        assert reg.tree_.value[node] == pytest.approx(y_reg[idx].mean(), abs=1e-12)
        assert reg.tree_.n_node_samples[node] == len(idx)
    np.testing.assert_allclose(reg.feature_importances_.sum(), 1.0)


def test_max_features_subsampling():
    X, y_reg, y_clf = make_data(2, n=200, d=8)
    a = DecisionTreeClassifier(max_features="sqrt", random_state=3).fit(X, y_clf)
    b = DecisionTreeClassifier(max_features="sqrt", random_state=3).fit(X, y_clf)
    np.testing.assert_array_equal(a.tree_.feature, b.tree_.feature)
    np.testing.assert_array_equal(a.tree_.threshold, b.tree_.threshold)
    trees = {
        tuple(DecisionTreeClassifier(max_features=1, random_state=s).fit(X, y_clf).tree_.feature)
        for s in range(5)
    }
    assert len(trees) > 1
    full = DecisionTreeRegressor(max_features=8, random_state=0).fit(X, y_reg)
    none = DecisionTreeRegressor(max_features=None).fit(X, y_reg)
    np.testing.assert_array_equal(full.tree_.feature, none.tree_.feature)
    np.testing.assert_array_equal(full.tree_.threshold, none.tree_.threshold)
    # constant features are skipped: with one informative feature and mtry = 1, the
    # draw continues past constant features, so the root always splits on feature 2
    Xc = np.zeros((50, 5))
    Xc[:, 2] = np.arange(50)
    yc = (np.arange(50) > 20).astype(int)
    for seed in range(10):
        t = DecisionTreeClassifier(max_features=1, random_state=seed).fit(Xc, yc)
        assert t.tree_.feature[0] == 2 and t.score(Xc, yc) == 1.0


@pytest.mark.parametrize("max_features, expected", [("sqrt", 3), ("log2", 3), (0.5, 5), (4, 4)])
def test_max_features_resolution(max_features, expected):
    from lipipy.ml.tree.cart import _resolve_max_features

    assert _resolve_max_features(max_features, 10) == expected


def test_tree_object_and_apply():
    X, y_reg, _ = make_data(4, n=100)
    reg = DecisionTreeRegressor(max_depth=3).fit(X, y_reg)
    t = reg.tree_
    assert isinstance(t, Tree)
    leaves = reg.apply(X)
    assert np.all(t.children_left[leaves] == -1)
    np.testing.assert_allclose(reg.predict(X), t.value[leaves])
    d = t.to_dict()
    assert set(d) >= {"children_left", "threshold", "value", "max_depth"}
    assert len(d["feature"]) == t.node_count
    with pytest.raises(ValueError):
        t.threshold[0] = 1.0  # read-only arrays


def test_string_labels_and_proba(rng):
    X = rng.normal(size=(120, 3))
    y = np.array(["lo", "mid", "hi"])[np.digitize(X[:, 0], [-0.5, 0.5])]
    clf = DecisionTreeClassifier(max_depth=3).fit(X, y)
    assert clf.classes_.tolist() == ["hi", "lo", "mid"]
    np.testing.assert_allclose(clf.predict_proba(X).sum(axis=1), 1.0)
    assert clf.score(X, y) > 0.95


def test_failed_fit_leaves_previous_state():
    X, y_reg, _ = make_data(0, n=50)
    reg = DecisionTreeRegressor(max_depth=2).fit(X, y_reg)
    tree = reg.tree_
    reg.set_params(min_samples_leaf=0)
    with pytest.raises(ValueError, match="min_samples_leaf"):
        reg.fit(X, y_reg + 1)
    assert reg.tree_ is tree
    fresh = DecisionTreeClassifier(criterion="mse")
    with pytest.raises(ValueError, match="criterion"):
        fresh.fit(X, y_reg > 0)
    assert not hasattr(fresh, "tree_") and not hasattr(fresh, "classes_")


def test_bad_inputs():
    X, y_reg, y_clf = make_data(0, n=40)
    bad = [
        ({"criterion": "gini"}, "criterion"),
        ({"max_depth": 0}, "max_depth"),
        ({"max_depth": 2.5}, "max_depth"),
        ({"min_samples_split": 1}, "min_samples_split"),
        ({"min_samples_split": 1.5}, "min_samples_split"),
        ({"min_samples_leaf": 0}, "min_samples_leaf"),
        ({"min_samples_leaf": 1.0}, "min_samples_leaf"),
        ({"max_features": 0}, "max_features"),
        ({"max_features": 6}, "max_features"),
        ({"max_features": 1.5}, "max_features"),
        ({"max_features": "auto"}, "max_features"),
        ({"min_impurity_decrease": -1.0}, "min_impurity_decrease"),
    ]
    for params, match in bad:
        with pytest.raises(ValueError, match=match):
            DecisionTreeRegressor(**params).fit(X, y_reg)
    with pytest.raises(TypeError, match="max_features"):
        DecisionTreeRegressor(max_features=[1]).fit(X, y_reg)
    with pytest.raises(ValueError, match="NaN"):
        DecisionTreeRegressor().fit(X, np.full(len(X), np.nan))
    with pytest.raises(ValueError, match="samples"):
        DecisionTreeClassifier().fit(X, y_clf[:-1])
    with pytest.raises(NotFittedError):
        DecisionTreeClassifier().predict(X)
    with pytest.raises(NotFittedError):
        DecisionTreeRegressor().get_depth()
    clf = DecisionTreeClassifier().fit(X, y_clf)
    with pytest.raises(ValueError, match="features"):
        clf.predict(X[:, :3])


def _predict_input(n, rng):
    model = DecisionTreeRegressor(max_depth=8).fit(*_train(rng))
    return model, rng.normal(size=(n, 5))


def _train(rng):
    X = rng.normal(size=(2000, 5))
    return X, np.sin(3 * X[:, 0]) + X[:, 1] + 0.1 * rng.normal(size=2000)


def test_predict_memory_is_linear_in_n():
    est = estimate_complexity(
        lambda model, X: model.predict(X),
        _predict_input,
        sizes=(10000, 20000, 40000, 80000),
        repeats=1,
    )
    assert 0.9 < est.space_exponent < 1.1


def test_fit_memory_is_linear_in_n():
    def make(n, rng):
        X = rng.normal(size=(n, 5))
        return X, X[:, 0] + rng.normal(size=n)

    est = estimate_complexity(
        lambda X, y: DecisionTreeRegressor(max_depth=6).fit(X, y),
        make,
        sizes=(4000, 8000, 16000, 32000),
        repeats=1,
    )
    assert est.space_exponent < 1.15


@pytest.mark.complexity
def test_fit_time_is_n_log_n_per_level():
    def make(n, rng):
        X = rng.normal(size=(n, 8))
        return X, (X[:, 0] + X[:, 1] * X[:, 2] + rng.normal(size=n) > 0).astype(int)

    est = estimate_complexity(
        lambda X, y: DecisionTreeClassifier(max_depth=6).fit(X, y),
        make,
        sizes=(20000, 40000, 80000, 160000),
        repeats=3,
    )
    assert est.time_exponent < 1.3  # n log n over this range fits ~1.1
