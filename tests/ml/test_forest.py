"""Random forests: reproducibility, statistical parity with scikit-learn, OOB, errors."""

import numpy as np
import pytest

from lipipy.core import NotFittedError
from lipipy.ml import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

# Stated statistical-parity margins (lipipy vs scikit-learn, same hyperparameters,
# 400 training / 200 test rows, 50 trees): for every dataset the mean test score
# over 5 seeds differs by at most MEAN_MARGIN, and each single run by at most
# RUN_MARGIN. Accuracy steps are 1/200 = 0.005; measured: mean <= 0.012, run <= 0.025.
MEAN_MARGIN = 0.03
RUN_MARGIN = 0.06


def dataset(kind, seed):
    ds = pytest.importorskip("sklearn.datasets")
    if kind == "classification":
        X, y = ds.make_classification(600, 8, n_informative=4, n_classes=3, random_state=seed)
    elif kind == "moons":
        X, y = ds.make_moons(600, noise=0.3, random_state=seed)
    elif kind == "friedman1":
        X, y = ds.make_friedman1(600, noise=1.0, random_state=seed)
    else:
        X, y = ds.make_regression(600, 6, n_informative=3, noise=10.0, random_state=seed)
    return X[:400], y[:400], X[400:], y[400:]


@pytest.mark.parametrize("kind", ["classification", "moons", "friedman1", "linear"])
def test_statistical_parity_with_sklearn(kind):
    ens = pytest.importorskip("sklearn.ensemble")
    is_clf = kind in ("classification", "moons")
    ours_cls = RandomForestClassifier if is_clf else RandomForestRegressor
    ref_cls = ens.RandomForestClassifier if is_clf else ens.RandomForestRegressor
    diffs, oob_diffs = [], []
    for seed in range(5):
        X, y, X_test, y_test = dataset(kind, seed)
        ours = ours_cls(n_estimators=50, oob_score=True, random_state=seed).fit(X, y)
        ref = ref_cls(n_estimators=50, oob_score=True, random_state=seed).fit(X, y)
        diffs.append(ours.score(X_test, y_test) - ref.score(X_test, y_test))
        oob_diffs.append(ours.oob_score_ - ref.oob_score_)
    assert abs(np.mean(diffs)) <= MEAN_MARGIN and np.max(np.abs(diffs)) <= RUN_MARGIN
    assert abs(np.mean(oob_diffs)) <= MEAN_MARGIN and np.max(np.abs(oob_diffs)) <= RUN_MARGIN


def test_exact_reproducibility_and_independent_tree_streams():
    X, y, X_test, _ = dataset("classification", 0)
    a = RandomForestClassifier(n_estimators=20, random_state=7).fit(X, y)
    b = RandomForestClassifier(n_estimators=20, random_state=7).fit(X, y)
    np.testing.assert_array_equal(a.predict_proba(X_test), b.predict_proba(X_test))
    c = RandomForestClassifier(n_estimators=20, random_state=8).fit(X, y)
    assert not np.array_equal(a.predict_proba(X_test), c.predict_proba(X_test))
    # spawn(): tree b's stream does not depend on how many trees are grown
    short = RandomForestClassifier(n_estimators=5, random_state=7).fit(X, y)
    for t_long, t_short in zip(a.estimators_[:5], short.estimators_, strict=True):
        np.testing.assert_array_equal(t_long.tree_.threshold, t_short.tree_.threshold)
    # trees differ from each other
    assert len({t.tree_.node_count for t in a.estimators_}) > 1


def test_no_randomness_reduces_to_a_single_tree():
    """bootstrap=False and all features: every tree is the deterministic CART tree."""
    X, y, X_test, _ = dataset("friedman1", 1)
    X = X.astype(np.float32).astype(np.float64)
    rf = RandomForestRegressor(
        n_estimators=3, bootstrap=False, max_features=None, min_samples_leaf=5, random_state=0
    ).fit(X, y)
    tree = DecisionTreeRegressor(min_samples_leaf=5).fit(X, y)
    np.testing.assert_allclose(rf.predict(X_test), tree.predict(X_test), rtol=1e-12)
    np.testing.assert_allclose(rf.feature_importances_, tree.feature_importances_, atol=1e-12)
    clf = RandomForestClassifier(n_estimators=2, bootstrap=False, max_features=None).fit(
        X[:, :3], y > 14
    )
    single = DecisionTreeClassifier().fit(X[:, :3], y > 14)
    np.testing.assert_array_equal(clf.predict(X_test[:, :3]), single.predict(X_test[:, :3]))


def test_oob_estimates_test_score():
    X, y, X_test, y_test = dataset("friedman1", 2)
    rf = RandomForestRegressor(n_estimators=60, oob_score=True, random_state=0).fit(X, y)
    assert abs(rf.oob_score_ - rf.score(X_test, y_test)) < 0.08
    assert rf.oob_prediction_.shape == (len(y),)
    # the OOB prediction of a row only uses trees that did not see it
    X, y, X_test, y_test = dataset("moons", 2)
    rf = RandomForestClassifier(n_estimators=60, oob_score=True, random_state=0).fit(X, y)
    assert abs(rf.oob_score_ - rf.score(X_test, y_test)) < 0.08
    np.testing.assert_allclose(rf.oob_decision_function_.sum(axis=1), 1.0)


def test_oob_with_few_trees_warns_and_ignores_unseen_rows():
    X, y, _, _ = dataset("moons", 3)
    with pytest.warns(UserWarning, match="no out-of-bag prediction"):
        rf = RandomForestClassifier(n_estimators=2, oob_score=True, random_state=0).fit(X, y)
    unseen = np.isnan(rf.oob_decision_function_[:, 0])
    assert unseen.any() and np.isfinite(rf.oob_score_)


def test_feature_importances():
    X, y, _, _ = dataset("friedman1", 0)  # only the first 5 of 10 features matter
    rf = RandomForestRegressor(n_estimators=30, random_state=0).fit(X, y)
    np.testing.assert_allclose(rf.feature_importances_.sum(), 1.0)
    assert set(np.argsort(rf.feature_importances_)[-4:]) <= {0, 1, 2, 3, 4}
    manual = np.mean([t.feature_importances_ for t in rf.estimators_], axis=0)
    np.testing.assert_allclose(rf.feature_importances_, manual / manual.sum())


def test_classifier_with_rare_class_and_string_labels():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = np.where(X[:, 0] > 2.0, "rare", np.where(X[:, 1] > 0, "b", "a"))
    rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    proba = rf.predict_proba(X)
    assert proba.shape == (100, len(rf.classes_))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert rf.score(X, y) > 0.95
    apply = rf.apply(X)
    assert apply.shape == (100, 10)


def test_failed_fit_leaves_previous_state():
    X, y, _, _ = dataset("moons", 0)
    rf = RandomForestClassifier(n_estimators=3, random_state=0).fit(X, y)
    trees = rf.estimators_
    rf.set_params(criterion="squared_error")
    with pytest.raises(ValueError, match="criterion"):
        rf.fit(X, y)
    assert rf.estimators_ is trees


def test_bad_inputs():
    X, y, _, _ = dataset("moons", 0)
    for ne in (0, 2.5, True):
        with pytest.raises(ValueError, match="n_estimators"):
            RandomForestClassifier(n_estimators=ne).fit(X, y)
    with pytest.raises(ValueError, match="bootstrap"):
        RandomForestClassifier(bootstrap=False, oob_score=True).fit(X, y)
    with pytest.raises(TypeError, match="bootstrap"):
        RandomForestRegressor(bootstrap="yes").fit(X, y)
    with pytest.raises(ValueError, match="max_features"):
        RandomForestRegressor(max_features=0).fit(X, y)
    with pytest.raises(NotFittedError):
        RandomForestClassifier().predict(X)
    with pytest.raises(NotFittedError):
        RandomForestRegressor().predict(X)
    rf = RandomForestRegressor(n_estimators=2).fit(X, y)
    with pytest.raises(ValueError, match="features"):
        rf.predict(X[:, :1])
