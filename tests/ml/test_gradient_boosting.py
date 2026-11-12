"""Gradient boosting: parity with scikit-learn on tie-free data, Newton steps, stability."""

import numpy as np
import pytest

from lipipy.core import NotFittedError
from lipipy.ml import GradientBoostingClassifier, GradientBoostingRegressor
from tests.ml._tree_helpers import check_splits_are_optimal


def make_data(seed, n=300, d=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32).astype(np.float64)  # sklearn uses float32
    y_reg = np.sin(2 * X[:, 0]) + X[:, 1] ** 2 + 0.3 * rng.normal(size=n)
    y_clf = (X[:, 0] + X[:, 1] * X[:, 2] + 0.5 * rng.normal(size=n) > 0).astype(int)
    X_test = rng.normal(size=(100, d)).astype(np.float32).astype(np.float64)
    return X, y_reg, y_clf, X_test


def sk_pair(is_clf, **params):
    ens = pytest.importorskip("sklearn.ensemble")
    # sklearn's trees use squared-error splits for boosting either way (its `criterion`
    # argument is deprecated and ignored since 1.9; older defaults differ: friedman_mse)
    cls = ens.GradientBoostingClassifier if is_clf else ens.GradientBoostingRegressor
    try:
        return cls(criterion="squared_error", random_state=0, **params)
    except TypeError:  # pragma: no cover - a future sklearn without the argument
        return cls(random_state=0, **params)


@pytest.mark.filterwarnings("ignore:The parameter `criterion` is deprecated:FutureWarning")
@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("is_clf", [False, True])
@pytest.mark.parametrize(
    "params",
    [
        {"n_estimators": 40, "min_samples_leaf": 10},
        {"n_estimators": 30, "max_depth": 2, "learning_rate": 0.5, "min_samples_leaf": 8},
        {"n_estimators": 25, "max_depth": 4, "min_samples_leaf": 12},
    ],
)
def test_parity_with_sklearn_on_tie_free_data(is_clf, params):
    """Parity to 1e-8 whenever every stage's tree has unique splits (lesson e).

    Regression residuals are continuous, so with >= 8 samples per leaf no split ties.
    Log-loss residuals y - p take few distinct values in early stages, so the gain
    depends on counts and ties occur in some fits: those are skipped here (every
    split is still checked to be optimal) and covered by the next test.
    """
    compared = 0
    for seed in range(6):
        X, y_reg, y_clf, X_test = make_data(seed)
        y = y_clf if is_clf else y_reg
        ours = (GradientBoostingClassifier if is_clf else GradientBoostingRegressor)(**params)
        ours.fit(X, y)
        ref = sk_pair(is_clf, **params).fit(X, y)
        tie_free = True
        F = np.full(len(y), ours.init_)
        for est in ours.estimators_:
            r = y - (1 / (1 + np.exp(-F)) if is_clf else F)
            tie_free &= check_splits_are_optimal(
                est.tree_, X, r, "squared_error", params["min_samples_leaf"]
            )
            F = F + params.get("learning_rate", 0.1) * est.predict(X)
        np.testing.assert_allclose(ours.decision_function(X) if is_clf else ours.predict(X), F)
        if not tie_free:
            continue
        compared += 1
        if is_clf:
            np.testing.assert_allclose(
                ours.decision_function(X_test), ref.decision_function(X_test), atol=1e-8
            )
            np.testing.assert_allclose(
                ours.predict_proba(X_test), ref.predict_proba(X_test), atol=1e-8
            )
            np.testing.assert_array_equal(ours.predict(X_test), ref.predict(X_test))
            for a, b in zip(
                ours.staged_predict_proba(X_test), ref.staged_predict_proba(X_test), strict=True
            ):
                np.testing.assert_allclose(a, b, atol=1e-8)
        else:
            np.testing.assert_allclose(ours.predict(X_test), ref.predict(X_test), atol=1e-8)
            for a, b in zip(ours.staged_predict(X_test), ref.staged_predict(X_test), strict=True):
                np.testing.assert_allclose(a, b, atol=1e-8)
        np.testing.assert_allclose(ours.train_score_, ref.train_score_, rtol=1e-10)
        np.testing.assert_allclose(ours.feature_importances_, ref.feature_importances_, atol=1e-10)
    assert compared >= (6 if not is_clf else 2)


@pytest.mark.filterwarnings("ignore:The parameter `criterion` is deprecated:FutureWarning")
@pytest.mark.parametrize("is_clf", [False, True])
def test_with_tied_splits_compare_only_training_loss(is_clf):
    """min_samples_leaf=1, depth 5: tied splits make the trees non-unique (lesson e).

    Regression: ties only occur in 2-sample nodes, where every tied candidate cuts off
    the same training rows, so the training trajectory is identical (rtol 1e-10).
    Classification: count ties can choose different rows; we assert every split is
    optimal and the training deviance is within 10% of scikit-learn's (measured <= 5.2%).
    """
    for seed in range(6):
        X, y_reg, y_clf, _ = make_data(seed)
        y = y_clf if is_clf else y_reg
        ours = (GradientBoostingClassifier if is_clf else GradientBoostingRegressor)(
            n_estimators=30, max_depth=5
        ).fit(X, y)
        ref = sk_pair(is_clf, n_estimators=30, max_depth=5).fit(X, y)
        if is_clf:
            np.testing.assert_allclose(ours.train_score_, ref.train_score_, rtol=0.1)
            F = np.full(len(y), ours.init_)
            for est in ours.estimators_[:5]:
                check_splits_are_optimal(est.tree_, X, y - 1 / (1 + np.exp(-F)), "squared_error")
                F = F + 0.1 * est.predict(X)
        else:
            np.testing.assert_allclose(ours.train_score_, ref.train_score_, rtol=1e-10)


def test_initial_predictions_and_newton_leaf_values():
    X, _, y, _ = make_data(1)
    gb = GradientBoostingClassifier(n_estimators=1, learning_rate=0.3).fit(X, y)
    p = y.mean()
    assert gb.init_ == pytest.approx(np.log(p / (1 - p)), rel=1e-14)
    tree = gb.estimators_[0].tree_
    leaves = tree.apply(X)
    r = y - p
    for leaf in np.unique(leaves):
        in_leaf = leaves == leaf
        gamma = r[in_leaf].sum() / (p * (1 - p) * in_leaf.sum())
        assert tree.value[leaf] == pytest.approx(gamma, rel=1e-12)
    np.testing.assert_allclose(gb.decision_function(X), gb.init_ + 0.3 * tree.value[leaves])

    X, y, _, _ = make_data(2)
    reg = GradientBoostingRegressor(n_estimators=1).fit(X, y)
    assert reg.init_ == pytest.approx(y.mean(), rel=1e-14)


def test_staged_predictions_and_monotone_training_loss():
    X, y, _, X_test = make_data(3)
    gb = GradientBoostingRegressor(n_estimators=60, learning_rate=0.2).fit(X, y)
    stages = list(gb.staged_predict(X_test))
    assert len(stages) == 60
    np.testing.assert_array_equal(stages[-1], gb.predict(X_test))
    # squared error with exact leaf means and 0 < nu <= 1 never increases the loss
    assert np.all(np.diff(gb.train_score_) <= 1e-12)
    assert gb.train_score_[0] < np.var(y)
    assert gb.score(X, y) > 0.95


def test_subsample_is_reproducible_and_uses_in_bag_rows():
    X, _, y, X_test = make_data(4)
    a = GradientBoostingClassifier(subsample=0.5, random_state=3).fit(X, y)
    b = GradientBoostingClassifier(subsample=0.5, random_state=3).fit(X, y)
    np.testing.assert_array_equal(a.predict_proba(X_test), b.predict_proba(X_test))
    c = GradientBoostingClassifier(subsample=0.5, random_state=4).fit(X, y)
    assert not np.array_equal(a.predict_proba(X_test), c.predict_proba(X_test))
    assert all(t.tree_.n_node_samples[0] == 150 for t in a.estimators_)
    assert a.score(X, y) > 0.85


@pytest.mark.filterwarnings("error")
def test_separable_classification_stays_finite():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] > 0).astype(int)
    gb = GradientBoostingClassifier(n_estimators=300, learning_rate=1.0).fit(X, y)
    F = gb.decision_function(X)
    # log-odds grow until the residual variance (~(1 - p)^2) drops below machine epsilon
    # and the stage tree becomes a single leaf (scikit-learn's impurity <= eps rule)
    assert np.all(np.isfinite(F)) and np.abs(F).min() > 15
    proba = gb.predict_proba(X)
    assert np.all(np.isfinite(proba)) and np.all((proba >= 0) & (proba <= 1))
    assert np.all(np.isfinite(gb.train_score_)) and gb.score(X, y) == 1.0


@pytest.mark.filterwarnings("error")
def test_extreme_class_imbalance_initial_log_odds():
    X = np.arange(10.0)[:, None]
    y = np.array([0] * 9 + [1])
    gb = GradientBoostingClassifier(n_estimators=5).fit(X, y)
    assert gb.init_ == pytest.approx(np.log(0.1 / 0.9))
    assert gb.predict(X).tolist()[-1] == 1


def test_string_labels_and_class_count_errors():
    X, _, y, _ = make_data(5)
    labels = np.array(["no", "yes"])[y]
    gb = GradientBoostingClassifier(n_estimators=10).fit(X, labels)
    assert set(gb.predict(X)) <= {"no", "yes"}
    for bad in (np.zeros(len(X)), np.arange(len(X)) % 3):
        with pytest.raises(ValueError, match="exactly 2 classes"):
            GradientBoostingClassifier().fit(X, bad)


def test_failed_fit_leaves_previous_state():
    X, y, _, _ = make_data(0)
    gb = GradientBoostingRegressor(n_estimators=3).fit(X, y)
    trees = gb.estimators_
    gb.set_params(max_depth=0)
    with pytest.raises(ValueError, match="max_depth"):
        gb.fit(X, y)
    assert gb.estimators_ is trees


def test_bad_inputs():
    X, y, yc, _ = make_data(0)
    for params, match in [
        ({"loss": "absolute_error"}, "loss"),
        ({"learning_rate": 0}, "learning_rate"),
        ({"learning_rate": np.inf}, "learning_rate"),
        ({"n_estimators": 0}, "n_estimators"),
        ({"subsample": 0.0}, "subsample"),
        ({"subsample": 1.5}, "subsample"),
        ({"min_samples_leaf": 0}, "min_samples_leaf"),
    ]:
        with pytest.raises(ValueError, match=match):
            GradientBoostingRegressor(**params).fit(X, y)
    with pytest.raises(ValueError, match="loss"):
        GradientBoostingClassifier(loss="exponential").fit(X, yc)
    with pytest.raises(NotFittedError):
        GradientBoostingRegressor().predict(X)
    with pytest.raises(NotFittedError):
        GradientBoostingClassifier().predict(X)
    gb = GradientBoostingRegressor(n_estimators=2).fit(X, y)
    with pytest.raises(ValueError, match="features"):
        gb.predict(X[:, :2])
