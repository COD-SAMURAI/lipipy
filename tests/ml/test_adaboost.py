"""AdaBoostClassifier (SAMME): parity with sklearn on float32-exact, tie-free data.

scikit-learn's trees cast X to float32 and visit features in a random order, so
the data use distinct multiples of 1/64 (exact in float32, no repeated values);
the boosted sample weights then make tied candidate splits practically
impossible. Stump optimality is also checked against a brute-force search.
"""

import numpy as np
import pytest

from lipipy.core import NotFittedError
from lipipy.ml import AdaBoostClassifier
from lipipy.ml.ensemble import DecisionStump
from lipipy.ml.ensemble.adaboost import _fit_stump


def make_data(seed, n=150, d=3, k=2):
    rng = np.random.default_rng(seed)
    X = rng.permutation(64 * 200)[: n * d].reshape(n, d) / 64.0  # distinct, float32-exact
    y = rng.integers(k, size=n)
    X[:, 0] += 40.0 * y  # signal in feature 0, still float32-exact
    assert np.array_equal(X.astype(np.float32).astype(np.float64), X)
    return X, y


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("learning_rate", [1.0, 0.5])
def test_parity_with_sklearn(seed, k, learning_rate):
    sk_ens = pytest.importorskip("sklearn.ensemble")
    X, y = make_data(seed, k=k)
    X_test, _ = make_data(seed + 100, k=k)
    kw = dict(n_estimators=30, learning_rate=learning_rate)
    ours = AdaBoostClassifier(**kw).fit(X, y)
    ref = sk_ens.AdaBoostClassifier(**kw, random_state=0).fit(X, y)
    np.testing.assert_allclose(ours.estimator_weights_, ref.estimator_weights_, rtol=1e-12)
    np.testing.assert_allclose(ours.estimator_errors_, ref.estimator_errors_, rtol=1e-12)
    assert len(ours.estimators_) == len(ref.estimators_)
    for a, b in zip(ours.estimators_, ref.estimators_, strict=True):
        assert a.feature == b.tree_.feature[0]
        assert a.threshold == b.tree_.threshold[0]
    np.testing.assert_array_equal(ours.predict(X_test), ref.predict(X_test))
    np.testing.assert_allclose(
        ours.decision_function(X_test), ref.decision_function(X_test), atol=1e-12
    )
    np.testing.assert_allclose(ours.predict_proba(X_test), ref.predict_proba(X_test), atol=1e-12)
    for a, b in zip(ours.staged_predict(X_test), ref.staged_predict(X_test), strict=True):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(
        ours.staged_predict_proba(X_test), ref.staged_predict_proba(X_test), strict=True
    ):
        np.testing.assert_allclose(a, b, atol=1e-12)


def brute_force_stump_score(X, y, w, k):
    """Best weighted Gini proxy over every split of every feature (no prefix sums)."""
    best = -np.inf
    for f in range(X.shape[1]):
        values = np.unique(X[:, f])
        for a, b in zip(values[:-1], values[1:], strict=True):
            left = X[:, f] <= a / 2 + b / 2
            score = 0.0
            for side in (left, ~left):
                counts = np.bincount(y[side], weights=w[side], minlength=k)
                score += np.sum(counts**2) / counts.sum()
            best = max(best, score)
    return best


@pytest.mark.parametrize("seed", range(10))
def test_stump_is_the_best_weighted_split(seed):
    X, y = make_data(seed, n=60, k=3)
    w = np.random.default_rng(seed).uniform(0.1, 1.0, size=60)
    order = np.argsort(X, axis=0, kind="stable").T
    stump = _fit_stump(X, order, np.eye(3)[y], w)
    left = X[:, stump.feature] <= stump.threshold
    score = sum(
        np.sum(np.bincount(y[s], weights=w[s], minlength=3) ** 2) / w[s].sum()
        for s in (left, ~left)
    )
    assert score == pytest.approx(brute_force_stump_score(X, y, w, 3), rel=1e-12)


def test_samme_weight_update_by_hand():
    X, y = make_data(0, n=40)
    clf = AdaBoostClassifier(n_estimators=2, learning_rate=0.7).fit(X, y)
    stump = clf.estimators_[0]
    wrong = stump.predict(X) != y
    err = wrong.mean()
    assert clf.estimator_errors_[0] == pytest.approx(err)
    assert clf.estimator_weights_[0] == pytest.approx(0.7 * np.log((1 - err) / err))
    w = np.where(wrong, np.exp(clf.estimator_weights_[0]), 1.0)
    w /= w.sum()
    assert clf.estimator_errors_[1] == pytest.approx(
        np.sum(w * (clf.estimators_[1].predict(X) != y))
    )


def test_perfect_first_learner_stops():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    clf = AdaBoostClassifier(n_estimators=10).fit(X, y)
    assert len(clf.estimators_) == 1
    assert clf.estimator_weights_.tolist() == [1.0] + [0.0] * 9
    assert clf.estimator_errors_[0] == 0.0
    assert clf.predict(X).tolist() == [0, 0, 1, 1]


def test_worse_than_chance_first_learner_raises():
    X = np.zeros((6, 2))  # no split possible: the stump predicts one class
    y = np.array([0, 1, 2, 0, 1, 2])
    clf = AdaBoostClassifier()
    with pytest.raises(ValueError, match="no better than chance"):
        clf.fit(X, y)
    assert not hasattr(clf, "estimators_")  # atomic: nothing assigned


def test_user_estimator_matches_default_stump():
    """Any learner with fit(X, y, sample_weight) works; sklearn's depth-1 tree equals our stump."""
    sk_tree = pytest.importorskip("sklearn.tree")
    X, y = make_data(3, k=3)
    ours = AdaBoostClassifier(n_estimators=15).fit(X, y)
    user = AdaBoostClassifier(
        estimator=sk_tree.DecisionTreeClassifier(max_depth=1), n_estimators=15, random_state=0
    ).fit(X, y)
    np.testing.assert_allclose(user.estimator_weights_, ours.estimator_weights_, rtol=1e-12)
    np.testing.assert_array_equal(user.predict(X), ours.predict(X))
    assert all(isinstance(e.random_state, int) for e in user.estimators_)


def test_stumps_handed_out_are_read_only():
    X, y = make_data(1)
    clf = AdaBoostClassifier(n_estimators=5).fit(X, y)
    stump = clf.estimators_[0]
    assert isinstance(stump, DecisionStump)
    with pytest.raises(ValueError):
        stump.value[0, 0] = 1.0
    with pytest.raises(AttributeError):
        stump.feature = 0
    assert stump.to_dict()["feature"] == stump.feature


def test_string_labels_and_score():
    X, y = make_data(2, k=3)
    names = np.array(["a", "b", "c"])[y]
    clf = AdaBoostClassifier(n_estimators=40).fit(X, names)
    ints = AdaBoostClassifier(n_estimators=40).fit(X, y)
    np.testing.assert_array_equal(clf.predict(X), np.array(["a", "b", "c"])[ints.predict(X)])
    assert clf.score(X, names) == ints.score(X, y) > 1 / 3


@pytest.mark.parametrize(
    ("params", "error"),
    [
        ({"n_estimators": 0}, ValueError),
        ({"learning_rate": 0.0}, ValueError),
        ({"learning_rate": np.inf}, ValueError),
        ({"estimator": object()}, TypeError),
    ],
)
def test_bad_params(params, error):
    X, y = make_data(0)
    with pytest.raises(error):
        AdaBoostClassifier(**params).fit(X, y)


def test_unfitted_feature_mismatch_and_one_class():
    X, y = make_data(0)
    with pytest.raises(NotFittedError):
        AdaBoostClassifier().predict(X)
    with pytest.raises(ValueError, match="2 classes"):
        AdaBoostClassifier().fit(X, np.zeros(len(X)))
    clf = AdaBoostClassifier(n_estimators=3).fit(X, y)
    with pytest.raises(ValueError, match="features"):
        clf.predict(X[:, :2])
