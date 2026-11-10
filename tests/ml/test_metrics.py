"""lipipy.ml.metrics: parity with sklearn.metrics, zero division, label subsets."""

import math

import numpy as np
import pytest

from lipipy.core import registry
from lipipy.ml import (
    PrecisionRecallF1,
    adjusted_rand_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_f1,
)

skm = pytest.importorskip("sklearn.metrics")


@pytest.fixture
def multiclass(rng):
    y_true = rng.integers(4, size=60)
    y_pred = np.where(rng.random(60) < 0.6, y_true, rng.integers(4, size=60))
    return y_true, y_pred


@pytest.mark.filterwarnings("ignore:A single label was found:UserWarning")  # sklearn, labels=[1]
def test_confusion_matrix_parity(multiclass):
    y_true, y_pred = multiclass
    np.testing.assert_array_equal(
        confusion_matrix(y_true, y_pred), skm.confusion_matrix(y_true, y_pred)
    )
    for labels in ([3, 1], [2, 0, 9], [1]):
        np.testing.assert_array_equal(
            confusion_matrix(y_true, y_pred, labels=labels),
            skm.confusion_matrix(y_true, y_pred, labels=labels),
        )


def test_confusion_matrix_strings():
    y_true = ["cat", "dog", "cat", "emu"]
    y_pred = ["dog", "dog", "cat", "cat"]
    np.testing.assert_array_equal(
        confusion_matrix(y_true, y_pred), skm.confusion_matrix(y_true, y_pred)
    )


def _sk_prf(y_true, y_pred, **kw):
    return skm.precision_recall_fscore_support(y_true, y_pred, **kw)


def _assert_prf(ours: PrecisionRecallF1, ref):
    p, r, f, s = ref
    np.testing.assert_allclose(ours.precision, p, rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose(ours.recall, r, rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose(ours.f1, f, rtol=1e-12, equal_nan=True)
    if s is None:
        assert ours.support is None
    else:
        np.testing.assert_array_equal(ours.support, s)


@pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
@pytest.mark.parametrize("labels", [None, [0, 2], [3, 1, 7], [5]])
@pytest.mark.parametrize("zero_division", [0.0, 1.0, np.nan])
def test_prf_multiclass_parity(multiclass, average, labels, zero_division):
    y_true, y_pred = multiclass
    kw = dict(average=average, labels=labels, zero_division=zero_division)
    _assert_prf(precision_recall_f1(y_true, y_pred, **kw), _sk_prf(y_true, y_pred, **kw))


@pytest.mark.parametrize("zero_division", [0.0, 1.0, np.nan])
@pytest.mark.parametrize(
    ("y_true", "y_pred", "pos_label"),
    [
        ([0, 1, 1, 0, 1], [0, 1, 0, 1, 1], 1),
        ([0, 1, 1, 0, 1], [0, 1, 0, 1, 1], 0),
        ([0, 0, 0, 0], [0, 0, 0, 0], 1),  # pos_label absent: everything zero-division
        ([0, 1, 0, 0], [0, 0, 0, 0], 1),  # no positive predictions
        ([0, 0, 0, 0], [0, 1, 0, 0], 1),  # no positive labels
        (["no", "yes", "yes"], ["yes", "yes", "no"], "yes"),
    ],
)
def test_prf_binary_parity(y_true, y_pred, pos_label, zero_division):
    kw = dict(average="binary", pos_label=pos_label, zero_division=zero_division)
    _assert_prf(precision_recall_f1(y_true, y_pred, **kw), _sk_prf(y_true, y_pred, **kw))


@pytest.mark.parametrize("zero_division", [0.0, 1.0, np.nan])
@pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
def test_prf_zero_support_labels(average, zero_division):
    # Label 2 never appears; label 3 is predicted but never true.
    y_true = [0, 1, 1, 0]
    y_pred = [0, 3, 1, 1]
    for labels in ([0, 1, 2, 3], [2], [2, 3]):
        kw = dict(average=average, labels=labels, zero_division=zero_division)
        _assert_prf(precision_recall_f1(y_true, y_pred, **kw), _sk_prf(y_true, y_pred, **kw))


def test_prf_result_object():
    r = precision_recall_f1([0, 1, 2, 2], [0, 2, 2, 2], average=None)
    assert r.labels.tolist() == [0, 1, 2]
    d = r.to_dict()
    assert d["support"] == [1, 1, 2] and d["precision"][0] == 1.0
    with pytest.raises(AttributeError):
        r.f1 = 0.0  # frozen
    assert precision_recall_f1([0, 1], [0, 1]).to_dict()["support"] is None


def test_prf_errors():
    with pytest.raises(ValueError, match="binary"):
        precision_recall_f1([0, 1, 2], [0, 1, 2])
    with pytest.raises(ValueError, match="pos_label"):
        precision_recall_f1(["a", "b"], ["a", "b"], pos_label=1)
    with pytest.raises(ValueError, match="average"):
        precision_recall_f1([0, 1], [0, 1], average="samples")
    with pytest.raises(ValueError, match="zero_division"):
        precision_recall_f1([0, 1], [0, 1], zero_division=0.5)
    with pytest.raises(ValueError, match="samples"):
        precision_recall_f1([0, 1], [0, 1, 1])
    with pytest.raises(ValueError, match="duplicates"):
        precision_recall_f1([0, 1], [0, 1], average="macro", labels=[1, 1])


def test_regression_metrics_parity(rng):
    y, p = rng.normal(size=50), rng.normal(size=50)
    assert mean_absolute_error(y, p) == pytest.approx(skm.mean_absolute_error(y, p), rel=1e-14)
    assert mean_squared_error(y, p) == pytest.approx(skm.mean_squared_error(y, p), rel=1e-14)
    with pytest.raises(ValueError, match="Shapes"):
        mean_absolute_error([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="empty"):
        mean_absolute_error([], [])


@pytest.mark.parametrize("seed", range(5))
def test_adjusted_rand_parity(seed):
    rng = np.random.default_rng(seed)
    a = rng.integers(5, size=200)
    b = np.where(rng.random(200) < 0.5, a, rng.integers(7, size=200))
    assert adjusted_rand_score(a, b) == pytest.approx(skm.adjusted_rand_score(a, b), rel=1e-12)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ([0, 0, 0], [1, 1, 1]),  # one cluster each
        ([0, 1, 2], [5, 6, 7]),  # all singletons
        ([0, 0, 1, 1], [0, 1, 2, 3]),
        ([0, 0, 0, 0], [0, 1, 2, 3]),
        (["x", "x", "y"], [1, 1, 2]),
        ([1], [2]),
    ],
)
def test_adjusted_rand_edge_cases(a, b):
    ours = adjusted_rand_score(a, b)
    ref = skm.adjusted_rand_score(a, b)
    assert ours == pytest.approx(ref) and not math.isnan(ours)


def test_adjusted_rand_large_n_no_overflow():
    n = 200_000
    a = np.arange(n) % 3
    b = np.arange(n) % 2
    assert adjusted_rand_score(a, b) == pytest.approx(skm.adjusted_rand_score(a, b), abs=1e-12)


def test_metrics_registered():
    for name in (
        "ml.confusion_matrix",
        "ml.precision_recall_f1",
        "ml.mean_absolute_error",
        "ml.adjusted_rand_score",
    ):
        assert "evaluation" in registry.get(name).tasks
