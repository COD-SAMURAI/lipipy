r"""Evaluation metrics for classification, regression and clustering.

Classification metrics are built from per-label counts. For a label :math:`\ell`

.. math::
    TP_\ell = \#\{i : y_i = \ell,\ \hat y_i = \ell\}, \quad
    P_\ell = \#\{i : \hat y_i = \ell\}, \quad
    S_\ell = \#\{i : y_i = \ell\},

so precision is :math:`TP_\ell / P_\ell`, recall :math:`TP_\ell / S_\ell` and
:math:`F_1 = 2\,TP_\ell / (P_\ell + S_\ell)` (the harmonic mean of precision and
recall written without dividing by zero first).

``mean_squared_error`` is :func:`lipipy.core.metrics.mean_squared_error`,
re-exported here so all metrics are importable from one place.

Reference: Manning, Raghavan & Schütze, *Introduction to Information Retrieval*,
§8.3; Hubert & Arabie (1985) for the adjusted Rand index.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from lipipy.core.complexity import Complexity
from lipipy.core.metrics import mean_squared_error
from lipipy.core.registry import register

__all__ = [
    "PrecisionRecallF1",
    "adjusted_rand_score",
    "confusion_matrix",
    "mean_absolute_error",
    "mean_squared_error",
    "precision_recall_f1",
]

_AVERAGES = ("binary", "micro", "macro", "weighted", None)


def _check_label_pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(
            f"y_true and y_pred must be 1D, got shapes {y_true.shape} and {y_pred.shape}."
        )
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"y_true has {y_true.shape[0]} samples but y_pred has {y_pred.shape[0]}.")
    if y_true.shape[0] == 0:
        raise ValueError("y_true and y_pred are empty.")
    return y_true, y_pred


def _encode(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Index of each value in ``labels`` (any order), or -1 if it is not a label."""
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    pos = np.clip(np.searchsorted(sorted_labels, values), 0, len(labels) - 1)
    found = sorted_labels[pos] == values
    return np.where(found, order[pos], -1)


def _resolve_labels(y_true, y_pred, labels) -> np.ndarray:
    if labels is None:
        return np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels must be a non-empty 1D sequence.")
    if np.unique(labels).size != labels.size:
        raise ValueError("labels must not contain duplicates.")
    return labels


@register(
    domain="ml",
    tasks=("classification", "evaluation"),
    complexity=Complexity(time="O(n log n + k^2)", space="O(n + k^2)"),
    references=("Manning, Raghavan & Schütze, Introduction to Information Retrieval, §8.3",),
    status="experimental",
)
def confusion_matrix(y_true, y_pred, *, labels=None) -> np.ndarray:
    r"""Confusion matrix :math:`C_{ij} = \#\{s : y_s = \ell_i,\ \hat y_s = \ell_j\}`.

    Rows are true labels and columns predicted labels.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        True and predicted labels (any comparable dtype).
    labels : array-like of shape (k,), optional
        Labels that index the rows and columns, in this order. Samples whose true
        or predicted label is not listed are ignored. Default: the sorted union of
        the labels in ``y_true`` and ``y_pred``.

    Returns
    -------
    ndarray of shape (k, k), dtype int64

    Complexity
    ----------
    Time O(n log n + k^2) (sorting to encode labels), extra space O(n + k^2).

    References
    ----------
    Manning, Raghavan & Schütze, *Introduction to Information Retrieval*, §8.3.

    Examples
    --------
    >>> confusion_matrix([0, 1, 1, 2], [0, 1, 2, 2]).tolist()
    [[1, 0, 0], [0, 1, 1], [0, 0, 1]]
    """
    y_true, y_pred = _check_label_pair(y_true, y_pred)
    labels = _resolve_labels(y_true, y_pred, labels)
    k = len(labels)
    i, j = _encode(y_true, labels), _encode(y_pred, labels)
    keep = (i >= 0) & (j >= 0)
    flat = np.bincount(i[keep] * k + j[keep], minlength=k * k)
    return flat.reshape(k, k).astype(np.int64)


@dataclass(frozen=True)
class PrecisionRecallF1:
    """Result of :func:`precision_recall_f1`.

    For an averaged result the scores are floats and ``support`` is ``None``
    (like scikit-learn). With ``average=None`` they are arrays aligned with
    ``labels`` and ``support`` holds the number of true samples per label.
    """

    precision: float | np.ndarray
    recall: float | np.ndarray
    f1: float | np.ndarray
    support: np.ndarray | None
    labels: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        def plain(v):
            return v.tolist() if isinstance(v, np.ndarray) else v

        return {
            "precision": plain(self.precision),
            "recall": plain(self.recall),
            "f1": plain(self.f1),
            "support": plain(self.support),
            "labels": plain(self.labels),
        }


def _safe_divide(num: np.ndarray, den: np.ndarray, zero_division: float) -> np.ndarray:
    out = np.full(num.shape, zero_division, dtype=np.float64)
    nonzero = den != 0
    out[nonzero] = num[nonzero] / den[nonzero]
    return out


def _nan_average(values: np.ndarray, weights: np.ndarray | None) -> float:
    """Weighted mean ignoring NaN entries; all-zero weights fall back to a plain mean."""
    keep = ~np.isnan(values)
    if not np.any(keep):
        return float("nan")
    values = values[keep]
    if weights is None or np.sum(weights[keep]) == 0:
        return float(np.mean(values))
    return float(np.average(values, weights=weights[keep]))


@register(
    domain="ml",
    tasks=("classification", "evaluation"),
    complexity=Complexity(time="O(n log n + k)", space="O(n + k)"),
    references=(
        "Manning, Raghavan & Schütze, Introduction to Information Retrieval, §8.3",
        "Sokolova & Lapalme (2009), A systematic analysis of performance measures "
        "for classification tasks, Information Processing & Management 45(4)",
    ),
    status="experimental",
)
def precision_recall_f1(
    y_true,
    y_pred,
    *,
    average: str | None = "binary",
    labels=None,
    pos_label: Any = 1,
    zero_division: float = 0.0,
) -> PrecisionRecallF1:
    r"""Precision, recall and :math:`F_1` score, per label or averaged.

    Per label :math:`\ell`: precision :math:`TP_\ell/P_\ell`, recall
    :math:`TP_\ell/S_\ell` and :math:`F_1 = 2TP_\ell/(P_\ell+S_\ell)`, where
    :math:`P_\ell` counts predictions of :math:`\ell` and :math:`S_\ell` (the
    *support*) counts true samples of :math:`\ell`. A ratio whose denominator is
    zero is set to ``zero_division``.

    Averaging over the selected labels:

    * ``"binary"``: only ``pos_label`` (the data may contain at most two labels);
    * ``"micro"``: sum :math:`TP, P, S` over labels, then divide;
    * ``"macro"``: unweighted mean of the per-label scores;
    * ``"weighted"``: mean weighted by support (plain mean if all supports are 0);
    * ``None``: return the per-label arrays.

    Averages ignore ``NaN`` entries (which appear only when ``zero_division=nan``).

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        True and predicted labels.
    average : {"binary", "micro", "macro", "weighted"} or None, default="binary"
        Averaging mode (see above).
    labels : array-like, optional
        Labels to include, in this order (ignored for ``"binary"``). Default: the
        sorted union of labels in ``y_true`` and ``y_pred``.
    pos_label : label, default=1
        Positive class for ``average="binary"``.
    zero_division : {0.0, 1.0, nan}, default=0.0
        Value used when a denominator is zero.

    Returns
    -------
    PrecisionRecallF1
        Frozen dataclass with ``precision``, ``recall``, ``f1``, ``support`` and
        ``labels``.

    Complexity
    ----------
    Time O(n log n + k), extra space O(n + k) for k labels.

    References
    ----------
    Manning, Raghavan & Schütze, *Introduction to Information Retrieval*, §8.3.

    Examples
    --------
    >>> r = precision_recall_f1([0, 1, 1, 0], [0, 1, 0, 1])
    >>> r.precision, r.recall, r.f1
    (0.5, 0.5, 0.5)
    >>> precision_recall_f1([0, 1, 2], [0, 2, 2], average="macro").f1
    0.5555555555555555
    """
    if average not in _AVERAGES:
        raise ValueError(f"average must be one of {_AVERAGES}, got {average!r}.")
    zero_division = float(zero_division)
    if not (zero_division in (0.0, 1.0) or math.isnan(zero_division)):
        raise ValueError(f"zero_division must be 0, 1 or nan, got {zero_division!r}.")
    y_true, y_pred = _check_label_pair(y_true, y_pred)

    if average == "binary":
        present = np.unique(np.concatenate([y_true, y_pred]))
        if len(present) > 2:
            raise ValueError(
                f"Target has {len(present)} labels but average='binary'. Choose "
                "average='micro', 'macro', 'weighted' or None."
            )
        if pos_label not in present.tolist() and len(present) >= 2:
            raise ValueError(
                f"pos_label={pos_label!r} is not a valid label; use one of {present.tolist()}."
            )
        labels = np.asarray([pos_label])
    else:
        labels = _resolve_labels(y_true, y_pred, labels)

    true_idx, pred_idx = _encode(y_true, labels), _encode(y_pred, labels)
    k = len(labels)
    hit = (true_idx == pred_idx) & (true_idx >= 0)
    tp = np.bincount(true_idx[hit], minlength=k).astype(np.float64)
    pred_sum = np.bincount(pred_idx[pred_idx >= 0], minlength=k).astype(np.float64)
    true_sum = np.bincount(true_idx[true_idx >= 0], minlength=k).astype(np.float64)
    support = true_sum.astype(np.int64)

    if average == "micro":
        tp = tp.sum(keepdims=True)
        pred_sum = pred_sum.sum(keepdims=True)
        true_sum = true_sum.sum(keepdims=True)

    precision = _safe_divide(tp, pred_sum, zero_division)
    recall = _safe_divide(tp, true_sum, zero_division)
    f1 = _safe_divide(2.0 * tp, pred_sum + true_sum, zero_division)

    if average is None:
        return PrecisionRecallF1(precision, recall, f1, support, labels)
    weights = support.astype(np.float64) if average == "weighted" else None
    return PrecisionRecallF1(
        _nan_average(precision, weights),
        _nan_average(recall, weights),
        _nan_average(f1, weights),
        None,
        labels,
    )


@register(
    domain="ml",
    tasks=("regression", "evaluation"),
    complexity=Complexity(time="O(n)", space="O(n)"),
    references=("Hastie, Tibshirani & Friedman, The Elements of Statistical Learning, §2.4",),
    status="experimental",
)
def mean_absolute_error(y_true, y_pred) -> float:
    r"""Mean absolute error :math:`\frac{1}{n}\sum_i |y_i - \hat y_i|`.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        Targets and predictions.

    Returns
    -------
    float

    Complexity
    ----------
    Time O(n), extra space O(n) for the residuals.

    References
    ----------
    Hastie, Tibshirani & Friedman, *ESL* 2e, §2.4.

    Examples
    --------
    >>> mean_absolute_error([1.0, 2.0, 3.0], [1.0, 1.0, 5.0])
    1.0
    """
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shapes differ: y_true {y_true.shape}, y_pred {y_pred.shape}.")
    if y_true.size == 0:
        raise ValueError("y_true and y_pred are empty.")
    return float(np.mean(np.abs(y_true - y_pred)))


@register(
    domain="ml",
    tasks=("clustering", "evaluation"),
    complexity=Complexity(time="O(n log n)", space="O(n)"),
    references=(
        "Hubert & Arabie (1985), Comparing partitions, Journal of Classification 2(1)",
        "Steinley (2004), Properties of the Hubert-Arabie adjusted Rand index, "
        "Psychological Methods 9(3)",
    ),
    status="experimental",
)
def adjusted_rand_score(labels_true, labels_pred) -> float:
    r"""Adjusted Rand index between two partitions (1 = identical, ~0 = random).

    Count pairs of samples: :math:`TP` pairs together in both partitions,
    :math:`TN` apart in both, :math:`FP`/:math:`FN` together in only one. With
    contingency counts :math:`n_{ij}` and marginals :math:`a_i, b_j`,

    .. math::
        \textstyle Q = \sum_{ij} n_{ij}^2, \quad
        TP = Q - n,\ FN = \sum_i a_i^2 - Q,\ FP = \sum_j b_j^2 - Q,\
        TN = n^2 - TP - FP - FN - n,

    .. math::
        \mathrm{ARI} = \frac{2\,(TP\cdot TN - FN\cdot FP)}
        {(TP+FN)(FN+TN) + (TP+FP)(FP+TN)} ,

    and ARI = 1 when :math:`FP = FN = 0` (identical partitions, including the
    degenerate all-one-cluster and all-singletons cases).

    Pair counts are exact integers (no floating-point cancellation).
    The contingency table is never stored densely: only its non-zero cells.

    Parameters
    ----------
    labels_true, labels_pred : array-like of shape (n_samples,)
        Two clusterings of the same samples (label values are arbitrary).

    Returns
    -------
    float

    Complexity
    ----------
    Time O(n log n) (sorting), extra space O(n).

    References
    ----------
    Hubert, L. & Arabie, P. (1985). Comparing partitions. *J. Classification*.

    Examples
    --------
    >>> adjusted_rand_score([0, 0, 1, 1], [1, 1, 0, 0])
    1.0
    >>> round(adjusted_rand_score([0, 0, 1, 1], [0, 1, 0, 1]), 3)
    -0.5
    """
    labels_true, labels_pred = _check_label_pair(labels_true, labels_pred)
    n = labels_true.shape[0]
    _, a_codes = np.unique(labels_true, return_inverse=True)
    _, b_codes = np.unique(labels_pred, return_inverse=True)
    pair_codes = a_codes.astype(np.int64) * (int(b_codes.max()) + 1) + b_codes
    _, cell_counts = np.unique(pair_codes, return_counts=True)

    def sum_squares(counts: np.ndarray) -> int:
        # int64 is exact while n^2 < 2^63; the products below use Python ints.
        return int(np.sum(counts.astype(np.int64) ** 2))

    q = sum_squares(cell_counts)
    tp = q - n
    fn = sum_squares(np.bincount(a_codes)) - q
    fp = sum_squares(np.bincount(b_codes)) - q
    tn = n * n - tp - fp - fn - n
    if fn == 0 and fp == 0:
        return 1.0
    return 2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
