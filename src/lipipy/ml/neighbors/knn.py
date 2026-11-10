r"""k-nearest neighbours classification and regression (brute force, chunked).

Model
-----
For a query :math:`x`, let :math:`N_k(x)` be the indices of the ``k`` training
points closest to :math:`x` under a metric :math:`\rho`:

* ``"euclidean"``: :math:`\rho(x, t) = \sqrt{\sum_j (x_j - t_j)^2}`
* ``"manhattan"``: :math:`\rho(x, t) = \sum_j |x_j - t_j|`

Each neighbour gets a weight :math:`w_i` = 1 (``weights="uniform"``) or
:math:`1/\rho(x, t_i)` (``weights="distance"``). If some neighbours are at
distance exactly 0, those get weight 1 and all others weight 0 (as in
scikit-learn), so an exact match decides the prediction.

* Classification: :math:`P(c \mid x) = \sum_{i\in N_k} w_i [y_i = c] / \sum_i w_i`
  and the prediction is :math:`\arg\max_c P(c\mid x)`.
* Regression: :math:`\hat y = \sum_{i\in N_k} w_i y_i / \sum_i w_i`.

Algorithm and memory
--------------------
Queries are processed in chunks of ``B = chunk_size`` rows. For each chunk we
build the ``B x n_train`` distance block one feature at a time
(:math:`D \mathrel{+}= |x_j - t_j|^p`), so no ``B x n_train x d`` temporary is
created and the distances are computed exactly (no
:math:`\lVert x\rVert^2 - 2x^\top t + \lVert t\rVert^2` cancellation). The ``k``
smallest entries per row are selected in O(n_train) with ``np.partition``. Extra
memory is therefore O(B · n_train), independent of the number of queries;
the full ``n_test x n_train`` matrix is never materialised.

Ties
----
When several training points are at the same distance as the k-th neighbour,
the ones with the **lowest training index** are kept (scikit-learn's brute-force
search does the same). Neighbours are returned sorted by distance, then index.
When classes tie in the vote, the class that sorts first in ``classes_`` wins.

Reference: Cover & Hart (1967), *Nearest neighbor pattern classification*, IEEE
Trans. Information Theory 13(1); Hastie, Tibshirani & Friedman, *ESL* 2e, §13.3.
"""

from __future__ import annotations

import numbers

import numpy as np

from lipipy.core.base import BaseEstimator, RegressorMixin
from lipipy.core.complexity import Complexity
from lipipy.core.registry import register
from lipipy.core.validation import check_X_y
from lipipy.ml._base import ClassifierMixin, check_n_features, check_X_labels

_WEIGHTS = ("uniform", "distance")
_METRICS = ("euclidean", "manhattan")
#: Default distance-block budget when ``chunk_size=None`` (float64 entries, ~16 MiB).
_DEFAULT_BLOCK_ELEMENTS = 2**21

_REFERENCES = (
    "Cover & Hart (1967), Nearest neighbor pattern classification, "
    "IEEE Transactions on Information Theory 13(1)",
    "Hastie, Tibshirani & Friedman, The Elements of Statistical Learning, 2nd ed., §13.3",
)
_COMPLEXITY = Complexity(
    time="O(n^2 d)",
    space="O(B n)",
    notes=(
        "predict on n queries against n training points: O(n_test n_train d) time; "
        "extra memory O(B n_train) for B = chunk_size, independent of n_test; fit is O(1)"
    ),
)


def _pairwise_block(Q: np.ndarray, T: np.ndarray, metric: str) -> np.ndarray:
    """Exact distances between rows of ``Q`` (B, d) and ``T`` (n, d), one feature at a time."""
    D = np.zeros((Q.shape[0], T.shape[0]))
    for j in range(Q.shape[1]):
        diff = Q[:, j, None] - T[None, :, j]
        if metric == "euclidean":
            D += diff * diff
        else:
            D += np.abs(diff)
    if metric == "euclidean":
        np.sqrt(D, out=D)
    return D


def _k_smallest(D: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Per row, the ``k`` smallest entries of ``D``; ties broken by lowest column index.

    Returns ``(distances, indices)``, each of shape (rows, k), sorted by
    (distance, index). Time O(rows · n), extra space O(rows · n).
    """
    rows, n = D.shape
    if k == n:
        idx = np.broadcast_to(np.arange(n), (rows, n))
    else:
        kth = np.partition(D, k - 1, axis=1)[:, k - 1 : k]  # k-th smallest value per row
        below = D < kth
        n_below = below.sum(axis=1, keepdims=True)
        # Among entries equal to the k-th value keep the lowest-index ones.
        at = D == kth
        take_at = at & (np.cumsum(at, axis=1) <= k - n_below)
        chosen = below | take_at
        idx = np.nonzero(chosen)[1].reshape(rows, k)  # row-major: ascending index
    dist = np.take_along_axis(D, idx, axis=1)
    order = np.argsort(dist, axis=1, kind="stable")  # stable keeps index order on ties
    return np.take_along_axis(dist, order, axis=1), np.take_along_axis(idx, order, axis=1)


def _neighbour_weights(dist: np.ndarray, weights: str) -> np.ndarray:
    """Weights per neighbour; exact-zero distances take all the weight in their row."""
    if weights == "uniform":
        return np.ones_like(dist)
    w = np.empty_like(dist)
    zero = dist == 0.0
    has_zero = zero.any(axis=1)
    w[has_zero] = zero[has_zero]
    with np.errstate(divide="ignore"):
        w[~has_zero] = 1.0 / dist[~has_zero]
    return w


class _KNeighborsBase(BaseEstimator):
    """Shared validation, storage and chunked neighbour search."""

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        metric: str = "euclidean",
        chunk_size: int | None = None,
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.chunk_size = chunk_size

    def _validate_params(self, n_train: int) -> None:
        k = self.n_neighbors
        if isinstance(k, bool) or not isinstance(k, numbers.Integral) or k < 1:
            raise ValueError(f"n_neighbors must be a positive integer, got {k!r}.")
        if k > n_train:
            raise ValueError(f"n_neighbors={k} is larger than the number of samples ({n_train}).")
        if self.weights not in _WEIGHTS:
            raise ValueError(f"weights must be one of {_WEIGHTS}, got {self.weights!r}.")
        if self.metric not in _METRICS:
            raise ValueError(f"metric must be one of {_METRICS}, got {self.metric!r}.")
        c = self.chunk_size
        if c is not None and (isinstance(c, bool) or not isinstance(c, numbers.Integral) or c < 1):
            raise ValueError(f"chunk_size must be None or a positive integer, got {c!r}.")

    def _store(self, X: np.ndarray, y: np.ndarray) -> None:
        self._validate_params(X.shape[0])
        self._fit_X = X.copy()  # later edits to the caller's array must not change the model
        self._fit_y = y.copy()
        self.n_features_in_ = X.shape[1]
        self.n_samples_fit_ = X.shape[0]

    def _rows_per_chunk(self) -> int:
        if self.chunk_size is not None:
            return int(self.chunk_size)
        return max(1, _DEFAULT_BLOCK_ELEMENTS // self.n_samples_fit_)

    def kneighbors(
        self, X, n_neighbors: int | None = None, return_distance: bool = True
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Find the nearest training points of each row of ``X``.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
        n_neighbors : int, optional
            Defaults to the ``n_neighbors`` hyperparameter.
        return_distance : bool, default=True

        Returns
        -------
        distances : ndarray of shape (n_queries, k)
            Only if ``return_distance``; sorted ascending.
        indices : ndarray of shape (n_queries, k)
            Indices into the training set.
        """
        self._check_is_fitted("_fit_X")
        X = check_n_features(X, self.n_features_in_, self)
        k = self.n_neighbors if n_neighbors is None else n_neighbors
        if (
            isinstance(k, bool)
            or not isinstance(k, numbers.Integral)
            or not 1 <= k <= len(self._fit_X)
        ):
            raise ValueError(f"n_neighbors must be an integer in [1, {len(self._fit_X)}], got {k}.")
        dist = np.empty((X.shape[0], k))
        idx = np.empty((X.shape[0], k), dtype=np.intp)
        step = self._rows_per_chunk()
        for start in range(0, X.shape[0], step):
            stop = start + step
            block = _pairwise_block(X[start:stop], self._fit_X, self.metric)
            dist[start:stop], idx[start:stop] = _k_smallest(block, k)
        return (dist, idx) if return_distance else idx


@register(
    domain="ml",
    tasks=("classification",),
    complexity=_COMPLEXITY,
    references=_REFERENCES,
    status="experimental",
)
class KNeighborsClassifier(ClassifierMixin, _KNeighborsBase):
    """k-nearest neighbours classifier (brute force with chunked distances).

    Predicts the (weighted) majority class among the ``k`` closest training
    points; see the module docstring for the weighting, tie and memory rules.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbours ``k`` (at most the number of training samples).
    weights : {"uniform", "distance"}, default="uniform"
        Equal votes, or votes weighted by inverse distance (exact matches win).
    metric : {"euclidean", "manhattan"}, default="euclidean"
        Distance between samples.
    chunk_size : int or None, default=None
        Number of query rows per distance block. ``None`` picks
        ``max(1, 2**21 // n_train)`` rows (about 16 MiB of float64 distances).

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Sorted class labels.
    n_features_in_ : int
    n_samples_fit_ : int

    Complexity
    ----------
    ``fit``: O(n d) to validate and store. ``predict``/``predict_proba`` on
    ``n_test`` queries: time O(n_test · n_train · d), extra memory
    O(chunk_size · n_train), independent of ``n_test``.

    References
    ----------
    Cover, T. & Hart, P. (1967). Nearest neighbor pattern classification.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.ml import KNeighborsClassifier
    >>> X = np.array([[0.0], [1.0], [10.0], [11.0]])
    >>> y = np.array(["a", "a", "b", "b"])
    >>> clf = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    >>> clf.predict([[0.5], [10.5]]).tolist()
    ['a', 'b']
    >>> clf.predict_proba([[0.5]]).round(3).tolist()
    [[0.667, 0.333]]
    """

    def fit(self, X, y) -> KNeighborsClassifier:
        """Store the training set (``X`` of shape (n, d), labels ``y``); return ``self``."""
        X, y = check_X_labels(X, y)
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self._store(X, y_encoded.ravel())
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities, shape (n_queries, n_classes), columns ordered as ``classes_``."""
        dist, idx = self.kneighbors(X)
        w = _neighbour_weights(dist, self.weights)
        labels = self._fit_y[idx]
        proba = np.zeros((idx.shape[0], len(self.classes_)))
        rows = np.repeat(np.arange(idx.shape[0]), idx.shape[1])
        np.add.at(proba, (rows, labels.ravel()), w.ravel())
        total = proba.sum(axis=1, keepdims=True)
        total[total == 0.0] = 1.0
        return proba / total

    def predict(self, X) -> np.ndarray:
        """Most probable class per query (ties go to the first class in ``classes_``)."""
        self._check_is_fitted("_fit_X")
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


@register(
    domain="ml",
    tasks=("regression",),
    complexity=_COMPLEXITY,
    references=_REFERENCES,
    status="experimental",
)
class KNeighborsRegressor(RegressorMixin, _KNeighborsBase):
    """k-nearest neighbours regressor (brute force with chunked distances).

    Predicts the (weighted) mean target of the ``k`` closest training points;
    see the module docstring for the weighting, tie and memory rules.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbours ``k`` (at most the number of training samples).
    weights : {"uniform", "distance"}, default="uniform"
        Plain mean, or mean weighted by inverse distance (exact matches win).
    metric : {"euclidean", "manhattan"}, default="euclidean"
        Distance between samples.
    chunk_size : int or None, default=None
        Number of query rows per distance block. ``None`` picks
        ``max(1, 2**21 // n_train)`` rows (about 16 MiB of float64 distances).

    Attributes
    ----------
    n_features_in_ : int
    n_samples_fit_ : int

    Complexity
    ----------
    ``fit``: O(n d) to validate and store. ``predict`` on ``n_test`` queries:
    time O(n_test · n_train · d), extra memory O(chunk_size · n_train).

    References
    ----------
    Hastie, Tibshirani & Friedman, *ESL* 2e, §2.3.2 and §13.3.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.ml import KNeighborsRegressor
    >>> X = np.array([[0.0], [1.0], [2.0], [3.0]])
    >>> y = np.array([0.0, 1.0, 4.0, 9.0])
    >>> KNeighborsRegressor(n_neighbors=2).fit(X, y).predict([[1.4]]).tolist()
    [2.5]
    """

    def fit(self, X, y) -> KNeighborsRegressor:
        """Store the training set (``X`` of shape (n, d), targets ``y`` of shape (n,))."""
        X, y = check_X_y(X, y)
        self._store(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        """Weighted mean of the neighbours' targets, shape (n_queries,)."""
        dist, idx = self.kneighbors(X)
        w = _neighbour_weights(dist, self.weights)
        return np.sum(w * self._fit_y[idx], axis=1) / np.sum(w, axis=1)
