r"""AdaBoost for classification with the multi-class SAMME algorithm.

Algorithm (SAMME, Zhu, Zou, Rosset & Hastie 2009)
-------------------------------------------------
Start with uniform sample weights :math:`w_i = 1/n`. For :math:`m = 1, \dots, M`:

1. fit a weak learner :math:`h_m` to the data weighted by :math:`w`;
2. weighted error :math:`\varepsilon_m = \sum_i w_i [h_m(x_i) \ne y_i] / \sum_i w_i`;
3. learner weight
   :math:`\alpha_m = \eta \bigl(\log\frac{1 - \varepsilon_m}{\varepsilon_m} + \log(k - 1)\bigr)`
   with learning rate :math:`\eta`;
4. :math:`w_i \leftarrow w_i \exp(\alpha_m [h_m(x_i) \ne y_i])`, then normalise
   (skipped after the last round).

Stopping rules (scikit-learn's): if :math:`\varepsilon_m = 0` the learner is
kept with :math:`\alpha_m = 1` and boosting stops; if
:math:`\varepsilon_m \ge 1 - 1/k` (no better than chance) the learner is discarded
and boosting stops (an error if it was the first one). Weights are clipped below
at machine epsilon before every fit, and boosting stops early if their sum
overflows.

The ensemble score of class :math:`c` is

.. math::
    f_c(x) = \frac{\sum_m \alpha_m \bigl([h_m(x) = c] - \tfrac{1}{k-1}[h_m(x) \ne c]\bigr)}
    {\sum_m \alpha_m},

``predict`` returns :math:`\arg\max_c f_c` and ``predict_proba`` is
:math:`\operatorname{softmax}(f / (k-1))` (eq. 15 of Zhu et al.). For two classes
``decision_function`` returns the single score :math:`f_1 - f_0` up to the
factor 1/2 used by scikit-learn: :math:`-f_0 + f_1`, positive meaning ``classes_[1]``.

Weak learner
------------
By default a weighted *decision stump* (a CART tree of depth one, Gini
criterion). For every feature the samples are presorted once; prefix sums of
the weighted one-hot labels give, for every split position between two distinct
values, the Gini proxy :math:`\sum_c L_c^2/W_L + \sum_c R_c^2/W_R` (weighted class
totals left and right). The best split maximises it; the threshold is the
midpoint of the two values. Ties (equal proxies in floating point) go to the
lowest feature index, then the smallest threshold. Each leaf predicts its
weighted majority class (lowest class index on ties).

Any other weak learner can be passed as ``estimator``: an unfitted object with
``fit(X, y, sample_weight=...)`` and ``predict(X)``. It is copied
(``copy.deepcopy``) for every round and trained on the class *indices*
``0..k-1``; if it has a ``random_state`` parameter, each copy gets a seed drawn
from ``random_state``.

Comparing with scikit-learn: scikit-learn's trees cast ``X`` to float32 and visit
features in a random order, so identical stumps need float32-representable
inputs without tied candidate splits.

References
----------
Zhu, Zou, Rosset & Hastie (2009), *Multi-class AdaBoost*, Statistics and Its
Interface 2; Freund & Schapire (1997), *A decision-theoretic generalization of
on-line learning and an application to boosting*, JCSS 55; Hastie, Tibshirani &
Friedman, *ESL* 2e, §10.1.
"""

from __future__ import annotations

import copy
import inspect
import numbers
import warnings
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from lipipy.core.base import BaseEstimator
from lipipy.core.complexity import Complexity
from lipipy.core.random import check_random_state
from lipipy.core.registry import register
from lipipy.ml._base import ClassifierMixin, check_n_features, check_X_labels
from lipipy.numerics import softmax

_EPS = float(np.finfo(np.float64).eps)


def _read_only(a: np.ndarray) -> np.ndarray:
    a = np.array(a, copy=True)
    a.flags.writeable = False
    return a


@dataclass(frozen=True)
class DecisionStump:
    """A fitted one-split tree: ``x[feature] <= threshold`` goes left.

    ``feature`` is -1 for a single leaf (no valid split). ``value`` has shape
    (2, k): the weighted class fractions of the left and right leaves (both rows
    equal for a single leaf). Arrays are read-only.
    """

    feature: int
    threshold: float
    value: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Class index of the leaf each row of ``X`` falls in."""
        if self.feature < 0:
            leaf = np.zeros(X.shape[0], dtype=np.intp)
        else:
            leaf = (X[:, self.feature] > self.threshold).astype(np.intp)
        return np.argmax(self.value, axis=1)[leaf]

    def to_dict(self) -> dict:
        return {"feature": self.feature, "threshold": self.threshold, "value": self.value.tolist()}


def _fit_stump(X: np.ndarray, order: np.ndarray, Y: np.ndarray, w: np.ndarray) -> DecisionStump:
    """Best weighted-Gini stump; ``order`` (d, n) presorted indices, ``Y`` one-hot (n, k)."""
    WY = Y * w[:, None]
    total = WY.sum(axis=0)
    W = total.sum()
    best = (-np.inf, -1, 0.0, -1)  # proxy, feature, threshold, position
    for f in range(X.shape[1]):
        idx = order[f]
        xs = X[idx, f]
        valid = xs[:-1] < xs[1:]
        if not valid.any():
            continue
        L = np.cumsum(WY[idx], axis=0)[:-1]  # left totals for "first p+1 samples"
        R = total - L
        wl = L.sum(axis=1)
        wr = W - wl
        with np.errstate(divide="ignore", invalid="ignore"):
            proxy = np.sum(L * L, axis=1) / wl + np.sum(R * R, axis=1) / wr
        proxy = np.where(valid & (wl > 0) & (wr > 0), proxy, -np.inf)
        p = int(np.argmax(proxy))  # first maximum: smallest threshold
        if proxy[p] > best[0]:  # strict: lower feature wins ties
            a, b = xs[p], xs[p + 1]
            thr = a / 2.0 + b / 2.0
            if thr == b:
                thr = a
            best = (proxy[p], f, float(thr), p)
    if best[1] < 0:
        frac = total / W
        return DecisionStump(-1, -2.0, _read_only(np.vstack([frac, frac])))
    f, thr = best[1], best[2]
    left = X[:, f] <= thr
    lw, rw = WY[left].sum(axis=0), WY[~left].sum(axis=0)
    return DecisionStump(f, thr, _read_only(np.vstack([lw / lw.sum(), rw / rw.sum()])))


@register(
    domain="ml",
    tasks=("classification",),
    complexity=Complexity(
        time="O(n d log n + T n d k)",
        space="O(n d + n k)",
        notes=(
            "default stumps: features presorted once, then T rounds of O(n d k) prefix "
            "sums; predict is O(T n) (O(T) stumps per sample)"
        ),
    ),
    references=(
        "Zhu, Zou, Rosset & Hastie (2009), Multi-class AdaBoost, Statistics and Its Interface 2",
        "Freund & Schapire (1997), A decision-theoretic generalization of on-line "
        "learning and an application to boosting, JCSS 55",
        "Hastie, Tibshirani & Friedman, The Elements of Statistical Learning, 2nd ed., §10.1",
    ),
    status="experimental",
)
class AdaBoostClassifier(ClassifierMixin, BaseEstimator):
    r"""AdaBoost (SAMME) on weighted decision stumps or a user-supplied learner.

    See the module docstring for the algorithm, stopping rules and the stump.

    Parameters
    ----------
    estimator : object or None, default=None
        Weak learner template with ``fit(X, y, sample_weight=...)`` and
        ``predict``; ``None`` uses a weighted decision stump.
    n_estimators : int, default=50
        Maximum number of boosting rounds ``M``.
    learning_rate : float, default=1.0
        Shrinkage :math:`\eta > 0` of every learner weight.
    random_state : None, int, SeedSequence or Generator, default=None
        Seeds passed to ``estimator`` copies that have a ``random_state``
        parameter (stumps are deterministic).

    Attributes
    ----------
    classes_ : ndarray of shape (k,)
    n_classes_ : int
    estimators_ : tuple
        Fitted weak learners (``DecisionStump`` by default).
    estimator_weights_ : ndarray of shape (n_estimators,)
        :math:`\alpha_m` (0 for rounds that did not run).
    estimator_errors_ : ndarray of shape (n_estimators,)
        :math:`\varepsilon_m` (1 for rounds that did not run).
    n_features_in_ : int

    Raises
    ------
    ValueError
        If the first weak learner is no better than chance.

    Complexity
    ----------
    ``fit`` with stumps: time O(n d log n) to presort plus O(n d k) per round, so
    O(n d log n + T n d k) for T rounds; extra space O(n d + n k). ``predict``:
    time O(T n), space O(n k).

    References
    ----------
    Zhu, J., Zou, H., Rosset, S. & Hastie, T. (2009). Multi-class AdaBoost.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.ml import AdaBoostClassifier
    >>> X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    >>> y = np.array([0, 0, 1, 1, 0, 0])
    >>> clf = AdaBoostClassifier(n_estimators=3).fit(X, y)
    >>> clf.predict(X).tolist()
    [0, 0, 1, 1, 0, 0]
    >>> [round(float(e), 4) for e in clf.estimator_errors_]
    [0.3333, 0.25, 0.1667]
    """

    def __init__(
        self,
        estimator=None,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        random_state=None,
    ):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

    def _check_params(self) -> None:
        m = self.n_estimators
        if isinstance(m, bool) or not isinstance(m, numbers.Integral) or m < 1:
            raise ValueError(f"n_estimators must be an integer >= 1, got {m!r}.")
        lr = self.learning_rate
        if isinstance(lr, bool) or not isinstance(lr, numbers.Real) or not 0 < lr < np.inf:
            raise ValueError(f"learning_rate must be a finite number > 0, got {lr!r}.")
        est = self.estimator
        if est is not None and not (
            callable(getattr(est, "fit", None)) and callable(getattr(est, "predict", None))
        ):
            raise TypeError("estimator must be None or have fit and predict methods.")

    def _make_learner(self, rng: np.random.Generator):
        learner = copy.deepcopy(self.estimator)
        params = inspect.signature(type(learner).__init__).parameters
        if "random_state" in params and hasattr(learner, "set_params"):
            learner.set_params(random_state=int(rng.integers(np.iinfo(np.int32).max)))
        return learner

    def fit(self, X, y) -> AdaBoostClassifier:
        """Boost on ``X`` (n, d) and labels ``y`` (n,); return ``self``."""
        self._check_params()
        X, y = check_X_labels(X, y)
        classes, y_idx = np.unique(y, return_inverse=True)
        k = len(classes)
        if k < 2:
            raise ValueError(f"AdaBoostClassifier needs at least 2 classes, got {k}.")
        n, M = X.shape[0], int(self.n_estimators)
        rng = check_random_state(self.random_state)
        if self.estimator is None:
            order = np.argsort(X, axis=0, kind="stable").T
            Y = np.eye(k)[y_idx]

        weights = np.zeros(M)
        errors = np.ones(M)
        learners = []
        w = np.full(n, 1.0 / n)
        for m in range(M):
            w = np.maximum(w, _EPS)
            if self.estimator is None:
                learner = _fit_stump(X, order, Y, w)
            else:
                learner = self._make_learner(rng)
                learner.fit(X, y_idx, sample_weight=w.copy())
            incorrect = learner.predict(X) != y_idx
            err = float(np.average(incorrect, weights=w))
            if err <= 0:
                learners.append(learner)
                weights[m], errors[m] = 1.0, 0.0
                break
            if err >= 1.0 - 1.0 / k:
                if not learners:
                    raise ValueError(
                        "The first weak learner is no better than chance "
                        f"(weighted error {err:.4g} >= 1 - 1/k); AdaBoost cannot be fit."
                    )
                break
            alpha = float(self.learning_rate) * (np.log((1.0 - err) / err) + np.log(k - 1.0))
            learners.append(learner)
            weights[m], errors[m] = alpha, err
            if m == M - 1:
                break
            w = np.exp(np.log(w) + alpha * incorrect)
            total = float(w.sum())
            if not np.isfinite(total) or total <= 0:
                warnings.warn(
                    f"Sample weights overflowed after round {m + 1}; boosting stopped early.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break
            w /= total

        self.classes_ = classes
        self.n_classes_ = k
        self.estimators_ = tuple(learners)
        self.estimator_weights_ = weights
        self.estimator_errors_ = errors
        self.n_features_in_ = X.shape[1]
        return self

    def _votes(self, X) -> Iterator[tuple[float, np.ndarray]]:
        """(alpha_m, one-vs-rest SAMME vote matrix (n, k)) for every fitted learner."""
        self._check_is_fitted("estimators_")
        X = check_n_features(X, self.n_features_in_, self)
        k = self.n_classes_
        for alpha, learner in zip(self.estimator_weights_, self.estimators_, strict=False):
            hit = learner.predict(X)[:, None] == np.arange(k)[None, :]
            yield alpha, np.where(hit, alpha, -alpha / (k - 1))

    def _finish(self, score: np.ndarray, norm: float) -> np.ndarray:
        score = score / norm
        if self.n_classes_ == 2:
            return score[:, 1] - score[:, 0]
        return score

    def decision_function(self, X) -> np.ndarray:
        """SAMME scores: (n,) for two classes (> 0 means ``classes_[1]``), else (n, k)."""
        score = sum(v for _, v in self._votes(X))
        return self._finish(score, float(self.estimator_weights_.sum()))

    def staged_decision_function(self, X) -> Iterator[np.ndarray]:
        """``decision_function`` of the first 1, 2, ... learners."""
        score, norm = None, 0.0
        for alpha, v in self._votes(X):
            norm += alpha
            score = v if score is None else score + v
            yield self._finish(score, norm)

    def _label(self, score: np.ndarray) -> np.ndarray:
        if self.n_classes_ == 2:
            return self.classes_[(score > 0).astype(np.intp)]
        return self.classes_[np.argmax(score, axis=1)]

    def predict(self, X) -> np.ndarray:
        """Class with the largest SAMME score."""
        return self._label(self.decision_function(X))

    def staged_predict(self, X) -> Iterator[np.ndarray]:
        """``predict`` of the first 1, 2, ... learners."""
        for score in self.staged_decision_function(X):
            yield self._label(score)

    def _proba(self, score: np.ndarray) -> np.ndarray:
        if self.n_classes_ == 2:
            return softmax(np.stack([-score, score], axis=1) / 2.0, axis=1)
        return softmax(score / (self.n_classes_ - 1), axis=1)

    def predict_proba(self, X) -> np.ndarray:
        """:math:`\\operatorname{softmax}(f/(k-1))`, shape (n, k)."""
        return self._proba(self.decision_function(X))

    def staged_predict_proba(self, X) -> Iterator[np.ndarray]:
        """``predict_proba`` of the first 1, 2, ... learners."""
        for score in self.staged_decision_function(X):
            yield self._proba(score)
