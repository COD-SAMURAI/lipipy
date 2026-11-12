r"""Gradient tree boosting (Friedman 2001) for regression and binary classification.

Model
-----
An additive model of :math:`M` (``n_estimators``) regression trees,

.. math::  F_M(x) = F_0 + \nu \sum_{m=1}^{M} h_m(x),

with learning rate :math:`\nu`. :math:`F` is the prediction itself (squared error)
or the log-odds :math:`\log\frac{P(y=1\mid x)}{P(y=0\mid x)}` (log loss).

Algorithm (Friedman 2001, Algorithms 2 and 5)
---------------------------------------------
1. :math:`F_0` minimises the loss with a constant: the mean :math:`\bar y`
   (squared error) or the log-odds :math:`\log\frac{\bar p}{1-\bar p}` of the
   positive-class frequency :math:`\bar p` (clipped to :math:`[\varepsilon,
   1-\varepsilon]`; log loss).
2. For :math:`m = 1..M`: compute the negative gradient (pseudo-residual)
   :math:`r_i = y_i - F_{m-1}(x_i)` (squared error) or
   :math:`r_i = y_i - \sigma(F_{m-1}(x_i))` (log loss); optionally draw a
   subsample of :math:`\lfloor\text{subsample}\cdot n\rfloor` rows without
   replacement (stochastic gradient boosting, Friedman 2002); fit a regression
   tree of depth ``max_depth`` to the residuals with squared-error splits.
3. Set the value of every leaf :math:`R_j` to a Newton step for the loss:
   the residual mean for squared error (exact), and

   .. math::  \gamma_j = \frac{\sum_{i \in R_j} r_i}{\sum_{i \in R_j} p_i (1 - p_i)},
              \qquad p_i = \sigma(F_{m-1}(x_i)),

   for log loss (0 if the denominator is below ``1e-150``).
4. Update :math:`F_m(x) = F_{m-1}(x) + \nu \gamma_{j(x)}`.

``train_score_[m]`` is the training (in-bag) loss after stage :math:`m`: the mean
squared error, or the binomial deviance :math:`2 \cdot` mean log loss (the
scikit-learn conventions). Probabilities use the stable
:func:`~lipipy.numerics.sigmoid` and the deviance uses
:func:`~lipipy.numerics.log_sigmoid`, so large scores never overflow.

References
----------
J. H. Friedman (2001), *Greedy function approximation: a gradient boosting
machine*, Annals of Statistics 29(5); J. H. Friedman (2002), *Stochastic gradient
boosting*, CSDA 38(4); Hastie, Tibshirani & Friedman, *ESL* 2e, §10.9-10.10.
"""

from __future__ import annotations

import numbers
from collections import deque
from collections.abc import Iterator

import numpy as np

from lipipy.core.base import BaseEstimator, RegressorMixin
from lipipy.core.complexity import Complexity
from lipipy.core.random import check_random_state
from lipipy.core.registry import register
from lipipy.core.validation import check_X_y
from lipipy.ml._base import ClassifierMixin, check_n_features, check_X_labels
from lipipy.ml.tree.cart import DecisionTreeRegressor, Tree
from lipipy.numerics import log_sigmoid, sigmoid

_REFERENCES = (
    "Friedman (2001), Greedy function approximation: a gradient boosting machine, "
    "Annals of Statistics 29(5)",
    "Friedman (2002), Stochastic gradient boosting, Computational Statistics & Data Analysis 38(4)",
    "Hastie, Tibshirani & Friedman, The Elements of Statistical Learning, 2nd ed., §10.10",
)
_COMPLEXITY = Complexity(
    time="O(T n d log n)",
    space="O(n d)",
    notes=(
        "per boosting stage, times n_estimators; T = max_depth; space is one tree's "
        "presorted index table plus O(n) scores; predict is O(n_estimators T) per "
        "sample with O(n) extra memory"
    ),
)
_EPS = float(np.finfo(np.float64).eps)


class _BaseGradientBoosting(BaseEstimator):
    def __init__(
        self,
        loss: str,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 1.0,
        max_depth: int | None = 3,
        min_samples_split: int | float = 2,
        min_samples_leaf: int | float = 1,
        max_features: int | float | str | None = None,
        random_state=None,
    ):
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    _loss_name = ""

    def _check_params(self) -> None:
        if self.loss != self._loss_name:
            raise ValueError(f"loss must be {self._loss_name!r}, got {self.loss!r}.")
        lr = self.learning_rate
        if isinstance(lr, bool) or not isinstance(lr, numbers.Real) or not (0 < lr < np.inf):
            raise ValueError(f"learning_rate must be a finite number > 0, got {lr!r}.")
        ne = self.n_estimators
        if isinstance(ne, bool) or not isinstance(ne, numbers.Integral) or ne < 1:
            raise ValueError(f"n_estimators must be an integer >= 1, got {ne!r}.")
        ss = self.subsample
        if isinstance(ss, bool) or not isinstance(ss, numbers.Real) or not (0 < ss <= 1):
            raise ValueError(f"subsample must be in (0, 1], got {ss!r}.")

    # Loss-specific pieces ------------------------------------------------------
    def _init_score(self, y: np.ndarray) -> float:
        raise NotImplementedError

    def _residual(self, y: np.ndarray, F: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _leaf_values(self, tree: Tree, leaf, r, F) -> np.ndarray:
        return tree.value  # squared error: the tree's leaf means are the Newton step

    def _loss_value(self, y: np.ndarray, F: np.ndarray) -> float:
        raise NotImplementedError

    # Shared fit -------------------------------------------------------------------
    def _boost(self, X: np.ndarray, y: np.ndarray) -> None:
        """Run all stages, then assign the fitted state (nothing is set on failure)."""
        self._check_params()
        n, d = X.shape
        lr = float(self.learning_rate)
        rng = check_random_state(self.random_state)
        template = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
        )
        n_inbag = max(1, int(float(self.subsample) * n))
        init = self._init_score(y)
        F = np.full(n, init)
        trees, train_score = [], np.empty(int(self.n_estimators))
        for m in range(int(self.n_estimators)):
            r = self._residual(y, F)
            if n_inbag < n:
                rows = np.sort(rng.choice(n, size=n_inbag, replace=False))
            else:
                rows = np.arange(n)
            tree = template._grow(X[rows], r[rows], 1, rng)
            leaf_in = tree.apply(X[rows])
            tree = tree.with_values(self._leaf_values(tree, leaf_in, r[rows], F[rows]))
            F = F + lr * tree.value[tree.apply(X)]
            train_score[m] = self._loss_value(y[rows], F[rows])
            est = DecisionTreeRegressor(**template.get_params())
            est._set_tree(tree, d)
            trees.append(est)

        relevant = [
            t.tree_.compute_feature_importances(d, normalize=False)
            for t in trees
            if t.tree_.node_count > 1
        ]
        importances = np.mean(relevant, axis=0) if relevant else np.zeros(d)
        total = importances.sum()
        self.init_ = float(init)
        self.estimators_ = trees
        self.train_score_ = train_score
        self.feature_importances_ = importances / total if total > 0 else importances
        self.n_features_in_ = d

    def _staged_scores(self, X) -> Iterator[np.ndarray]:
        self._check_is_fitted("estimators_")
        X = check_n_features(X, self.n_features_in_, self)
        F = np.full(X.shape[0], self.init_)
        lr = float(self.learning_rate)
        for t in self.estimators_:
            F = F + lr * t.tree_.value[t.tree_.apply(X)]
            yield F

    def _scores(self, X) -> np.ndarray:
        return deque(self._staged_scores(X), maxlen=1)[0]  # last stage, O(n) memory


_GB_PARAMS_DOC = """
    learning_rate : float, default=0.1
        Shrinkage :math:`\\nu > 0` applied to every tree.
    n_estimators : int, default=100
        Number of boosting stages ``M``.
    subsample : float, default=1.0
        Fraction of rows (drawn without replacement) used by each stage; < 1 gives
        stochastic gradient boosting.
    max_depth : int or None, default=3
        Depth of every tree.
    min_samples_split : int or float, default=2
    min_samples_leaf : int or float, default=1
    max_features : int, float, "sqrt", "log2" or None, default=None
        Features searched per node by each tree.
    random_state : None, int, SeedSequence or Generator
        Seed for subsampling and feature subsampling.
"""


@register(
    domain="ml",
    tasks=("regression",),
    complexity=_COMPLEXITY,
    references=_REFERENCES,
    status="experimental",
)
class GradientBoostingRegressor(RegressorMixin, _BaseGradientBoosting):
    __doc__ = (
        r"""Gradient boosting of regression trees with squared-error loss.

    Parameters
    ----------
    loss : {"squared_error"}, default="squared_error\""""
        + _GB_PARAMS_DOC
        + r"""
    Attributes
    ----------
    init_ : float
        Initial prediction :math:`F_0 = \bar y`.
    estimators_ : list of DecisionTreeRegressor
    train_score_ : ndarray of shape (n_estimators,)
        In-bag mean squared error after each stage.
    feature_importances_ : ndarray of shape (d,)
        Mean over stages of the (unnormalised) tree importances, normalised.
    n_features_in_ : int

    Complexity
    ----------
    ``fit``: time O(n_estimators · T n d log n) with T = ``max_depth``, extra
    space O(n d). ``predict``: O(n_estimators · T) per sample, extra space O(n).

    References
    ----------
    Friedman (2001), Algorithm 2 (LS_Boost); *ESL* 2e, §10.10.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.ml import GradientBoostingRegressor
    >>> rng = np.random.default_rng(0)
    >>> X = rng.uniform(-2, 2, size=(200, 2))
    >>> y = np.sin(X[:, 0]) + 0.5 * X[:, 1]
    >>> gb = GradientBoostingRegressor(n_estimators=200).fit(X, y)
    >>> bool(gb.score(X, y) > 0.99), bool(np.all(np.diff(gb.train_score_) <= 0))
    (True, True)
    """
    )

    _loss_name = "squared_error"

    def __init__(
        self,
        loss: str = "squared_error",
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 1.0,
        max_depth: int | None = 3,
        min_samples_split: int | float = 2,
        min_samples_leaf: int | float = 1,
        max_features: int | float | str | None = None,
        random_state=None,
    ):
        super().__init__(
            loss,
            learning_rate,
            n_estimators,
            subsample,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_features,
            random_state,
        )

    def _init_score(self, y):
        return float(np.mean(y))

    def _residual(self, y, F):
        return y - F

    def _loss_value(self, y, F):
        return float(np.mean((y - F) ** 2))

    def fit(self, X, y) -> GradientBoostingRegressor:
        """Fit ``n_estimators`` stages to ``X`` (n, d) and targets ``y`` (n,)."""
        X, y = check_X_y(X, y)
        self._boost(X, y)
        return self

    def staged_predict(self, X) -> Iterator[np.ndarray]:
        """Yield the predictions after each stage (``n_estimators`` arrays of shape (n,))."""
        for F in self._staged_scores(X):
            yield F.copy()

    def predict(self, X) -> np.ndarray:
        """Prediction :math:`F_M(x)`."""
        return self._scores(X)


@register(
    domain="ml",
    tasks=("classification",),
    complexity=_COMPLEXITY,
    references=_REFERENCES,
    status="experimental",
)
class GradientBoostingClassifier(ClassifierMixin, _BaseGradientBoosting):
    __doc__ = (
        r"""Binary gradient boosting with log loss and Newton leaf values.

    Parameters
    ----------
    loss : {"log_loss"}, default="log_loss\""""
        + _GB_PARAMS_DOC
        + r"""
    Attributes
    ----------
    classes_ : ndarray of shape (2,)
    init_ : float
        Initial log-odds :math:`F_0`.
    estimators_ : list of DecisionTreeRegressor
        Trees whose leaf values are the Newton steps :math:`\gamma_j`.
    train_score_ : ndarray of shape (n_estimators,)
        In-bag binomial deviance (2 × mean log loss) after each stage.
    feature_importances_ : ndarray of shape (d,)
    n_features_in_ : int

    Raises
    ------
    ValueError
        If ``y`` does not have exactly two classes.

    Complexity
    ----------
    ``fit``: time O(n_estimators · T n d log n) with T = ``max_depth``, extra
    space O(n d). ``predict*``: O(n_estimators · T) per sample, extra space O(n).

    References
    ----------
    Friedman (2001), Algorithm 5 (L2_TreeBoost); *ESL* 2e, §10.10.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.ml import GradientBoostingClassifier
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 2))
    >>> y = (X[:, 0] * X[:, 1] > 0).astype(int)
    >>> gb = GradientBoostingClassifier(n_estimators=100).fit(X, y)
    >>> bool(gb.score(X, y) > 0.95), gb.predict_proba(X[:3]).shape
    (True, (3, 2))
    """
    )

    _loss_name = "log_loss"

    def __init__(
        self,
        loss: str = "log_loss",
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 1.0,
        max_depth: int | None = 3,
        min_samples_split: int | float = 2,
        min_samples_leaf: int | float = 1,
        max_features: int | float | str | None = None,
        random_state=None,
    ):
        super().__init__(
            loss,
            learning_rate,
            n_estimators,
            subsample,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_features,
            random_state,
        )

    def _init_score(self, y):
        p = float(np.clip(np.mean(y), _EPS, 1.0 - _EPS))
        return float(np.log(p / (1.0 - p)))

    def _residual(self, y, F):
        return y - sigmoid(F)

    def _leaf_values(self, tree, leaf, r, F):
        p = sigmoid(F)
        size = tree.node_count
        num = np.bincount(leaf, weights=r, minlength=size)
        den = np.bincount(leaf, weights=p * (1.0 - p), minlength=size)
        safe = np.abs(den) >= 1e-150
        values = np.zeros(size)
        values[safe] = num[safe] / den[safe]
        return values

    def _loss_value(self, y, F):
        sign = 2.0 * y - 1.0
        return float(-2.0 * np.mean(log_sigmoid(sign * F)))

    def fit(self, X, y) -> GradientBoostingClassifier:
        """Fit ``n_estimators`` stages to ``X`` (n, d) and binary labels ``y`` (n,)."""
        X, y = check_X_labels(X, y)
        classes, y_idx = np.unique(y, return_inverse=True)
        if len(classes) != 2:
            raise ValueError(
                f"GradientBoostingClassifier supports exactly 2 classes, got {len(classes)}."
            )
        self._boost(X, y_idx.astype(np.float64))
        self.classes_ = classes
        return self

    def decision_function(self, X) -> np.ndarray:
        """Log-odds :math:`F_M(x)` of the second class, shape (n,)."""
        return self._scores(X)

    def staged_decision_function(self, X) -> Iterator[np.ndarray]:
        """Yield the log-odds after each stage."""
        for F in self._staged_scores(X):
            yield F.copy()

    @staticmethod
    def _proba(F: np.ndarray) -> np.ndarray:
        return np.column_stack([sigmoid(-F), sigmoid(F)])

    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities :math:`[1-\\sigma(F), \\sigma(F)]`, shape (n, 2)."""
        return self._proba(self._scores(X))

    def staged_predict_proba(self, X) -> Iterator[np.ndarray]:
        """Yield the class probabilities after each stage."""
        for F in self._staged_scores(X):
            yield self._proba(F)

    def predict(self, X) -> np.ndarray:
        """Second class where :math:`F_M(x) > 0`, else the first."""
        F = self._scores(X)  # checks fitted first
        return self.classes_[(F > 0).astype(np.intp)]

    def staged_predict(self, X) -> Iterator[np.ndarray]:
        """Yield the predicted labels after each stage."""
        for F in self._staged_scores(X):
            yield self.classes_[(F > 0).astype(np.intp)]
