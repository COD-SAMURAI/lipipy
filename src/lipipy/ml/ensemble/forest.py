r"""Random forests: bagged CART trees with per-node feature subsampling.

Model
-----
A random forest averages :math:`B` (``n_estimators``) decision trees,

.. math::  \hat p(c \mid x) = \frac1B \sum_{b=1}^{B} \hat p_b(c \mid x)
           \quad\text{(classification)}, \qquad
           \hat f(x) = \frac1B \sum_{b=1}^{B} \hat f_b(x) \quad\text{(regression)} .

Tree :math:`b` is grown to full depth (by default) on a bootstrap sample (``n``
rows drawn with replacement) and, at every node, searches only a random subset of
``max_features`` features. Both sources of randomness decorrelate the trees, so
averaging reduces variance much more than bagging alone (Breiman 2001; ESL §15.2:
the variance of an average of :math:`B` trees with pairwise correlation
:math:`\rho` is :math:`\rho\sigma^2 + (1-\rho)\sigma^2/B`).

Randomness
----------
Every tree gets its own generator from :func:`lipipy.core.spawn`, which draws its
bootstrap sample and its feature subsets, so a fixed ``random_state`` reproduces
the forest exactly and trees are statistically independent.

Out-of-bag estimate
-------------------
About :math:`(1 - 1/n)^n \approx e^{-1} \approx 36.8\%` of the rows are left out of
each bootstrap sample. Predicting each training row with only the trees that did
not see it gives an almost unbiased estimate of the test score (``oob_score_``:
accuracy or :math:`R^2`).

Feature importance
------------------
The mean over trees (with at least one split) of each tree's normalised impurity
importance, renormalised to sum to one (scikit-learn's definition).

References
----------
Breiman (2001), *Random forests*, Machine Learning 45(1); Hastie, Tibshirani &
Friedman, *ESL* 2e, ch. 15.
"""

from __future__ import annotations

import numbers
import warnings

import numpy as np

from lipipy.core.base import BaseEstimator, RegressorMixin
from lipipy.core.complexity import Complexity
from lipipy.core.metrics import accuracy_score, r2_score
from lipipy.core.random import spawn
from lipipy.core.registry import register
from lipipy.core.validation import check_X_y
from lipipy.ml._base import ClassifierMixin, check_n_features, check_X_labels
from lipipy.ml.tree.cart import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    _BaseDecisionTree,
)

_REFERENCES = (
    "Breiman (2001), Random forests, Machine Learning 45(1)",
    "Hastie, Tibshirani & Friedman, The Elements of Statistical Learning, 2nd ed., ch. 15",
)
_COMPLEXITY = Complexity(
    time="O(T n d log n)",
    space="O(n d)",
    notes=(
        "per tree, times n_estimators: T = depth of each tree (O(log n) for balanced "
        "fully grown trees, n in the worst case); space is one tree's presorted index "
        "table plus the stored trees (O(n) nodes each); predict is O(n_estimators T) "
        "per sample with O(n k) extra memory"
    ),
)


class _BaseForest(BaseEstimator):
    _tree_class: type[_BaseDecisionTree] = _BaseDecisionTree

    def _check_forest_params(self) -> None:
        ne = self.n_estimators
        if isinstance(ne, bool) or not isinstance(ne, numbers.Integral) or ne < 1:
            raise ValueError(f"n_estimators must be an integer >= 1, got {ne!r}.")
        for name in ("bootstrap", "oob_score"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise TypeError(f"{name} must be a bool, got {getattr(self, name)!r}.")
        if self.oob_score and not self.bootstrap:
            raise ValueError("oob_score=True requires bootstrap=True.")

    def _make_tree(self) -> _BaseDecisionTree:
        return self._tree_class(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
        )

    def _fit_trees(self, X: np.ndarray, y: np.ndarray, n_classes: int):
        """Grow every tree; return (trees, in-bag masks or None). Mutates nothing."""
        self._check_forest_params()
        n = X.shape[0]
        trees, in_bag = [], []
        for rng in spawn(self.random_state, int(self.n_estimators)):
            tree = self._make_tree()
            if self.bootstrap:
                idx = rng.integers(0, n, size=n)
                tree._set_tree(tree._grow(X[idx], y[idx], n_classes, rng), X.shape[1])
                mask = np.zeros(n, dtype=bool)
                mask[idx] = True
                in_bag.append(mask)
            else:
                tree._set_tree(tree._grow(X, y, n_classes, rng), X.shape[1])
            trees.append(tree)
        return trees, (in_bag if self.bootstrap else None)

    @staticmethod
    def _mean_importances(trees, d: int) -> np.ndarray:
        relevant = [t.feature_importances_ for t in trees if t.tree_.node_count > 1]
        if not relevant:
            return np.zeros(d)
        mean = np.mean(relevant, axis=0)
        return mean / mean.sum()

    @staticmethod
    def _oob_rows(in_bag: list[np.ndarray], n: int) -> np.ndarray:
        n_oob = np.zeros(n, dtype=np.intp)
        for mask in in_bag:
            n_oob += ~mask
        if np.any(n_oob == 0):
            warnings.warn(
                f"{int(np.sum(n_oob == 0))} samples were in every bootstrap sample and "
                "have no out-of-bag prediction; they are left out of oob_score_ "
                "(use more estimators).",
                UserWarning,
                stacklevel=4,
            )
        return n_oob

    def _validated(self, X) -> np.ndarray:
        self._check_is_fitted("estimators_")
        return check_n_features(X, self.n_features_in_, self)

    def apply(self, X) -> np.ndarray:
        """Leaf index of every row in every tree, shape (n, n_estimators)."""
        X = self._validated(X)
        return np.column_stack([t.tree_.apply(X) for t in self.estimators_])


_FOREST_PARAMS_DOC = """
    max_depth : int or None, default=None
        Maximum depth of each tree; ``None`` grows fully.
    min_samples_split : int or float, default=2
    min_samples_leaf : int or float, default=1
    bootstrap : bool, default=True
        Grow each tree on a bootstrap sample (else on all rows).
    oob_score : bool, default=False
        Compute the out-of-bag score ``oob_score_`` (requires ``bootstrap``).
    random_state : None, int, SeedSequence or Generator
        Seed; each tree gets an independent stream via :func:`lipipy.core.spawn`.
"""


@register(
    domain="ml",
    tasks=("classification",),
    complexity=_COMPLEXITY,
    references=_REFERENCES,
    status="experimental",
)
class RandomForestClassifier(ClassifierMixin, _BaseForest):
    __doc__ = (
        r"""Random forest of CART classification trees (soft voting).

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees ``B``.
    criterion : {"gini", "entropy"}, default="gini"
    max_features : int, float, "sqrt", "log2" or None, default="sqrt"
        Features searched per node (see :class:`~lipipy.ml.DecisionTreeClassifier`)."""
        + _FOREST_PARAMS_DOC
        + r"""
    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
    classes_ : ndarray of shape (k,)
    feature_importances_ : ndarray of shape (d,)
    oob_score_ : float
        Out-of-bag accuracy (only with ``oob_score=True``).
    oob_decision_function_ : ndarray of shape (n, k)
        Out-of-bag class probabilities (NaN rows for samples never out of bag).
    n_features_in_ : int

    Complexity
    ----------
    ``fit``: time O(n_estimators · T n d log n), T the tree depth; extra space
    O(n d) while growing a tree plus the stored trees. ``predict``: time
    O(n_estimators · T) per sample, extra space O(n k).

    References
    ----------
    Breiman (2001), *Random forests*; *ESL* 2e, ch. 15.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.ml import RandomForestClassifier
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 4))
    >>> y = (X[:, 0] + X[:, 1] > 0).astype(int)
    >>> rf = RandomForestClassifier(n_estimators=25, random_state=0).fit(X, y)
    >>> bool(rf.score(X, y) == 1.0), rf.predict_proba(X[:2]).shape
    (True, (2, 2))
    """
    )

    _tree_class = DecisionTreeClassifier

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int | float = 2,
        min_samples_leaf: int | float = 1,
        max_features: int | float | str | None = "sqrt",
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state

    def fit(self, X, y) -> RandomForestClassifier:
        """Grow the forest on ``X`` (n, d) and labels ``y`` (n,); return ``self``."""
        X, y = check_X_labels(X, y)
        classes, y_idx = np.unique(y, return_inverse=True)
        y_idx = y_idx.astype(np.intp)
        k = len(classes)
        trees, in_bag = self._fit_trees(X, y_idx, k)
        for t in trees:
            t.classes_, t.n_classes_ = classes, k

        oob = {}
        if self.oob_score:
            n = X.shape[0]
            n_oob = self._oob_rows(in_bag, n)
            proba = np.zeros((n, k))
            for t, mask in zip(trees, in_bag, strict=True):
                rows = ~mask
                proba[rows] += t.tree_.value[t.tree_.apply(X[rows])]
            seen = n_oob > 0
            with np.errstate(invalid="ignore", divide="ignore"):
                proba = proba / n_oob[:, None]
            proba[~seen] = np.nan
            oob["oob_decision_function_"] = proba
            oob["oob_score_"] = (
                accuracy_score(y_idx[seen], np.argmax(proba[seen], axis=1))
                if seen.any()
                else float("nan")
            )

        self.estimators_ = trees
        self.classes_ = classes
        self.n_classes_ = k
        self.n_features_in_ = X.shape[1]
        self.feature_importances_ = self._mean_importances(trees, X.shape[1])
        for name, value in oob.items():
            setattr(self, name, value)
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Mean of the trees' leaf class fractions, shape (n, k)."""
        X = self._validated(X)
        proba = np.zeros((X.shape[0], self.n_classes_))
        for t in self.estimators_:
            proba += t.tree_.value[t.tree_.apply(X)]
        return proba / len(self.estimators_)

    def predict(self, X) -> np.ndarray:
        """Class with the highest mean probability (lowest index on ties)."""
        proba = self.predict_proba(X)  # checks fitted first
        return self.classes_[np.argmax(proba, axis=1)]


@register(
    domain="ml",
    tasks=("regression",),
    complexity=_COMPLEXITY,
    references=_REFERENCES,
    status="experimental",
)
class RandomForestRegressor(RegressorMixin, _BaseForest):
    __doc__ = (
        r"""Random forest of CART regression trees (mean of the trees).

    Parameters
    ----------
    n_estimators : int, default=100
    criterion : {"squared_error"}, default="squared_error"
    max_features : int, float, "sqrt", "log2" or None, default=1.0
        Features searched per node; the default 1.0 uses all of them (bagging),
        as scikit-learn."""
        + _FOREST_PARAMS_DOC
        + r"""
    Attributes
    ----------
    estimators_ : list of DecisionTreeRegressor
    feature_importances_ : ndarray of shape (d,)
    oob_score_ : float
        Out-of-bag :math:`R^2` (only with ``oob_score=True``).
    oob_prediction_ : ndarray of shape (n,)
        Out-of-bag predictions (NaN for samples never out of bag).
    n_features_in_ : int

    Complexity
    ----------
    ``fit``: time O(n_estimators · T n d log n); extra space O(n d) plus the trees.
    ``predict``: time O(n_estimators · T) per sample, extra space O(n).

    References
    ----------
    Breiman (2001), *Random forests*; *ESL* 2e, ch. 15.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.ml import RandomForestRegressor
    >>> rng = np.random.default_rng(0)
    >>> X = rng.uniform(-2, 2, size=(300, 2))
    >>> y = X[:, 0] ** 2 + 0.1 * rng.normal(size=300)
    >>> rf = RandomForestRegressor(n_estimators=30, oob_score=True, random_state=0).fit(X, y)
    >>> bool(rf.oob_score_ > 0.9)
    True
    """
    )

    _tree_class = DecisionTreeRegressor

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "squared_error",
        max_depth: int | None = None,
        min_samples_split: int | float = 2,
        min_samples_leaf: int | float = 1,
        max_features: int | float | str | None = 1.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state

    def fit(self, X, y) -> RandomForestRegressor:
        """Grow the forest on ``X`` (n, d) and targets ``y`` (n,); return ``self``."""
        X, y = check_X_y(X, y)
        trees, in_bag = self._fit_trees(X, y, 1)

        oob = {}
        if self.oob_score:
            n = X.shape[0]
            n_oob = self._oob_rows(in_bag, n)
            pred = np.zeros(n)
            for t, mask in zip(trees, in_bag, strict=True):
                rows = ~mask
                pred[rows] += t.tree_.value[t.tree_.apply(X[rows])]
            seen = n_oob > 0
            with np.errstate(invalid="ignore", divide="ignore"):
                pred = pred / n_oob
            pred[~seen] = np.nan
            oob["oob_prediction_"] = pred
            oob["oob_score_"] = r2_score(y[seen], pred[seen]) if seen.any() else float("nan")

        self.estimators_ = trees
        self.n_features_in_ = X.shape[1]
        self.feature_importances_ = self._mean_importances(trees, X.shape[1])
        for name, value in oob.items():
            setattr(self, name, value)
        return self

    def predict(self, X) -> np.ndarray:
        """Mean prediction of the trees."""
        X = self._validated(X)
        pred = np.zeros(X.shape[0])
        for t in self.estimators_:
            pred += t.tree_.value[t.tree_.apply(X)]
        return pred / len(self.estimators_)
