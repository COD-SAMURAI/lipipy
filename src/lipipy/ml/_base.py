"""Shared building blocks for :mod:`lipipy.ml` estimators.

``lipipy.core`` provides :class:`~lipipy.core.BaseEstimator` and
:class:`~lipipy.core.RegressorMixin`. The mixins here complete the scikit-learn
style API for classifiers and transformers. ``ClassifierMixin`` is a candidate to
move into ``lipipy.core`` once a second domain needs it.
"""

from __future__ import annotations

import numpy as np

from lipipy.core.metrics import accuracy_score
from lipipy.core.validation import check_array


class ClassifierMixin:
    """Adds an accuracy ``score`` method to classifiers.

    Examples
    --------
    >>> import numpy as np
    >>> class Constant(ClassifierMixin):
    ...     def predict(self, X):
    ...         return np.zeros(len(X), dtype=int)
    >>> Constant().score([[0.0], [1.0]], [0, 1])
    0.5
    """

    def score(self, X, y) -> float:
        """Fraction of samples in ``X`` whose predicted label equals ``y``."""
        return accuracy_score(np.asarray(y), self.predict(X))


class TransformerMixin:
    """Adds ``fit_transform`` (``fit`` followed by ``transform``) to transformers.

    Examples
    --------
    >>> import numpy as np
    >>> class Centre(TransformerMixin):
    ...     def fit(self, X, y=None):
    ...         self.mean_ = np.mean(X, axis=0)
    ...         return self
    ...     def transform(self, X):
    ...         return np.asarray(X) - self.mean_
    >>> Centre().fit_transform([[1.0], [3.0]]).ravel().tolist()
    [-1.0, 1.0]
    """

    def fit_transform(self, X, y=None) -> np.ndarray:
        """Fit to ``X`` and return the transformed ``X``."""
        return self.fit(X, y).transform(X)


def check_n_features(X, n_features_in: int, estimator: object) -> np.ndarray:
    """Validate ``X`` as a finite 2D float array with ``n_features_in`` columns."""
    X = check_array(X)
    if X.shape[1] != n_features_in:
        raise ValueError(
            f"X has {X.shape[1]} features, but {type(estimator).__name__} "
            f"was fitted with {n_features_in}."
        )
    return X


def check_X_labels(X, y) -> tuple[np.ndarray, np.ndarray]:
    """Validate a float feature matrix and a 1D label vector of any dtype."""
    X = check_array(X)
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"Expected 1D target y, got shape {y.shape}.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} samples but y has {y.shape[0]}.")
    if y.dtype.kind == "f" and not np.all(np.isfinite(y)):
        raise ValueError("y contains NaN or infinity.")
    return X, y
