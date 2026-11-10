r"""Gaussian naive Bayes classifier.

Model
-----
Features are assumed conditionally independent given the class, each Gaussian:

.. math::
    p(x \mid c) = \prod_{j=1}^{d} \mathcal N(x_j;\ \theta_{cj}, \sigma^2_{cj}),
    \qquad P(c \mid x) \propto \pi_c\, p(x \mid c).

Fitting is closed form: :math:`\theta_{cj}` and :math:`\sigma^2_{cj}` are the
per-class mean and (biased, ``ddof=0``) variance of feature ``j``, and the prior
:math:`\pi_c` is the class frequency unless given. Every variance is increased by
:math:`\varepsilon = \texttt{var\_smoothing}\cdot\max_j \operatorname{Var}(x_j)` so
constant features do not cause division by zero.

Prediction in log space
-----------------------
The joint log-likelihood is

.. math::
    \log \pi_c - \tfrac12 \sum_j \log(2\pi\sigma^2_{cj})
    - \tfrac12 \sum_j \frac{(x_j - \theta_{cj})^2}{\sigma^2_{cj}} ,

and is normalised with the log-sum-exp trick,
:math:`\log P(c\mid x) = a_c - (m + \log\sum_{c'} e^{a_{c'} - m})` with
:math:`m = \max_{c'} a_{c'}`. Multiplying densities directly would underflow to 0
for far-away points or many features; in log space the result stays finite.

Reference: Hastie, Tibshirani & Friedman, *ESL* 2e, §6.6.3; Chan, Golub & LeVeque
(1983) for the variance computation used by scikit-learn.
"""

from __future__ import annotations

import numbers

import numpy as np

from lipipy.core.base import BaseEstimator
from lipipy.core.complexity import Complexity
from lipipy.core.registry import register
from lipipy.ml._base import ClassifierMixin, check_n_features, check_X_labels


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    """Stable :math:`\\log\\sum\\exp` along ``axis`` (keeps dims)."""
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)  # all -inf row: avoid inf - inf
    return m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))


@register(
    domain="ml",
    tasks=("classification",),
    complexity=Complexity(
        time="O(n k d)",
        space="O(n d + n k)",
        notes="fit is O(n d) after sorting labels (O(n log n)); predict is O(n k d)",
    ),
    references=(
        "Hastie, Tibshirani & Friedman, The Elements of Statistical Learning, 2nd ed., §6.6.3",
        "Zhang (2004), The optimality of naive Bayes, FLAIRS",
    ),
    status="experimental",
)
class GaussianNB(ClassifierMixin, BaseEstimator):
    r"""Gaussian naive Bayes with log-space prediction.

    Parameters
    ----------
    var_smoothing : float, default=1e-9
        Fraction of the largest feature variance added to every variance.
    priors : array-like of shape (n_classes,), optional
        Class prior probabilities (non-negative, summing to 1), in the order of
        ``classes_``. Default: class frequencies.

    Attributes
    ----------
    classes_ : ndarray of shape (k,)
    class_count_ : ndarray of shape (k,)
    class_prior_ : ndarray of shape (k,)
    theta_ : ndarray of shape (k, d)
        Per-class feature means.
    var_ : ndarray of shape (k, d)
        Per-class feature variances plus ``epsilon_``.
    epsilon_ : float
    n_features_in_ : int

    Complexity
    ----------
    ``fit``: time O(n log n + n d) (label sorting, one pass for the moments),
    extra space O(n d) for the per-class row selections. ``predict*``: time
    O(n k d), extra space O(n d + n k).

    References
    ----------
    Hastie, Tibshirani & Friedman, *ESL* 2e, §6.6.3.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.ml import GaussianNB
    >>> X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    >>> y = np.array([0, 0, 1, 1])
    >>> nb = GaussianNB().fit(X, y)
    >>> nb.predict([[-1.5], [3.0]]).tolist()
    [0, 1]
    >>> nb.predict_proba([[0.0]]).round(6).tolist()
    [[0.5, 0.5]]
    """

    def __init__(self, var_smoothing: float = 1e-9, priors=None):
        self.var_smoothing = var_smoothing
        self.priors = priors

    def fit(self, X, y) -> GaussianNB:
        """Estimate class priors, means and variances from ``X`` (n, d) and labels ``y``."""
        vs = self.var_smoothing
        if isinstance(vs, bool) or not isinstance(vs, numbers.Real) or not vs >= 0:
            raise ValueError(f"var_smoothing must be a number >= 0, got {vs!r}.")
        X, y = check_X_labels(X, y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        k, d = len(self.classes_), X.shape[1]

        self.epsilon_ = float(vs) * float(np.var(X, axis=0).max())
        self.theta_ = np.empty((k, d))
        self.var_ = np.empty((k, d))
        for c in range(k):
            Xc = X[y_idx == c]
            self.theta_[c] = Xc.mean(axis=0)
            self.var_[c] = Xc.var(axis=0)
        self.var_ += self.epsilon_
        self.class_count_ = np.bincount(y_idx, minlength=k).astype(np.float64)

        if self.priors is None:
            self.class_prior_ = self.class_count_ / self.class_count_.sum()
        else:
            priors = np.asarray(self.priors, dtype=np.float64)
            if priors.shape != (k,):
                raise ValueError(f"priors has shape {priors.shape}, expected ({k},).")
            if np.any(priors < 0) or not np.isclose(priors.sum(), 1.0):
                raise ValueError("priors must be non-negative and sum to 1.")
            self.class_prior_ = priors
        if np.any(self.var_ == 0):
            raise ValueError(
                "A class has a zero-variance feature and var_smoothing gives no "
                "smoothing (all features constant or var_smoothing=0); use var_smoothing > 0."
            )
        self.n_features_in_ = d
        return self

    def _joint_log_likelihood(self, X) -> np.ndarray:
        self._check_is_fitted("theta_")
        X = check_n_features(X, self.n_features_in_, self)
        jll = np.empty((X.shape[0], len(self.classes_)))
        with np.errstate(divide="ignore"):  # a zero prior gives log 0 = -inf
            log_prior = np.log(self.class_prior_)
        for c in range(len(self.classes_)):
            log_norm = -0.5 * np.sum(np.log(2.0 * np.pi * self.var_[c]))
            mahalanobis = np.sum((X - self.theta_[c]) ** 2 / self.var_[c], axis=1)
            jll[:, c] = log_prior[c] + log_norm - 0.5 * mahalanobis
        return jll

    def predict_log_proba(self, X) -> np.ndarray:
        """Log posterior :math:`\\log P(c\\mid x)`, shape (n, k), columns as ``classes_``."""
        jll = self._joint_log_likelihood(X)
        return jll - _logsumexp(jll, axis=1)

    def predict_proba(self, X) -> np.ndarray:
        """Posterior :math:`P(c\\mid x)`, shape (n, k), columns as ``classes_``."""
        return np.exp(self.predict_log_proba(X))

    def predict(self, X) -> np.ndarray:
        """Class with the highest posterior for each row of ``X``."""
        self._check_is_fitted("theta_")
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]
