r"""Linear and quadratic discriminant analysis.

Generative model
----------------
Each class :math:`c` with prior :math:`\pi_c` generates Gaussian features
:math:`x \mid c \sim \mathcal N(\mu_c, \Sigma_c)`. Bayes' rule gives the log posterior
up to a constant,

.. math::
    \delta_c(x) = \log\pi_c - \tfrac12 \log\lvert\Sigma_c\rvert
    - \tfrac12 (x - \mu_c)^\top \Sigma_c^{-1} (x - \mu_c),

and the posterior is :math:`P(c\mid x) = \operatorname{softmax}_c \delta_c(x)`
(computed with log-sum-exp).

LDA (shared covariance, ``solver="svd"``)
-----------------------------------------
With one pooled within-class covariance :math:`\Sigma` (ML estimate, divided by
:math:`n`) the quadratic terms cancel and :math:`\delta_c` is linear in :math:`x`.
Following scikit-learn's SVD solver, :math:`\Sigma` is never formed or inverted:

1. Centre each sample on its class mean, scale each feature by its within-class
   standard deviation :math:`s_j` and by :math:`1/\sqrt n`, and take the SVD
   :math:`U S V^\top`. The whitening map is :math:`W = \mathrm{diag}(1/s)\, V_r S_r^{-1}`,
   keeping the :math:`r` singular values above ``tol`` (so :math:`W^\top \Sigma W = I`).
2. In the whitened space, the class means weighted by
   :math:`\sqrt{n \pi_c / (k-1)}` around the overall mean :math:`\bar x = \sum_c \pi_c
   \mu_c` are decomposed again by SVD; its right singular vectors are the
   discriminant directions, ordered by between-class variance. The ratios of the
   squared singular values are ``explained_variance_ratio_``, and
   ``scalings_`` :math:`= W V'` keeps the directions with singular value above
   ``tol`` times the largest.
3. In the whitened coordinates :math:`z = (x - \bar x)^\top` ``scalings_`` the
   classifier is the nearest class mean with a prior term:
   :math:`\delta_c = z^\top m_c - \tfrac12\lVert m_c\rVert^2 + \log\pi_c`.

``transform`` returns :math:`z` restricted to ``n_components`` directions.

Sign convention: singular vectors are defined only up to sign, so each
discriminant direction (each column of ``transform``'s output) may be negated
relative to another implementation; the predictions and probabilities are not
affected. We keep NumPy's LAPACK output as is, like scikit-learn.

QDA (one covariance per class)
------------------------------
Each class's centred data is decomposed :math:`X_c - \mu_c = U S V^\top`; the
covariance eigenvalues are :math:`S^2 / n_c` (ML estimate), shrunk to
:math:`(1 - r) S^2/n_c + r` by ``reg_param`` :math:`r`, and

.. math::
    (x - \mu_c)^\top \Sigma_c^{-1} (x - \mu_c) = \lVert (x - \mu_c)^\top V
    \mathrm{diag}(\lambda)^{-1/2} \rVert^2, \qquad
    \log\lvert\Sigma_c\rvert = \textstyle\sum \log\lambda .

A class whose regularised covariance has an eigenvalue :math:`\le` ``tol`` (for
example a class with no more samples than features, or a constant feature with
``reg_param=0``) makes :math:`\Sigma_c` singular and is rejected with an error,
like scikit-learn.

References
----------
Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, 2nd ed.,
§4.3 (eq. 4.10-4.12) and §4.3.3 (reduced-rank LDA); Fisher (1936), *The use of
multiple measurements in taxonomic problems*, Annals of Eugenics 7.
"""

from __future__ import annotations

import numbers

import numpy as np

from lipipy.core.base import BaseEstimator
from lipipy.core.complexity import Complexity
from lipipy.core.registry import register
from lipipy.ml._base import ClassifierMixin, TransformerMixin, check_n_features, check_X_labels
from lipipy.numerics import logsumexp, sigmoid, softmax

_REFERENCES = (
    "Hastie, Tibshirani & Friedman, The Elements of Statistical Learning, 2nd ed., §4.3",
    "Fisher (1936), The use of multiple measurements in taxonomic problems, Annals of Eugenics 7",
)


def _is_real(v) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, bool)


def _resolve_priors(priors, counts: np.ndarray) -> np.ndarray:
    if priors is None:
        return counts / counts.sum()
    p = np.asarray(priors, dtype=np.float64)
    if p.shape != counts.shape:
        raise ValueError(f"priors has shape {p.shape}, expected {counts.shape}.")
    if not np.all(np.isfinite(p)) or np.any(p < 0) or not np.isclose(p.sum(), 1.0):
        raise ValueError("priors must be finite, non-negative and sum to 1.")
    return p.copy()


def _encode_classes(X, y):
    X, y = check_X_labels(X, y)
    classes, y_idx = np.unique(y, return_inverse=True)
    if len(classes) < 2:
        raise ValueError(f"Discriminant analysis needs at least 2 classes, got {len(classes)}.")
    return X, classes, y_idx


@register(
    domain="ml",
    tasks=("classification", "dimensionality-reduction"),
    complexity=Complexity(
        time="O(n d min(n, d) + k d^2)",
        space="O(n d)",
        notes="SVD of the n x d within-class-centred data, then of the k x r class means",
    ),
    references=_REFERENCES,
    status="experimental",
)
class LinearDiscriminantAnalysis(ClassifierMixin, TransformerMixin, BaseEstimator):
    r"""Linear discriminant analysis (shared covariance), SVD solver.

    Fits Gaussian classes with a common covariance and classifies with Bayes'
    rule; also projects onto the at most :math:`k-1` most discriminative
    directions. See the module docstring for the algorithm and sign convention.

    Parameters
    ----------
    solver : {"svd"}, default="svd"
        Only the SVD solver (no covariance matrix is formed) is implemented.
    priors : array-like of shape (k,), optional
        Class priors (non-negative, sum to 1); default: class frequencies.
    n_components : int, optional
        Number of directions kept by ``transform``, at most ``min(k - 1, d)``.
    tol : float, default=1e-4
        Singular values below ``tol`` (whitening step) or below ``tol`` times the
        largest (between-class step) are treated as zero.

    Attributes
    ----------
    classes_, priors_, means_ : class labels, priors and means (k, d)
    xbar_ : ndarray of shape (d,)
        Prior-weighted overall mean.
    scalings_ : ndarray of shape (d, r)
        Whitening and discriminant directions.
    explained_variance_ratio_ : ndarray of shape (n_components,)
        Between-class variance explained by each direction.
    coef_ : ndarray of shape (1, d) for two classes, else (k, d)
    intercept_ : ndarray of shape (1,) or (k,)
    n_features_in_ : int

    Complexity
    ----------
    ``fit``: time O(n d min(n, d) + k d^2) for the two SVDs, extra space O(n d)
    for the centred copy. ``predict``/``transform``: time O(n d k), space O(n k).

    References
    ----------
    Hastie, Tibshirani & Friedman, *ESL* 2e, §4.3.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.ml import LinearDiscriminantAnalysis
    >>> X = np.array([[-2.0, 0.0], [-1.0, 1.0], [-1.5, -1.0], [1.0, 0.0], [2.0, 1.0], [1.5, -1.0]])
    >>> y = np.array([0, 0, 0, 1, 1, 1])
    >>> lda = LinearDiscriminantAnalysis().fit(X, y)
    >>> lda.predict([[-1.0, 0.0], [1.0, 0.5]]).tolist()
    [0, 1]
    >>> lda.explained_variance_ratio_.tolist(), lda.transform(X).shape
    ([1.0], (6, 1))
    """

    def __init__(self, solver: str = "svd", priors=None, n_components=None, tol: float = 1e-4):
        self.solver = solver
        self.priors = priors
        self.n_components = n_components
        self.tol = tol

    def fit(self, X, y) -> LinearDiscriminantAnalysis:
        """Fit class means, whitening and discriminant directions; return ``self``."""
        if self.solver != "svd":
            raise ValueError(f"solver must be 'svd', got {self.solver!r}.")
        if not _is_real(self.tol) or not 0 <= self.tol < np.inf:
            raise ValueError(f"tol must be a finite number >= 0, got {self.tol!r}.")
        X, classes, y_idx = _encode_classes(X, y)
        n, d = X.shape
        k = len(classes)
        if n <= k:
            raise ValueError("The number of samples must be more than the number of classes.")
        max_components = min(k - 1, d)
        m = self.n_components
        if m is None:
            m = max_components
        elif isinstance(m, bool) or not isinstance(m, numbers.Integral) or not 0 <= m:
            raise ValueError(f"n_components must be None or an integer >= 0, got {m!r}.")
        elif m > max_components:
            raise ValueError(
                f"n_components={m} cannot be larger than min(n_features, n_classes - 1)="
                f"{max_components}."
            )
        counts = np.bincount(y_idx, minlength=k).astype(np.float64)
        priors = _resolve_priors(self.priors, counts)

        means = np.zeros((k, d))
        np.add.at(means, y_idx, X)
        means /= counts[:, None]
        Xc = X - means[y_idx]
        xbar = priors @ means

        # 1) whiten the within-class scatter
        std = Xc.std(axis=0)
        std[std == 0] = 1.0
        _, S, Vt = np.linalg.svd(Xc / std / np.sqrt(n), full_matrices=False)
        rank = int(np.sum(S > self.tol))
        if rank == 0:
            raise ValueError("All within-class variance is zero (below tol); LDA is undefined.")
        whiten = (Vt[:rank] / std).T / S[:rank]
        # 2) between-class directions in the whitened space
        weights = np.sqrt(n * priors / (k - 1))
        M = (weights[:, None] * (means - xbar)) @ whiten
        _, S2, Vt2 = np.linalg.svd(M, full_matrices=False)
        ratio = (S2**2 / np.sum(S2**2))[:m] if m > 0 else np.empty(0)
        rank2 = int(np.sum(S2 > self.tol * S2[0]))
        scalings = whiten @ Vt2.T[:, :rank2]
        # 3) nearest whitened mean with prior term, mapped back to x
        coef = (means - xbar) @ scalings
        intercept = -0.5 * np.sum(coef**2, axis=1) + np.log(priors)
        coef = coef @ scalings.T
        intercept = intercept - xbar @ coef.T
        if k == 2:
            coef = (coef[1] - coef[0])[None, :]
            intercept = np.array([intercept[1] - intercept[0]])

        self.classes_ = classes
        self.priors_ = priors
        self.means_ = means
        self.xbar_ = xbar
        self.scalings_ = scalings
        self.explained_variance_ratio_ = ratio
        self.coef_ = coef
        self.intercept_ = intercept
        self._n_components = m
        self.n_features_in_ = d
        return self

    def decision_function(self, X) -> np.ndarray:
        """Linear scores ``X @ coef_.T + intercept_``; shape (n,) for two classes."""
        self._check_is_fitted("coef_")
        X = check_n_features(X, self.n_features_in_, self)
        scores = X @ self.coef_.T + self.intercept_
        return scores[:, 0] if len(self.classes_) == 2 else scores

    def predict_proba(self, X) -> np.ndarray:
        """Posterior class probabilities, shape (n, k)."""
        scores = self.decision_function(X)
        if len(self.classes_) == 2:
            p = sigmoid(scores)
            return np.stack([1.0 - p, p], axis=1)
        return softmax(scores, axis=1)

    def predict_log_proba(self, X) -> np.ndarray:
        """Log posterior class probabilities, shape (n, k)."""
        scores = self.decision_function(X)
        if len(self.classes_) == 2:
            scores = np.stack([np.zeros_like(scores), scores], axis=1)
        return scores - logsumexp(scores, axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        """Class with the highest posterior."""
        scores = self.decision_function(X)
        if len(self.classes_) == 2:
            return self.classes_[(scores > 0).astype(np.intp)]
        return self.classes_[np.argmax(scores, axis=1)]

    def transform(self, X) -> np.ndarray:
        """Project onto the discriminant directions, shape (n, n_components)."""
        self._check_is_fitted("scalings_")
        X = check_n_features(X, self.n_features_in_, self)
        return ((X - self.xbar_) @ self.scalings_)[:, : self._n_components]


@register(
    domain="ml",
    tasks=("classification",),
    complexity=Complexity(
        time="O(n d min(n, d) + n k d^2)",
        space="O(n d + k d^2)",
        notes="one SVD per class at fit; predict costs O(n k d^2)",
    ),
    references=_REFERENCES,
    status="experimental",
)
class QuadraticDiscriminantAnalysis(ClassifierMixin, BaseEstimator):
    r"""Quadratic discriminant analysis (one Gaussian covariance per class).

    See the module docstring for the model and how singular class covariances
    are handled.

    Parameters
    ----------
    reg_param : float in [0, 1], default=0.0
        Shrinks each class covariance's eigenvalues: :math:`(1-r)\lambda + r`.
    priors : array-like of shape (k,), optional
        Class priors (non-negative, sum to 1); default: class frequencies.
    tol : float, default=1e-4
        A regularised eigenvalue at or below ``tol`` means the covariance is singular.

    Attributes
    ----------
    classes_, priors_, means_ : class labels, priors and means (k, d)
    rotations_ : list of k ndarrays of shape (d, d)
        Eigenvectors of each class covariance (columns).
    scalings_ : list of k ndarrays of shape (d,)
        Regularised eigenvalues of each class covariance.
    n_features_in_ : int

    Raises
    ------
    ValueError
        If a class has a single sample, or a class covariance is singular.

    Complexity
    ----------
    ``fit``: time O(n d min(n, d)) for the per-class SVDs, extra space
    O(n d + k d^2). ``predict*``: time O(n k d^2), space O(n d + n k).

    References
    ----------
    Hastie, Tibshirani & Friedman, *ESL* 2e, §4.3 (eq. 4.12).

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.ml import QuadraticDiscriminantAnalysis
    >>> X = np.array([[-1.0], [0.0], [1.0], [-3.0], [0.5], [4.0]])
    >>> y = np.array([0, 0, 0, 1, 1, 1])
    >>> qda = QuadraticDiscriminantAnalysis().fit(X, y)
    >>> qda.predict([[0.2], [6.0]]).tolist()
    [0, 1]
    """

    def __init__(self, reg_param: float = 0.0, priors=None, tol: float = 1e-4):
        self.reg_param = reg_param
        self.priors = priors
        self.tol = tol

    def fit(self, X, y) -> QuadraticDiscriminantAnalysis:
        """Fit one Gaussian per class; return ``self``."""
        r = self.reg_param
        if not _is_real(r) or not 0.0 <= r <= 1.0:
            raise ValueError(f"reg_param must be a number in [0, 1], got {r!r}.")
        if not _is_real(self.tol) or not 0 <= self.tol < np.inf:
            raise ValueError(f"tol must be a finite number >= 0, got {self.tol!r}.")
        X, classes, y_idx = _encode_classes(X, y)
        d = X.shape[1]
        counts = np.bincount(y_idx, minlength=len(classes)).astype(np.float64)
        priors = _resolve_priors(self.priors, counts)
        means, rotations, scalings = [], [], []
        for c, label in enumerate(classes):
            Xg = X[y_idx == c]
            if len(Xg) == 1:
                raise ValueError(
                    f"Class {label!r} has a single sample; its covariance is undefined."
                )
            mu = Xg.mean(axis=0)
            _, S, Vt = np.linalg.svd(Xg - mu, full_matrices=False)
            lam = (1.0 - r) * S**2 / len(Xg) + r
            if len(lam) < d or np.sum(lam > self.tol) < d:
                raise ValueError(
                    f"The covariance of class {label!r} is singular (a class needs more "
                    "samples than features and no constant direction); increase reg_param."
                )
            means.append(mu)
            rotations.append(Vt.T)
            scalings.append(lam)

        self.classes_ = classes
        self.priors_ = priors
        self.means_ = np.asarray(means)
        self.rotations_ = rotations
        self.scalings_ = scalings
        self.n_features_in_ = d
        return self

    def _log_joint(self, X) -> np.ndarray:
        self._check_is_fitted("rotations_")
        X = check_n_features(X, self.n_features_in_, self)
        out = np.empty((X.shape[0], len(self.classes_)))
        with np.errstate(divide="ignore"):  # a zero prior gives -inf
            log_prior = np.log(self.priors_)
        for c in range(len(self.classes_)):
            Z = (X - self.means_[c]) @ (self.rotations_[c] / np.sqrt(self.scalings_[c]))
            out[:, c] = -0.5 * (np.sum(Z**2, axis=1) + np.sum(np.log(self.scalings_[c])))
            out[:, c] += log_prior[c]
        return out

    def decision_function(self, X) -> np.ndarray:
        r"""Log posterior up to a constant, (n, k); for two classes
        :math:`\delta_1 - \delta_0`, shape (n,)."""
        scores = self._log_joint(X)
        return scores[:, 1] - scores[:, 0] if len(self.classes_) == 2 else scores

    def predict_log_proba(self, X) -> np.ndarray:
        """Log posterior class probabilities, shape (n, k)."""
        scores = self._log_joint(X)
        return scores - logsumexp(scores, axis=1, keepdims=True)

    def predict_proba(self, X) -> np.ndarray:
        """Posterior class probabilities, shape (n, k)."""
        return np.exp(self.predict_log_proba(X))

    def predict(self, X) -> np.ndarray:
        """Class with the highest posterior (lowest class index on ties)."""
        scores = self._log_joint(X)  # checks fitted first
        return self.classes_[np.argmax(scores, axis=1)]
