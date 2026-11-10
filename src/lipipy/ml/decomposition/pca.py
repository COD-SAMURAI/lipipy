r"""Principal component analysis via the singular value decomposition.

Model
-----
Centre the data, :math:`X_c = X - \bar x`, and take the thin SVD

.. math::  X_c = U \Sigma V^\top, \qquad \sigma_1 \ge \sigma_2 \ge \dots \ge 0 .

The rows of :math:`V^\top` are the principal axes (unit directions of maximal
variance), and the variance explained by axis :math:`i` is
:math:`\lambda_i = \sigma_i^2 / (n - 1)`, the eigenvalues of the sample covariance
:math:`X_c^\top X_c / (n-1)`. Keeping the first ``m`` axes :math:`V_m`:

* ``transform``:  :math:`Z = (X - \bar x) V_m` (divided by :math:`\sqrt{\lambda_i}`
  per column if ``whiten``),
* ``inverse_transform``:  :math:`\hat X = Z V_m^\top + \bar x` (times
  :math:`\sqrt{\lambda_i}` first if ``whiten``). With all components it is exact.

Sign convention
---------------
Singular vectors are unique only up to sign. To make results deterministic we
flip each pair :math:`(u_i, v_i)` so that the entry of largest absolute value in
:math:`v_i` (the component) is positive. This is scikit-learn's
``svd_flip(u, v, u_based_decision=False)``, used by ``PCA`` since version 1.5.

Choosing ``n_components``
-------------------------
* ``None``: all :math:`\min(n, d)` components;
* int ``m``: the first ``m``;
* float ``p`` in (0, 1): the smallest ``m`` whose cumulative explained-variance
  ratio exceeds ``p``.

Reference: Jolliffe, *Principal Component Analysis*, 2nd ed. (2002);
Hastie, Tibshirani & Friedman, *ESL* 2e, §14.5.1.
"""

from __future__ import annotations

import numbers

import numpy as np

from lipipy.core.base import BaseEstimator
from lipipy.core.complexity import Complexity
from lipipy.core.registry import register
from lipipy.core.validation import check_array
from lipipy.ml._base import TransformerMixin, check_n_features


def _flip_signs(U: np.ndarray, Vt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""Make the largest-magnitude entry of each row of ``Vt`` positive (``svd_flip``).

    Column ``i`` of ``U`` is flipped together with row ``i`` of ``Vt``, so
    :math:`U \Sigma V^\top` is unchanged.
    """
    biggest = np.argmax(np.abs(Vt), axis=1)
    signs = np.sign(Vt[np.arange(Vt.shape[0]), biggest])
    signs[signs == 0] = 1.0
    return U * signs, Vt * signs[:, None]


@register(
    domain="ml",
    tasks=("dimensionality-reduction",),
    complexity=Complexity(
        time="O(n d min(n, d))",
        space="O(n d)",
        notes="thin SVD of the centred n x d data; transform is O(n d m) for m components",
    ),
    references=(
        "Jolliffe, Principal Component Analysis, 2nd ed., Springer (2002)",
        "Hastie, Tibshirani & Friedman, The Elements of Statistical Learning, 2nd ed., §14.5.1",
    ),
    status="experimental",
)
class PCA(TransformerMixin, BaseEstimator):
    r"""Principal component analysis (exact, via SVD of the centred data).

    Parameters
    ----------
    n_components : None, int or float in (0, 1), default=None
        Number of components to keep: all, a count, or the variance fraction to
        explain (see the module docstring).
    whiten : bool, default=False
        Scale each projected component to unit variance.

    Attributes
    ----------
    components_ : ndarray of shape (m, d)
        Principal axes (rows), sign-normalised as described above.
    explained_variance_ : ndarray of shape (m,)
        :math:`\sigma_i^2/(n-1)`.
    explained_variance_ratio_ : ndarray of shape (m,)
        Fraction of total variance per component.
    singular_values_ : ndarray of shape (m,)
    mean_ : ndarray of shape (d,)
    n_components_ : int
    noise_variance_ : float
        Mean of the discarded :math:`\lambda_i` (0 if none are discarded).
    n_features_in_ : int
    n_samples_ : int

    Complexity
    ----------
    ``fit``: time O(n d min(n, d)) for the thin SVD, extra space O(n d).
    ``transform``/``inverse_transform``: time O(n d m), space O(n m).
    ``fit_transform`` reuses the SVD (:math:`U_m\Sigma_m`) instead of projecting again.

    References
    ----------
    Jolliffe, I. T. (2002). *Principal Component Analysis*, 2nd ed.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.ml import PCA
    >>> X = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    >>> pca = PCA(n_components=1).fit(X)
    >>> pca.components_.round(6).tolist(), pca.explained_variance_ratio_.round(6).tolist()
    ([[0.707107, 0.707107]], [1.0])
    >>> pca.transform(X).round(6).ravel().tolist()
    [-1.414214, 0.0, 1.414214]
    """

    def __init__(self, n_components: int | float | None = None, whiten: bool = False):
        self.n_components = n_components
        self.whiten = whiten

    def _resolve_n_components(self, ratio: np.ndarray, max_components: int) -> int:
        m = self.n_components
        if m is None:
            return max_components
        if isinstance(m, bool) or not isinstance(m, numbers.Real):
            raise TypeError(f"n_components must be None, an int or a float, got {m!r}.")
        if isinstance(m, numbers.Integral):
            if not 0 <= m <= max_components:
                raise ValueError(
                    f"n_components={m} must be between 0 and min(n_samples, n_features)="
                    f"{max_components}."
                )
            return int(m)
        if not 0.0 < m < 1.0:
            raise ValueError(f"A float n_components must be in (0, 1), got {m!r}.")
        cumulative = np.cumsum(ratio)
        return min(int(np.searchsorted(cumulative, m, side="right")) + 1, max_components)

    def fit(self, X, y=None) -> PCA:
        """Fit the principal axes on ``X`` of shape (n, d); ``y`` is ignored."""
        self._fit(X)
        return self

    def fit_transform(self, X, y=None) -> np.ndarray:
        r"""Fit and return the projected training data.

        Uses the SVD directly: :math:`X_c V_m = U_m \Sigma_m`, and the whitened
        version :math:`U_m \sqrt{n-1}`. This is well defined even for components
        with zero variance, where dividing by :math:`\sqrt{\lambda_i}` is not.
        """
        U = self._fit(X)
        if self.whiten:
            return U * np.sqrt(self.n_samples_ - 1)
        return U * self.singular_values_

    def _fit(self, X) -> np.ndarray:
        """Fit the model and return the kept left singular vectors ``U_m`` (n, m)."""
        X = check_array(X)
        n, d = X.shape
        if n < 2:
            raise ValueError("PCA needs at least 2 samples to estimate a variance.")
        self.mean_ = X.mean(axis=0)
        U, S, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        U, Vt = _flip_signs(U, Vt)

        variance = S**2 / (n - 1)
        total = variance.sum()
        ratio = variance / total if total > 0 else np.zeros_like(variance)
        m = self._resolve_n_components(ratio, min(n, d))

        self.components_ = Vt[:m]
        self.explained_variance_ = variance[:m]
        self.explained_variance_ratio_ = ratio[:m]
        self.singular_values_ = S[:m]
        self.n_components_ = m
        self.noise_variance_ = float(variance[m:].mean()) if m < min(n, d) else 0.0
        self.n_features_in_ = d
        self.n_samples_ = n
        return U[:, :m]

    def _whiten_scale(self) -> np.ndarray:
        scale = np.sqrt(self.explained_variance_)
        return np.maximum(scale, np.finfo(np.float64).eps)  # avoid dividing by 0

    def transform(self, X) -> np.ndarray:
        """Project ``X`` (n, d) onto the components; returns shape (n, n_components_)."""
        self._check_is_fitted("components_")
        X = check_n_features(X, self.n_features_in_, self)
        Z = (X - self.mean_) @ self.components_.T
        return Z / self._whiten_scale() if self.whiten else Z

    def inverse_transform(self, Z) -> np.ndarray:
        """Map projected data (n, n_components_) back to feature space (n, d)."""
        self._check_is_fitted("components_")
        Z = check_array(Z)
        if Z.shape[1] != self.n_components_:
            raise ValueError(
                f"Z has {Z.shape[1]} columns, but PCA has {self.n_components_} components."
            )
        if self.whiten:
            Z = Z * self._whiten_scale()
        return Z @ self.components_ + self.mean_
