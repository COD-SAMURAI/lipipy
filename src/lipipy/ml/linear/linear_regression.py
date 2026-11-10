r"""Ordinary least squares linear regression.

Model
-----
.. math::  \hat y = X w + b

We pick ``w`` and ``b`` to minimise the mean squared error

.. math::  L(w, b) = \frac{1}{n} \lVert y - Xw - b \rVert_2^2 .

Solvers
-------
``"lstsq"`` (default)
    Closed form. Centring ``X`` and ``y`` removes the intercept, then we solve
    :math:`\min_w \lVert X_c w - y_c \rVert^2` with an SVD-based least-squares
    routine. This is the numerically stable version of the normal equation
    :math:`w = (X^\top X)^{-1} X^\top y` and also handles rank-deficient ``X``.
    The intercept is recovered as :math:`b = \bar y - \bar x^\top w`.

``"gd"``
    Batch gradient descent, included for teaching. The gradients are

    .. math::
        \nabla_w L = -\tfrac{2}{n} X^\top (y - \hat y), \qquad
        \nabla_b L = -\tfrac{2}{n} \textstyle\sum_i (y_i - \hat y_i).

    Features are standardised internally so a single learning rate works
    across datasets, and the weights are mapped back to the original scale.

Reference: Hastie, Tibshirani & Friedman, *The Elements of Statistical
Learning*, 2nd ed., section 3.2.
"""

from __future__ import annotations

import numpy as np

from lipipy.core.base import BaseEstimator, RegressorMixin
from lipipy.core.complexity import Complexity
from lipipy.core.registry import register
from lipipy.core.validation import check_array, check_X_y


@register(
    domain="ml",
    tasks=("regression",),
    complexity=Complexity(
        time="O(n d^2 + d^3)",
        space="O(n d)",
        notes="solver='gd' costs O(T n d) time; both keep one centred copy of X",
    ),
    references=(
        "Hastie, Tibshirani & Friedman, The Elements of Statistical Learning, 2nd ed., §3.2",
    ),
    status="stable",
)
class LinearRegression(RegressorMixin, BaseEstimator):
    """Ordinary least squares linear regression.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to learn a bias term ``b``.
    solver : {"lstsq", "gd"}, default="lstsq"
        ``"lstsq"`` solves in closed form; ``"gd"`` uses batch gradient descent.
    learning_rate : float, default=0.1
        Step size for ``solver="gd"``.
    max_iter : int, default=10_000
        Maximum number of iterations for ``solver="gd"``.
    tol : float, default=1e-10
        ``solver="gd"`` stops when the loss improves by less than ``tol``.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Learned weights ``w``.
    intercept_ : float
        Learned bias ``b`` (0.0 when ``fit_intercept=False``).
    n_iter_ : int
        Iterations run (1 for ``"lstsq"``).
    loss_history_ : list of float
        Training MSE per iteration (``"gd"`` only, empty otherwise).

    Complexity
    ----------
    ``solver="lstsq"``: time O(n d^2 + d^3), extra space O(n d).
    ``solver="gd"``: time O(T n d) for T iterations, extra space O(n d).

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.ml import LinearRegression
    >>> X = np.array([[1.0], [2.0], [3.0]])
    >>> y = 2 * X[:, 0] + 1
    >>> model = LinearRegression().fit(X, y)
    >>> round(float(model.coef_[0]), 6), round(model.intercept_, 6)
    (2.0, 1.0)
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        solver: str = "lstsq",
        learning_rate: float = 0.1,
        max_iter: int = 10_000,
        tol: float = 1e-10,
    ):
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y) -> LinearRegression:
        X, y = check_X_y(X, y)
        if self.solver == "lstsq":
            self._fit_lstsq(X, y)
        elif self.solver == "gd":
            self._fit_gd(X, y)
        else:
            raise ValueError(f"Unknown solver {self.solver!r}; use 'lstsq' or 'gd'.")
        return self

    def _fit_lstsq(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.fit_intercept:
            x_mean, y_mean = X.mean(axis=0), y.mean()
            w, *_ = np.linalg.lstsq(X - x_mean, y - y_mean, rcond=None)
            self.intercept_ = float(y_mean - x_mean @ w)
        else:
            w, *_ = np.linalg.lstsq(X, y, rcond=None)
            self.intercept_ = 0.0
        self.coef_ = w
        self.n_iter_ = 1
        self.loss_history_ = []

    def _fit_gd(self, X: np.ndarray, y: np.ndarray) -> None:
        n, d = X.shape
        # Standardise so one learning rate fits all datasets.
        mu = X.mean(axis=0) if self.fit_intercept else np.zeros(d)
        sigma = X.std(axis=0)
        sigma[sigma == 0] = 1.0
        Z = (X - mu) / sigma

        w, b = np.zeros(d), 0.0
        prev_loss = np.inf
        self.loss_history_ = []
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):  # noqa: B007 (used after loop)
            residual = y - (Z @ w + b)
            loss = float(np.mean(residual**2))
            self.loss_history_.append(loss)
            if prev_loss - loss < self.tol:
                break
            prev_loss = loss
            w += self.learning_rate * (2.0 / n) * (Z.T @ residual)
            if self.fit_intercept:
                b += self.learning_rate * (2.0 / n) * residual.sum()

        # Map back to the original feature scale.
        self.coef_ = w / sigma
        self.intercept_ = float(b - mu @ self.coef_) if self.fit_intercept else 0.0
        self.n_iter_ = n_iter

    def predict(self, X) -> np.ndarray:
        self._check_is_fitted("coef_")
        X = check_array(X)
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                f"X has {X.shape[1]} features, but the model was fitted with {self.coef_.shape[0]}."
            )
        return X @ self.coef_ + self.intercept_
