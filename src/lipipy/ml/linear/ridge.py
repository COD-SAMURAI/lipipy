r"""Ridge regression (L2-regularised least squares).

Model
-----
.. math::  \hat y = X w + b

``w`` and ``b`` minimise the penalised residual sum of squares

.. math::  L(w, b) = \lVert y - Xw - b \rVert_2^2 + \alpha \lVert w \rVert_2^2 ,

where the intercept ``b`` is not penalised.

Solution
--------
Centring :math:`X_c = X - \bar x` and :math:`y_c = y - \bar y` removes the
intercept. Setting the gradient to zero gives the regularised normal equations

.. math::  (X_c^\top X_c + \alpha I)\, w = X_c^\top y_c ,

whose matrix is symmetric positive definite for :math:`\alpha > 0`. We factor it
with Cholesky, :math:`L L^\top = X_c^\top X_c + \alpha I`, and solve the two
triangular systems. The intercept is :math:`b = \bar y - \bar x^\top w`.

With :math:`\alpha = 0` this is ordinary least squares and is accepted only when
:math:`X_c^\top X_c` is non-singular (``X_c`` has full column rank); otherwise use
:class:`~lipipy.ml.LinearRegression`, which handles rank deficiency.

Reference: Hoerl & Kennard (1970), *Ridge regression: biased estimation for
nonorthogonal problems*, Technometrics 12(1); Hastie, Tibshirani & Friedman,
*The Elements of Statistical Learning*, 2nd ed., §3.4.1.
"""

from __future__ import annotations

import numbers

import numpy as np

from lipipy.core.base import BaseEstimator, RegressorMixin
from lipipy.core.complexity import Complexity
from lipipy.core.registry import register
from lipipy.core.validation import check_X_y
from lipipy.ml._base import check_n_features


def _solve_spd(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve ``A x = b`` for symmetric positive definite ``A`` via Cholesky.

    Raises :class:`numpy.linalg.LinAlgError` if ``A`` is not positive definite.
    """
    L = np.linalg.cholesky(A)  # A = L L^T
    z = np.linalg.solve(L, b)  # forward substitution  L z = b
    return np.linalg.solve(L.T, z)  # back substitution  L^T x = z


@register(
    domain="ml",
    tasks=("regression",),
    complexity=Complexity(
        time="O(n d^2 + d^3)",
        space="O(n d + d^2)",
        notes="forming X^T X costs O(n d^2), Cholesky O(d^3); one centred copy of X",
    ),
    references=(
        "Hoerl & Kennard (1970), Ridge regression: biased estimation for nonorthogonal "
        "problems, Technometrics 12(1)",
        "Hastie, Tibshirani & Friedman, The Elements of Statistical Learning, 2nd ed., §3.4.1",
    ),
    status="experimental",
)
class Ridge(RegressorMixin, BaseEstimator):
    r"""Linear least squares with an L2 penalty on the weights.

    Minimises :math:`\lVert y - Xw - b\rVert^2 + \alpha\lVert w\rVert^2` in closed
    form by solving :math:`(X_c^\top X_c + \alpha I) w = X_c^\top y_c` with a
    Cholesky factorisation (see the module docstring for the derivation).

    Parameters
    ----------
    alpha : float, default=1.0
        Regularisation strength, ``alpha >= 0``. ``alpha=0`` is plain least squares
        and requires ``X`` (centred, if ``fit_intercept``) to have full column rank.
    fit_intercept : bool, default=True
        Whether to learn an unpenalised bias term ``b``.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Learned weights ``w``.
    intercept_ : float
        Learned bias ``b`` (0.0 when ``fit_intercept=False``).
    n_features_in_ : int
        Number of features seen in ``fit``.

    Complexity
    ----------
    Time O(n d^2 + d^3): O(n d^2) to form :math:`X^\top X`, O(d^3) for Cholesky.
    Extra space O(n d + d^2) for the centred copy of ``X`` and the Gram matrix.

    References
    ----------
    Hoerl, A. E. & Kennard, R. W. (1970). Ridge regression. *Technometrics* 12(1).

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.ml import Ridge
    >>> X = np.array([[0.0], [1.0], [2.0]])
    >>> y = np.array([0.0, 1.0, 2.0])
    >>> model = Ridge(alpha=1.0).fit(X, y)
    >>> round(float(model.coef_[0]), 6), round(model.intercept_, 6)
    (0.666667, 0.333333)
    """

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y) -> Ridge:
        """Fit the model to features ``X`` (n, d) and targets ``y`` (n,); return ``self``."""
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, numbers.Real):
            raise TypeError(f"alpha must be a real number, got {type(self.alpha).__name__}.")
        alpha = float(self.alpha)
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError(f"alpha must be a finite number >= 0, got {self.alpha!r}.")
        X, y = check_X_y(X, y)
        d = X.shape[1]

        if self.fit_intercept:
            x_mean, y_mean = X.mean(axis=0), y.mean()
            Xc, yc = X - x_mean, y - y_mean
        else:
            x_mean, y_mean = np.zeros(d), 0.0
            Xc, yc = X, y

        gram = Xc.T @ Xc
        gram[np.diag_indices(d)] += alpha
        rhs = Xc.T @ yc
        singular_msg = (
            "X^T X is singular (X has linearly dependent columns"
            + (" after centring" if self.fit_intercept else "")
            + "), so alpha=0 has no unique solution. Use alpha > 0 or LinearRegression."
        )
        if alpha == 0.0 and np.linalg.matrix_rank(Xc) < d:
            raise ValueError(singular_msg)
        try:
            w = _solve_spd(gram, rhs)
        except np.linalg.LinAlgError:
            raise ValueError(singular_msg) from None

        self.coef_ = w
        self.intercept_ = float(y_mean - x_mean @ w)
        self.n_features_in_ = d
        return self

    def predict(self, X) -> np.ndarray:
        """Predict targets :math:`Xw + b` for ``X`` of shape (n, d)."""
        self._check_is_fitted("coef_")
        X = check_n_features(X, self.n_features_in_, self)
        return X @ self.coef_ + self.intercept_
