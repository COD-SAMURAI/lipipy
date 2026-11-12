r"""L2-regularised logistic regression (binomial and multinomial / softmax).

Model
-----
For two classes (``multi_class="auto"``) the probability of the second class is

.. math::  P(y = 1 \mid x) = \sigma(w^\top x + b), \qquad \sigma(z) = 1/(1 + e^{-z}).

For :math:`k \ge 3` classes (or ``multi_class="multinomial"``) every class has its
own weights and the probabilities are the softmax of the logits

.. math::  z_c = w_c^\top x + b_c, \qquad
           P(y = c \mid x) = \frac{e^{z_c}}{\sum_{c'} e^{z_{c'}}} .

Objective
---------
The parameters minimise the penalised negative log-likelihood (scikit-learn's
objective; the intercepts are not penalised)

.. math::  \tfrac12 \lVert W \rVert_F^2 + C \sum_{i=1}^{n} \ell_i ,
           \qquad \ell_i = -\log P(y_i \mid x_i) .

The solver minimises the same objective divided by :math:`C n`,

.. math::  f(\theta) = \frac1n \sum_i \ell_i + \frac{1}{2 C n} \lVert W \rVert_F^2 ,

so that ``tol`` is a bound on a gradient of order one whatever ``C`` and ``n`` are
(scikit-learn's ``lbfgs`` solver scales the problem the same way).

Losses in log space
-------------------
With :math:`s_i = 2 y_i - 1` the binomial loss is :math:`\ell_i = -\log\sigma(s_i z_i)`,
computed with :func:`~lipipy.numerics.log_sigmoid`; the multinomial loss is
:math:`\ell_i = -\operatorname{log\_softmax}(z_i)_{y_i}`, computed with
:func:`~lipipy.numerics.log_softmax`. Neither exponentiates a large logit, so the
objective stays finite for separable data and extreme logits.

Gradients
---------
With :math:`r_i = \sigma(z_i) - y_i` (binomial) or
:math:`R = \operatorname{softmax}(Z) - Y` (multinomial, :math:`Y` one-hot),

.. math::  \nabla_W f = \tfrac1n R^\top X + \tfrac{1}{Cn} W, \qquad
           \nabla_b f = \tfrac1n \textstyle\sum_i r_i .

The objective is strictly convex in :math:`W` and is minimised with
:func:`~lipipy.numerics.minimize_lbfgs`, starting from zero.

Newton-CG polish at the floating-point limit
--------------------------------------------
L-BFGS accepts a step only if it decreases :math:`f` (Armijo). Near the optimum the
decrease, about :math:`\lVert g\rVert^2/\lambda`, falls below the rounding error of
:math:`f \approx O(1)` once :math:`\lVert g\rVert \approx 10^{-8}`, so the line search
stalls just above a tight ``tol`` (where exactly depends on the platform's BLAS).
Gradients do not suffer from this cancellation, so we finish with Newton steps
:math:`H s = -g` solved by conjugate gradients from exact Hessian-vector products

.. math::  H v = \tfrac1n X^\top \bigl(D \odot (X v_W + v_b)\bigr) + \tfrac{1}{Cn} v_W,

with :math:`D_i = p_i(1-p_i)` (binomial) or, per sample, the softmax Jacobian
:math:`\operatorname{diag}(p_i) - p_i p_i^\top` (multinomial). A Newton step is
accepted only if it reduces :math:`\lVert g\rVert_\infty`; near the optimum each one
reduces it quadratically (Nocedal & Wright §7.1). Each product costs O(n d k) time
and O(n k) memory, like a gradient.

Uniqueness of the multinomial intercepts
----------------------------------------
Adding the same constant to every :math:`b_c` leaves the softmax unchanged and the
intercepts are not penalised, so they are defined only up to that shift. We report
the representative with :math:`\sum_c b_c = 0`. (Starting from zero, every gradient
of the intercepts already sums to zero, so the iterates stay in that subspace; the
final centring only removes rounding drift.)

References
----------
Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, 2nd ed.,
§4.4; Bishop, *Pattern Recognition and Machine Learning* (2006), §4.3.2-4.3.4;
Nocedal & Wright, *Numerical Optimization*, 2nd ed., Alg. 7.4-7.5 (L-BFGS).
"""

from __future__ import annotations

import numbers
import warnings
from functools import partial

import numpy as np

from lipipy.core.base import BaseEstimator
from lipipy.core.complexity import Complexity
from lipipy.core.errors import ConvergenceWarning
from lipipy.core.registry import register
from lipipy.ml._base import ClassifierMixin, check_n_features, check_X_labels
from lipipy.numerics import log_sigmoid, log_softmax, minimize_lbfgs, sigmoid

_MULTI_CLASS = ("auto", "multinomial")
_CG_MAX_ITER = 50  # per Newton step; with <= 10 steps the polish is O(n d k) work


class _Objective:
    """Penalised mean log-loss and its gradient, evaluated together and cached.

    L-BFGS asks for ``fun`` at every trial point of the line search and for ``jac``
    only at the accepted point, which is always the last ``fun`` evaluation, so the
    cache turns every gradient call into a lookup.
    """

    def __init__(self, X, y_idx, n_classes, multinomial, fit_intercept, C):
        self.X = X
        self.n, self.d = X.shape
        self.k = n_classes if multinomial else 1
        self.multinomial = multinomial
        self.fit_intercept = fit_intercept
        self.l2 = 1.0 / (C * self.n)
        if multinomial:
            self.y_idx = y_idx
        else:
            self.y = y_idx.astype(np.float64)  # 0 / 1
            self.sign = 2.0 * self.y - 1.0  # -1 / +1
        self._key: bytes | None = None
        self._value = (0.0, np.empty(0))

    @property
    def size(self) -> int:
        return self.k * self.d + (self.k if self.fit_intercept else 0)

    def unpack(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        W = theta[: self.k * self.d].reshape(self.k, self.d)
        b = theta[self.k * self.d :] if self.fit_intercept else np.zeros(self.k)
        return W, b

    def _evaluate(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        key = theta.tobytes()
        if key == self._key:
            return self._value
        W, b = self.unpack(theta)
        Z = self.X @ W.T + b  # (n, k): the only O(n k) temporaries are Z and R
        if self.multinomial:
            log_p = log_softmax(Z, axis=1)
            loss = -float(np.mean(log_p[np.arange(self.n), self.y_idx]))
            R = np.exp(log_p)  # softmax(Z)
            R[np.arange(self.n), self.y_idx] -= 1.0
        else:
            z = Z[:, 0]
            loss = -float(np.mean(log_sigmoid(self.sign * z)))
            R = (sigmoid(z) - self.y)[:, None]
        loss += 0.5 * self.l2 * float(np.sum(W * W))
        grad_W = (R.T @ self.X) / self.n + self.l2 * W
        if self.fit_intercept:
            grad = np.concatenate([grad_W.ravel(), R.mean(axis=0)])
        else:
            grad = grad_W.ravel()
        self._key, self._value = key, (loss, grad)
        return loss, grad

    def hessp(self, theta: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Hessian-vector product :math:`H(\theta) v` in O(n d k) time, O(n k) memory."""
        W, b = self.unpack(theta)
        VW, vb = self.unpack(v)
        Z = self.X @ W.T + b
        U = self.X @ VW.T + vb  # directional derivative of the logits
        if self.multinomial:
            P = np.exp(log_softmax(Z, axis=1))
            HU = P * U
            HU -= P * HU.sum(axis=1, keepdims=True)  # (diag(p) - p p^T) u, row-wise
        else:
            p = sigmoid(Z[:, 0])
            HU = (p * (1.0 - p))[:, None] * U
        HW = (HU.T @ self.X) / self.n + self.l2 * VW
        if self.fit_intercept:
            return np.concatenate([HW.ravel(), HU.mean(axis=0)])
        return HW.ravel()

    def fun(self, theta: np.ndarray) -> float:
        return self._evaluate(theta)[0]

    def jac(self, theta: np.ndarray) -> np.ndarray:
        return self._evaluate(theta)[1]


def _conjugate_gradient(hessp, g: np.ndarray, max_iter: int) -> np.ndarray:
    """Approximately solve ``H s = -g`` for positive semi-definite ``H`` (CG, N&W Alg. 5.2)."""
    s = np.zeros_like(g)
    r = -g.copy()  # residual -g - H s
    p = r.copy()
    rr = float(r @ r)
    stop = 1e-12 * rr
    for _ in range(max_iter):
        Hp = hessp(p)
        curvature = float(p @ Hp)
        if not curvature > 0:  # flat direction (e.g. the multinomial intercept shift)
            break
        alpha = rr / curvature
        s += alpha * p
        r -= alpha * Hp
        rr_new = float(r @ r)
        if rr_new <= stop:
            break
        p = r + (rr_new / rr) * p
        rr = rr_new
    return s


def _newton_polish(obj: _Objective, theta: np.ndarray, tol: float, max_steps: int = 10):
    """Newton-CG steps from ``theta``, each kept only if it lowers the gradient norm."""
    g = obj.jac(theta)
    steps = 0
    while steps < max_steps and np.max(np.abs(g)) > tol:
        s = _conjugate_gradient(partial(obj.hessp, theta), g, max_iter=_CG_MAX_ITER)
        theta_new = theta + s
        g_new = obj.jac(theta_new)
        if not np.max(np.abs(g_new)) < np.max(np.abs(g)):
            break
        theta, g = theta_new, g_new
        steps += 1
    return theta, g, steps


@register(
    domain="ml",
    tasks=("classification",),
    complexity=Complexity(
        time="O(T n d k)",
        space="O(n k + d k)",
        notes=(
            "T L-BFGS iterations, each O(n d k) for the logits and the gradient plus "
            "line-search evaluations of the same cost; k = 1 for the binomial model; "
            "the L-BFGS memory adds O(m d k) with m = 10; the optional Newton-CG "
            "polish adds at most 10 x 50 Hessian-vector products, each O(n d k) time "
            "and O(n k) memory"
        ),
    ),
    references=(
        "Hastie, Tibshirani & Friedman, The Elements of Statistical Learning, 2nd ed., §4.4",
        "Bishop (2006), Pattern Recognition and Machine Learning, §4.3.2-4.3.4",
        "Liu & Nocedal (1989), On the limited memory BFGS method for large scale "
        "optimization, Mathematical Programming 45",
    ),
    status="experimental",
)
class LogisticRegression(ClassifierMixin, BaseEstimator):
    r"""Logistic regression with an L2 penalty, fitted by L-BFGS.

    Minimises :math:`\tfrac12\lVert W\rVert^2 + C\sum_i -\log P(y_i\mid x_i)` (the
    intercept is not penalised) with analytic gradients and log-space losses; see
    the module docstring for the derivation.

    Parameters
    ----------
    C : float, default=1.0
        Inverse regularisation strength, ``C > 0``; smaller is stronger.
    fit_intercept : bool, default=True
        Whether to learn unpenalised intercepts.
    multi_class : {"auto", "multinomial"}, default="auto"
        ``"auto"``: binomial model for two classes, multinomial (softmax) otherwise.
        ``"multinomial"``: softmax model for any number of classes ≥ 2.
    max_iter : int, default=1000
        Maximum number of L-BFGS iterations.
    tol : float, default=1e-8
        Stop when the infinity norm of the gradient of the scaled objective
        :math:`f` (module docstring) is at most ``tol``. If L-BFGS stalls at the
        floating-point resolution of :math:`f` first, Newton-CG steps finish the
        job (module docstring), so ``tol`` is met unless a warning is issued. The
        bound is absolute and the gradient scales
        with the features, so very small features need a proportionally smaller
        ``tol``; features many orders of magnitude below one also make the problem
        ill-conditioned (the curvature in W is tiny next to the intercept's), which
        limits the attainable accuracy in float64: standardise them.

    Attributes
    ----------
    classes_ : ndarray of shape (k,)
        Sorted class labels.
    coef_ : ndarray of shape (1, d) for the binomial model, (k, d) otherwise
    intercept_ : ndarray of shape (1,) or (k,)
        Zeros when ``fit_intercept=False``; multinomial intercepts sum to zero.
    n_iter_ : int
        L-BFGS iterations plus accepted Newton-CG polish steps.
    n_features_in_ : int

    Raises
    ------
    ValueError
        For invalid hyperparameters, invalid ``X``/``y`` or fewer than two classes.

    Warns
    -----
    ConvergenceWarning
        If the gradient infinity-norm is still above ``tol`` after ``max_iter``
        L-BFGS iterations, or after the Newton-CG polish stops making progress.

    Complexity
    ----------
    ``fit``: time O(n d k) per L-BFGS iteration (logits ``X W^T`` and the gradient
    ``R^T X``), plus at most 500 Hessian-vector products of the same cost in the
    Newton-CG polish; extra space O(n k + d k): the logits and residuals, never a
    copy of ``X`` or an n × n matrix. ``predict*``: time O(n d k), extra space O(n k).

    References
    ----------
    Hastie, Tibshirani & Friedman, *ESL* 2e, §4.4; Bishop (2006), §4.3.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.ml import LogisticRegression
    >>> X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    >>> y = np.array([0, 0, 1, 1])
    >>> clf = LogisticRegression(C=1.0).fit(X, y)
    >>> clf.predict([[-3.0], [0.5]]).tolist()
    [0, 1]
    >>> clf.predict_proba([[0.0]]).round(6).tolist()
    [[0.5, 0.5]]
    """

    def __init__(
        self,
        C: float = 1.0,
        fit_intercept: bool = True,
        multi_class: str = "auto",
        max_iter: int = 1000,
        tol: float = 1e-8,
    ):
        self.C = C
        self.fit_intercept = fit_intercept
        self.multi_class = multi_class
        self.max_iter = max_iter
        self.tol = tol

    def _check_params(self) -> None:
        C = self.C
        if isinstance(C, bool) or not isinstance(C, numbers.Real) or not (0 < C < np.inf):
            raise ValueError(f"C must be a finite number > 0, got {C!r}.")
        if not isinstance(self.fit_intercept, (bool, np.bool_)):
            raise TypeError(f"fit_intercept must be a bool, got {self.fit_intercept!r}.")
        if self.multi_class not in _MULTI_CLASS:
            raise ValueError(
                f"multi_class must be one of {_MULTI_CLASS}, got {self.multi_class!r}."
            )
        it = self.max_iter
        if isinstance(it, bool) or not isinstance(it, numbers.Integral) or it < 1:
            raise ValueError(f"max_iter must be an integer >= 1, got {it!r}.")
        tol = self.tol
        if isinstance(tol, bool) or not isinstance(tol, numbers.Real) or not (0 < tol < np.inf):
            raise ValueError(f"tol must be a finite number > 0, got {tol!r}.")

    def fit(self, X, y) -> LogisticRegression:
        """Fit to features ``X`` (n, d) and labels ``y`` (n,); return ``self``."""
        self._check_params()
        X, y = check_X_labels(X, y)
        classes, y_idx = np.unique(y, return_inverse=True)
        if len(classes) < 2:
            raise ValueError(
                f"LogisticRegression needs samples of at least 2 classes, got {len(classes)}."
            )
        multinomial = self.multi_class == "multinomial" or len(classes) > 2
        obj = _Objective(X, y_idx, len(classes), multinomial, bool(self.fit_intercept), self.C)

        with warnings.catch_warnings():
            # minimize_lbfgs warns from inside the solver; we re-issue one warning that
            # names this estimator and its parameters instead.
            warnings.simplefilter("ignore", ConvergenceWarning)
            res = minimize_lbfgs(
                obj.fun,
                np.zeros(obj.size),
                obj.jac,
                tol=float(self.tol),
                maxiter=int(self.max_iter),
            )
        theta, grad, n_newton = res.x, res.grad, 0
        if not res.success and res.nit < int(self.max_iter):
            # The line search stalled at the floating-point resolution of f (module
            # docstring): finish with gradient-based Newton-CG steps.
            theta, grad, n_newton = _newton_polish(obj, theta, float(self.tol))
        if not np.max(np.abs(grad)) <= self.tol:
            warnings.warn(
                f"LogisticRegression did not converge ({res.message}; gradient "
                f"infinity-norm {np.max(np.abs(grad)):.3g} > tol={self.tol}). "
                "Increase max_iter, or scale the features.",
                ConvergenceWarning,
                stacklevel=2,
            )
        W, b = obj.unpack(theta)
        b = b.copy()
        if multinomial and self.fit_intercept:
            b -= b.mean()  # pick the unique representative with sum(b) = 0

        # All state is assigned only after the whole fit succeeded.
        self.classes_ = classes
        self.coef_ = W.copy()
        self.intercept_ = b
        self.n_iter_ = int(res.nit) + n_newton
        self.n_features_in_ = X.shape[1]
        return self

    def decision_function(self, X) -> np.ndarray:
        """Logits: shape (n,) for the binomial model, (n, k) for the multinomial one."""
        self._check_is_fitted("coef_")
        X = check_n_features(X, self.n_features_in_, self)
        Z = X @ self.coef_.T + self.intercept_
        return Z[:, 0] if Z.shape[1] == 1 else Z

    def predict_log_proba(self, X) -> np.ndarray:
        """Log class probabilities, shape (n, k), columns ordered as ``classes_``."""
        Z = self.decision_function(X)
        if Z.ndim == 1:
            return np.column_stack([log_sigmoid(-Z), log_sigmoid(Z)])
        return log_softmax(Z, axis=1)

    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities, shape (n, k), columns ordered as ``classes_``."""
        return np.exp(self.predict_log_proba(X))

    def predict(self, X) -> np.ndarray:
        """Most probable class for each row of ``X``."""
        Z = self.decision_function(X)
        idx = (Z > 0).astype(np.intp) if Z.ndim == 1 else np.argmax(Z, axis=1)
        return self.classes_[idx]
