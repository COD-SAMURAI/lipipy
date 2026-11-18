r"""Gaussian mixture models fitted by expectation-maximisation (EM).

Model
-----
.. math::
    p(x) = \sum_{k=1}^{K} \pi_k\, \mathcal N(x;\ \mu_k, \Sigma_k),
    \qquad \pi_k \ge 0,\ \textstyle\sum_k \pi_k = 1 .

``covariance_type`` restricts :math:`\Sigma_k`: ``"full"`` (any SPD matrix per
component), ``"tied"`` (one shared matrix), ``"diag"`` (diagonal per component),
``"spherical"`` (:math:`\sigma_k^2 I`).

EM in log space
---------------
Each iteration first evaluates the log joint
:math:`\ell_{ik} = \log\pi_k + \log\mathcal N(x_i; \mu_k, \Sigma_k)` (E-step):

.. math::
    \log p(x_i) = \operatorname{logsumexp}_k \ell_{ik}, \qquad
    \log r_{ik} = \ell_{ik} - \log p(x_i),

so no density is ever exponentiated before normalisation and far-away points do
not underflow. The Gaussian log density uses the upper-triangular Cholesky factor
:math:`U` of the precision, :math:`U U^\top = \Sigma^{-1}`:
:math:`\log\mathcal N = -\tfrac12\bigl(d\log 2\pi + \lVert (x-\mu)^\top U\rVert^2\bigr)
+ \sum_j \log U_{jj}`. The M-step maximises the expected complete log-likelihood:

.. math::
    n_k = \textstyle\sum_i r_{ik} + 10\varepsilon_{mach},\quad
    \pi_k = n_k / \textstyle\sum_j n_j,\quad
    \mu_k = \tfrac{1}{n_k}\textstyle\sum_i r_{ik} x_i,\quad
    \Sigma_k = \tfrac{1}{n_k}\textstyle\sum_i r_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top + \lambda I,

(``"tied"``: :math:`\Sigma = \sum_k n_k\Sigma_k / \sum_k n_k`; ``"diag"``: the
diagonal only; ``"spherical"``: the mean of the diagonal), where
:math:`\lambda` = ``reg_covar`` keeps the covariances positive definite and the
tiny :math:`10\varepsilon_{mach}` avoids dividing by zero for an empty component
(both as in scikit-learn). Scatter matrices are formed from centred data
:math:`x_i - \mu_k` rather than :math:`\sum x x^\top - n\mu\mu^\top`, which avoids
cancellation.

The *lower bound* reported per iteration is the mean log-likelihood
:math:`\frac1n\sum_i \log p(x_i)` of the parameters at the start of that
iteration. EM never decreases it when ``reg_covar = 0``; with ``reg_covar > 0``
the M-step is no longer the exact maximiser and tiny decreases are possible. EM
stops when the lower bound changes by less than ``tol``.

Initialisation
--------------
Responsibilities come from k-means labels (``init_params="kmeans"``,
:class:`~lipipy.ml.KMeans` with one k-means++ start) or random uniform rows
normalised to sum to one (``"random"``); one M-step turns them into parameters.
``weights_init``, ``means_init`` and ``precisions_init`` override the
corresponding parameters. With all three given the start is fully determined.
``n_init`` restarts keep the run with the highest final lower bound.

Model selection
---------------
With :math:`L` the total log-likelihood and :math:`p` the number of free
parameters (:math:`K-1` weights, :math:`Kd` means and :math:`Kd(d+1)/2`,
:math:`d(d+1)/2`, :math:`Kd` or :math:`K` covariance parameters),
:math:`\mathrm{BIC} = -2L + p\log n` and :math:`\mathrm{AIC} = -2L + 2p`.

References
----------
Dempster, Laird & Rubin (1977), *Maximum likelihood from incomplete data via the
EM algorithm*, JRSS B 39(1); Bishop (2006), *Pattern Recognition and Machine
Learning*, §9.2; Schwarz (1978), *Estimating the dimension of a model*, Annals of
Statistics 6(2); Akaike (1974), IEEE Trans. Automatic Control 19(6).
"""

from __future__ import annotations

import numbers
import warnings
from dataclasses import dataclass

import numpy as np

from lipipy.core.base import BaseEstimator
from lipipy.core.complexity import Complexity
from lipipy.core.errors import ConvergenceWarning
from lipipy.core.random import check_random_state
from lipipy.core.registry import register
from lipipy.core.validation import check_array
from lipipy.ml._base import check_n_features
from lipipy.ml.cluster.kmeans import KMeans
from lipipy.numerics import logsumexp

_COV_TYPES = ("full", "tied", "diag", "spherical")
_INITS = ("kmeans", "random")
_SINGULAR = (
    "Fitting the mixture model failed because some components have an ill-defined "
    "covariance (for instance caused by singleton or collapsed samples). Decrease "
    "n_components, increase reg_covar, or scale the input data."
)


@dataclass(frozen=True)
class _Params:
    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray | None
    prec_chol: np.ndarray


def _chol_upper_from_covariance(cov: np.ndarray) -> np.ndarray:
    """Upper-triangular U with U U^T = cov^{-1} (U = L^{-T} for cov = L L^T)."""
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        raise ValueError(_SINGULAR) from None
    if not np.all(np.diag(L) > 0):
        raise ValueError(_SINGULAR)
    return np.linalg.solve(L, np.eye(len(cov))).T


def _chol_upper_from_precision(prec: np.ndarray) -> np.ndarray:
    """Upper-triangular U with U U^T = prec: reverse rows/cols, Cholesky, reverse back."""
    try:
        L = np.linalg.cholesky(prec[::-1, ::-1])
    except np.linalg.LinAlgError:
        raise ValueError("precisions_init must be symmetric positive definite.") from None
    return L[::-1, ::-1].copy()


def _precision_cholesky(cov: np.ndarray, cov_type: str) -> np.ndarray:
    if cov_type == "full":
        return np.stack([_chol_upper_from_covariance(c) for c in cov])
    if cov_type == "tied":
        return _chol_upper_from_covariance(cov)
    if np.any(cov <= 0) or not np.all(np.isfinite(cov)):
        raise ValueError(_SINGULAR)
    return 1.0 / np.sqrt(cov)


def _estimate_parameters(X: np.ndarray, resp: np.ndarray, reg: float, cov_type: str):
    """M-step: (unnormalised weights n_k, means, covariances)."""
    nk = resp.sum(axis=0) + 10 * np.finfo(np.float64).eps
    means = (resp.T @ X) / nk[:, None]
    K, d = means.shape
    if cov_type in ("full", "tied"):
        cov = np.empty((K, d, d))
        for k in range(K):
            diff = X - means[k]
            cov[k] = (resp[:, k] * diff.T) @ diff / nk[k]
        if cov_type == "tied":
            cov = np.tensordot(nk, cov, axes=1) / nk.sum()
            cov[np.diag_indices(d)] += reg
        else:
            cov[:, np.arange(d), np.arange(d)] += reg
    else:
        cov = np.empty((K, d))
        for k in range(K):
            diff = X - means[k]
            cov[k] = resp[:, k] @ (diff * diff) / nk[k]
        cov += reg
        if cov_type == "spherical":
            cov = cov.mean(axis=1)
    return nk, means, cov


def _log_gaussian(X: np.ndarray, means: np.ndarray, U: np.ndarray, cov_type: str) -> np.ndarray:
    """log N(x_i; mu_k, Sigma_k) for all i, k, shape (n, K)."""
    n, d = X.shape
    K = means.shape[0]
    out = np.empty((n, K))
    for k in range(K):
        diff = X - means[k]
        if cov_type == "full":
            Uk = U[k]
            y = diff @ Uk
            log_det = np.sum(np.log(np.diag(Uk)))
        elif cov_type == "tied":
            y = diff @ U
            log_det = np.sum(np.log(np.diag(U)))
        elif cov_type == "diag":
            y = diff * U[k]
            log_det = np.sum(np.log(U[k]))
        else:
            y = diff * U[k]
            log_det = d * np.log(U[k])
        out[:, k] = -0.5 * (d * np.log(2 * np.pi) + np.einsum("ij,ij->i", y, y)) + log_det
    return out


def _is_int(v) -> bool:
    return isinstance(v, numbers.Integral) and not isinstance(v, bool)


def _is_real(v) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, bool)


@register(
    domain="ml",
    tasks=("clustering", "density-estimation"),
    complexity=Complexity(
        time="O(T n k d^2 + T k d^3)",
        space="O(n k + n d + k d^2)",
        notes=(
            "per initialisation (times n_init), T EM iterations, k components; "
            "'full'/'tied' cost O(n k d^2) per iteration plus O(k d^3) Cholesky "
            "factorisations; 'diag'/'spherical' O(n k d); k-means init adds its own cost"
        ),
    ),
    references=(
        "Dempster, Laird & Rubin (1977), Maximum likelihood from incomplete data via "
        "the EM algorithm, JRSS B 39(1)",
        "Bishop (2006), Pattern Recognition and Machine Learning, §9.2",
        "Schwarz (1978), Estimating the dimension of a model, Annals of Statistics 6(2)",
    ),
    status="experimental",
)
class GaussianMixture(BaseEstimator):
    r"""Gaussian mixture model fitted by EM in log space.

    See the module docstring for the model, the E- and M-steps, initialisation
    and BIC/AIC.

    Parameters
    ----------
    n_components : int, default=1
    covariance_type : {"full", "tied", "diag", "spherical"}, default="full"
    tol : float, default=1e-3
        Stop when the lower bound (mean log-likelihood) changes by less than ``tol``.
    reg_covar : float, default=1e-6
        Added to the covariance diagonals, ``>= 0``.
    max_iter : int, default=100
        EM iterations per initialisation (>= 1).
    n_init : int, default=1
        Initialisations; the best final lower bound wins.
    init_params : {"kmeans", "random"}, default="kmeans"
    weights_init : array-like of shape (k,), optional
    means_init : array-like of shape (k, d), optional
    precisions_init : array-like, optional
        Inverse covariances: (k, d, d) for "full", (d, d) for "tied", (k, d) for
        "diag", (k,) for "spherical".
    random_state : None, int, SeedSequence or Generator, default=None
        Seeds the initialisation, and ``sample``.

    Attributes
    ----------
    weights_ : ndarray of shape (k,)
    means_ : ndarray of shape (k, d)
    covariances_ : ndarray
        Shape as for ``precisions_init``.
    precisions_ : ndarray
    precisions_cholesky_ : ndarray
        Upper-triangular factors U with U U^T = precision (square roots for
        "diag"/"spherical").
    converged_ : bool
    n_iter_ : int
    lower_bound_ : float
        Lower bound of the best run when it stopped.
    lower_bounds_ : ndarray
        Lower bound at every iteration of the best run.
    n_features_in_ : int

    Raises
    ------
    ValueError
        For invalid parameters, or when a covariance stops being positive
        definite (increase ``reg_covar``).

    Warns
    -----
    ConvergenceWarning
        If the best run reached ``max_iter`` without meeting ``tol``.

    Complexity
    ----------
    Per initialisation and EM iteration: time O(n k d²) plus O(k d³) for the
    Cholesky factors ("full"/"tied"), or O(n k d) ("diag"/"spherical"); extra
    space O(n k + n d + k d²).

    References
    ----------
    Dempster, Laird & Rubin (1977); Bishop (2006), §9.2.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.ml import GaussianMixture
    >>> rng = np.random.default_rng(0)
    >>> X = np.concatenate([rng.normal(-5, 1, (100, 1)), rng.normal(5, 1, (100, 1))])
    >>> gm = GaussianMixture(n_components=2, random_state=0).fit(X)
    >>> np.sort(gm.means_.ravel()).round(1).tolist()
    [-4.9, 4.9]
    >>> gm.predict([[-6.0], [6.0]]).tolist() == gm.predict([[-4.0], [4.0]]).tolist()
    True
    """

    def __init__(
        self,
        n_components: int = 1,
        *,
        covariance_type: str = "full",
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: str = "kmeans",
        weights_init=None,
        means_init=None,
        precisions_init=None,
        random_state=None,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init
        self.random_state = random_state

    # -- validation -------------------------------------------------------------
    def _check_params(self, X: np.ndarray):
        n, d = X.shape
        K = self.n_components
        if not _is_int(K) or not 1 <= K <= n:
            raise ValueError(f"n_components must be an integer in [1, {n}], got {K!r}.")
        if self.covariance_type not in _COV_TYPES:
            raise ValueError(f"covariance_type must be one of {_COV_TYPES}.")
        if self.init_params not in _INITS:
            raise ValueError(f"init_params must be one of {_INITS}.")
        for name in ("tol", "reg_covar"):
            v = getattr(self, name)
            if not _is_real(v) or not 0 <= v < np.inf:
                raise ValueError(f"{name} must be a finite number >= 0, got {v!r}.")
        for name in ("max_iter", "n_init"):
            v = getattr(self, name)
            if not _is_int(v) or v < 1:
                raise ValueError(f"{name} must be an integer >= 1, got {v!r}.")
        weights = means = prec_chol = None
        if self.weights_init is not None:
            weights = np.asarray(self.weights_init, dtype=np.float64)
            if weights.shape != (K,) or not np.all(np.isfinite(weights)):
                raise ValueError(f"weights_init must be a finite array of shape ({K},).")
            if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
                raise ValueError("weights_init must be non-negative and sum to 1.")
            weights = weights.copy()
        if self.means_init is not None:
            means = check_array(self.means_init)
            if means.shape != (K, d):
                raise ValueError(f"means_init must have shape ({K}, {d}), got {means.shape}.")
            means = means.copy()
        if self.precisions_init is not None:
            prec_chol = self._check_precisions(np.asarray(self.precisions_init, float), K, d)
        return weights, means, prec_chol

    def _check_precisions(self, P: np.ndarray, K: int, d: int) -> np.ndarray:
        shapes = {"full": (K, d, d), "tied": (d, d), "diag": (K, d), "spherical": (K,)}
        shape = shapes[self.covariance_type]
        if P.shape != shape or not np.all(np.isfinite(P)):
            raise ValueError(f"precisions_init must be a finite array of shape {shape}.")
        if self.covariance_type in ("diag", "spherical"):
            if np.any(P <= 0):
                raise ValueError("precisions_init must be positive.")
            return np.sqrt(P)
        mats = P if self.covariance_type == "full" else P[None]
        if not np.allclose(mats, np.swapaxes(mats, 1, 2)):
            raise ValueError("precisions_init must be symmetric positive definite.")
        U = np.stack([_chol_upper_from_precision(m) for m in mats])
        return U if self.covariance_type == "full" else U[0]

    # -- EM ---------------------------------------------------------------------
    def _e_step(self, X: np.ndarray, p: _Params) -> tuple[float, np.ndarray]:
        with np.errstate(divide="ignore"):
            joint = _log_gaussian(X, p.means, p.prec_chol, self.covariance_type) + np.log(p.weights)
        log_norm = logsumexp(joint, axis=1, keepdims=True)
        return float(np.mean(log_norm)), joint - log_norm

    def _m_step(self, X: np.ndarray, log_resp: np.ndarray) -> _Params:
        nk, means, cov = _estimate_parameters(
            X, np.exp(log_resp), float(self.reg_covar), self.covariance_type
        )
        return _Params(nk / nk.sum(), means, cov, _precision_cholesky(cov, self.covariance_type))

    def _initial(self, X, rng, weights, means, prec_chol) -> _Params:
        n, K = X.shape[0], int(self.n_components)
        if self.init_params == "kmeans":
            labels = KMeans(n_clusters=K, n_init=1, random_state=rng).fit(X).labels_
            resp = np.zeros((n, K))
            resp[np.arange(n), labels] = 1.0
        else:
            resp = rng.uniform(size=(n, K))
            resp /= resp.sum(axis=1, keepdims=True)
        nk, mu, cov = _estimate_parameters(X, resp, float(self.reg_covar), self.covariance_type)
        w = nk / n if weights is None else weights
        mu = mu if means is None else means
        if prec_chol is None:
            return _Params(w, mu, cov, _precision_cholesky(cov, self.covariance_type))
        return _Params(w, mu, None, prec_chol)

    def fit(self, X, y=None) -> GaussianMixture:
        """Estimate the mixture by EM on ``X`` of shape (n, d); ``y`` is ignored."""
        X = check_array(X)
        weights, means, prec_chol = self._check_params(X)
        rng = check_random_state(self.random_state)
        best = None
        for _ in range(int(self.n_init)):
            params = self._initial(X, rng, weights, means, prec_chol)
            bound, bounds, converged, n_iter = -np.inf, [], False, 0
            for n_iter in range(1, int(self.max_iter) + 1):  # noqa: B007 (used after loop)
                previous = bound
                bound, log_resp = self._e_step(X, params)
                params = self._m_step(X, log_resp)
                bounds.append(bound)
                if abs(bound - previous) < self.tol:
                    converged = True
                    break
            if best is None or bound > best[1]:
                best = (params, bound, bounds, converged, n_iter)
        params, bound, bounds, converged, n_iter = best
        if not converged:
            warnings.warn(
                "The best initialisation did not converge within max_iter; increase "
                "max_iter or tol, try other initial parameters, or check for degenerate data.",
                ConvergenceWarning,
                stacklevel=2,
            )
        if self.covariance_type == "full":
            precisions = params.prec_chol @ np.swapaxes(params.prec_chol, 1, 2)
        elif self.covariance_type == "tied":
            precisions = params.prec_chol @ params.prec_chol.T
        else:
            precisions = params.prec_chol**2

        self.weights_ = params.weights
        self.means_ = params.means
        self.covariances_ = params.covariances
        self.precisions_cholesky_ = params.prec_chol
        self.precisions_ = precisions
        self.converged_ = converged
        self.n_iter_ = n_iter
        self.lower_bound_ = bound
        self.lower_bounds_ = np.asarray(bounds)
        self.n_features_in_ = X.shape[1]
        return self

    # -- inference --------------------------------------------------------------
    def _params(self) -> _Params:
        self._check_is_fitted("means_")
        return _Params(self.weights_, self.means_, self.covariances_, self.precisions_cholesky_)

    def _joint(self, X) -> np.ndarray:
        p = self._params()
        X = check_n_features(X, self.n_features_in_, self)
        with np.errstate(divide="ignore"):
            return _log_gaussian(X, p.means, p.prec_chol, self.covariance_type) + np.log(p.weights)

    def score_samples(self, X) -> np.ndarray:
        """Log density :math:`\\log p(x_i)` of every row, shape (n,)."""
        return logsumexp(self._joint(X), axis=1)

    def score(self, X, y=None) -> float:
        """Mean log density of the rows of ``X``."""
        return float(np.mean(self.score_samples(X)))

    def predict_proba(self, X) -> np.ndarray:
        """Responsibilities :math:`P(k\\mid x_i)`, shape (n, k)."""
        joint = self._joint(X)
        return np.exp(joint - logsumexp(joint, axis=1, keepdims=True))

    def predict(self, X) -> np.ndarray:
        """Most probable component per row."""
        return np.argmax(self._joint(X), axis=1)

    def fit_predict(self, X, y=None) -> np.ndarray:
        """Fit, then ``predict`` the training data."""
        return self.fit(X).predict(X)

    def _n_parameters(self) -> int:
        K, d = self.means_.shape
        cov = {"full": K * d * (d + 1) // 2, "tied": d * (d + 1) // 2, "diag": K * d}.get(
            self.covariance_type, K
        )
        return int(cov + K * d + K - 1)

    def bic(self, X) -> float:
        """Bayesian information criterion :math:`-2L + p\\log n` (lower is better)."""
        n = check_n_features(X, self.n_features_in_, self).shape[0]
        return -2.0 * self.score(X) * n + self._n_parameters() * np.log(n)

    def aic(self, X) -> float:
        """Akaike information criterion :math:`-2L + 2p` (lower is better)."""
        n = check_n_features(X, self.n_features_in_, self).shape[0]
        return -2.0 * self.score(X) * n + 2.0 * self._n_parameters()

    def sample(self, n_samples: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Draw ``n_samples`` points; returns ``(X, component labels)``.

        Uses ``random_state``, so repeated calls with an integer seed repeat the draw.
        """
        p = self._params()
        if not _is_int(n_samples) or n_samples < 1:
            raise ValueError(f"n_samples must be an integer >= 1, got {n_samples!r}.")
        rng = check_random_state(self.random_state)
        counts = rng.multinomial(n_samples, p.weights / p.weights.sum())
        K, d = p.means.shape
        cov = self.covariances_  # always set: every fit runs at least one M-step
        out, labels = [], []
        for k in range(K):
            z = rng.standard_normal((counts[k], d))
            if self.covariance_type == "full":
                z = z @ np.linalg.cholesky(cov[k]).T
            elif self.covariance_type == "tied":
                z = z @ np.linalg.cholesky(cov).T
            else:
                z = z * np.sqrt(cov[k])
            out.append(p.means[k] + z)
            labels.append(np.full(counts[k], k))
        return np.concatenate(out), np.concatenate(labels)
