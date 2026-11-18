r"""C-support vector classification solved with SMO (LIBSVM's algorithm).

Primal and dual
---------------
For labels :math:`y_i \in \{-1, +1\}` and a feature map :math:`\phi` with kernel
:math:`K(x, z) = \phi(x)^\top \phi(z)`, the soft-margin SVM solves

.. math::
    \min_{w, b, \xi}\ \tfrac12 \lVert w\rVert^2 + C \sum_i \xi_i
    \quad\text{s.t.}\quad y_i (w^\top \phi(x_i) + b) \ge 1 - \xi_i,\ \xi_i \ge 0 ,

whose dual, with :math:`Q_{ij} = y_i y_j K(x_i, x_j)`, is

.. math::
    \min_{\alpha}\ f(\alpha) = \tfrac12 \alpha^\top Q \alpha - \mathbf 1^\top \alpha
    \quad\text{s.t.}\quad 0 \le \alpha_i \le C,\ y^\top \alpha = 0 .

The decision function is :math:`f(x) = \sum_i y_i \alpha_i K(x_i, x) - \rho`; the
training points with :math:`\alpha_i > 0` are the *support vectors*.

SMO with second-order working-set selection
-------------------------------------------
Sequential minimal optimisation changes two multipliers per iteration, the
smallest change that keeps :math:`y^\top\alpha = 0`. With the gradient
:math:`G = Q\alpha - \mathbf 1` (initially :math:`\alpha = 0,\ G = -\mathbf 1`) and

.. math::
    I_{up} = \{t : y_t = 1, \alpha_t < C\} \cup \{t : y_t = -1, \alpha_t > 0\},\quad
    I_{low} = \{t : y_t = 1, \alpha_t > 0\} \cup \{t : y_t = -1, \alpha_t < C\},

the pair is chosen as in Fan, Chen & Lin (2005), the rule LIBSVM uses:

.. math::
    i = \arg\max_{t \in I_{up}} -y_t G_t, \qquad
    j = \arg\min_{t \in I_{low},\ -y_t G_t < -y_i G_i} -\frac{b_{it}^2}{a_{it}},

with :math:`b_{it} = -y_i G_i + y_t G_t > 0` and
:math:`a_{it} = K_{ii} + K_{tt} - 2K_{it}` (replaced by :math:`\tau = 10^{-12}` if
not positive). :math:`-b^2/a` is the exact decrease of :math:`f` along the pair's
feasible direction, so this second-order rule picks the pair that improves the
objective most among those that contain the maximal violator :math:`i`. The
two-variable subproblem is solved in closed form and clipped to the box (LIBSVM's
update), and :math:`G` is updated with two kernel columns. Iteration stops when the
maximal KKT violation :math:`m(\alpha) - M(\alpha) = \max_{I_{up}} -y G +
\max_{I_{low}} y G` is below ``tol``. Ties in the arg max / arg min are resolved
as LIBSVM does (the last index wins).

The offset is :math:`\rho = \operatorname{mean}_{0<\alpha_t<C} y_t G_t` over free
multipliers, or the midpoint of the feasible interval if there are none (LIBSVM's
``calculate_rho``).

Kernel cache
------------
Kernel columns :math:`K(X, x_t)` are computed on demand and kept in a
least-recently-used cache holding at most
:math:`\max(2, \lfloor \text{cache\_size} \cdot 2^{20} / (8 n_p) \rfloor)`
columns of :math:`n_p` float64 values, where :math:`n_p` is the number of samples
in the binary problem. So the cache never uses more than ``cache_size`` MiB
(except for the two columns every iteration needs), and the full
:math:`n_p \times n_p` kernel matrix is never formed unless it fits.

Multiclass: one-vs-one
----------------------
For :math:`k` classes a binary SVM is trained for every pair :math:`(a, b)`,
:math:`a < b`, with :math:`y = +1` for class :math:`a`. ``predict`` takes a vote: each
pair votes for :math:`a` if its decision value is positive, else for :math:`b`; ties go
to the class that sorts first. ``decision_function_shape="ovr"`` (the default)
turns the pairwise values into per-class scores like scikit-learn
(votes plus a confidence term in :math:`(-1/3, 1/3)`); ``"ovo"`` returns the
:math:`k(k-1)/2` pairwise values. The attribute layout (``support_`` grouped by
class, ``dual_coef_`` of shape :math:`(k-1, n_{SV})`, ``intercept_`` per pair)
follows LIBSVM / scikit-learn, and for two classes the signs are flipped so that a
positive decision value means the second class.

Uniqueness
----------
The decision values are unique (the dual is convex and the primal :math:`w` is
unique), but the multipliers need not be: with duplicated points, or when no
multiplier is free, several :math:`\alpha` give the same classifier.

References
----------
Platt (1998), *Sequential minimal optimization*, MSR-TR-98-14; Fan, Chen & Lin
(2005), *Working set selection using second order information for training SVM*,
JMLR 6; Chang & Lin (2011), *LIBSVM: a library for support vector machines*, ACM
TIST 2(3); Cortes & Vapnik (1995), *Support-vector networks*, Machine Learning 20.
"""

from __future__ import annotations

import numbers
import warnings
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from lipipy.core.base import BaseEstimator
from lipipy.core.complexity import Complexity
from lipipy.core.errors import ConvergenceWarning
from lipipy.core.registry import register
from lipipy.ml._base import ClassifierMixin, check_n_features, check_X_labels

_KERNELS = ("linear", "poly", "rbf", "sigmoid")
_TAU = 1e-12


def _is_real(v) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, bool)


def _is_int(v) -> bool:
    return isinstance(v, numbers.Integral) and not isinstance(v, bool)


@dataclass(frozen=True)
class _Kernel:
    """A kernel function with resolved hyperparameters."""

    name: str
    gamma: float
    degree: int
    coef0: float

    def __call__(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Kernel matrix between rows of ``A`` (m, d) and ``B`` (p, d)."""
        if self.name == "linear":
            return A @ B.T
        if self.name == "poly":
            return (self.gamma * (A @ B.T) + self.coef0) ** self.degree
        if self.name == "sigmoid":
            return np.tanh(self.gamma * (A @ B.T) + self.coef0)
        sq = (
            np.einsum("ij,ij->i", A, A)[:, None]
            - 2.0 * (A @ B.T)
            + np.einsum("ij,ij->i", B, B)[None, :]
        )
        return np.exp(-self.gamma * np.maximum(sq, 0.0))

    def diagonal(self, A: np.ndarray) -> np.ndarray:
        """:math:`K(a_i, a_i)` for every row, O(n d)."""
        if self.name == "rbf":
            return np.ones(A.shape[0])
        sq = np.einsum("ij,ij->i", A, A)
        if self.name == "linear":
            return sq
        if self.name == "poly":
            return (self.gamma * sq + self.coef0) ** self.degree
        return np.tanh(self.gamma * sq + self.coef0)


class _ColumnCache:
    """LRU cache of kernel columns ``K(X, x_t)`` bounded to ``max_columns`` columns."""

    def __init__(self, X: np.ndarray, kernel: _Kernel, max_columns: int):
        self.X = X
        self.kernel = kernel
        self.max_columns = max_columns
        self._columns: OrderedDict[int, np.ndarray] = OrderedDict()

    def column(self, t: int) -> np.ndarray:
        col = self._columns.get(t)
        if col is not None:
            self._columns.move_to_end(t)
            return col
        col = self.kernel(self.X, self.X[t : t + 1])[:, 0]
        self._columns[t] = col
        if len(self._columns) > self.max_columns:
            self._columns.popitem(last=False)
        return col


@dataclass(frozen=True)
class _BinarySolution:
    alpha: np.ndarray  # multipliers, shape (n_p,)
    rho: float
    n_iter: int
    converged: bool


def _smo(
    X: np.ndarray, y: np.ndarray, C: float, kernel: _Kernel, tol: float, max_iter: int, cache_mb
) -> _BinarySolution:
    """Solve the C-SVC dual for ``y`` in {-1, +1} with LIBSVM's SMO (module docstring)."""
    n = X.shape[0]
    max_columns = max(2, int(cache_mb * 2**20) // (8 * n))
    cache = _ColumnCache(X, kernel, max_columns)
    QD = kernel.diagonal(X)
    alpha = np.zeros(n)
    G = -np.ones(n)
    pos = y > 0
    n_iter = 0
    converged = False
    while max_iter < 0 or n_iter < max_iter:
        at_upper = alpha >= C
        at_lower = alpha <= 0.0
        up = np.where(pos, ~at_upper, ~at_lower)  # I_up
        low = np.where(pos, ~at_lower, ~at_upper)  # I_low
        minus_yG = -y * G
        if not up.any() or not low.any():
            converged = True
            break
        # i: maximal violator in I_up (last index on ties, like LIBSVM's >=).
        cand = np.where(up, minus_yG, -np.inf)
        i = n - 1 - int(np.argmax(cand[::-1]))
        g_max = cand[i]
        g_max2 = float(np.max(np.where(low, -minus_yG, -np.inf)))
        if g_max + g_max2 < tol:
            converged = True
            break
        # j: second-order choice among I_low with a positive gradient difference.
        K_i = cache.column(i)
        b = g_max - minus_yG  # = -y_i G_i + y_t G_t
        a = QD[i] + QD - 2.0 * K_i
        a = np.where(a > 0, a, _TAU)
        valid = low & (b > 0)
        if not valid.any():
            converged = True
            break
        obj = np.where(valid, -(b * b) / a, np.inf)
        j = n - 1 - int(np.argmin(obj[::-1]))
        K_j = cache.column(j)

        # Two-variable subproblem in closed form, clipped to the box (LIBSVM).
        yi, yj = y[i], y[j]
        Qij = yi * yj * K_i[j]
        old_i, old_j = alpha[i], alpha[j]
        if yi != yj:
            quad = QD[i] + QD[j] + 2.0 * Qij
            quad = quad if quad > 0 else _TAU
            delta = (-G[i] - G[j]) / quad
            diff = old_i - old_j
            ai, aj = old_i + delta, old_j + delta
            if diff > 0:
                if aj < 0:
                    aj, ai = 0.0, diff
            elif ai < 0:
                ai, aj = 0.0, -diff
            if diff > 0:  # C_i - C_j == 0
                if ai > C:
                    ai, aj = C, C - diff
            elif aj > C:
                aj, ai = C, C + diff
        else:
            quad = QD[i] + QD[j] - 2.0 * Qij
            quad = quad if quad > 0 else _TAU
            delta = (G[i] - G[j]) / quad
            total = old_i + old_j
            ai, aj = old_i - delta, old_j + delta
            if total > C:
                if ai > C:
                    ai, aj = C, total - C
            elif aj < 0:
                aj, ai = 0.0, total
            if total > C:
                if aj > C:
                    aj, ai = C, total - C
            elif ai < 0:
                ai, aj = 0.0, total
        alpha[i], alpha[j] = ai, aj
        # G += Q_i (ai - old_i) + Q_j (aj - old_j), with Q_t = y_t y * K_t
        G += y * (K_i * (yi * (ai - old_i)) + K_j * (yj * (aj - old_j)))
        n_iter += 1

    # Offset rho (LIBSVM calculate_rho).
    yG = y * G
    upper = alpha >= C
    lower = alpha <= 0.0
    free = ~upper & ~lower
    if free.any():
        rho = float(np.mean(yG[free]))
    else:
        ub_mask = (upper & ~pos) | (lower & pos)
        lb_mask = (upper & pos) | (lower & ~pos)
        ub = float(np.min(yG[ub_mask])) if ub_mask.any() else np.inf
        lb = float(np.max(yG[lb_mask])) if lb_mask.any() else -np.inf
        rho = (ub + lb) / 2.0
    return _BinarySolution(alpha, rho, n_iter, converged)


def _ovr_decision(pairwise: np.ndarray, n_classes: int) -> np.ndarray:
    """scikit-learn's one-vs-rest scores from one-vs-one decision values."""
    n = pairwise.shape[0]
    votes = np.zeros((n, n_classes))
    confidence = np.zeros((n, n_classes))
    p = 0
    for a in range(n_classes):
        for b in range(a + 1, n_classes):
            confidence[:, a] += pairwise[:, p]
            confidence[:, b] -= pairwise[:, p]
            votes[pairwise[:, p] >= 0, a] += 1  # scikit-learn counts 0 for the first class
            votes[pairwise[:, p] < 0, b] += 1
            p += 1
    return votes + confidence / (3.0 * (np.abs(confidence) + 1.0))


@register(
    domain="ml",
    tasks=("classification",),
    complexity=Complexity(
        time="O(T n d)",
        space="O(B n + n d)",
        notes=(
            "T SMO iterations per binary problem, each O(n) plus at most two kernel "
            "columns O(n d) on cache misses; k(k-1)/2 one-vs-one problems over the "
            "classes' samples; B = cached kernel columns <= cache_size MiB / (8 n); "
            "predict is O(n_SV d) per sample"
        ),
    ),
    references=(
        "Platt (1998), Sequential minimal optimization, Microsoft Research MSR-TR-98-14",
        "Fan, Chen & Lin (2005), Working set selection using second order information "
        "for training support vector machines, JMLR 6",
        "Chang & Lin (2011), LIBSVM: a library for support vector machines, ACM TIST 2(3)",
        "Cortes & Vapnik (1995), Support-vector networks, Machine Learning 20",
    ),
    status="experimental",
)
class SVC(ClassifierMixin, BaseEstimator):
    r"""C-support vector classifier (SMO with second-order working-set selection).

    Solves the soft-margin dual with LIBSVM's SMO algorithm for every pair of
    classes (one-vs-one); see the module docstring for the derivation, the
    stopping rule, the kernel cache bound and the multiclass layout.

    Kernels: ``"linear"`` :math:`x^\top z`; ``"poly"``
    :math:`(\gamma x^\top z + c_0)^{p}`; ``"rbf"``
    :math:`\exp(-\gamma \lVert x - z\rVert^2)`; ``"sigmoid"``
    :math:`\tanh(\gamma x^\top z + c_0)`.

    Parameters
    ----------
    C : float, default=1.0
        Penalty on margin violations, ``C > 0``.
    kernel : {"linear", "poly", "rbf", "sigmoid"}, default="rbf"
    degree : int, default=3
        Degree ``p`` of the polynomial kernel (>= 0).
    gamma : {"scale", "auto"} or float, default="scale"
        ``"scale"``: :math:`1 / (d \operatorname{Var}(X))` (1 if the variance of all
        entries of ``X`` is 0); ``"auto"``: :math:`1/d`; or a number > 0.
    coef0 : float, default=0.0
        :math:`c_0` in the poly and sigmoid kernels.
    tol : float, default=1e-3
        Stop when the maximal KKT violation :math:`m(\alpha) - M(\alpha)` < ``tol``.
    max_iter : int, default=-1
        SMO iterations per binary problem; -1 means no limit (SMO with this
        working-set rule always converges).
    cache_size : float, default=200
        Kernel-cache budget in MiB per binary problem.
    decision_function_shape : {"ovr", "ovo"}, default="ovr"
        Shape of ``decision_function`` for more than two classes.
    random_state : None, int, SeedSequence or Generator, default=None
        Accepted for compatibility with scikit-learn's signature. SMO is
        deterministic, so it has no effect.

    Attributes
    ----------
    classes_ : ndarray of shape (k,)
    support_ : ndarray of shape (n_SV,)
        Indices of the support vectors, grouped by class then by index.
    support_vectors_ : ndarray of shape (n_SV, d)
    n_support_ : ndarray of shape (k,)
        Support vectors per class.
    dual_coef_ : ndarray of shape (k - 1, n_SV)
        :math:`y_i \alpha_i` in LIBSVM's one-vs-one layout.
    intercept_ : ndarray of shape (k (k - 1) / 2,)
        :math:`-\rho` per pair (:math:`\rho` for two classes, see above).
    n_iter_ : ndarray of shape (k (k - 1) / 2,)
        SMO iterations per pair.
    shape_fit_ : tuple
    n_features_in_ : int

    Warns
    -----
    ConvergenceWarning
        If a binary problem stops at ``max_iter`` before reaching ``tol``.

    Complexity
    ----------
    ``fit``: time O(T n d) per binary problem for T SMO iterations (every
    iteration scans the n multipliers and needs at most two new kernel columns,
    O(n d) each); extra space O(B n + n d) with B the cached columns (at most
    ``cache_size`` MiB). ``decision_function``/``predict``: time
    O(n n_SV d), space O(n n_SV).

    References
    ----------
    Fan, R.-E., Chen, P.-H. & Lin, C.-J. (2005). JMLR 6, 1889-1918.
    Chang, C.-C. & Lin, C.-J. (2011). LIBSVM. ACM TIST 2(3).

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.ml import SVC
    >>> X = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0], [4.0, 4.0]])
    >>> y = np.array([0, 0, 1, 1])
    >>> clf = SVC(kernel="linear", C=10.0).fit(X, y)
    >>> clf.support_.tolist(), clf.predict([[0.5, 0.0], [3.5, 4.0]]).tolist()
    ([1, 2], [0, 1])
    >>> abs(float(clf.decision_function([[2.0, 2.0]])[0])) < 1e-12  # midway: on the boundary
    True
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        degree: int = 3,
        gamma: str | float = "scale",
        coef0: float = 0.0,
        tol: float = 1e-3,
        max_iter: int = -1,
        cache_size: float = 200.0,
        decision_function_shape: str = "ovr",
        random_state=None,
    ):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.cache_size = cache_size
        self.decision_function_shape = decision_function_shape
        self.random_state = random_state

    def _resolve_kernel(self, X: np.ndarray) -> _Kernel:
        if self.kernel not in _KERNELS:
            raise ValueError(f"kernel must be one of {_KERNELS}, got {self.kernel!r}.")
        if not _is_int(self.degree) or self.degree < 0:
            raise ValueError(f"degree must be an integer >= 0, got {self.degree!r}.")
        if not _is_real(self.coef0) or not np.isfinite(self.coef0):
            raise ValueError(f"coef0 must be a finite number, got {self.coef0!r}.")
        d = X.shape[1]
        if self.gamma == "scale":
            var = float(X.var())
            gamma = 1.0 / (d * var) if var != 0 else 1.0
        elif self.gamma == "auto":
            gamma = 1.0 / d
        elif _is_real(self.gamma) and 0 < self.gamma < np.inf:
            gamma = float(self.gamma)
        else:
            raise ValueError(f"gamma must be 'scale', 'auto' or a number > 0, got {self.gamma!r}.")
        return _Kernel(self.kernel, gamma, int(self.degree), float(self.coef0))

    def _check_params(self) -> None:
        if not _is_real(self.C) or not 0 < self.C < np.inf:
            raise ValueError(f"C must be a finite number > 0, got {self.C!r}.")
        if not _is_real(self.tol) or not 0 < self.tol < np.inf:
            raise ValueError(f"tol must be a finite number > 0, got {self.tol!r}.")
        if not _is_int(self.max_iter) or (self.max_iter < 1 and self.max_iter != -1):
            raise ValueError(f"max_iter must be -1 or an integer >= 1, got {self.max_iter!r}.")
        if not _is_real(self.cache_size) or not 0 < self.cache_size < np.inf:
            raise ValueError(f"cache_size must be a number > 0, got {self.cache_size!r}.")
        if self.decision_function_shape not in ("ovr", "ovo"):
            raise ValueError(
                "decision_function_shape must be 'ovr' or 'ovo', got "
                f"{self.decision_function_shape!r}."
            )

    def fit(self, X, y) -> SVC:
        """Fit one binary SVM per pair of classes on ``X`` (n, d) and labels ``y``."""
        self._check_params()
        X, y = check_X_labels(X, y)
        classes, y_idx = np.unique(y, return_inverse=True)
        k = len(classes)
        if k < 2:
            raise ValueError(f"SVC needs samples of at least 2 classes, got {k}.")
        kernel = self._resolve_kernel(X)
        C, tol = float(self.C), float(self.tol)

        members = [np.flatnonzero(y_idx == c) for c in range(k)]
        pair_alpha: list[tuple[np.ndarray, np.ndarray]] = []  # (alpha_a, alpha_b) per pair
        rhos, iters, failed = [], [], []
        for a in range(k):
            for b in range(a + 1, k):
                idx = np.concatenate([members[a], members[b]])
                yy = np.where(np.arange(len(idx)) < len(members[a]), 1.0, -1.0)
                sol = _smo(X[idx], yy, C, kernel, tol, int(self.max_iter), self.cache_size)
                pair_alpha.append((sol.alpha[: len(members[a])], sol.alpha[len(members[a]) :]))
                rhos.append(sol.rho)
                iters.append(sol.n_iter)
                if not sol.converged:
                    failed.append((classes[a], classes[b]))

        # A sample is a support vector if it has a non-zero multiplier in any pair.
        is_sv = [np.zeros(len(m), dtype=bool) for m in members]
        p = 0
        for a in range(k):
            for b in range(a + 1, k):
                is_sv[a] |= pair_alpha[p][0] > 0
                is_sv[b] |= pair_alpha[p][1] > 0
                p += 1
        sv_pos = [np.cumsum(s) - 1 for s in is_sv]  # position within the class block
        n_support = np.array([int(s.sum()) for s in is_sv])
        start = np.concatenate([[0], np.cumsum(n_support)[:-1]])
        dual = np.zeros((k - 1, int(n_support.sum())))
        p = 0
        for a in range(k):
            for b in range(a + 1, k):
                alpha_a, alpha_b = pair_alpha[p]
                nz = alpha_a > 0
                dual[b - 1, start[a] + sv_pos[a][nz]] = alpha_a[nz]  # y = +1
                nz = alpha_b > 0
                dual[a, start[b] + sv_pos[b][nz]] = -alpha_b[nz]  # y = -1
                p += 1
        support = np.concatenate([members[c][is_sv[c]] for c in range(k)])
        intercept = -np.asarray(rhos)
        if k == 2:  # positive decision value = second class (scikit-learn convention)
            dual, intercept = -dual, -intercept

        if failed:
            warnings.warn(
                f"SVC stopped at max_iter={self.max_iter} before reaching tol={self.tol} "
                f"for class pairs {failed}.",
                ConvergenceWarning,
                stacklevel=2,
            )
        # Assign state only after everything succeeded.
        self._kernel = kernel
        self._gamma = kernel.gamma
        self.classes_ = classes
        self.support_ = support
        self.support_vectors_ = X[support].copy()
        self.n_support_ = n_support
        self.dual_coef_ = dual
        self.intercept_ = intercept
        self.n_iter_ = np.asarray(iters)
        self.shape_fit_ = X.shape
        self.n_features_in_ = X.shape[1]
        return self

    def _pairwise_decision(self, X) -> np.ndarray:
        """LIBSVM one-vs-one decision values, shape (n, k(k-1)/2), positive = first class."""
        self._check_is_fitted("support_vectors_")
        X = check_n_features(X, self.n_features_in_, self)
        K = self._kernel(X, self.support_vectors_)
        k = len(self.classes_)
        dual = -self.dual_coef_ if k == 2 else self.dual_coef_
        rho = self.intercept_ if k == 2 else -self.intercept_
        start = np.concatenate([[0], np.cumsum(self.n_support_)])
        out = np.empty((X.shape[0], len(rho)))
        p = 0
        for a in range(k):
            for b in range(a + 1, k):
                sa, sb = slice(start[a], start[a + 1]), slice(start[b], start[b + 1])
                out[:, p] = K[:, sa] @ dual[b - 1, sa] + K[:, sb] @ dual[a, sb] - rho[p]
                p += 1
        return out

    def decision_function(self, X) -> np.ndarray:
        """Signed distances: shape (n,) for two classes (> 0 means ``classes_[1]``),
        (n, k) for ``"ovr"`` or (n, k(k-1)/2) for ``"ovo"`` otherwise."""
        pairwise = self._pairwise_decision(X)
        k = len(self.classes_)
        if k == 2:
            return -pairwise[:, 0]
        if self.decision_function_shape == "ovr":
            return _ovr_decision(pairwise, k)
        return pairwise

    def predict(self, X) -> np.ndarray:
        """One-vs-one vote (ties go to the class that sorts first)."""
        pairwise = self._pairwise_decision(X)
        k = len(self.classes_)
        votes = np.zeros((pairwise.shape[0], k), dtype=np.int64)
        p = 0
        for a in range(k):
            for b in range(a + 1, k):
                votes[:, a] += pairwise[:, p] > 0
                votes[:, b] += pairwise[:, p] <= 0
                p += 1
        return self.classes_[np.argmax(votes, axis=1)]
