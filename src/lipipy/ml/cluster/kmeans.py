r"""k-means clustering with Lloyd's algorithm and k-means++ seeding.

Objective
---------
Partition ``n`` points into ``k`` clusters with centres :math:`\mu_1..\mu_k`
minimising the *inertia* (within-cluster sum of squares)

.. math::  J = \sum_{i=1}^{n} \min_{j} \lVert x_i - \mu_j \rVert_2^2 .

Lloyd's algorithm
-----------------
Alternate two steps, each of which can only decrease :math:`J`:

1. **Assign**: :math:`\ell_i = \arg\min_j \lVert x_i - \mu_j\rVert^2`.
2. **Update**: :math:`\mu_j = \text{mean}\{x_i : \ell_i = j\}`.

Stop when the labels no longer change, or when the total squared centre shift
:math:`\sum_j \lVert \mu_j^{new} - \mu_j\rVert^2` is at most
``tol * mean(var(X, axis=0))`` (the tolerance is relative to the data scale, as in
scikit-learn). If the loop stops on the shift test, labels are recomputed once
against the final centres so ``labels_`` and ``cluster_centers_`` agree.

**Empty clusters.** If a cluster loses all its points in step 1, it is re-seeded
with the point that is farthest from its own (previous) centre; that point is
removed from its old cluster's mean. With several empty clusters the farthest
points are used in decreasing order of distance. This is scikit-learn's rule; if
all points sit exactly on their centres (fewer distinct points than ``k``) the
empty centre is left where it was and a :class:`ConvergenceWarning` is raised.

Seeding (k-means++)
-------------------
The first centre is a uniformly random point. Each next centre is sampled with
probability proportional to :math:`D(x)^2`, the squared distance to the nearest
centre chosen so far. We use the *greedy* variant (as scikit-learn): draw
:math:`2 + \lfloor\ln k\rfloor` candidates and keep the one that reduces the
potential :math:`\sum_x D(x)^2` the most.

Numerics: like scikit-learn we subtract the data mean before iterating (and add it
back to the centres), and squared distances are accumulated exactly as
:math:`\sum_d (x_d - \mu_d)^2` (no :math:`\lVert x\rVert^2 - 2x^\top\mu + \lVert\mu\rVert^2`
cancellation).

References: Lloyd (1982), *Least squares quantization in PCM*, IEEE Trans.
Information Theory 28(2); Arthur & Vassilvitskii (2007), *k-means++: the advantages
of careful seeding*, SODA.
"""

from __future__ import annotations

import numbers
import warnings

import numpy as np

from lipipy.core.base import BaseEstimator
from lipipy.core.complexity import Complexity
from lipipy.core.errors import ConvergenceWarning
from lipipy.core.random import check_random_state
from lipipy.core.registry import register
from lipipy.core.validation import check_array
from lipipy.ml._base import TransformerMixin, check_n_features


def _squared_distances(X: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Exact squared Euclidean distances, shape (n, k), one centre at a time."""
    D = np.empty((X.shape[0], centres.shape[0]))
    for j, mu in enumerate(centres):
        diff = X - mu
        D[:, j] = np.einsum("ij,ij->i", diff, diff)
    return D


def _kmeans_plusplus(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Greedy k-means++ seeding (Arthur & Vassilvitskii 2007). Time O(n k d log k)."""
    n = X.shape[0]
    n_trials = 2 + int(np.log(k))
    centres = np.empty((k, X.shape[1]))
    centres[0] = X[rng.integers(n)]
    closest = _squared_distances(X, centres[:1])[:, 0]  # D(x)^2
    potential = closest.sum()
    for c in range(1, k):
        draws = rng.random(n_trials) * potential
        candidates = np.searchsorted(np.cumsum(closest), draws)
        candidates = np.minimum(candidates, n - 1)  # guard against round-off at the top
        # D(x)^2 if each candidate were added, shape (n, n_trials)
        trial = np.minimum(closest[:, None], _squared_distances(X, X[candidates]))
        best = int(np.argmin(trial.sum(axis=0)))
        centres[c] = X[candidates[best]]
        closest = trial[:, best]
        potential = closest.sum()
    return centres


def _update_centres(X: np.ndarray, labels: np.ndarray, old: np.ndarray) -> tuple[np.ndarray, bool]:
    """Cluster means, re-seeding empty clusters from the farthest points.

    Returns ``(centres, stuck)`` where ``stuck`` is True when an empty cluster
    could not be re-seeded because every point sits exactly on its centre.
    """
    k, d = old.shape
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    sums = np.zeros((k, d))
    np.add.at(sums, labels, X)
    empty = np.flatnonzero(counts == 0)
    stuck = False
    if empty.size:
        diff = X - old[labels]
        dist = np.einsum("ij,ij->i", diff, diff)
        if dist.max() == 0.0:
            stuck = True
            sums[empty] = old[empty]  # keep the old centre (count is treated as 1 below)
            counts[empty] = 1.0
        else:
            far = np.argsort(-dist, kind="stable")[: empty.size]  # farthest first
            for j, i in zip(empty, far, strict=True):
                src = labels[i]
                sums[src] -= X[i]
                counts[src] -= 1.0
                sums[j] = X[i]
                counts[j] = 1.0
    # A donor cluster left with no points gets centre 0 (the data mean, since X is
    # centred); the next assignment step treats it as empty again.
    return sums / np.maximum(counts, 1.0)[:, None], stuck


@register(
    domain="ml",
    tasks=("clustering",),
    complexity=Complexity(
        time="O(T n k d)",
        space="O(n (k + d))",
        notes=(
            "per initialisation (times n_init); k-means++ seeding adds O(n k d log k); "
            "space is the n x k distance matrix plus one centred copy of X"
        ),
    ),
    references=(
        "Lloyd (1982), Least squares quantization in PCM, IEEE Transactions on "
        "Information Theory 28(2)",
        "Arthur & Vassilvitskii (2007), k-means++: the advantages of careful seeding, SODA",
    ),
    status="experimental",
)
class KMeans(TransformerMixin, BaseEstimator):
    r"""k-means clustering (Lloyd's algorithm, k-means++ seeding).

    Minimises the inertia :math:`\sum_i \min_j \lVert x_i - \mu_j\rVert^2`; the
    module docstring describes the iteration, stopping rule, empty-cluster
    handling and seeding.

    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters ``k`` (at most the number of samples).
    init : "k-means++" or ndarray of shape (k, d), default="k-means++"
        Seeding method, or explicit initial centres. With an array, the algorithm
        is deterministic and runs once (``n_init`` is ignored).
    n_init : int, default=10
        Number of k-means++ restarts; the run with the lowest inertia is kept.
    max_iter : int, default=300
        Maximum Lloyd iterations per run.
    tol : float, default=1e-4
        Relative tolerance on the squared centre shift (scaled by the mean
        feature variance).
    random_state : None, int, SeedSequence or Generator, default=None
        Seed for k-means++. Same seed ⇒ same result.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (k, d)
    labels_ : ndarray of shape (n,)
        Index of the closest centre for each training point.
    inertia_ : float
        Sum of squared distances to the closest centre.
    n_iter_ : int
        Lloyd iterations of the best run.
    n_features_in_ : int

    Complexity
    ----------
    Time O(T n k d) per initialisation for T iterations (times ``n_init``), plus
    O(n k d log k) for k-means++. Extra space O(n (k + d)): the n x k distance
    matrix and a centred copy of ``X``.

    References
    ----------
    Lloyd, S. (1982). Least squares quantization in PCM.
    Arthur, D. & Vassilvitskii, S. (2007). k-means++.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.ml import KMeans
    >>> X = np.array([[0.0, 0.0], [0.0, 1.0], [10.0, 0.0], [10.0, 1.0]])
    >>> km = KMeans(n_clusters=2, init=np.array([[0.0, 0.0], [10.0, 0.0]])).fit(X)
    >>> km.labels_.tolist(), km.cluster_centers_.tolist(), km.inertia_
    ([0, 0, 1, 1], [[0.0, 0.5], [10.0, 0.5]], 1.0)
    >>> km.predict([[9.0, 0.0]]).tolist()
    [1]
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init: str | np.ndarray = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    @staticmethod
    def _check_positive_int(value, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        return int(value)

    def _initial_centres(self, X: np.ndarray, k: int) -> np.ndarray | None:
        """Validated explicit init array, or None for k-means++."""
        if isinstance(self.init, str):
            if self.init != "k-means++":
                raise ValueError(f"init must be 'k-means++' or an array, got {self.init!r}.")
            return None
        centres = check_array(self.init)
        if centres.shape != (k, X.shape[1]):
            raise ValueError(f"init array has shape {centres.shape}, expected ({k}, {X.shape[1]}).")
        return centres

    def _lloyd(
        self, X: np.ndarray, centres: np.ndarray, tol: float
    ) -> tuple[np.ndarray, np.ndarray, float, int, bool]:
        """One Lloyd run from ``centres``; returns centres, labels, inertia, n_iter, stuck."""
        labels_old = np.full(X.shape[0], -1)
        stuck = False
        converged_strictly = False
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):  # noqa: B007 (used after loop)
            labels = np.argmin(_squared_distances(X, centres), axis=1)
            new_centres, stuck_now = _update_centres(X, labels, centres)
            stuck |= stuck_now
            shift = float(np.sum((new_centres - centres) ** 2))
            centres = new_centres
            if np.array_equal(labels, labels_old):
                converged_strictly = True
                break
            if shift <= tol:
                break
            labels_old = labels
        D = _squared_distances(X, centres)
        if not converged_strictly:
            labels = np.argmin(D, axis=1)  # make labels consistent with final centres
        inertia = float(D[np.arange(X.shape[0]), labels].sum())
        return centres, labels, inertia, n_iter, stuck

    def fit(self, X, y=None) -> KMeans:
        """Cluster ``X`` of shape (n, d); ``y`` is ignored. Returns ``self``."""
        X = check_array(X)
        n = X.shape[0]
        k = self._check_positive_int(self.n_clusters, "n_clusters")
        if k > n:
            raise ValueError(f"n_clusters={k} is larger than the number of samples ({n}).")
        self._check_positive_int(self.max_iter, "max_iter")
        if isinstance(self.tol, bool) or not isinstance(self.tol, numbers.Real) or self.tol < 0:
            raise ValueError(f"tol must be a number >= 0, got {self.tol!r}.")
        init = self._initial_centres(X, k)
        n_init = 1 if init is not None else self._check_positive_int(self.n_init, "n_init")
        rng = check_random_state(self.random_state)

        x_mean = X.mean(axis=0)
        Xc = X - x_mean
        tol = float(np.mean(np.var(X, axis=0))) * float(self.tol)

        best = None
        for _ in range(n_init):
            start = init - x_mean if init is not None else _kmeans_plusplus(Xc, k, rng)
            run = self._lloyd(Xc, start, tol)
            if best is None or run[2] < best[2]:
                best = run
        centres, labels, inertia, n_iter, stuck = best

        if stuck or np.unique(labels).size < k:
            warnings.warn(
                f"Number of distinct clusters ({np.unique(labels).size}) found smaller than "
                f"n_clusters ({k}). Possibly due to duplicate points in X.",
                ConvergenceWarning,
                stacklevel=2,
            )
        self.cluster_centers_ = centres + x_mean
        self.labels_ = labels
        self.inertia_ = inertia
        self.n_iter_ = n_iter
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X) -> np.ndarray:
        """Euclidean distance from each row of ``X`` to each centre, shape (n, k)."""
        self._check_is_fitted("cluster_centers_")
        X = check_n_features(X, self.n_features_in_, self)
        return np.sqrt(_squared_distances(X, self.cluster_centers_))

    def predict(self, X) -> np.ndarray:
        """Index of the closest centre for each row of ``X``."""
        self._check_is_fitted("cluster_centers_")
        X = check_n_features(X, self.n_features_in_, self)
        return np.argmin(_squared_distances(X, self.cluster_centers_), axis=1)

    def fit_predict(self, X, y=None) -> np.ndarray:
        """Fit and return ``labels_``."""
        return self.fit(X).labels_
