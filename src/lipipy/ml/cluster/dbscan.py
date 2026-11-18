r"""DBSCAN: density-based clustering with noise.

Definitions
-----------
With :math:`N_\varepsilon(x) = \{z : d(x, z) \le \varepsilon\}` (including
:math:`x` itself):

* :math:`x` is a **core** point if :math:`|N_\varepsilon(x)| \ge` ``min_samples``;
* core points within :math:`\varepsilon` of each other are in the same cluster
  (clusters are the connected components of this graph);
* a non-core point within :math:`\varepsilon` of a core point is a **border**
  point of that cluster; every other point is **noise** (label -1).

Border rule (documented, deterministic)
---------------------------------------
A border point can be within :math:`\varepsilon` of core points of several
clusters, and DBSCAN's result then depends on the processing order. Clusters
are numbered by their smallest core-point index (cluster 0 holds the
lowest-index core point), and a border point joins the cluster with the smallest
number among those it touches. This is exactly what scikit-learn's DBSCAN
produces, since it grows whole clusters one at a time from core points in
index order.

Algorithm and memory
--------------------
Distances are computed in blocks of ``B = chunk_size`` rows against all
points (exactly, one feature at a time), so no :math:`n \times n` matrix or
neighbour list is stored:

1. count neighbours per point (one pass) to find the core points;
2. for blocks of core points, their core neighbours are merged into one
   component with a vectorised union-find whose roots are always the smallest
   index in the component;
3. for blocks of non-core points, the smallest component root among their core
   neighbours gives the border label.

Extra memory is O(B n); time is O(n² d) (three passes in the worst case).

References
----------
Ester, Kriegel, Sander & Xu (1996), *A density-based algorithm for discovering
clusters in large spatial databases with noise*, KDD; Schubert, Sander, Ester,
Kriegel & Xu (2017), *DBSCAN revisited, revisited*, ACM TODS 42(3).
"""

from __future__ import annotations

import numbers

import numpy as np

from lipipy.core.base import BaseEstimator
from lipipy.core.complexity import Complexity
from lipipy.core.registry import register
from lipipy.core.validation import check_array

_METRICS = ("euclidean", "manhattan")
_DEFAULT_BLOCK_ELEMENTS = 2**21  # ~16 MiB of float64 distances per block


def _distance_block(Q: np.ndarray, T: np.ndarray, metric: str) -> np.ndarray:
    D = np.zeros((Q.shape[0], T.shape[0]))
    for j in range(Q.shape[1]):
        diff = Q[:, j, None] - T[None, :, j]
        D += diff * diff if metric == "euclidean" else np.abs(diff)
    return np.sqrt(D, out=D) if metric == "euclidean" else D


def _compress(parent: np.ndarray) -> np.ndarray:
    """Point every node at its root (pointer jumping); parents always point down in index."""
    while True:
        grand = parent[parent]
        if np.array_equal(grand, parent):
            return parent
        parent = grand


def _union(parent: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Merge the components of every edge (u_i, v_i); roots stay the smallest index."""
    while True:
        parent = _compress(parent)
        ru, rv = parent[u], parent[v]
        differ = ru != rv
        if not differ.any():
            return parent
        hi = np.maximum(ru[differ], rv[differ])
        lo = np.minimum(ru[differ], rv[differ])
        np.minimum.at(parent, hi, lo)


@register(
    domain="ml",
    tasks=("clustering",),
    complexity=Complexity(
        time="O(n^2 d)",
        space="O(B n)",
        notes="B = chunk_size rows of distances at a time; no neighbour lists are stored",
    ),
    references=(
        "Ester, Kriegel, Sander & Xu (1996), A density-based algorithm for discovering "
        "clusters in large spatial databases with noise, KDD",
        "Schubert, Sander, Ester, Kriegel & Xu (2017), DBSCAN revisited, revisited, "
        "ACM Transactions on Database Systems 42(3)",
    ),
    status="experimental",
)
class DBSCAN(BaseEstimator):
    r"""Density-based spatial clustering of applications with noise.

    See the module docstring for the definitions, the border-point rule and the
    chunked algorithm.

    Parameters
    ----------
    eps : float, default=0.5
        Neighbourhood radius (inclusive), ``eps > 0``.
    min_samples : int, default=5
        Neighbours (including the point itself) needed for a core point.
    metric : {"euclidean", "manhattan"}, default="euclidean"
    chunk_size : int or None, default=None
        Rows per distance block; ``None`` uses ``max(1, 2**21 // n)`` (~16 MiB).

    Attributes
    ----------
    labels_ : ndarray of shape (n,)
        Cluster index per sample, -1 for noise.
    core_sample_indices_ : ndarray
        Sorted indices of the core points.
    components_ : ndarray of shape (n_core, d)
        The core points.
    n_features_in_ : int

    Complexity
    ----------
    Time O(n² d); extra space O(B n) for B = ``chunk_size``.

    References
    ----------
    Ester, M., Kriegel, H.-P., Sander, J. & Xu, X. (1996). KDD.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.ml import DBSCAN
    >>> X = np.array([[0.0], [0.5], [1.0], [10.0], [10.5], [11.0], [30.0]])
    >>> db = DBSCAN(eps=0.6, min_samples=2).fit(X)
    >>> db.labels_.tolist(), db.core_sample_indices_.tolist()
    ([0, 0, 0, 1, 1, 1, -1], [0, 1, 2, 3, 4, 5])
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
        chunk_size: int | None = None,
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.chunk_size = chunk_size

    def _check_params(self) -> None:
        e = self.eps
        if isinstance(e, bool) or not isinstance(e, numbers.Real) or not 0 < e < np.inf:
            raise ValueError(f"eps must be a finite number > 0, got {e!r}.")
        m = self.min_samples
        if isinstance(m, bool) or not isinstance(m, numbers.Integral) or m < 1:
            raise ValueError(f"min_samples must be an integer >= 1, got {m!r}.")
        if self.metric not in _METRICS:
            raise ValueError(f"metric must be one of {_METRICS}, got {self.metric!r}.")
        c = self.chunk_size
        if c is not None and (isinstance(c, bool) or not isinstance(c, numbers.Integral) or c < 1):
            raise ValueError(f"chunk_size must be None or a positive integer, got {c!r}.")

    def fit(self, X, y=None) -> DBSCAN:
        """Cluster ``X`` of shape (n, d); ``y`` is ignored. Returns ``self``."""
        self._check_params()
        X = check_array(X)
        n = X.shape[0]
        eps, metric = float(self.eps), self.metric
        step = int(self.chunk_size) if self.chunk_size else max(1, _DEFAULT_BLOCK_ELEMENTS // n)

        # 1) neighbour counts -> core points
        counts = np.empty(n, dtype=np.int64)
        for s in range(0, n, step):
            counts[s : s + step] = np.sum(_distance_block(X[s : s + step], X, metric) <= eps, 1)
        core = np.flatnonzero(counts >= self.min_samples)
        Xc = X[core]

        # 2) connected components of the core graph (indices into `core`)
        parent = np.arange(len(core))
        for s in range(0, len(core), step):
            rows, cols = np.nonzero(_distance_block(Xc[s : s + step], Xc, metric) <= eps)
            keep = rows + s < cols  # each undirected edge once
            parent = _union(parent, rows[keep] + s, cols[keep])
        parent = _compress(parent)
        roots, comp = np.unique(parent, return_inverse=True)  # roots ascending = by min index

        labels = np.full(n, -1, dtype=np.intp)
        labels[core] = comp
        # 3) border points: the smallest-numbered cluster among their core neighbours
        others = np.setdiff1d(np.arange(n), core)
        if len(core):
            for s in range(0, len(others), step):
                rows = others[s : s + step]
                near = _distance_block(X[rows], Xc, metric) <= eps
                best = np.where(near, comp[None, :], len(roots)).min(axis=1)
                hit = best < len(roots)
                labels[rows[hit]] = best[hit]

        self.labels_ = labels
        self.core_sample_indices_ = core
        self.components_ = Xc.copy()
        self.n_features_in_ = X.shape[1]
        return self

    def fit_predict(self, X, y=None) -> np.ndarray:
        """Fit and return ``labels_``."""
        return self.fit(X).labels_
