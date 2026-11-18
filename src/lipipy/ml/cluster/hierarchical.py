r"""Agglomerative (bottom-up) hierarchical clustering.

Model
-----
Start with every sample as its own cluster and repeatedly merge the two closest
clusters until one remains. The distance between clusters is defined by the
*linkage*. With Euclidean point distances :math:`d(a, b)`:

* ``"single"``: :math:`\min_{a\in A, b\in B} d(a, b)`
* ``"complete"``: :math:`\max_{a\in A, b\in B} d(a, b)`
* ``"average"``: mean of :math:`d(a, b)` over all pairs (UPGMA)
* ``"ward"``: :math:`\sqrt{\tfrac{2|A||B|}{|A|+|B|}}\,\lVert \bar a - \bar b\rVert`,
  the increase in within-cluster sum of squares (times 2, square-rooted).

Lance-Williams updates
----------------------
After merging :math:`I` and :math:`J` (sizes :math:`n_I, n_J`), the distance to any
other cluster :math:`K` follows from the old distances alone:

.. math::
    \text{single: } \min(d_{KI}, d_{KJ}),\quad
    \text{complete: } \max(d_{KI}, d_{KJ}),\quad
    \text{average: } \frac{n_I d_{KI} + n_J d_{KJ}}{n_I + n_J},

.. math::
    \text{ward: } \sqrt{\frac{(n_K + n_I) d_{KI}^2 + (n_K + n_J) d_{KJ}^2 - n_K d_{IJ}^2}
    {n_I + n_J + n_K}} .

Nearest-neighbour chain
-----------------------
All four linkages are *reducible* (merging cannot bring the new cluster closer
to a third cluster than its parts were), so mutual nearest neighbours can be
merged in any order. The NN-chain algorithm follows nearest neighbours from a
cluster until two clusters are each other's nearest neighbour, merges them, and
continues from the rest of the chain. Each row scan is O(n), the chain is
extended O(n) times in total, so the whole tree costs O(n²) time. When the
previous chain element ties with another candidate it is kept (this prevents
cycles). The merges are then stably sorted by height and relabelled, exactly as
SciPy does, giving the SciPy linkage matrix.

Merge order is unique only when all heights are distinct; with equal heights
any order among them is a valid dendrogram.

Memory guard
------------
The algorithm keeps the full :math:`n\times n` float64 distance matrix (O(n²)
memory; computing it needs a second n × n temporary, so the peak is about
:math:`16 n^2` bytes, 1.6 GB at the limit). ``fit`` raises
:class:`~lipipy.core.ResourceLimitError` before allocating anything when
:math:`n >` ``MAX_SAMPLES`` = 10 000.

Flat clusters
-------------
With ``n_clusters`` the first :math:`n -` ``n_clusters`` merges are applied;
with ``distance_threshold`` the number of clusters is
:math:`1 + \#\{\text{merges with height} \ge \text{threshold}\}` (scikit-learn's rule).
Labels are numbered by the smallest sample index in each cluster (cluster 0
contains sample 0).

References
----------
Müllner (2011), *Modern hierarchical, agglomerative clustering algorithms*,
arXiv:1109.2378; Lance & Williams (1967), *A general theory of classificatory
sorting strategies*, The Computer Journal 9(4); Murtagh (1983), *A survey of
recent advances in hierarchical clustering algorithms*, The Computer Journal 26(4).
"""

from __future__ import annotations

import numbers

import numpy as np

from lipipy.core.base import BaseEstimator
from lipipy.core.complexity import Complexity
from lipipy.core.errors import ResourceLimitError
from lipipy.core.registry import register
from lipipy.core.validation import check_array

#: Largest n accepted by :class:`AgglomerativeClustering` (n x n float64 matrix).
MAX_SAMPLES = 10_000
_LINKAGES = ("ward", "complete", "average", "single")


def _pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    """Exact Euclidean distances, summing squared differences feature by feature."""
    n = X.shape[0]
    D = np.zeros((n, n))
    for j in range(X.shape[1]):
        diff = X[:, j, None] - X[None, :, j]
        D += diff * diff
    return np.sqrt(D, out=D)


def _lance_williams(linkage: str, d_xi, d_yi, d_xy: float, nx: float, ny: float, ni):
    """Distance from every cluster i to the merge of x and y (SciPy's formulas)."""
    if linkage == "single":
        return np.minimum(d_xi, d_yi)
    if linkage == "complete":
        return np.maximum(d_xi, d_yi)
    if linkage == "average":
        return (nx * d_xi + ny * d_yi) / (nx + ny)
    t = 1.0 / (nx + ny + ni)
    return np.sqrt((ni + nx) * t * d_xi * d_xi + (ni + ny) * t * d_yi * d_yi - ni * t * d_xy * d_xy)


def _nn_chain(D: np.ndarray, linkage: str) -> np.ndarray:
    """Unsorted merges (x, y, height, size) by the NN-chain algorithm; modifies ``D``."""
    n = D.shape[0]
    size = np.ones(n)
    active = np.ones(n, dtype=bool)
    np.fill_diagonal(D, np.inf)
    Z = np.empty((n - 1, 4))
    chain: list[int] = []
    for k in range(n - 1):
        if not chain:
            chain.append(int(np.argmax(active)))  # first active cluster
        while True:
            x = chain[-1]
            row = np.where(active, D[x], np.inf)
            y = int(np.argmin(row))  # first minimum
            best = row[y]
            if len(chain) > 1:
                prev = chain[-2]
                if not best < D[x, prev]:  # keep the previous element on ties
                    y, best = prev, D[x, prev]
                    break
            chain.append(y)
        chain.pop()
        chain.pop()
        x, y = min(x, y), max(x, y)
        nx, ny = size[x], size[y]
        Z[k] = (x, y, best, nx + ny)
        # merged cluster lives at y; x becomes inactive
        others = active.copy()
        others[[x, y]] = False
        new = _lance_williams(linkage, D[x, others], D[y, others], best, nx, ny, size[others])
        D[y, others] = new
        D[others, y] = new
        active[x] = False
        size[x], size[y] = 0.0, nx + ny
    return Z


def _relabel(Z: np.ndarray, n: int) -> np.ndarray:
    """Stable-sort merges by height and renumber clusters SciPy-style (n + step)."""
    Z = Z[np.argsort(Z[:, 2], kind="stable")]
    parent = np.arange(2 * n - 1)
    next_label = n

    def find(a: int) -> int:
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:  # path compression
            parent[a], a = root, parent[a]
        return root

    out = Z.copy()
    for i in range(n - 1):
        rx, ry = find(int(Z[i, 0])), find(int(Z[i, 1]))
        out[i, 0], out[i, 1] = min(rx, ry), max(rx, ry)
        parent[rx] = parent[ry] = next_label
        next_label += 1
    return out


def _cut(children: np.ndarray, n: int, n_merges: int) -> np.ndarray:
    """Flat labels after the first ``n_merges`` merges, numbered by smallest member."""
    parent = np.arange(2 * n - 1)
    for i in range(n_merges):
        a, b = children[i]
        parent[a] = parent[b] = n + i
    root = np.arange(n)
    while True:  # follow parents to the top (pointer jumping, O(n log n))
        nxt = parent[root]
        if np.array_equal(nxt, root):
            break
        root = nxt
    _, first = np.unique(root, return_index=True)
    order = np.argsort(first)  # clusters by smallest member index
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return rank[np.unique(root, return_inverse=True)[1]]


@register(
    domain="ml",
    tasks=("clustering",),
    complexity=Complexity(
        time="O(n^2 d)",
        space="O(n^2)",
        notes=(
            "O(n^2 d) for the distance matrix, then O(n^2) for the NN-chain merges; "
            "the n x n matrix is kept (peak ~16 n^2 bytes), so n <= MAX_SAMPLES = 10000"
        ),
    ),
    references=(
        "Müllner (2011), Modern hierarchical, agglomerative clustering algorithms, arXiv:1109.2378",
        "Lance & Williams (1967), A general theory of classificatory sorting strategies, "
        "The Computer Journal 9(4)",
    ),
    status="experimental",
)
class AgglomerativeClustering(BaseEstimator):
    r"""Agglomerative clustering with Lance-Williams updates (NN-chain).

    See the module docstring for the linkages, the algorithm, the memory guard
    and how flat clusters are formed.

    Parameters
    ----------
    n_clusters : int or None, default=2
        Number of flat clusters; must be ``None`` when ``distance_threshold`` is set.
    distance_threshold : float or None, default=None
        Merges at or above this height are not applied.
    linkage : {"ward", "complete", "average", "single"}, default="ward"

    Attributes
    ----------
    children_ : ndarray of shape (n - 1, 2)
        Clusters merged at each step; ids < n are samples, id n + i is the
        cluster formed at step i.
    distances_ : ndarray of shape (n - 1,)
        Merge heights (non-decreasing).
    labels_ : ndarray of shape (n,)
    n_clusters_ : int
    n_leaves_ : int
    n_features_in_ : int

    Raises
    ------
    ResourceLimitError
        If n > ``MAX_SAMPLES`` (10 000).

    Complexity
    ----------
    Time O(n² d) for the distances plus O(n²) for the merges; extra space O(n²)
    (peak about 16 n² bytes).

    References
    ----------
    Müllner, D. (2011). arXiv:1109.2378.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.ml import AgglomerativeClustering
    >>> X = np.array([[0.0], [1.0], [5.0], [6.5], [20.0]])
    >>> model = AgglomerativeClustering(n_clusters=3, linkage="single").fit(X)
    >>> model.labels_.tolist()
    [0, 0, 1, 1, 2]
    >>> model.linkage_matrix().tolist()
    [[0.0, 1.0, 1.0, 2.0], [2.0, 3.0, 1.5, 2.0], [5.0, 6.0, 4.0, 4.0], [4.0, 7.0, 13.5, 5.0]]
    """

    def __init__(
        self,
        n_clusters: int | None = 2,
        *,
        distance_threshold: float | None = None,
        linkage: str = "ward",
    ):
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold
        self.linkage = linkage

    def fit(self, X, y=None) -> AgglomerativeClustering:
        """Build the full merge tree of ``X`` (n, d) and cut it; ``y`` is ignored."""
        if self.linkage not in _LINKAGES:
            raise ValueError(f"linkage must be one of {_LINKAGES}, got {self.linkage!r}.")
        if (self.n_clusters is None) == (self.distance_threshold is None):
            raise ValueError("Set exactly one of n_clusters and distance_threshold.")
        X = check_array(X)
        n = X.shape[0]
        if self.n_clusters is not None:
            k = self.n_clusters
            if isinstance(k, bool) or not isinstance(k, numbers.Integral) or not 1 <= k <= n:
                raise ValueError(f"n_clusters must be an integer in [1, {n}], got {k!r}.")
        else:
            t = self.distance_threshold
            if isinstance(t, bool) or not isinstance(t, numbers.Real) or not t >= 0:
                raise ValueError(f"distance_threshold must be a number >= 0, got {t!r}.")
        if n > MAX_SAMPLES:
            raise ResourceLimitError(
                f"AgglomerativeClustering keeps an n x n distance matrix; n={n} exceeds "
                f"MAX_SAMPLES={MAX_SAMPLES} (about {16 * n * n / 1e9:.1f} GB peak)."
            )
        if n == 1:
            Z = np.empty((0, 4))
        else:
            Z = _relabel(_nn_chain(_pairwise_euclidean(X), self.linkage), n)
        children = Z[:, :2].astype(np.intp)
        distances = Z[:, 2].copy()
        if self.n_clusters is not None:
            n_clusters = int(self.n_clusters)
        else:
            n_clusters = int(np.count_nonzero(distances >= self.distance_threshold)) + 1
        labels = _cut(children, n, n - n_clusters)

        self._Z = Z
        self.children_ = children
        self.distances_ = distances
        self.labels_ = labels
        self.n_clusters_ = n_clusters
        self.n_leaves_ = n
        self.n_features_in_ = X.shape[1]
        return self

    def fit_predict(self, X, y=None) -> np.ndarray:
        """Fit and return ``labels_``."""
        return self.fit(X).labels_

    def linkage_matrix(self) -> np.ndarray:
        """SciPy linkage matrix ``Z`` of shape (n - 1, 4): merged ids, height, size.

        A copy, usable with ``scipy.cluster.hierarchy`` (``dendrogram``, ``fcluster``).
        """
        self._check_is_fitted("_Z")
        return self._Z.copy()
