r"""CART decision trees for classification and regression.

Model
-----
A binary tree partitions feature space into axis-aligned boxes. Every internal
node :math:`t` holds a test :math:`x_{j(t)} \le \theta_t`; samples that pass go to
the left child, the others to the right. Each leaf predicts a constant: the class
fractions of its training samples (classification) or their mean (regression).

Impurity
--------
For a node with :math:`n_t` samples and class fractions :math:`p_c`, or targets
:math:`y_i` with mean :math:`\bar y`,

.. math::
    \text{gini} = 1 - \sum_c p_c^2, \qquad
    \text{entropy} = -\sum_c p_c \log_2 p_c, \qquad
    \text{squared\_error} = \frac1{n_t}\sum_{i \in t} (y_i - \bar y)^2 .

Greedy split search
-------------------
A split of node :math:`t` into :math:`t_L, t_R` is chosen to maximise the impurity
decrease :math:`n_t I(t) - n_L I(t_L) - n_R I(t_R)`; equivalently (dropping
the constant :math:`n_t I(t)`) to maximise a cheaper *proxy*:

* gini: :math:`\sum_c L_c^2 / n_L + \sum_c R_c^2 / n_R` (class counts :math:`L_c, R_c`);
* entropy: :math:`\sum_c L_c\ln L_c - n_L\ln n_L + \sum_c R_c\ln R_c - n_R\ln n_R`;
* squared error: :math:`S_L^2\, n_t / (n_L n_R)` with :math:`S_L` the sum of the
  *centred* targets :math:`y_i - \bar y` on the left (centring removes the
  cancellation in the textbook form :math:`S_L^2/n_L + S_R^2/n_R`).

For every candidate feature the node's samples are visited in sorted order; the
proxy of all :math:`n_t - 1` "left = first :math:`p` samples" partitions follows
from prefix sums, and only positions between two *distinct* consecutive values
with at least ``min_samples_leaf`` samples on each side are valid. The threshold
is the midpoint of the two values, computed as :math:`a/2 + b/2` (as
scikit-learn, so it never overflows); if rounding makes it equal to :math:`b` the
lower value :math:`a` is used, so that :math:`a \le \theta < b` always holds.

Tie-breaking rule (documented, deterministic): among candidates with equal proxy
values in floating point, the split with the lowest feature index wins, and within
a feature the one with the smallest threshold.

Stopping
--------
A node becomes a leaf when ``depth >= max_depth``, :math:`n_t <`
``min_samples_split``, :math:`n_t < 2\,` ``min_samples_leaf``, its impurity is
:math:`\le \varepsilon` (machine epsilon), no valid split exists, or the weighted
impurity decrease
:math:`\tfrac{n_t}{n}\bigl(I(t) - \tfrac{n_L}{n_t} I(t_L) - \tfrac{n_R}{n_t} I(t_R)\bigr)`
plus :math:`\varepsilon` is below ``min_impurity_decrease`` (scikit-learn's rules).

Implementation: level-wise growth
---------------------------------
Every feature is argsorted once (O(n d log n)). Each level then processes all
open nodes together: for each feature the presorted sample list is restricted to
open nodes and stably grouped by node, which keeps the within-node sorted order,
so every level costs O(n d log n) NumPy work with only O(d) Python iterations
instead of a Python loop over nodes. Nodes are finally renumbered in depth-first,
left-first preorder, the numbering scikit-learn uses.

Feature subsampling (``max_features`` < d): every node draws a random order of the
features and evaluates the first ``max_features`` of them, continuing past that
number only while every drawn feature was constant in the node (scikit-learn's
rule). Constant features are skipped.

Feature importance
------------------
:math:`\mathrm{imp}_j \propto \sum_{t:\,j(t)=j} n_t I(t) - n_L I(t_L) - n_R I(t_R)`,
normalised to sum to one (mean decrease in impurity, Breiman 2001).

Differences from scikit-learn: we split on the float64 values (scikit-learn
casts ``X`` to float32 first, so identical trees need float32-representable
inputs); values are distinct when they differ at all (scikit-learn merges values
closer than ``1e-7``); no sample weights, missing values, ``max_leaf_nodes`` or
pruning; ``tree_.value`` has shape (nodes, k) for classifiers and (nodes,) for
regressors.

References
----------
Breiman, Friedman, Olshen & Stone (1984), *Classification and Regression Trees*;
Hastie, Tibshirani & Friedman, *ESL* 2e, §9.2; Breiman (2001), *Random forests*,
Machine Learning 45 (impurity importance).
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Any

import numpy as np

from lipipy.core.base import BaseEstimator, RegressorMixin
from lipipy.core.complexity import Complexity
from lipipy.core.random import check_random_state
from lipipy.core.registry import register
from lipipy.core.validation import check_X_y
from lipipy.ml._base import ClassifierMixin, check_n_features, check_X_labels

TREE_LEAF = -1
TREE_UNDEFINED = -2
_EPS = float(np.finfo(np.float64).eps)
_CLF_CRITERIA = ("gini", "entropy")
_REG_CRITERIA = ("squared_error",)
_REFERENCES = (
    "Breiman, Friedman, Olshen & Stone (1984), Classification and Regression Trees, Wadsworth",
    "Hastie, Tibshirani & Friedman, The Elements of Statistical Learning, 2nd ed., §9.2",
)
_TREE_COMPLEXITY = Complexity(
    time="O(T n d log n)",
    space="O(n d)",
    notes=(
        "fit; T = depth of the tree (levels grown), each level sorts every feature's "
        "samples by node, O(n d log n), plus O(n d k) for class-count prefix sums "
        "(k = 1 for regression); space is the presorted index table; predict and "
        "apply are O(T) per sample with O(n) extra memory"
    ),
)


# ---------------------------------------------------------------------------
# Tree storage
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Tree:
    """Array-based binary tree (node ``0`` is the root, nodes in preorder).

    Attributes
    ----------
    children_left, children_right : ndarray of int, shape (node_count,)
        Child node ids, ``-1`` for leaves.
    feature : ndarray of int, shape (node_count,)
        Feature tested at each node, ``-2`` for leaves.
    threshold : ndarray of float, shape (node_count,)
        Test :math:`x_{feature} \\le threshold` sends a sample left; ``-2.0`` for leaves.
    value : ndarray of float
        Class fractions, shape (node_count, k), for classifiers; mean target,
        shape (node_count,), for regressors (leaf values for boosting).
    impurity : ndarray of float, shape (node_count,)
    n_node_samples : ndarray of int, shape (node_count,)
    max_depth : int
        Depth of the deepest leaf (the root has depth 0).

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.ml import DecisionTreeRegressor
    >>> t = DecisionTreeRegressor().fit([[0.0], [1.0]], [0.0, 2.0]).tree_
    >>> t.node_count, float(t.threshold[0]), t.value.tolist()
    (3, 0.5, [1.0, 0.0, 2.0])
    """

    children_left: np.ndarray
    children_right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    value: np.ndarray
    impurity: np.ndarray
    n_node_samples: np.ndarray
    max_depth: int

    def __post_init__(self) -> None:
        for name in (
            "children_left",
            "children_right",
            "feature",
            "threshold",
            "value",
            "impurity",
            "n_node_samples",
        ):
            getattr(self, name).setflags(write=False)

    @property
    def node_count(self) -> int:
        return int(self.children_left.shape[0])

    @property
    def n_leaves(self) -> int:
        return int(np.count_nonzero(self.children_left == TREE_LEAF))

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Leaf id of every row of a validated float array ``X`` (O(n) memory)."""
        node = np.zeros(X.shape[0], dtype=np.intp)
        active = np.arange(X.shape[0])
        while active.size:
            nd = node[active]
            internal = self.children_left[nd] != TREE_LEAF
            active, nd = active[internal], nd[internal]
            go_left = X[active, self.feature[nd]] <= self.threshold[nd]
            node[active] = np.where(go_left, self.children_left[nd], self.children_right[nd])
        return node

    def compute_feature_importances(self, n_features: int, normalize: bool = True) -> np.ndarray:
        """Mean decrease in impurity per feature, weighted by n_t / n.

        Normalised to sum to one when ``normalize`` (all zeros for a single-leaf tree).
        """
        internal = np.flatnonzero(self.children_left != TREE_LEAF)
        left, right = self.children_left[internal], self.children_right[internal]
        w = self.n_node_samples.astype(np.float64)
        decrease = (
            w[internal] * self.impurity[internal]
            - w[left] * self.impurity[left]
            - w[right] * self.impurity[right]
        )
        importances = np.bincount(self.feature[internal], weights=decrease, minlength=n_features)
        importances = importances / w[0]
        total = importances.sum()
        return importances / total if normalize and total > 0 else importances

    def with_values(self, value: np.ndarray) -> Tree:
        """Copy of the tree with different node values (used by gradient boosting)."""
        return Tree(
            self.children_left,
            self.children_right,
            self.feature,
            self.threshold,
            np.array(value, dtype=np.float64),
            self.impurity,
            self.n_node_samples,
            self.max_depth,
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly description (nested lists)."""
        return {
            "children_left": self.children_left.tolist(),
            "children_right": self.children_right.tolist(),
            "feature": self.feature.tolist(),
            "threshold": self.threshold.tolist(),
            "value": self.value.tolist(),
            "impurity": self.impurity.tolist(),
            "n_node_samples": self.n_node_samples.tolist(),
            "max_depth": self.max_depth,
        }


# ---------------------------------------------------------------------------
# Hyperparameter checks (shared with the ensembles)
# ---------------------------------------------------------------------------
def _is_int(v) -> bool:
    return isinstance(v, numbers.Integral) and not isinstance(v, (bool, np.bool_))


def _is_real(v) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, (bool, np.bool_))


@dataclass(frozen=True)
class _GrowParams:
    max_depth: int
    min_samples_split: int
    min_samples_leaf: int
    max_features: int
    min_impurity_decrease: float


def _resolve_max_features(max_features, d: int) -> int:
    if max_features is None:
        return d
    if isinstance(max_features, str):
        if max_features == "sqrt":
            return max(1, int(np.sqrt(d)))
        if max_features == "log2":
            return max(1, int(np.log2(d)))
        raise ValueError(
            f"max_features must be 'sqrt', 'log2', an int or a float; got {max_features!r}."
        )
    if _is_int(max_features):
        if not 1 <= max_features <= d:
            raise ValueError(f"max_features={max_features} must be in [1, n_features={d}].")
        return int(max_features)
    if _is_real(max_features):
        if not 0.0 < max_features <= 1.0:
            raise ValueError(f"float max_features must be in (0, 1], got {max_features!r}.")
        return max(1, int(max_features * d))
    raise TypeError(f"max_features must be None, a str, an int or a float; got {max_features!r}.")


def _resolve_grow_params(
    n: int,
    d: int,
    max_depth,
    min_samples_split,
    min_samples_leaf,
    max_features,
    min_impurity_decrease,
) -> _GrowParams:
    """Validate tree hyperparameters and turn fractions into sample counts."""
    if max_depth is None:
        depth = np.iinfo(np.int32).max
    elif _is_int(max_depth) and max_depth >= 1:
        depth = int(max_depth)
    else:
        raise ValueError(f"max_depth must be None or an integer >= 1, got {max_depth!r}.")

    if _is_int(min_samples_split):
        if min_samples_split < 2:
            raise ValueError(f"int min_samples_split must be >= 2, got {min_samples_split}.")
        mss = int(min_samples_split)
    elif _is_real(min_samples_split) and 0.0 < min_samples_split <= 1.0:
        mss = max(2, int(np.ceil(min_samples_split * n)))
    else:
        raise ValueError(
            "min_samples_split must be an int >= 2 or a float in (0, 1], "
            f"got {min_samples_split!r}."
        )

    if _is_int(min_samples_leaf):
        if min_samples_leaf < 1:
            raise ValueError(f"int min_samples_leaf must be >= 1, got {min_samples_leaf}.")
        msl = int(min_samples_leaf)
    elif _is_real(min_samples_leaf) and 0.0 < min_samples_leaf < 1.0:
        msl = max(1, int(np.ceil(min_samples_leaf * n)))
    else:
        raise ValueError(
            f"min_samples_leaf must be an int >= 1 or a float in (0, 1), got {min_samples_leaf!r}."
        )

    mid = min_impurity_decrease
    if not _is_real(mid) or not (0.0 <= mid < np.inf):
        raise ValueError(f"min_impurity_decrease must be a finite number >= 0, got {mid!r}.")
    return _GrowParams(depth, mss, msl, _resolve_max_features(max_features, d), float(mid))


# ---------------------------------------------------------------------------
# Node statistics
# ---------------------------------------------------------------------------
def _xlogx(c: np.ndarray) -> np.ndarray:
    """:math:`c \\ln c` with :math:`0 \\ln 0 = 0` for non-negative counts."""
    return c * np.log(np.maximum(c, 1.0))


class _Criterion:
    """Node statistics and the split proxy for one criterion.

    ``y`` is the class index (classification) or the float target (regression).
    """

    def __init__(self, criterion: str, y: np.ndarray, n_classes: int):
        self.criterion = criterion
        self.y = y
        self.k = n_classes
        self.is_clf = criterion in _CLF_CRITERIA

    def node_stats(self, idx: np.ndarray, slot: np.ndarray, F: int):
        """Per-slot sample counts, values and impurities of samples ``idx`` in ``slot``."""
        n_s = np.bincount(slot, minlength=F)
        y = self.y[idx]
        if self.is_clf:
            counts = np.bincount(slot * self.k + y, minlength=F * self.k).reshape(F, self.k)
            counts = counts.astype(np.float64)
            denom = np.maximum(n_s, 1).astype(np.float64)[:, None]
            if self.criterion == "gini":
                impurity = 1.0 - np.sum(counts * counts, axis=1) / (denom[:, 0] * denom[:, 0])
            else:
                p = counts / denom
                with np.errstate(divide="ignore", invalid="ignore"):
                    plogp = np.where(p > 0, p * (np.log(p) / np.log(2.0)), 0.0)
                impurity = -np.sum(plogp, axis=1)
            return n_s, counts / denom, impurity, counts
        denom = np.maximum(n_s, 1).astype(np.float64)
        mean = np.bincount(slot, weights=y, minlength=F) / denom
        resid = y - mean[slot]
        impurity = np.bincount(slot, weights=resid * resid, minlength=F) / denom
        return n_s, mean, impurity, mean

    def proxy(self, o, s, starts, n_s, nl, nr, totals) -> np.ndarray:
        """Split proxy at every position of the node-grouped, value-sorted samples ``o``."""
        nr_safe = np.maximum(nr, 1).astype(np.float64)
        nl_f = nl.astype(np.float64)
        if not self.is_clf:
            yc = self.y[o] - totals[s]  # centred on the node mean
            cs = np.cumsum(yc)
            pre = cs[starts] - yc[starts]
            SL = cs - pre[s]
            return SL * SL * (n_s[s] / (nl_f * nr_safe))
        y = self.y[o]
        acc_l = np.zeros(o.size)
        acc_r = np.zeros(o.size)
        for c in range(self.k):
            ind = y == c
            cs = np.cumsum(ind)
            pre = cs[starts] - ind[starts]
            L = (cs - pre[s]).astype(np.float64)
            R = totals[s, c] - L
            if self.criterion == "gini":
                acc_l += L * L
                acc_r += R * R
            else:
                acc_l += _xlogx(L)
                acc_r += _xlogx(R)
        if self.criterion == "gini":
            return acc_l / nl_f + acc_r / nr_safe
        return acc_l - _xlogx(nl_f) + acc_r - _xlogx(nr.astype(np.float64))


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
def _choose_features(constant: np.ndarray, mtry: int, rng: np.random.Generator) -> np.ndarray:
    """(F, d) mask of features each node evaluates (see module docstring)."""
    F, d = constant.shape
    eligible = np.zeros((F, d), dtype=bool)
    for t in range(F):
        perm = rng.permutation(d)
        nonconst = ~constant[t, perm]
        first = int(np.argmax(nonconst)) if nonconst.any() else d - 1
        drawn = perm[: max(mtry, first + 1)]
        eligible[t, drawn] = ~constant[t, drawn]
    return eligible


def build_tree(
    X: np.ndarray,
    y: np.ndarray,
    criterion: str,
    n_classes: int,
    params: _GrowParams,
    rng: np.random.Generator,
) -> Tree:
    """Grow a CART tree on validated ``X`` (n, d) and encoded ``y``.

    ``y`` holds class indices in ``[0, n_classes)`` for classification criteria and
    float targets for ``"squared_error"``.
    """
    n, d = X.shape
    crit = _Criterion(criterion, y, n_classes)
    order = np.argsort(X, axis=0, kind="stable").T  # (d, n): presorted sample ids
    all_idx = np.arange(n)

    # Per-node storage in creation order (renumbered to preorder at the end).
    left: list[int] = []
    right: list[int] = []
    feature: list[int] = []
    threshold: list[float] = []
    values: list[np.ndarray] = []
    impurity: list[float] = []
    n_samples: list[int] = []
    depth: list[int] = []

    def add_nodes(n_s, vals, imps, dep) -> np.ndarray:
        first = len(left)
        m = len(n_s)
        left.extend([TREE_LEAF] * m)
        right.extend([TREE_LEAF] * m)
        feature.extend([TREE_UNDEFINED] * m)
        threshold.extend([float(TREE_UNDEFINED)] * m)
        values.extend(list(vals))
        impurity.extend(imps.tolist())
        n_samples.extend(n_s.tolist())
        depth.extend([dep] * m)
        return np.arange(first, first + m)

    def can_split(ids: np.ndarray) -> np.ndarray:
        return np.array(
            [
                depth[i] < params.max_depth
                and n_samples[i] >= params.min_samples_split
                and n_samples[i] >= 2 * params.min_samples_leaf
                and impurity[i] > _EPS
                for i in ids
            ],
            dtype=bool,
        )

    n_s, vals, imps, _ = crit.node_stats(all_idx, np.zeros(n, dtype=np.intp), 1)
    root = add_nodes(n_s, vals, imps, 0)
    root_impurity_scale = float(n)
    node_of = np.zeros(n, dtype=np.intp)
    frontier = root[can_split(root)]

    while frontier.size:
        F = frontier.size
        node_slot = np.full(len(left), -1, dtype=np.intp)
        node_slot[frontier] = np.arange(F)
        sample_slot = node_slot[node_of]
        active = sample_slot >= 0
        act_idx = all_idx[active]
        n_s, _, node_imp, totals = crit.node_stats(act_idx, sample_slot[active], F)
        starts = np.concatenate([[0], np.cumsum(n_s)[:-1]])

        if params.max_features < d:
            grouped = act_idx[np.argsort(sample_slot[active], kind="stable")]
            Xg = X[grouped]
            constant = np.minimum.reduceat(Xg, starts, axis=0) == np.maximum.reduceat(
                Xg, starts, axis=0
            )
            eligible = _choose_features(constant, params.max_features, rng)
            del Xg
        else:
            eligible = np.ones((F, d), dtype=bool)

        best_proxy = np.full(F, -np.inf)
        best_feat = np.full(F, TREE_UNDEFINED, dtype=np.intp)
        best_thr = np.zeros(F)
        for f in range(d):
            if not eligible[:, f].any():
                continue
            o = order[f]
            o = o[active[o]]
            g = np.argsort(sample_slot[o], kind="stable")  # group by node, keep x order
            o = o[g]
            s = sample_slot[o]
            x = X[o, f]
            m = o.size
            nl = np.arange(1, m + 1) - starts[s]
            nr = n_s[s] - nl
            valid = (nl >= params.min_samples_leaf) & (nr >= params.min_samples_leaf)
            valid &= eligible[s, f]
            valid[:-1] &= x[:-1] < x[1:]
            valid[-1] = False
            proxy = np.where(valid, crit.proxy(o, s, starts, n_s, nl, nr, totals), -np.inf)
            seg_max = np.maximum.reduceat(proxy, starts)
            candidates = np.where(valid & (proxy == seg_max[s]), np.arange(m), m)
            first = np.minimum.reduceat(candidates, starts)
            better = (first < m) & (seg_max > best_proxy)
            if better.any():
                p = first[better]
                a, b = x[p], x[p + 1]
                thr = a / 2.0 + b / 2.0
                thr = np.where(thr == b, a, thr)  # keep a <= thr < b under rounding
                best_proxy[better] = seg_max[better]
                best_feat[better] = f
                best_thr[better] = thr

        # Partition the samples of nodes that found a split and score the split.
        has_split = best_feat >= 0
        slot_a = sample_slot[active]
        go_left = np.zeros(act_idx.size, dtype=bool)
        rows = has_split[slot_a]
        go_left[rows] = X[act_idx[rows], best_feat[slot_a[rows]]] <= best_thr[slot_a[rows]]
        child_slot = 2 * slot_a + (~go_left)
        c_n, c_vals, c_imp, _ = crit.node_stats(act_idx, child_slot, 2 * F)
        nL, nR = c_n[0::2].astype(np.float64), c_n[1::2].astype(np.float64)
        nT = n_s.astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            improvement = (nT / root_impurity_scale) * (
                node_imp - nL / nT * c_imp[0::2] - nR / nT * c_imp[1::2]
            )
        split = has_split & (improvement + _EPS >= params.min_impurity_decrease)

        next_frontier = []
        child_ids = np.full(2 * F, -1, dtype=np.intp)
        for t in np.flatnonzero(split):
            node = int(frontier[t])
            ids = add_nodes(
                c_n[2 * t : 2 * t + 2],
                c_vals[2 * t : 2 * t + 2],
                c_imp[2 * t : 2 * t + 2],
                depth[node] + 1,
            )
            left[node], right[node] = int(ids[0]), int(ids[1])
            feature[node] = int(best_feat[t])
            threshold[node] = float(best_thr[t])
            child_ids[2 * t : 2 * t + 2] = ids
            next_frontier.extend(ids[can_split(ids)].tolist())
        moved = split[slot_a]
        node_of[act_idx[moved]] = child_ids[child_slot[moved]]
        frontier = np.asarray(next_frontier, dtype=np.intp)

    return _to_preorder(left, right, feature, threshold, values, impurity, n_samples, depth)


def _to_preorder(left, right, feature, threshold, values, impurity, n_samples, depth) -> Tree:
    """Renumber nodes depth-first, left child first (scikit-learn's numbering)."""
    order = []
    stack = [0]
    while stack:
        node = stack.pop()
        order.append(node)
        if left[node] != TREE_LEAF:
            stack.append(right[node])
            stack.append(left[node])
    order_arr = np.asarray(order, dtype=np.intp)
    new_id = np.empty(len(order), dtype=np.intp)
    new_id[order_arr] = np.arange(len(order))

    def remap(children):
        c = np.asarray(children, dtype=np.intp)[order_arr]
        return np.where(c == TREE_LEAF, TREE_LEAF, new_id[np.maximum(c, 0)])

    return Tree(
        children_left=remap(left),
        children_right=remap(right),
        feature=np.asarray(feature, dtype=np.intp)[order_arr],
        threshold=np.asarray(threshold, dtype=np.float64)[order_arr],
        value=np.asarray(values, dtype=np.float64)[order_arr],
        impurity=np.asarray(impurity, dtype=np.float64)[order_arr],
        n_node_samples=np.asarray(n_samples, dtype=np.intp)[order_arr],
        max_depth=int(max(depth)),
    )


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------
class _BaseDecisionTree(BaseEstimator):
    _criteria: tuple[str, ...] = ()

    def __init__(
        self,
        criterion: str,
        max_depth: int | None = None,
        min_samples_split: int | float = 2,
        min_samples_leaf: int | float = 1,
        max_features: int | float | str | None = None,
        min_impurity_decrease: float = 0.0,
        random_state=None,
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state

    def _grow(self, X: np.ndarray, y: np.ndarray, n_classes: int, rng=None) -> Tree:
        """Validate hyperparameters and grow on validated, encoded data."""
        if self.criterion not in self._criteria:
            raise ValueError(f"criterion must be one of {self._criteria}, got {self.criterion!r}.")
        params = _resolve_grow_params(
            X.shape[0],
            X.shape[1],
            self.max_depth,
            self.min_samples_split,
            self.min_samples_leaf,
            self.max_features,
            self.min_impurity_decrease,
        )
        rng = check_random_state(self.random_state if rng is None else rng)
        return build_tree(X, y, self.criterion, n_classes, params, rng)

    def _set_tree(self, tree: Tree, n_features: int) -> None:
        self.n_features_in_ = n_features
        self.tree_ = tree
        self.feature_importances_ = tree.compute_feature_importances(n_features)

    def _validated(self, X) -> np.ndarray:
        self._check_is_fitted("tree_")
        return check_n_features(X, self.n_features_in_, self)

    def apply(self, X) -> np.ndarray:
        """Index (into ``tree_`` arrays) of the leaf each row of ``X`` ends in."""
        X = self._validated(X)
        return self.tree_.apply(X)

    def get_depth(self) -> int:
        """Depth of the tree (a single leaf has depth 0)."""
        self._check_is_fitted("tree_")
        return self.tree_.max_depth

    def get_n_leaves(self) -> int:
        """Number of leaves."""
        self._check_is_fitted("tree_")
        return self.tree_.n_leaves


_TREE_PARAMS_DOC = """
    max_depth : int or None, default=None
        Maximum depth; ``None`` grows until the other rules stop it.
    min_samples_split : int or float, default=2
        Minimum samples needed to split a node (a float is a fraction of n, rounded up).
    min_samples_leaf : int or float, default=1
        Minimum samples in each child of a split (a float is a fraction of n, rounded up).
    max_features : int, float, "sqrt", "log2" or None, default=None
        Number of features evaluated per node (see the module docstring); ``None``
        uses all ``d``.
    min_impurity_decrease : float, default=0.0
        Split only if the weighted impurity decrease is at least this.
    random_state : None, int, SeedSequence or Generator
        Seed for the per-node feature subsampling (unused when all features are used).
"""


@register(
    domain="ml",
    tasks=("classification",),
    complexity=_TREE_COMPLEXITY,
    references=_REFERENCES,
    status="experimental",
)
class DecisionTreeClassifier(ClassifierMixin, _BaseDecisionTree):
    __doc__ = (
        r"""CART classification tree (Gini or entropy splits).

    Parameters
    ----------
    criterion : {"gini", "entropy"}, default="gini"
        Node impurity (see the module docstring)."""
        + _TREE_PARAMS_DOC
        + r"""
    Attributes
    ----------
    classes_ : ndarray of shape (k,)
    n_classes_ : int
    tree_ : Tree
        Array-based tree; ``tree_.value`` holds class fractions, shape (nodes, k).
    feature_importances_ : ndarray of shape (d,)
        Normalised mean decrease in impurity.
    n_features_in_ : int

    Complexity
    ----------
    ``fit``: time O(T n d log n) for a tree of depth T (each level sorts every
    feature's samples by node), plus O(T n d k) for the class-count prefix sums;
    extra space O(n d) for the presorted index table. ``predict``, ``apply``:
    time O(T) per sample, extra space O(n).

    References
    ----------
    Breiman, Friedman, Olshen & Stone (1984); Hastie, Tibshirani & Friedman,
    *ESL* 2e, §9.2.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.ml import DecisionTreeClassifier
    >>> X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]])
    >>> y = np.array([0, 0, 1, 1])
    >>> clf = DecisionTreeClassifier().fit(X, y)
    >>> int(clf.tree_.feature[0]), float(clf.tree_.threshold[0]), clf.get_depth()
    (0, 1.5, 1)
    >>> clf.predict([[0.5, 0.0], [2.5, 1.0]]).tolist()
    [0, 1]
    """
    )

    _criteria = _CLF_CRITERIA

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int | float = 2,
        min_samples_leaf: int | float = 1,
        max_features: int | float | str | None = None,
        min_impurity_decrease: float = 0.0,
        random_state=None,
    ):
        super().__init__(
            criterion,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_features,
            min_impurity_decrease,
            random_state,
        )

    def fit(self, X, y) -> DecisionTreeClassifier:
        """Grow the tree on ``X`` (n, d) and labels ``y`` (n,); return ``self``."""
        X, y = check_X_labels(X, y)
        classes, y_idx = np.unique(y, return_inverse=True)
        tree = self._grow(X, y_idx.astype(np.intp), len(classes))
        self.classes_ = classes
        self.n_classes_ = len(classes)
        self._set_tree(tree, X.shape[1])
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Class fractions of the leaf each row falls in, shape (n, k)."""
        leaves = self.apply(X)  # checks fitted first
        return self.tree_.value[leaves]

    def predict(self, X) -> np.ndarray:
        """Majority class of the leaf (lowest class index on ties)."""
        proba = self.predict_proba(X)  # checks fitted first
        return self.classes_[np.argmax(proba, axis=1)]


@register(
    domain="ml",
    tasks=("regression",),
    complexity=_TREE_COMPLEXITY,
    references=_REFERENCES,
    status="experimental",
)
class DecisionTreeRegressor(RegressorMixin, _BaseDecisionTree):
    __doc__ = (
        r"""CART regression tree (squared-error splits, leaf means).

    Parameters
    ----------
    criterion : {"squared_error"}, default="squared_error"
        Node impurity: the variance of the node's targets."""
        + _TREE_PARAMS_DOC
        + r"""
    Attributes
    ----------
    tree_ : Tree
        Array-based tree; ``tree_.value`` holds node means, shape (nodes,).
    feature_importances_ : ndarray of shape (d,)
    n_features_in_ : int

    Complexity
    ----------
    ``fit``: time O(T n d log n) for a tree of depth T, extra space O(n d).
    ``predict``, ``apply``: time O(T) per sample, extra space O(n).

    References
    ----------
    Breiman, Friedman, Olshen & Stone (1984); *ESL* 2e, §9.2.2.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.ml import DecisionTreeRegressor
    >>> X = np.array([[0.0], [1.0], [2.0], [3.0]])
    >>> reg = DecisionTreeRegressor(max_depth=1).fit(X, [1.0, 1.0, 5.0, 7.0])
    >>> float(reg.tree_.threshold[0]), reg.predict([[0.2], [2.9]]).tolist()
    (1.5, [1.0, 6.0])
    """
    )

    _criteria = _REG_CRITERIA

    def __init__(
        self,
        criterion: str = "squared_error",
        max_depth: int | None = None,
        min_samples_split: int | float = 2,
        min_samples_leaf: int | float = 1,
        max_features: int | float | str | None = None,
        min_impurity_decrease: float = 0.0,
        random_state=None,
    ):
        super().__init__(
            criterion,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_features,
            min_impurity_decrease,
            random_state,
        )

    def fit(self, X, y) -> DecisionTreeRegressor:
        """Grow the tree on ``X`` (n, d) and targets ``y`` (n,); return ``self``."""
        X, y = check_X_y(X, y)
        tree = self._grow(X, y, 1)
        self._set_tree(tree, X.shape[1])
        return self

    def predict(self, X) -> np.ndarray:
        """Mean training target of the leaf each row falls in."""
        leaves = self.apply(X)  # checks fitted first
        return self.tree_.value[leaves]
