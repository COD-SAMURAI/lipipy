"""Independent brute-force checks for CART trees (used by the tree/forest/boosting tests).

Nothing here imports the implementation's split search: impurities and gains are
recomputed from scratch with plain NumPy over every candidate split.
"""

from __future__ import annotations

import numpy as np


def impurity(y: np.ndarray, criterion: str) -> float:
    if criterion == "squared_error":
        return float(np.mean((y - y.mean()) ** 2))
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    if criterion == "gini":
        return float(1.0 - np.sum(p * p))
    return float(-np.sum(p * np.log2(p)))


def node_samples(tree, X: np.ndarray) -> list[np.ndarray]:
    """Indices of the training rows that reach every node (by explicit traversal)."""
    members: list[np.ndarray] = [np.empty(0, dtype=int)] * tree.node_count
    stack = [(0, np.arange(X.shape[0]))]
    while stack:
        node, idx = stack.pop()
        members[node] = idx
        if tree.children_left[node] != -1:
            go_left = X[idx, tree.feature[node]] <= tree.threshold[node]
            stack.append((int(tree.children_left[node]), idx[go_left]))
            stack.append((int(tree.children_right[node]), idx[~go_left]))
    return members


def _weighted_child_impurity(left: np.ndarray, Y: np.ndarray, criterion: str) -> np.ndarray:
    """n_L I(L) + n_R I(R) for every row of the boolean (candidates, m) matrix ``left``."""
    nl = left.sum(axis=1).astype(float)
    nr = left.shape[1] - nl
    if criterion == "squared_error":
        y = Y[:, 0]
        s_l, q_l = left @ y, left @ (y * y)
        s_r, q_r = y.sum() - s_l, (y * y).sum() - q_l
        return (q_l - s_l**2 / nl) + (q_r - s_r**2 / nr)
    cl = left.astype(float) @ Y  # class counts on the left, (candidates, k)
    cr = Y.sum(axis=0) - cl
    out = np.zeros(len(nl))
    for counts, size in ((cl, nl), (cr, nr)):
        p = counts / size[:, None]
        if criterion == "gini":
            out += size * (1.0 - np.sum(p * p, axis=1))
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                out -= size * np.sum(np.where(p > 0, p * np.log2(p), 0.0), axis=1)
    return out


def all_split_gains(Xn, yn, criterion, min_samples_leaf):
    """[(gain, feature, threshold)] for every valid split of one node (brute force)."""
    n = len(yn)
    base = n * impurity(yn, criterion)
    if criterion == "squared_error":
        Y = yn.astype(float)[:, None]
    else:
        Y = (yn[:, None] == np.unique(yn)[None, :]).astype(float)
    out = []
    for f in range(Xn.shape[1]):
        values = np.unique(Xn[:, f])
        if len(values) < 2:
            continue
        lows, highs = values[:-1], values[1:]
        left = Xn[None, :, f] <= lows[:, None]  # one row per candidate threshold
        nl = left.sum(axis=1)
        ok = (nl >= min_samples_leaf) & (n - nl >= min_samples_leaf)
        if not ok.any():
            continue
        gains = base - _weighted_child_impurity(left[ok], Y, criterion)
        thresholds = lows[ok] / 2.0 + highs[ok] / 2.0
        out += [(float(g), f, float(t)) for g, t in zip(gains, thresholds, strict=True)]
    return out


def check_splits_are_optimal(tree, X, y, criterion, min_samples_leaf=1, rtol=1e-9):
    """Assert every split of ``tree`` maximises the gain; return True if no node had a tie.

    A tie is a second candidate (another feature or threshold) whose gain is within
    ``rtol`` (relative to the node's total impurity) of the best one.
    """
    tie_free = True
    for node, idx in enumerate(node_samples(tree, X)):
        if tree.children_left[node] == -1:
            continue
        cands = all_split_gains(X[idx], y[idx], criterion, min_samples_leaf)
        best = max(g for g, _, _ in cands)
        scale = max(len(idx) * impurity(y[idx], criterion), 1e-300)
        f, thr = tree.feature[node], tree.threshold[node]
        ours = [g for g, cf, ct in cands if cf == f and ct == thr]
        assert ours, f"node {node}: split ({f}, {thr}) is not a valid candidate"
        assert ours[0] >= best - rtol * scale, f"node {node}: gain {ours[0]} < best {best}"
        if sum(g >= best - rtol * scale for g, _, _ in cands) > 1:
            tie_free = False
    return tie_free
