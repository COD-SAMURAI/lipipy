"""Private validation, result and action-selection helpers shared by ``lipipy.rl``."""

from __future__ import annotations

import numbers
from collections.abc import Callable
from dataclasses import fields
from typing import Any

import numpy as np


class ResultMixin:
    """``to_dict`` for frozen result dataclasses (arrays become nested lists)."""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):  # type: ignore[arg-type]
            value = getattr(self, f.name)
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, np.generic):
                value = value.item()
            out[f.name] = value
        return out


def freeze(*arrays: np.ndarray) -> None:
    """Mark arrays stored in a frozen result as read-only."""
    for a in arrays:
        a.setflags(write=False)


def is_int(v) -> bool:
    return isinstance(v, numbers.Integral) and not isinstance(v, (bool, np.bool_))


def is_real(v) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, (bool, np.bool_))


def check_int(v, name: str, minimum: int = 1) -> int:
    if not is_int(v) or v < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {v!r}.")
    return int(v)


def check_gamma(gamma) -> float:
    if not is_real(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be a number in [0, 1], got {gamma!r}.")
    return float(gamma)


def check_unit(v, name: str) -> float:
    """A number in [0, 1] (probabilities, epsilon, lambda)."""
    if not is_real(v) or not 0.0 <= v <= 1.0:
        raise ValueError(f"{name} must be a number in [0, 1], got {v!r}.")
    return float(v)


def check_positive(v, name: str, allow_zero: bool = False) -> float:
    ok = is_real(v) and np.isfinite(v) and (v >= 0 if allow_zero else v > 0)
    if not ok:
        bound = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be a finite number {bound}, got {v!r}.")
    return float(v)


StepSize = float | Callable[[int], float]


def check_step_size(alpha, name: str = "alpha") -> Callable[[int], float]:
    """A constant in (0, 1] or a callable ``n -> alpha_n`` of the visit count n >= 1."""
    if callable(alpha):
        return alpha
    a = check_positive(alpha, name)
    if a > 1.0:
        raise ValueError(f"{name} must be in (0, 1], got {alpha!r}.")
    return lambda n: a


def argmax_random(values: np.ndarray, rng: np.random.Generator) -> int:
    """Index of a maximum of ``values``, ties broken uniformly at random."""
    best = np.flatnonzero(values == values.max())
    return int(best[0]) if best.size == 1 else int(rng.choice(best))


def epsilon_greedy(q: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    """With probability epsilon a uniform action, otherwise a greedy one (random ties)."""
    if epsilon > 0.0 and rng.random() < epsilon:
        return int(rng.integers(q.size))
    return argmax_random(q, rng)


def epsilon_greedy_probs(q: np.ndarray, epsilon: float) -> np.ndarray:
    """Action probabilities of the epsilon-greedy policy (greedy mass shared by ties)."""
    probs = np.full(q.size, epsilon / q.size)
    best = q == q.max()
    probs[best] += (1.0 - epsilon) / np.count_nonzero(best)
    return probs


def greedy_policy(Q: np.ndarray) -> np.ndarray:
    """Deterministic greedy policy, lowest action index on ties, shape (S,)."""
    return np.argmax(Q, axis=1)


def seed_from(rng: np.random.Generator) -> int:
    """An integer seed drawn from ``rng`` (Gymnasium's ``reset`` accepts only ints)."""
    return int(rng.integers(0, 2**63 - 1))
