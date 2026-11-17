"""Private helpers shared by the lipipy.ai modules (validation, JSON export)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def jsonable(value: Any) -> Any:
    """Turn tuples, NumPy values and nested containers into JSON-friendly objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (frozenset, set)):
        return sorted((jsonable(v) for v in value), key=repr)
    if isinstance(value, float) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def check_callable(func: Any, name: str) -> None:
    if not callable(func):
        raise TypeError(f"{name} must be callable, got {type(func).__name__}")


def check_int(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def check_float(value: Any, name: str, *, low: float | None = None, high: float | None = None,
                low_open: bool = False) -> float:  # fmt: skip
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if low is not None and (out < low or (low_open and out == low)):
        raise ValueError(f"{name} must be {'>' if low_open else '>='} {low}, got {out}")
    if high is not None and out > high:
        raise ValueError(f"{name} must be <= {high}, got {out}")
    return out
