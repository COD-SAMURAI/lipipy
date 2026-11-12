"""Private helpers shared by the wave-2 numerics modules (validation, JSON export)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike


def jsonable(value: Any) -> Any:
    """Turn NumPy arrays / scalars (incl. complex) into JSON-friendly Python objects."""
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {"real": value.real.tolist(), "imag": value.imag.tolist()}
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def check_callable(func: Any, name: str) -> None:
    if not callable(func):
        raise TypeError(f"{name} must be callable, got {type(func).__name__}")


def as_real(x: ArrayLike, name: str, *, ndim: int | None = None, finite: bool = True) -> np.ndarray:
    """Convert to a float64 array with clear errors for non-numeric or complex input."""
    if isinstance(x, (str, bytes)):
        raise TypeError(f"{name} must be numeric, got {type(x).__name__}")
    if np.iscomplexobj(x):
        raise TypeError(f"{name} must be real-valued, got a complex array")
    try:
        arr = np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be convertible to a float64 array: {exc}") from None
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {arr.shape}")
    if finite and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinity")
    return arr


def as_finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def check_positive_int(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)
