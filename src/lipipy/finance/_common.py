"""Private validation and result helpers shared by the finance modules."""

from __future__ import annotations

import numbers
from dataclasses import fields
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

_EPS = float(np.finfo(np.float64).eps)


def as_float_array(x: ArrayLike, name: str) -> np.ndarray:
    """Convert to a float64 array, turning conversion failures into TypeError."""
    if isinstance(x, (str, bytes)):
        raise TypeError(f"{name} must be numeric, got {type(x).__name__}")
    try:
        return np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric array-like: {exc}") from None


def check_finite(a: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN or infinity")


def check_series(x: ArrayLike, name: str = "returns", min_rows: int = 1) -> np.ndarray:
    """A finite float64 array of shape (T,) or (T, A) with at least ``min_rows`` rows."""
    a = as_float_array(x, name)
    if a.ndim not in (1, 2):
        raise ValueError(f"{name} must have shape (T,) or (T, A), got shape {a.shape}")
    if a.shape[0] < min_rows or a.size == 0:
        raise ValueError(f"{name} needs at least {min_rows} row(s), got shape {a.shape}")
    check_finite(a, name)
    return a


def check_prices(x: ArrayLike, name: str = "prices", min_rows: int = 2) -> np.ndarray:
    p = check_series(x, name, min_rows)
    if np.any(p <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    return p


def check_aligned(b: ArrayLike, a: np.ndarray, name: str) -> np.ndarray:
    """A finite series of shape (T,) or a.shape that broadcasts against ``a`` row-wise."""
    b = check_series(b, name)
    if b.shape[0] != a.shape[0]:
        raise ValueError(f"{name} has {b.shape[0]} rows but returns has {a.shape[0]}")
    if b.ndim == 1 and a.ndim == 2:
        return b[:, None]
    if b.shape != a.shape:
        raise ValueError(f"{name} must have shape ({a.shape[0]},) or {a.shape}, got {b.shape}")
    return b


def check_rate(x: ArrayLike, a: np.ndarray, name: str) -> np.ndarray | float:
    """A scalar per-period rate, or a series aligned with ``a``."""
    arr = as_float_array(x, name)
    if arr.ndim == 0:
        check_finite(arr, name)
        return float(arr)
    return check_aligned(arr, a, name)


def check_positive_number(x: Any, name: str) -> float:
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(x).__name__}")
    v = float(x)
    if not (np.isfinite(v) and v > 0.0):
        raise ValueError(f"{name} must be a positive finite number, got {x!r}")
    return v


def check_real(x: Any, name: str) -> float:
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(x).__name__}")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite, got {x!r}")
    return v


def check_int(x: Any, name: str, minimum: int) -> int:
    if isinstance(x, bool) or not isinstance(x, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(x).__name__}")
    if int(x) < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {x!r}")
    return int(x)


def check_level(level: Any) -> float:
    v = check_real(level, "level")
    if not 0.0 < v < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level!r}")
    return v


def scalar_or_array(a: np.ndarray) -> Any:
    """0-d arrays become Python floats; other arrays are returned as is."""
    a = np.asarray(a)
    return float(a) if a.ndim == 0 else a


def is_negligible(spread: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """True where a dispersion is zero up to rounding of data of magnitude ``scale``.

    A constant series has a computed standard deviation of a few ulps of its values,
    not exactly zero; anything below ``64 eps * scale`` is treated as zero.
    """
    return np.asarray(spread) <= 64.0 * _EPS * np.asarray(scale)


def safe_ratio(num: np.ndarray, den: np.ndarray, zero_den: np.ndarray) -> np.ndarray:
    """``num / den``, with ``+-inf`` (or NaN for 0/0) where ``zero_den`` is True."""
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    zero_den = np.asarray(zero_den, dtype=bool)
    num, den, zero_den = np.broadcast_arrays(num, den, zero_den)
    out = np.empty(num.shape)
    ok = ~zero_den
    out[ok] = num[ok] / den[ok]
    out[zero_den] = np.where(num[zero_den] == 0.0, np.nan, np.copysign(np.inf, num[zero_den]))
    return out


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    return value


class DictMixin:
    """``to_dict`` for frozen result dataclasses (arrays become lists)."""

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dictionary (arrays become lists)."""
        return {f.name: jsonable(getattr(self, f.name)) for f in fields(self)}  # type: ignore[arg-type]
