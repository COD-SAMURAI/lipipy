"""Input validation helpers."""

from __future__ import annotations

import numpy as np


def check_array(X, *, ensure_2d: bool = True, dtype=np.float64) -> np.ndarray:
    """Convert ``X`` to a finite NumPy array, optionally requiring 2 dimensions."""
    X = np.asarray(X, dtype=dtype)
    if ensure_2d:
        if X.ndim == 1:
            raise ValueError(
                "Expected a 2D array, got 1D. Reshape with X.reshape(-1, 1) for a single "
                "feature or X.reshape(1, -1) for a single sample."
            )
        if X.ndim != 2:
            raise ValueError(f"Expected a 2D array, got {X.ndim}D.")
    if X.size == 0:
        raise ValueError("Input array is empty.")
    if not np.all(np.isfinite(X)):
        raise ValueError("Input contains NaN or infinity.")
    return X


def check_X_y(X, y) -> tuple[np.ndarray, np.ndarray]:
    """Validate a feature matrix and a 1D target vector with matching lengths."""
    X = check_array(X)
    y = check_array(y, ensure_2d=False)
    if y.ndim != 1:
        raise ValueError(f"Expected 1D target y, got shape {y.shape}.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} samples but y has {y.shape[0]}.")
    return X, y
