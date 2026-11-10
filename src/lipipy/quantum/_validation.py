"""Private input checks shared by the quantum modules.

Public functions validate once at their boundary with these helpers; internal
code then trusts its inputs.
"""

from __future__ import annotations

import numbers
from collections.abc import Iterable

import numpy as np

from lipipy.core.errors import ResourceLimitError

#: Default largest statevector a simulator will allocate (16·2^26 bytes = 1 GiB).
MAX_QUBITS = 26


def format_bytes(num_bytes: int) -> str:
    """Human-readable size with binary prefixes, e.g. ``'1.0 GiB'``."""
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"):
        if size < 1024 or unit == "EiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")  # pragma: no cover


def check_int(value: object, name: str, *, minimum: int | None = None) -> int:
    """Return ``value`` as an ``int`` (bools rejected), optionally bounded below."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def check_num_qubits(num_qubits: object, max_qubits: int = MAX_QUBITS) -> int:
    """Validate a qubit count and refuse it, before any allocation, if too large."""
    q = check_int(num_qubits, "num_qubits", minimum=1)
    max_qubits = check_int(max_qubits, "max_qubits", minimum=1)
    if q > max_qubits:
        need = 16 * 2**q
        raise ResourceLimitError(
            f"A statevector of {q} qubits needs 16·2^{q} bytes = {format_bytes(need)} "
            f"(complex128), above max_qubits={max_qubits}. Raise max_qubits if you have "
            "the memory, or use a matrix-product-state simulator (memory O(q χ^2))."
        )
    return q


def check_angle(value: object, name: str) -> float:
    """A finite real angle in radians."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return value


def check_state(state: object, name: str = "state") -> tuple[np.ndarray, int]:
    """Return ``(state as complex128 1-D array, number of qubits)``."""
    psi = np.asarray(state, dtype=np.complex128)
    if psi.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {psi.shape}")
    size = psi.shape[0]
    if size < 2 or size & (size - 1):
        raise ValueError(f"{name} length must be a power of two >= 2, got {size}")
    if not np.all(np.isfinite(psi)):
        raise ValueError(f"{name} contains NaN or infinity")
    return psi, size.bit_length() - 1


def check_square_matrix(matrix: object, name: str = "gate") -> tuple[np.ndarray, int]:
    """Return ``(matrix as complex128, number of qubits k)`` for a 2^k x 2^k matrix."""
    U = np.asarray(matrix, dtype=np.complex128)
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {U.shape}")
    size = U.shape[0]
    if size < 2 or size & (size - 1):
        raise ValueError(f"{name} dimension must be a power of two >= 2, got {size}")
    if not np.all(np.isfinite(U)):
        raise ValueError(f"{name} contains NaN or infinity")
    return U, size.bit_length() - 1


def check_qubits(
    qubits: Iterable[object] | object, num_qubits: int, name: str = "targets"
) -> tuple[int, ...]:
    """Distinct qubit indices in ``range(num_qubits)``; a single int is accepted."""
    if isinstance(qubits, numbers.Integral) and not isinstance(qubits, bool):
        qubits = (qubits,)
    if isinstance(qubits, (str, bytes)) or not isinstance(qubits, Iterable):
        raise TypeError(f"{name} must be an int or a sequence of ints")
    out = tuple(check_int(t, name) for t in qubits)
    if not out:
        raise ValueError(f"{name} must not be empty")
    for t in out:
        if not 0 <= t < num_qubits:
            raise ValueError(f"qubit index {t} out of range for {num_qubits} qubits")
    if len(set(out)) != len(out):
        raise ValueError(f"{name} contains duplicate qubits: {out}")
    return out
