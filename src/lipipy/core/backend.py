"""Array-backend dispatch (NumPy, CuPy, PyTorch, JAX).

lipipy code that should run on any backend asks for the caller's array namespace
and uses only Array API standard functions on it::

    xp = get_namespace(X)
    centred = X - xp.mean(X, axis=0)

With only NumPy installed, NumPy arrays and Python sequences are supported. Install
``lipipy[backends]`` (array-api-compat) to accept CuPy, PyTorch and JAX arrays; the
result then stays on the caller's device. See ARCHITECTURE.md §4.2.
"""

from __future__ import annotations

import numbers
from types import ModuleType
from typing import Any

import numpy as np

from lipipy.core.errors import MissingDependencyError
from lipipy.core.imports import is_available

_HAS_COMPAT = is_available("array_api_compat")


def _is_numpy_like(x: Any) -> bool:
    return isinstance(x, np.ndarray | np.generic | list | tuple | numbers.Number)


def get_namespace(*arrays: Any) -> ModuleType:
    """Return the array namespace shared by ``arrays``.

    Raises
    ------
    TypeError
        If the arrays come from different libraries.
    MissingDependencyError
        If a non-NumPy array is passed without ``lipipy[backends]`` installed.

    Examples
    --------
    >>> import numpy as np
    >>> xp = get_namespace(np.ones(3))
    >>> float(xp.sum(np.ones(3)))
    3.0
    """
    arrays = tuple(a for a in arrays if a is not None)
    if all(_is_numpy_like(a) for a in arrays):
        if _HAS_COMPAT:
            import array_api_compat

            return array_api_compat.array_namespace(*[np.asarray(a) for a in arrays])
        return np
    if not _HAS_COMPAT:
        kinds = sorted({type(a).__module__.split(".")[0] for a in arrays})
        raise MissingDependencyError(
            f"Arrays from {kinds} need array-api-compat. "
            "Install it with: pip install 'lipipy[backends]'"
        )
    import array_api_compat

    return array_api_compat.array_namespace(*arrays)


def backend_name(xp: ModuleType) -> str:
    """Short library name of a namespace, e.g. ``'numpy'``, ``'torch'``, ``'cupy'``."""
    name = xp.__name__
    if name.startswith("array_api_compat."):
        name = name.split(".", 1)[1]
    return name.split(".")[0]


def to_numpy(x: Any) -> np.ndarray:
    """Copy any supported array to a NumPy array on the host (CPU)."""
    if isinstance(x, np.ndarray):
        return x
    if _HAS_COMPAT:
        import array_api_compat

        if array_api_compat.is_torch_array(x):
            return x.detach().cpu().numpy()
        if array_api_compat.is_cupy_array(x):
            return x.get()
    return np.asarray(x)
