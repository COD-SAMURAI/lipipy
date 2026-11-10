"""Optional-dependency handling.

Heavy dependencies (torch, qiskit, …) are imported only inside the code that needs
them, through :func:`require`, so ``import lipipy`` stays light and a missing
package produces an actionable message.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from lipipy.core.errors import MissingDependencyError


def require(module: str, *, extra: str, purpose: str = "") -> ModuleType:
    """Import ``module`` or raise :class:`MissingDependencyError` with an install hint.

    Examples
    --------
    >>> np = require("numpy", extra="core")
    >>> np.__name__
    'numpy'
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        what = f" for {purpose}" if purpose else ""
        raise MissingDependencyError(
            f"{module!r} is required{what}. Install it with: pip install 'lipipy[{extra}]'"
        ) from exc


def is_available(module: str) -> bool:
    """Return True if ``module`` can be imported (does not raise)."""
    try:
        importlib.import_module(module)
    except ImportError:
        return False
    return True
