"""Layer 0: shared services used by every lipipy subpackage (NumPy only).

See ARCHITECTURE.md §4 for the design of each service.
"""

from lipipy.core.backend import backend_name, get_namespace, to_numpy
from lipipy.core.base import BaseEstimator, RegressorMixin
from lipipy.core.complexity import SYMBOLS, Complexity
from lipipy.core.errors import (
    ConvergenceWarning,
    LipipyError,
    MissingDependencyError,
    NotFittedError,
    RegistryError,
    ResourceLimitError,
)
from lipipy.core.imports import is_available, require
from lipipy.core.random import check_random_state, spawn
from lipipy.core.registry import AlgorithmInfo, Registry, register, registry
from lipipy.core.validation import check_array, check_X_y

__all__ = [
    "SYMBOLS",
    "AlgorithmInfo",
    "BaseEstimator",
    "Complexity",
    "ConvergenceWarning",
    "LipipyError",
    "MissingDependencyError",
    "NotFittedError",
    "RegistryError",
    "Registry",
    "RegressorMixin",
    "ResourceLimitError",
    "backend_name",
    "check_X_y",
    "check_array",
    "check_random_state",
    "get_namespace",
    "is_available",
    "register",
    "registry",
    "require",
    "spawn",
    "to_numpy",
]
