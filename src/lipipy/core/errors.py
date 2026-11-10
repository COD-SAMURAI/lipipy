"""The lipipy exception and warning hierarchy.

Catch :class:`LipipyError` to handle any error raised deliberately by lipipy.
"""

from __future__ import annotations


class LipipyError(Exception):
    """Base class for all lipipy errors."""


class NotFittedError(LipipyError, RuntimeError):
    """An estimator was used before ``fit`` was called."""


class MissingDependencyError(LipipyError, ImportError):
    """An optional dependency needed by this feature is not installed."""


class RegistryError(LipipyError, KeyError):
    """An algorithm name is unknown or registered twice."""

    def __str__(self) -> str:  # KeyError would otherwise wrap the message in quotes
        return str(self.args[0]) if self.args else ""


class ConvergenceWarning(UserWarning):
    """An iterative algorithm stopped before meeting its tolerance."""


class ResourceLimitError(LipipyError, ValueError):
    """A request would exceed a configured time or memory limit.

    Raised before allocating, e.g. when a quantum statevector for too many qubits
    is requested.
    """
