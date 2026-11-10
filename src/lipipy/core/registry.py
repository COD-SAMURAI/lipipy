"""Central catalog of every lipipy algorithm.

The registry is the single source of truth for discovery, documentation, agent
tooling, benchmarks and plugins (see ARCHITECTURE.md §4.1).

Register an algorithm with :func:`register`::

    @register(
        domain="ml",
        tasks=("regression",),
        complexity=Complexity(time="O(n d^2 + d^3)", space="O(n d + d^2)"),
        references=("Hastie et al., ESL 2e, §3.2",),
        status="stable",
    )
    class LinearRegression: ...

Then query it::

    >>> import lipipy
    >>> info = lipipy.registry.get("ml.LinearRegression")
    >>> info.domain, info.tasks
    ('ml', ('regression',))

Cost: ``get`` is O(1); ``find`` and ``to_json`` are O(R) for R registered entries.
The first ``find``/``get``/iteration imports every subpackage once (and the
``lipipy.plugins`` entry points) so all algorithms are visible; subpackages whose
optional dependencies are missing are skipped. Set the environment variable
``LIPIPY_DISABLE_PLUGINS=1`` to skip third-party plugins.
"""

from __future__ import annotations

import importlib
import json
import os
import pkgutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any, Literal, TypeVar

from lipipy.core.complexity import Complexity
from lipipy.core.errors import MissingDependencyError, RegistryError

Status = Literal["experimental", "stable", "deprecated"]
_STATUSES = ("experimental", "stable", "deprecated")
PLUGIN_GROUP = "lipipy.plugins"
DISABLE_PLUGINS_ENV = "LIPIPY_DISABLE_PLUGINS"

T = TypeVar("T", bound=Callable[..., Any])


@dataclass(frozen=True)
class AlgorithmInfo:
    """Metadata describing one registered algorithm."""

    name: str
    obj: Callable[..., Any] = field(repr=False, compare=False)
    domain: str
    tasks: tuple[str, ...]
    complexity: Complexity
    references: tuple[str, ...] = ()
    backends: tuple[str, ...] = ("numpy",)
    status: Status = "experimental"

    @property
    def import_path(self) -> str:
        return f"{self.obj.__module__}.{self.obj.__qualname__}"

    @property
    def summary(self) -> str:
        doc = (self.obj.__doc__ or "").strip()
        return doc.splitlines()[0] if doc else ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly description (for docs and AI agents)."""
        return {
            "name": self.name,
            "summary": self.summary,
            "domain": self.domain,
            "tasks": list(self.tasks),
            "complexity": self.complexity.to_dict(),
            "references": list(self.references),
            "backends": list(self.backends),
            "status": self.status,
            "import_path": self.import_path,
        }


class Registry:
    """A name → :class:`AlgorithmInfo` mapping with search helpers."""

    def __init__(self) -> None:
        self._entries: dict[str, AlgorithmInfo] = {}
        self._discovered = False

    # -- registration -----------------------------------------------------------
    def add(self, info: AlgorithmInfo) -> None:
        existing = self._entries.get(info.name)
        if existing is not None and existing.obj is not info.obj:
            raise RegistryError(f"Algorithm {info.name!r} is already registered.")
        self._entries[info.name] = info

    # -- discovery --------------------------------------------------------------
    def discover(self) -> None:
        """Import all lipipy subpackages and plugins so their algorithms register."""
        if self._discovered:
            return
        self._discovered = True
        import lipipy

        for mod in pkgutil.walk_packages(lipipy.__path__, prefix="lipipy."):
            if mod.name.rsplit(".", 1)[-1].startswith("_"):
                continue
            try:
                importlib.import_module(mod.name)
            except (ImportError, MissingDependencyError):
                continue  # optional dependency not installed
        if os.environ.get(DISABLE_PLUGINS_ENV, "").strip().lower() in {"1", "true", "yes"}:
            return
        # Plugins are code from packages the user installed; they run on import.
        for ep in entry_points(group=PLUGIN_GROUP):
            try:
                ep.load()
            except (ImportError, MissingDependencyError):
                continue

    # -- queries ----------------------------------------------------------------
    def get(self, name: str) -> AlgorithmInfo:
        if name not in self._entries:
            self.discover()
        try:
            return self._entries[name]
        except KeyError:
            raise RegistryError(f"No algorithm named {name!r}.") from None

    def find(
        self,
        *,
        domain: str | None = None,
        task: str | None = None,
        text: str | None = None,
        status: Status | None = None,
        backend: str | None = None,
    ) -> list[AlgorithmInfo]:
        """Return algorithms matching every given filter, sorted by name.

        ``text`` is a case-insensitive substring search over name and summary.
        """
        self.discover()
        needle = text.lower() if text else None
        out = []
        for info in self._entries.values():
            if domain is not None and info.domain != domain:
                continue
            if task is not None and task not in info.tasks:
                continue
            if status is not None and info.status != status:
                continue
            if backend is not None and backend not in info.backends:
                continue
            if needle and needle not in f"{info.name} {info.summary}".lower():
                continue
            out.append(info)
        return sorted(out, key=lambda i: i.name)

    def to_json(self, **filters: Any) -> str:
        """Machine-readable catalog (all entries, or those matching ``find`` filters)."""
        return json.dumps([i.to_dict() for i in self.find(**filters)], indent=2)

    def __iter__(self) -> Iterator[AlgorithmInfo]:
        return iter(self.find())

    def __len__(self) -> int:
        self.discover()
        return len(self._entries)

    def __contains__(self, name: object) -> bool:
        self.discover()
        return name in self._entries


#: The global registry instance (also available as ``lipipy.registry``).
registry = Registry()


def register(
    *,
    domain: str,
    complexity: Complexity,
    tasks: tuple[str, ...] = (),
    references: tuple[str, ...] = (),
    backends: tuple[str, ...] = ("numpy",),
    status: Status = "experimental",
    name: str | None = None,
) -> Callable[[T], T]:
    """Class/function decorator that adds an algorithm to :data:`registry`.

    The registered name defaults to ``"<domain>.<object name>"``. The decorated
    object is returned unchanged, with its metadata attached as ``__lipipy_info__``.
    """
    if status not in _STATUSES:
        raise ValueError(f"status must be one of {_STATUSES}, got {status!r}")
    if not isinstance(complexity, Complexity):
        raise TypeError("complexity must be a lipipy.core.Complexity instance")

    def decorator(obj: T) -> T:
        info = AlgorithmInfo(
            name=name or f"{domain}.{obj.__name__}",
            obj=obj,
            domain=domain,
            tasks=tuple(tasks),
            complexity=complexity,
            references=tuple(references),
            backends=tuple(backends),
            status=status,
        )
        registry.add(info)
        obj.__lipipy_info__ = info  # type: ignore[attr-defined]
        return obj

    return decorator
