"""lipipy: readable, verified, complexity-documented implementations of the
algorithms behind computing, from numerical methods to ML, deep learning, AI agents,
quantum computing and quantitative finance.

Layers (a package imports only from lower layers; see ARCHITECTURE.md §3)
---------------------------------------------------------------------------
0  core       registry, complexity metadata, backends, randomness, errors, validation
1  numerics   linear algebra, optimization, integration, ODEs, FFT, statistics
1  bench      empirical time/space complexity measurement
2  ml         classic machine learning (NumPy only)
2  ai         classical AI: search, logic, planning, reasoning under uncertainty
2  dl         deep learning building blocks (``lipipy[dl]``)
2  quantum    quantum circuits, simulators and algorithms
2  finance    returns, risk, portfolios, pricing and backtesting
3  rl         reinforcement learning
3  research   modern architectures from recent papers
3  interpret  interpretability and analysis tools
4  agents     LLM agents, tools, reasoning, AGI research directions

Subpackages load lazily on first use, so ``import lipipy`` stays fast::

    >>> import lipipy
    >>> lipipy.registry.get("ml.LinearRegression").domain
    'ml'
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from lipipy.core.registry import registry

__version__ = "0.1.0.dev0"

SUBPACKAGES = (
    "core",
    "numerics",
    "bench",
    "ml",
    "ai",
    "dl",
    "quantum",
    "finance",
    "rl",
    "research",
    "interpret",
    "agents",
)

__all__ = ["SUBPACKAGES", "__version__", "registry", *SUBPACKAGES]

if TYPE_CHECKING:  # pragma: no cover - for static analysers only
    from lipipy import (  # noqa: F401
        agents,
        ai,
        bench,
        core,
        dl,
        finance,
        interpret,
        ml,
        numerics,
        quantum,
        research,
        rl,
    )


def __getattr__(name: str) -> Any:
    if name in SUBPACKAGES:
        module = importlib.import_module(f"lipipy.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'lipipy' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(SUBPACKAGES))
