"""Big-O complexity metadata.

Every registered algorithm declares its cost with :class:`Complexity`, written in
terms of the shared variable glossary :data:`SYMBOLS` so that costs are comparable
across domains. Space means *extra* memory beyond inputs and outputs.

Declared complexity is checked empirically with
:func:`lipipy.bench.estimate_complexity`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

#: Shared variable names for complexity expressions (see ARCHITECTURE.md §6.1).
SYMBOLS: dict[str, str] = {
    "n": "number of samples / rows",
    "d": "number of features / dimension",
    "k": "clusters, classes or neighbours",
    "T": "iterations or time steps",
    "L": "sequence length",
    "h": "hidden width",
    "V": "graph vertices",
    "E": "graph edges",
    "q": "qubits",
    "g": "quantum gates",
    "χ": "bond dimension (tensor networks)",
    "A": "assets",
    "P": "Monte Carlo paths / samples",
    "B": "batch size",
}


@dataclass(frozen=True)
class Complexity:
    """Declared asymptotic cost of an algorithm.

    Parameters
    ----------
    time : str
        Big-O time, e.g. ``"O(n d^2 + d^3)"``.
    space : str
        Big-O extra memory, e.g. ``"O(d^2)"``.
    notes : str, optional
        Caveats, e.g. which phase dominates or what changes with a solver option.

    Examples
    --------
    >>> c = Complexity(time="O(n log n)", space="O(n)")
    >>> str(c)
    'time O(n log n), space O(n)'
    """

    time: str
    space: str
    notes: str = ""

    def __post_init__(self) -> None:
        for field_name in ("time", "space"):
            value = getattr(self, field_name).strip()
            if not (value.startswith("O(") and value.endswith(")")):
                raise ValueError(f"Complexity.{field_name} must look like 'O(...)', got {value!r}")

    def __str__(self) -> str:
        text = f"time {self.time}, space {self.space}"
        return f"{text} ({self.notes})" if self.notes else text

    def to_dict(self) -> dict[str, str]:
        return asdict(self)
