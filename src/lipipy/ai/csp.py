r"""Constraint satisfaction problems: backtracking, arc consistency, min-conflicts.

A CSP has variables :math:`X_1..X_n`, finite ordered domains :math:`D_i` and
constraints, each a scope :math:`(X_{i_1}, \dots, X_{i_k})` with a predicate over the
values of its scope. A solution assigns every variable a value of its domain so that
every predicate holds.

Backtracking search
-------------------
Depth-first over partial assignments. The next variable is chosen by **MRV** (fewest
remaining values) with the **degree** heuristic (most constraints with unassigned
variables) as tie-break, then the order of ``variables``. Values are tried in **LCV**
order (the value that rules out the fewest values of neighbouring variables through
binary constraints first; ties keep domain order). After each assignment an
inference step prunes the remaining domains:

* ``"forward_checking"`` - every constraint on the new variable whose scope has
  exactly one unassigned variable left loses the values of that variable that
  violate it (for binary constraints this is classic forward checking);
* ``"mac"`` - forward checking on the non-binary constraints, then AC-3 on the binary
  constraints starting from the arcs of the new variable's neighbours, propagating
  every change (maintaining arc consistency);
* ``"none"`` - only the consistency check of fully assigned constraints.

A wiped-out domain triggers a backtrack. Every returned solution is re-checked
against all constraints.

AC-3
----
A binary constraint :math:`C(X, Y)` is arc consistent from :math:`X` when every
:math:`x \in D_X` has a support :math:`y \in D_Y` with :math:`C(x, y)`. AC-3 keeps a
queue of arcs; revising :math:`(X, C)` deletes unsupported values of :math:`X`, and if
anything was deleted every arc :math:`(Z, C')` with :math:`Y \ne Z`, :math:`X` in
:math:`C'` is queued again. The result is the unique largest arc-consistent
sub-domain (the arc-consistent closure). Constraints of arity other than 2 are not
used by AC-3 (forward checking handles them).

Min-conflicts
-------------
Local search on complete assignments: start from a greedy assignment, then
repeatedly pick a random conflicted variable and give it the value with the fewest
violated constraints (ties broken uniformly at random).

References
----------
S. Russell & P. Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed. (2020),
ch. 6. A. K. Mackworth, "Consistency in networks of relations", *Artificial
Intelligence* 8 (1977) 99-118. S. Minton, M. D. Johnston, A. B. Philips & P. Laird,
"Minimizing conflicts", *Artificial Intelligence* 58 (1992) 161-205.
R. M. Haralick & G. L. Elliott, "Increasing tree search efficiency for constraint
satisfaction problems", *Artificial Intelligence* 14 (1980) 263-313.
"""

from __future__ import annotations

import itertools
from collections import deque
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from lipipy.ai._common import check_callable, check_float, check_int, jsonable
from lipipy.core.complexity import Complexity
from lipipy.core.random import RandomState, check_random_state
from lipipy.core.registry import register

__all__ = [
    "AC3Result",
    "CSP",
    "CSPResult",
    "Constraint",
    "ac3",
    "all_different",
    "backtracking_search",
    "min_conflicts",
]

_AIMA = "Russell & Norvig, Artificial Intelligence: A Modern Approach, 4th ed. (2020), ch. 6"
_MACKWORTH = "Mackworth, Artificial Intelligence 8 (1977) 99-118"
_MINTON = "Minton, Johnston, Philips & Laird, Artificial Intelligence 58 (1992) 161-205"
_HARALICK = "Haralick & Elliott, Artificial Intelligence 14 (1980) 263-313"
_INFERENCE = ("forward_checking", "mac", "none")

Domains = dict[Hashable, list[Any]]


@dataclass(frozen=True)
class Constraint:
    """A constraint: ``predicate(*values)`` must hold for the values of ``scope``.

    Examples
    --------
    >>> c = Constraint(("x", "y"), lambda x, y: x < y)
    >>> c.is_satisfied({"x": 1, "y": 2}), c.arity
    (True, 2)
    """

    scope: tuple[Hashable, ...]
    predicate: Callable[..., bool]

    def __post_init__(self) -> None:
        if not isinstance(self.scope, tuple) or len(self.scope) == 0:
            raise ValueError("scope must be a non-empty tuple of variables")
        if len(set(self.scope)) != len(self.scope):
            raise ValueError(f"scope has repeated variables: {self.scope}")
        check_callable(self.predicate, "predicate")

    @property
    def arity(self) -> int:
        return len(self.scope)

    def is_satisfied(self, assignment: Mapping[Hashable, Any]) -> bool:
        """Whether the predicate holds (the whole scope must be assigned)."""
        return bool(self.predicate(*(assignment[v] for v in self.scope)))


def all_different(variables: Iterable[Hashable]) -> list[Constraint]:
    """Pairwise ``!=`` constraints over ``variables`` (a binary decomposition of AllDiff).

    Examples
    --------
    >>> len(all_different(["a", "b", "c"]))
    3
    """
    vs = list(variables)
    return [Constraint((a, b), _ne) for a, b in itertools.combinations(vs, 2)]


def _ne(a: Any, b: Any) -> bool:
    return a != b


class CSP:
    """A constraint satisfaction problem (immutable after construction).

    Parameters
    ----------
    variables : sequence of hashable
        Distinct variables; their order breaks heuristic ties.
    domains : mapping
        ``{variable: iterable of values}``; the iteration order is kept.
    constraints : iterable of Constraint
        Constraints over the variables (any arity >= 1).

    Examples
    --------
    >>> csp = CSP(["a", "b"], {"a": [1, 2], "b": [1, 2]}, all_different(["a", "b"]))
    >>> backtracking_search(csp, all_solutions=True).n_solutions
    2
    """

    def __init__(
        self,
        variables: Sequence[Hashable],
        domains: Mapping[Hashable, Iterable[Any]],
        constraints: Iterable[Constraint] = (),
    ) -> None:
        vs = tuple(variables)
        if len(set(vs)) != len(vs):
            raise ValueError("variables must be distinct")
        if not isinstance(domains, Mapping):
            raise TypeError("domains must be a mapping from variable to values")
        missing = [v for v in vs if v not in domains]
        if missing:
            raise ValueError(f"no domain for variables {missing}")
        doms = {}
        for v in vs:
            vals = list(domains[v])
            if len(vals) != len(set(map(_hashable_key, vals))):
                raise ValueError(f"domain of {v!r} has repeated values")
            doms[v] = vals
        cons = tuple(constraints)
        known = set(vs)
        for c in cons:
            if not isinstance(c, Constraint):
                raise TypeError(f"constraints must be Constraint objects, got {type(c).__name__}")
            unknown = [v for v in c.scope if v not in known]
            if unknown:
                raise ValueError(f"constraint scope mentions unknown variables {unknown}")
        self.variables = vs
        self._domains = doms
        self.constraints = cons
        self._by_var: dict[Hashable, list[Constraint]] = {v: [] for v in vs}
        for c in cons:
            for v in c.scope:
                self._by_var[v].append(c)
        self._neighbors = {
            v: tuple(dict.fromkeys(u for c in self._by_var[v] for u in c.scope if u != v))
            for v in vs
        }

    @property
    def domains(self) -> Domains:
        """A fresh copy of the domains (mutating it does not change the CSP)."""
        return {v: list(d) for v, d in self._domains.items()}

    def constraints_on(self, var: Hashable) -> tuple[Constraint, ...]:
        return tuple(self._by_var[var])

    def neighbors(self, var: Hashable) -> tuple[Hashable, ...]:
        return self._neighbors[var]

    def is_solution(self, assignment: Mapping[Hashable, Any]) -> bool:
        """Every variable assigned a value of its domain and every constraint satisfied."""
        if set(assignment) != set(self.variables):
            return False
        for v in self.variables:
            if not any(_same(assignment[v], x) for x in self._domains[v]):
                return False
        return all(c.is_satisfied(assignment) for c in self.constraints)

    def n_conflicts(self, var: Hashable, value: Any, assignment: Mapping[Hashable, Any]) -> int:
        """Constraints on ``var`` violated if ``var = value`` (fully assigned scopes only)."""
        trial = dict(assignment)
        trial[var] = value
        return sum(
            1
            for c in self._by_var[var]
            if all(u in trial for u in c.scope) and not c.is_satisfied(trial)
        )


def _hashable_key(value: Any) -> Any:
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _same(a: Any, b: Any) -> bool:
    return a is b or a == b


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CSPResult:
    """Outcome of :func:`backtracking_search` or :func:`min_conflicts`.

    Attributes
    ----------
    solution : dict or None
        The first solution found (``None`` if none).
    solutions : tuple of dict
        All solutions found (only filled with ``all_solutions=True``; otherwise the
        first one, if any).
    n_solutions : int
        Number of solutions found.
    nodes : int
        Assignments tried (backtracking) or value changes (min-conflicts).
    backtracks : int
        Dead ends (backtracking) or 0 (min-conflicts).
    found : bool
        Whether at least one solution was found.
    """

    solution: dict[Hashable, Any] | None
    solutions: tuple[dict[Hashable, Any], ...]
    n_solutions: int
    nodes: int
    backtracks: int
    found: bool

    def __post_init__(self) -> None:
        if self.found != (self.solution is not None) or self.found != (self.n_solutions > 0):
            raise ValueError("found, solution and n_solutions disagree")
        if self.nodes < 0 or self.backtracks < 0:
            raise ValueError("counters must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "solution": jsonable(self.solution),
            "solutions": [jsonable(s) for s in self.solutions],
            "n_solutions": self.n_solutions,
            "nodes": self.nodes,
            "backtracks": self.backtracks,
            "found": self.found,
        }


@dataclass(frozen=True)
class AC3Result:
    """Outcome of :func:`ac3`: the arc-consistent ``domains`` and whether none is empty."""

    domains: dict[Hashable, list[Any]]
    consistent: bool
    revisions: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "domains": jsonable(self.domains),
            "consistent": self.consistent,
            "revisions": self.revisions,
        }


# ---------------------------------------------------------------------------
# arc consistency
# ---------------------------------------------------------------------------
def _revise(domains: Domains, var: Hashable, c: Constraint) -> bool:
    """Remove values of ``var`` without support through binary constraint ``c``."""
    first = c.scope[0] == var
    other = c.scope[1] if first else c.scope[0]
    pred = c.predicate
    others = domains[other]
    kept = [x for x in domains[var] if any((pred(x, y) if first else pred(y, x)) for y in others)]
    if len(kept) == len(domains[var]):
        return False
    domains[var] = kept
    return True


def _ac3_queue(csp: CSP, domains: Domains, queue: deque[tuple[Hashable, Constraint]]) -> tuple[
    bool, int
]:  # fmt: skip
    queued = set((v, id(c)) for v, c in queue)
    revisions = 0
    while queue:
        var, c = queue.popleft()
        queued.discard((var, id(c)))
        revisions += 1
        if _revise(domains, var, c):
            if not domains[var]:
                return False, revisions
            for c2 in csp.constraints_on(var):
                if c2.arity != 2 or c2 is c:
                    continue
                z = c2.scope[1] if c2.scope[0] == var else c2.scope[0]
                if (z, id(c2)) not in queued:
                    queue.append((z, c2))
                    queued.add((z, id(c2)))
    return True, revisions


@register(
    domain="ai",
    tasks=("constraint-satisfaction", "arc-consistency"),
    complexity=Complexity(
        time="O(E k^3)",
        space="O(E + n k)",
        notes="E binary constraints, k = largest domain size, n variables "
        "(each arc is re-queued at most k times, a revision costs O(k^2))",
    ),
    references=(_MACKWORTH, _AIMA + ", §6.2.2"),
)
def ac3(csp: CSP, domains: Mapping[Hashable, Iterable[Any]] | None = None) -> AC3Result:
    r"""Make the binary constraints of ``csp`` arc consistent (AC-3).

    Parameters
    ----------
    csp : CSP
    domains : mapping, optional
        Starting domains (default: the CSP's). They are copied, never mutated.

    Returns
    -------
    AC3Result
        ``domains``: the arc-consistent closure (values keep their order);
        ``consistent``: False if some domain became empty (then the CSP has no
        solution); ``revisions``: number of arc revisions.

    Complexity
    ----------
    Time :math:`O(Ek^3)` for E binary constraints and domain size k; space
    :math:`O(E + nk)`.

    References
    ----------
    Mackworth (1977); Russell & Norvig (2020), §6.2.2.

    Examples
    --------
    >>> csp = CSP(["x", "y"], {"x": [1, 2, 3], "y": [1, 2, 3]},
    ...           [Constraint(("x", "y"), lambda x, y: x < y)])
    >>> ac3(csp).domains
    {'x': [1, 2], 'y': [2, 3]}
    """
    if not isinstance(csp, CSP):
        raise TypeError(f"csp must be a CSP, got {type(csp).__name__}")
    if domains is None:
        doms = csp.domains
    else:
        if set(domains) != set(csp.variables):
            raise ValueError("domains must have exactly the CSP's variables")
        doms = {v: list(domains[v]) for v in csp.variables}
    if any(not d for d in doms.values()):
        return AC3Result(doms, False, 0)
    queue = deque((v, c) for c in csp.constraints if c.arity == 2 for v in c.scope)
    ok, revisions = _ac3_queue(csp, doms, queue)
    return AC3Result(doms, ok, revisions)


# ---------------------------------------------------------------------------
# backtracking
# ---------------------------------------------------------------------------
class _Search:
    def __init__(self, csp: CSP, inference: str, all_solutions: bool, limit: int | None):
        self.csp = csp
        self.inference = inference
        self.all_solutions = all_solutions
        self.limit = limit
        self.solutions: list[dict[Hashable, Any]] = []
        self.nodes = 0
        self.backtracks = 0

    def select_variable(self, assignment: dict, domains: Domains) -> Hashable:
        unassigned = [v for v in self.csp.variables if v not in assignment]
        smallest = min(len(domains[v]) for v in unassigned)
        tied = [v for v in unassigned if len(domains[v]) == smallest]  # MRV
        if len(tied) == 1:
            return tied[0]

        def degree(v: Hashable) -> int:
            return sum(
                1
                for c in self.csp.constraints_on(v)
                if any(u != v and u not in assignment for u in c.scope)
            )

        return max(tied, key=degree)  # first maximum keeps the order of `variables`

    def order_values(self, var: Hashable, assignment: dict, domains: Domains) -> list[Any]:
        def ruled_out(x: Any) -> int:
            count = 0
            for c in self.csp.constraints_on(var):
                if c.arity != 2:
                    continue
                first = c.scope[0] == var
                other = c.scope[1] if first else c.scope[0]
                if other in assignment:
                    continue
                count += sum(
                    1
                    for y in domains[other]
                    if not (c.predicate(x, y) if first else c.predicate(y, x))
                )
            return count

        values = domains[var]
        return (
            [x for _, _, x in sorted((ruled_out(x), i, x) for i, x in enumerate(values))]
            if (len(values) > 1)
            else list(values)
        )

    def consistent(self, var: Hashable, assignment: dict) -> bool:
        for c in self.csp.constraints_on(var):
            if all(u in assignment for u in c.scope) and not c.is_satisfied(assignment):
                return False
        return True

    def infer(self, var: Hashable, assignment: dict, domains: Domains) -> Domains | None:
        """Return pruned copies of the domains, or None on a wipe-out."""
        new = dict(domains)
        new[var] = [assignment[var]]
        if self.inference == "none":
            return new
        for c in self.csp.constraints_on(var):
            if self.inference == "mac" and c.arity == 2:
                continue  # AC-3 below revises these and propagates the changes
            free = [u for u in c.scope if u not in assignment]
            if len(free) != 1:
                continue
            u = free[0]
            trial = dict(assignment)
            kept = []
            for y in new[u]:
                trial[u] = y
                if c.is_satisfied(trial):
                    kept.append(y)
            if len(kept) != len(new[u]):
                if not kept:
                    return None
                new[u] = kept
        if self.inference == "mac":
            queue = deque(
                (u, c)
                for c in self.csp.constraints_on(var)
                if c.arity == 2
                for u in c.scope
                if u != var and u not in assignment
            )
            # _revise replaces domain lists instead of editing them, so the parent's
            # lists (shared by `new`) are never modified.
            ok, _ = _ac3_queue(self.csp, new, queue)
            if not ok:
                return None
        return new

    def run(self, assignment: dict, domains: Domains) -> bool:
        """Depth-first search; returns True to stop (enough solutions found)."""
        if len(assignment) == len(self.csp.variables):
            solution = dict(assignment)
            if not self.csp.is_solution(solution):  # defensive re-check of every constraint
                raise RuntimeError("internal error: search produced an invalid solution")
            self.solutions.append(solution)
            if not self.all_solutions:
                return True
            return self.limit is not None and len(self.solutions) >= self.limit
        var = self.select_variable(assignment, domains)
        for value in self.order_values(var, assignment, domains):
            self.nodes += 1
            assignment[var] = value
            if self.consistent(var, assignment):
                pruned = self.infer(var, assignment, domains)
                if pruned is not None and self.run(assignment, pruned):
                    return True
            del assignment[var]
        self.backtracks += 1
        return False


@register(
    domain="ai",
    tasks=("constraint-satisfaction", "search"),
    complexity=Complexity(
        time="O(k^n)",
        space="O(n^2 k)",
        notes="n variables, k = largest domain size (worst case; heuristics and "
        "inference prune heavily); space holds one domain copy per search level",
    ),
    references=(_AIMA + ", §6.3", _HARALICK, _MACKWORTH),
)
def backtracking_search(
    csp: CSP,
    inference: Literal["forward_checking", "mac", "none"] = "forward_checking",
    all_solutions: bool = False,
    max_solutions: int | None = None,
) -> CSPResult:
    r"""Backtracking search with MRV + degree, LCV and forward checking or MAC.

    Parameters
    ----------
    csp : CSP
    inference : {"forward_checking", "mac", "none"}, default="forward_checking"
        Pruning after each assignment (see the module docstring).
    all_solutions : bool, default=False
        Enumerate every solution instead of stopping at the first.
    max_solutions : int, optional
        With ``all_solutions``, stop after this many (>= 1).

    Returns
    -------
    CSPResult
        Every returned solution satisfies every constraint (checked).

    Complexity
    ----------
    Worst-case time :math:`O(k^n)` for n variables with domains of size k; space
    :math:`O(n^2 k)` (one domain copy per level of the recursion).

    References
    ----------
    Russell & Norvig (2020), §6.3; Haralick & Elliott (1980); Mackworth (1977).

    Examples
    --------
    >>> regions = ["WA", "NT", "SA", "Q", "NSW", "V", "T"]
    >>> borders = [("WA", "NT"), ("WA", "SA"), ("NT", "SA"), ("NT", "Q"), ("SA", "Q"),
    ...            ("SA", "NSW"), ("SA", "V"), ("Q", "NSW"), ("NSW", "V")]
    >>> csp = CSP(regions, {r: ["red", "green", "blue"] for r in regions},
    ...           [Constraint(b, lambda x, y: x != y) for b in borders])
    >>> res = backtracking_search(csp, inference="mac")
    >>> res.found, csp.is_solution(res.solution)
    (True, True)
    >>> backtracking_search(csp, all_solutions=True).n_solutions
    18
    """
    if not isinstance(csp, CSP):
        raise TypeError(f"csp must be a CSP, got {type(csp).__name__}")
    if inference not in _INFERENCE:
        raise ValueError(f"inference must be one of {_INFERENCE}, got {inference!r}")
    if not isinstance(all_solutions, bool):
        raise TypeError("all_solutions must be a bool")
    limit = None if max_solutions is None else check_int(max_solutions, "max_solutions", 1)
    search = _Search(csp, inference, all_solutions, limit)
    domains = csp.domains
    # Unary constraints are enforced up front (they have no other variable to wait for).
    for c in csp.constraints:
        if c.arity == 1:
            v = c.scope[0]
            domains[v] = [x for x in domains[v] if c.predicate(x)]
    ok = all(domains[v] for v in csp.variables)
    if ok and inference == "mac":
        queue = deque((v, c) for c in csp.constraints if c.arity == 2 for v in c.scope)
        ok, _ = _ac3_queue(csp, domains, queue)
    if ok:
        search.run({}, domains)
    sols = tuple(search.solutions)
    return CSPResult(
        solution=sols[0] if sols else None,
        solutions=sols,
        n_solutions=len(sols),
        nodes=search.nodes,
        backtracks=search.backtracks,
        found=bool(sols),
    )


# ---------------------------------------------------------------------------
# min-conflicts
# ---------------------------------------------------------------------------
@register(
    domain="ai",
    tasks=("constraint-satisfaction", "local-search"),
    complexity=Complexity(
        time="O(T k c)",
        space="O(n)",
        notes="T = max_steps, k = largest domain size, c = cost of counting the "
        "constraints on one variable; n variables",
    ),
    references=(_MINTON, _AIMA + ", §6.4"),
)
def min_conflicts(
    csp: CSP,
    max_steps: int = 10_000,
    random_state: RandomState = None,
    p_random: float = 0.05,
) -> CSPResult:
    r"""Min-conflicts local search for a complete consistent assignment.

    Starts from a greedy assignment (each variable in turn takes a value with the
    fewest conflicts with those already set; random ties), then for up to
    ``max_steps`` steps moves a random conflicted variable to a value with the fewest
    conflicts (random ties). With probability ``p_random`` the chosen variable gets a
    uniformly random value instead (a random-walk step); without it the search can be
    trapped for ever in a local minimum. Incomplete: ``found=False`` if the budget runs
    out (it cannot prove that no solution exists).

    Parameters
    ----------
    csp : CSP
    max_steps : int, default=10000
        Maximum number of repair steps (>= 0).
    random_state : None, int, SeedSequence or Generator
    p_random : float, default=0.05
        Probability of a random-walk step, in [0, 1].

    Returns
    -------
    CSPResult
        ``nodes`` = repair steps taken; ``backtracks`` = 0.

    Complexity
    ----------
    Time :math:`O(T k c)` for T steps, domain size k and c = cost of checking the
    constraints on one variable; space :math:`O(n)`.

    References
    ----------
    Minton et al. (1992); Russell & Norvig (2020), §6.4.

    Examples
    --------
    >>> cols = list(range(8))
    >>> queens = CSP(cols, {c: list(range(8)) for c in cols},
    ...              [Constraint((a, b), lambda r1, r2, d=b - a: r1 != r2 and abs(r1 - r2) != d)
    ...               for a, b in __import__("itertools").combinations(cols, 2)])
    >>> res = min_conflicts(queens, max_steps=2000, random_state=0)
    >>> res.found, queens.is_solution(res.solution)
    (True, True)
    """
    if not isinstance(csp, CSP):
        raise TypeError(f"csp must be a CSP, got {type(csp).__name__}")
    max_steps = check_int(max_steps, "max_steps")
    p_random = check_float(p_random, "p_random", low=0.0, high=1.0)
    rng = check_random_state(random_state)
    domains = csp.domains
    if any(not domains[v] for v in csp.variables):
        return CSPResult(None, (), 0, 0, 0, False)

    def best_values(var: Hashable, assignment: dict) -> list[Any]:
        scores = [csp.n_conflicts(var, x, assignment) for x in domains[var]]
        low = min(scores)
        return [x for x, s in zip(domains[var], scores, strict=True) if s == low]

    assignment: dict[Hashable, Any] = {}
    for v in csp.variables:
        choices = best_values(v, assignment)
        assignment[v] = choices[int(rng.integers(len(choices)))]
    steps = 0
    while True:
        conflicted = [v for v in csp.variables if csp.n_conflicts(v, assignment[v], assignment) > 0]
        if not conflicted:
            solution = dict(assignment)
            if not csp.is_solution(solution):
                raise RuntimeError("internal error: min_conflicts produced an invalid solution")
            return CSPResult(solution, (solution,), 1, steps, 0, True)
        if steps >= max_steps:
            return CSPResult(None, (), 0, steps, 0, False)
        var = conflicted[int(rng.integers(len(conflicted)))]
        choices = domains[var] if rng.random() < p_random else best_values(var, assignment)
        assignment[var] = choices[int(rng.integers(len(choices)))]
        steps += 1
