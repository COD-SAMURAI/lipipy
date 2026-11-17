r"""State-space search: uninformed, informed and bidirectional.

Problems
--------
A search problem (:class:`SearchProblem`) has an ``initial_state``, ``actions(s)``,
``result(s, a)``, ``step_cost(s, a, s')`` (finite and :math:`\ge 0`) and ``is_goal(s)``;
an optional ``heuristic(s)`` estimates the remaining cost. States must be hashable.
:class:`GraphProblem` builds one from an adjacency mapping and :class:`SlidingPuzzle`
is the classic :math:`n \times n` sliding-tile puzzle.

Notation for costs: :math:`b` = branching factor (maximum number of actions), :math:`d`
= depth of the shallowest goal, :math:`m` = maximum depth of the state space,
:math:`C^*` = optimal cost, :math:`\varepsilon` = smallest step cost, :math:`V, E` =
states and transitions reachable from the start.

Algorithms
----------
All are **graph searches** with a reached set (except the depth-limited family and
IDA*, which are tree searches that only reject states already on the current path, so
they use :math:`O(bd)` memory).

* ``breadth_first`` - FIFO frontier, goal test when a state is generated; fewest actions.
* ``depth_first`` - LIFO frontier. Without a limit it is a graph search; with
  ``depth_limit`` it is depth-limited tree search (path-cycle check), which cannot miss
  a goal within the limit.
* ``iterative_deepening`` - depth-limited search for limits 0, 1, 2, ...; fewest actions.
* ``uniform_cost`` - Dijkstra: priority :math:`g(n)`; optimal for step costs :math:`\ge 0`.
* ``greedy_best_first`` - priority :math:`h(n)`; fast, not optimal.
* ``astar`` - priority :math:`f = g + h`; optimal when :math:`h` is admissible
  (:math:`h \le h^*`). States are **reopened** when a cheaper path is found, so
  admissible but inconsistent heuristics stay optimal.
* ``weighted_astar`` - priority :math:`g + w h`, :math:`w \ge 1`; returns a cost
  :math:`\le w\,C^*` for admissible :math:`h`.
* ``ida_star`` - depth-first contours of increasing :math:`f`-threshold; optimal for
  admissible :math:`h` in :math:`O(bd)` memory.
* ``bidirectional_bfs`` - alternates full BFS layers from the start and (via
  ``predecessors``) from the single goal state, always growing the smaller frontier;
  returns a path with the fewest actions.

Goal tests: BFS-type searches test on generation; priority-queue searches test on
expansion (needed for optimality).

Tie-breaking (documented so results are reproducible): actions are generated in the
order ``actions(s)`` returns them; FIFO among equal priorities for ``uniform_cost`` and
``greedy_best_first``; ``astar`` / ``weighted_astar`` break :math:`f`-ties by smaller
:math:`h` (deeper nodes first), then FIFO. Several optimal paths may exist; which one
is returned depends only on these rules.

References
----------
S. Russell & P. Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed. (2020),
ch. 3. P. E. Hart, N. J. Nilsson & B. Raphael, "A formal basis for the heuristic
determination of minimum cost paths", *IEEE Trans. SSC* 4 (1968) 100-107. R. E. Korf,
"Depth-first iterative-deepening", *Artificial Intelligence* 27 (1985) 97-109.
I. Pohl, "Heuristic search viewed as path finding in a graph", *Artificial
Intelligence* 1 (1970) 193-204.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections import deque
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

from lipipy.ai._common import check_float, check_int, jsonable
from lipipy.core.complexity import Complexity
from lipipy.core.registry import register

__all__ = [
    "GraphProblem",
    "SearchProblem",
    "SearchResult",
    "SlidingPuzzle",
    "astar",
    "bidirectional_bfs",
    "breadth_first",
    "depth_first",
    "greedy_best_first",
    "ida_star",
    "iterative_deepening",
    "uniform_cost",
    "weighted_astar",
]

_AIMA = "Russell & Norvig, Artificial Intelligence: A Modern Approach, 4th ed. (2020), ch. 3"
_HART = "Hart, Nilsson & Raphael, IEEE Trans. SSC 4 (1968) 100-107"
_KORF = "Korf, Artificial Intelligence 27 (1985) 97-109"
_POHL = "Pohl, Artificial Intelligence 1 (1970) 193-204"
_NOTATION = "b = branching factor, d = depth of the shallowest goal, m = maximum depth"

Heuristic = Callable[[Any], float]


# ---------------------------------------------------------------------------
# problem definitions
# ---------------------------------------------------------------------------
@runtime_checkable
class SearchProblem(Protocol):
    """Interface of a search problem (structural: any object with these members works).

    ``heuristic(state)`` is optional; searches that need one take a ``heuristic``
    argument and otherwise fall back to ``problem.heuristic`` or :math:`h = 0`.
    """

    initial_state: Hashable

    def actions(self, state: Any) -> Iterable[Any]: ...

    def result(self, state: Any, action: Any) -> Any: ...

    def step_cost(self, state: Any, action: Any, next_state: Any) -> float: ...

    def is_goal(self, state: Any) -> bool: ...


class GraphProblem:
    """Shortest path between two nodes of a directed graph given as an adjacency mapping.

    Parameters
    ----------
    graph : mapping
        ``{node: {neighbour: cost}}`` (weighted) or ``{node: [neighbour, ...]}`` (unit
        costs). Edges are directed; list both directions for an undirected graph.
        Costs must be finite and :math:`\\ge 0`.
    start, goal : hashable
        Start and goal nodes (must appear in the graph).
    heuristic : callable or mapping, optional
        Estimate of the remaining cost from each node (missing nodes get 0).

    Notes
    -----
    Actions are the neighbour nodes themselves: ``result(s, a) == a``. The graph is
    copied at construction, so later changes to ``graph`` do not affect the problem.

    Examples
    --------
    >>> p = GraphProblem({"A": {"B": 1, "C": 4}, "B": {"C": 1}}, "A", "C")
    >>> astar(p).path, astar(p).cost
    (('A', 'B', 'C'), 2.0)
    """

    def __init__(
        self,
        graph: Mapping[Hashable, Mapping[Hashable, float] | Iterable[Hashable]],
        start: Hashable,
        goal: Hashable,
        heuristic: Heuristic | Mapping[Hashable, float] | None = None,
    ) -> None:
        if not isinstance(graph, Mapping):
            raise TypeError(f"graph must be a mapping, got {type(graph).__name__}")
        adj: dict[Hashable, dict[Hashable, float]] = {}
        for u, nbrs in graph.items():
            row = adj.setdefault(u, {})
            items = nbrs.items() if isinstance(nbrs, Mapping) else ((v, 1.0) for v in nbrs)
            for v, c in items:
                cost = check_float(c, f"cost of edge {u!r}->{v!r}", low=0.0)
                row[v] = cost
                adj.setdefault(v, {})
        for name, node in (("start", start), ("goal", goal)):
            if node not in adj:
                raise ValueError(f"{name} node {node!r} is not in the graph")
        rev: dict[Hashable, list[Hashable]] = {u: [] for u in adj}
        for u, row in adj.items():
            for v in row:
                rev[v].append(u)
        if heuristic is not None and not (callable(heuristic) or isinstance(heuristic, Mapping)):
            raise TypeError("heuristic must be callable, a mapping or None")
        self._adj = adj
        self._rev = rev
        self.initial_state = start
        self.goal_state = goal
        self._h = dict(heuristic) if isinstance(heuristic, Mapping) else heuristic

    @property
    def nodes(self) -> tuple[Hashable, ...]:
        """All nodes, in insertion order."""
        return tuple(self._adj)

    def actions(self, state: Hashable) -> list[Hashable]:
        return list(self._adj[state])

    def result(self, state: Hashable, action: Hashable) -> Hashable:
        return action

    def step_cost(self, state: Hashable, action: Hashable, next_state: Hashable) -> float:
        return self._adj[state][next_state]

    def is_goal(self, state: Hashable) -> bool:
        return state == self.goal_state

    def predecessors(self, state: Hashable) -> list[tuple[Hashable, Hashable]]:
        """``(previous_state, action)`` pairs leading into ``state``."""
        return [(u, state) for u in self._rev[state]]

    def heuristic(self, state: Hashable) -> float:
        if self._h is None:
            return 0.0
        if isinstance(self._h, dict):
            return float(self._h.get(state, 0.0))
        return float(self._h(state))


class SlidingPuzzle:
    """The :math:`n \\times n` sliding-tile puzzle (8-puzzle for ``n = 3``).

    States are tuples of length :math:`n^2` in row-major order with ``0`` for the
    blank. Actions ``"U", "D", "L", "R"`` move the blank; every step costs 1.
    ``heuristic`` is the Manhattan distance of the tiles to their goal cells
    (admissible and consistent).

    Parameters
    ----------
    start : sequence of int
        A permutation of ``0 .. n^2 - 1``.
    goal : sequence of int, optional
        Default ``(1, 2, ..., n^2 - 1, 0)``.

    Examples
    --------
    >>> p = SlidingPuzzle((1, 2, 3, 4, 5, 6, 0, 7, 8))
    >>> p.is_solvable(), astar(p).cost
    (True, 2.0)
    """

    _MOVES = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}

    def __init__(self, start: Iterable[int], goal: Iterable[int] | None = None) -> None:
        tiles = tuple(int(t) for t in start)
        n = math.isqrt(len(tiles))
        if n < 2 or n * n != len(tiles) or sorted(tiles) != list(range(n * n)):
            raise ValueError("start must be a permutation of 0 .. n^2 - 1 with n >= 2")
        target = tuple(range(1, n * n)) + (0,) if goal is None else tuple(int(t) for t in goal)
        if sorted(target) != list(range(n * n)):
            raise ValueError("goal must be a permutation of the same tiles")
        self.n = n
        self.initial_state = tiles
        self.goal_state = target
        self._goal_pos = {t: divmod(i, n) for i, t in enumerate(target)}

    def actions(self, state: tuple[int, ...]) -> list[str]:
        r, c = divmod(state.index(0), self.n)
        return [
            a
            for a, (dr, dc) in self._MOVES.items()
            if 0 <= r + dr < self.n and 0 <= c + dc < self.n
        ]

    def result(self, state: tuple[int, ...], action: str) -> tuple[int, ...]:
        i = state.index(0)
        dr, dc = self._MOVES[action]
        j = i + dr * self.n + dc
        s = list(state)
        s[i], s[j] = s[j], s[i]
        return tuple(s)

    def step_cost(self, state: Any, action: Any, next_state: Any) -> float:
        return 1.0

    def is_goal(self, state: tuple[int, ...]) -> bool:
        return state == self.goal_state

    def predecessors(self, state: tuple[int, ...]) -> list[tuple[tuple[int, ...], str]]:
        """Moves are reversible: the predecessor via ``a`` is the result of the opposite move."""
        opposite = {"U": "D", "D": "U", "L": "R", "R": "L"}
        return [(self.result(state, a), opposite[a]) for a in self.actions(state)]

    def heuristic(self, state: tuple[int, ...]) -> float:
        total = 0
        for i, t in enumerate(state):
            if t:
                r, c = divmod(i, self.n)
                gr, gc = self._goal_pos[t]
                total += abs(r - gr) + abs(c - gc)
        return float(total)

    def is_solvable(self) -> bool:
        """Parity test: the goal is reachable iff the permutation parities agree.

        Moving the blank swaps two cells, so the parity of the permutation of all
        :math:`n^2` symbols flips with every move, as does the parity of the blank's
        Manhattan distance from a fixed cell; their sum is invariant, and states with
        equal invariant are reachable (Johnson & Story, 1879).
        """

        def invariant(state: tuple[int, ...]) -> int:
            perm = [self._index_in_goal[t] for t in state]
            inv = sum(1 for i in range(len(perm)) for j in range(i) if perm[j] > perm[i])
            r, c = divmod(state.index(0), self.n)
            return (inv + r + c) % 2

        return invariant(self.initial_state) == invariant(self.goal_state)

    @property
    def _index_in_goal(self) -> dict[int, int]:
        return {t: i for i, t in enumerate(self.goal_state)}


# ---------------------------------------------------------------------------
# results and helpers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SearchResult:
    """Outcome of a search.

    Attributes
    ----------
    path : tuple
        States from the initial state to the goal (empty if not found).
    actions : tuple
        Actions along ``path`` (``len(path) - 1`` of them).
    cost : float
        Sum of step costs along ``path`` (``inf`` if not found).
    nodes_expanded : int
        Number of node expansions (children generated); re-expansions count again.
    max_frontier : int
        Largest size reached by the frontier data structure (for heaps, including
        stale entries; for depth-first methods, the stack depth).
    found : bool
        Whether a goal was reached.
    """

    path: tuple[Any, ...]
    cost: float
    nodes_expanded: int
    max_frontier: int
    found: bool
    actions: tuple[Any, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.found != bool(self.path):
            raise ValueError("found must be True exactly when path is non-empty")
        if self.found and len(self.actions) != len(self.path) - 1:
            raise ValueError("actions must have one entry per step of path")
        if self.nodes_expanded < 0 or self.max_frontier < 0:
            raise ValueError("counters must be non-negative")
        if not self.found and self.cost != math.inf:
            raise ValueError("cost must be inf when no path was found")

    def to_dict(self) -> dict[str, Any]:
        return {k: jsonable(v) for k, v in asdict(self).items()}


def _check_problem(problem: Any) -> None:
    if not isinstance(problem, SearchProblem):
        raise TypeError(
            "problem must provide initial_state, actions, result, step_cost and is_goal"
        )


def _step(problem: Any, s: Any, a: Any) -> tuple[Any, float]:
    s2 = problem.result(s, a)
    c = problem.step_cost(s, a, s2)
    if isinstance(c, bool) or not isinstance(c, (int, float)) or not (0 <= c < math.inf):
        try:
            c = float(c)
        except (TypeError, ValueError):
            raise TypeError(f"step_cost must return a number, got {type(c).__name__}") from None
        if not 0 <= c < math.inf:
            raise ValueError(f"step costs must be finite and >= 0, got {c} for {s!r} -{a!r}->")
    return s2, float(c)


def _heuristic_fn(problem: Any, heuristic: Heuristic | None) -> Heuristic:
    if heuristic is not None:
        if not callable(heuristic):
            raise TypeError("heuristic must be callable")
        h = heuristic
    elif callable(getattr(problem, "heuristic", None)):
        h = problem.heuristic
    else:
        return lambda s: 0.0

    def checked(s: Any) -> float:
        v = float(h(s))
        if not 0 <= v < math.inf:
            raise ValueError(f"heuristic must be finite and >= 0, got {v} for {s!r}")
        return v

    return checked


def _found(
    parents: Mapping[Any, tuple[Any, Any, float] | None], goal: Any, expanded: int, frontier: int
) -> SearchResult:
    states, actions, cost = [goal], [], 0.0
    node = parents[goal]
    costs = []
    while node is not None:
        prev, action, c = node
        states.append(prev)
        actions.append(action)
        costs.append(c)
        node = parents[prev]
    cost = math.fsum(costs)
    return SearchResult(
        tuple(reversed(states)), cost, expanded, frontier, True, tuple(reversed(actions))
    )


def _not_found(expanded: int, frontier: int) -> SearchResult:
    return SearchResult((), math.inf, expanded, frontier, False, ())


# ---------------------------------------------------------------------------
# uninformed search
# ---------------------------------------------------------------------------
@register(
    domain="ai",
    tasks=("search", "uninformed-search"),
    complexity=Complexity(time="O(b^d)", space="O(b^d)", notes=_NOTATION + "; O(V + E) on graphs"),
    references=(_AIMA + ", §3.4.1",),
)
def breadth_first(problem: SearchProblem) -> SearchResult:
    r"""Breadth-first graph search: returns a path with the fewest actions.

    Parameters
    ----------
    problem : SearchProblem

    Returns
    -------
    SearchResult
        ``cost`` is the sum of step costs along the returned path (optimal only for
        uniform step costs).

    Complexity
    ----------
    Time and space :math:`O(b^d)`; :math:`O(V + E)` on an explicit graph.

    References
    ----------
    Russell & Norvig (2020), §3.4.1.

    Examples
    --------
    >>> g = {1: [2, 3], 2: [4], 3: [4], 4: []}
    >>> breadth_first(GraphProblem(g, 1, 4)).path
    (1, 2, 4)
    """
    _check_problem(problem)
    start = problem.initial_state
    parents: dict[Any, tuple[Any, Any, float] | None] = {start: None}
    if problem.is_goal(start):
        return _found(parents, start, 0, 1)
    frontier = deque([start])
    expanded, max_frontier = 0, 1
    while frontier:
        s = frontier.popleft()
        expanded += 1
        for a in list(problem.actions(s)):
            s2, c = _step(problem, s, a)
            if s2 in parents:
                continue
            parents[s2] = (s, a, c)
            if problem.is_goal(s2):
                return _found(parents, s2, expanded, max(max_frontier, len(frontier) + 1))
            frontier.append(s2)
        max_frontier = max(max_frontier, len(frontier))
    return _not_found(expanded, max_frontier)


def _depth_limited(problem: Any, limit: int) -> tuple[SearchResult, bool]:
    """Depth-limited tree search with path-cycle check; returns (result, cutoff)."""
    start = problem.initial_state
    if problem.is_goal(start):
        return _found({start: None}, start, 0, 1), False
    if limit == 0:
        return _not_found(0, 1), bool(list(problem.actions(start)))
    stack = [(start, iter(list(problem.actions(start))))]
    on_path = {start}
    steps: list[tuple[Any, Any, float]] = []  # (prev, action, cost) along the stack
    expanded, max_frontier, cutoff = 1, 1, False
    while stack:
        s, it = stack[-1]
        a = next(it, _SENTINEL)
        if a is _SENTINEL:
            stack.pop()
            on_path.discard(s)
            if steps:
                steps.pop()
            continue
        s2, c = _step(problem, s, a)
        if s2 in on_path:
            continue
        if problem.is_goal(s2):
            return _found_from_steps(start, [*steps, (s, a, c)], s2, expanded, max_frontier), False
        if len(stack) < limit:
            stack.append((s2, iter(list(problem.actions(s2)))))
            on_path.add(s2)
            steps.append((s, a, c))
            expanded += 1
            max_frontier = max(max_frontier, len(stack))
        else:
            cutoff = True
    return _not_found(expanded, max_frontier), cutoff


_SENTINEL = object()


def _found_from_steps(
    start: Any, steps: list[tuple[Any, Any, float]], goal: Any, expanded: int, frontier: int
) -> SearchResult:
    states = [prev for prev, _, _ in steps] + [goal]  # steps[0][0] is start
    actions = [a for _, a, _ in steps]
    cost = math.fsum(c for _, _, c in steps)
    return SearchResult(tuple(states), cost, expanded, frontier, True, tuple(actions))


@register(
    domain="ai",
    tasks=("search", "uninformed-search"),
    complexity=Complexity(
        time="O(b^m)",
        space="O(b m)",
        notes=_NOTATION + "; with depth_limit l: time O(b^l), space O(b l); graph search "
        "without a limit uses O(V) memory for the reached set",
    ),
    references=(_AIMA + ", §3.4.3-3.4.4",),
)
def depth_first(problem: SearchProblem, depth_limit: int | None = None) -> SearchResult:
    r"""Depth-first search, optionally depth-limited.

    Without ``depth_limit``: graph search with a LIFO frontier and a reached set
    (terminates on finite state spaces, not optimal). With ``depth_limit = l``:
    depth-limited tree search that only rejects states already on the current path;
    it finds a goal whenever one is reachable within ``l`` actions.

    Parameters
    ----------
    problem : SearchProblem
    depth_limit : int, optional
        Maximum number of actions (>= 0).

    Returns
    -------
    SearchResult

    Complexity
    ----------
    Without a limit: time O(V + E), space O(V). With limit l: time :math:`O(b^l)`,
    space :math:`O(bl)`.

    References
    ----------
    Russell & Norvig (2020), §3.4.3-3.4.4.

    Examples
    --------
    >>> g = {1: [2, 3], 2: [4], 3: [5], 4: [], 5: []}
    >>> depth_first(GraphProblem(g, 1, 5)).path
    (1, 3, 5)
    >>> depth_first(GraphProblem(g, 1, 5), depth_limit=1).found
    False
    """
    _check_problem(problem)
    if depth_limit is not None:
        return _depth_limited(problem, check_int(depth_limit, "depth_limit"))[0]
    start = problem.initial_state
    parents: dict[Any, tuple[Any, Any, float] | None] = {start: None}
    stack = [start]
    expanded, max_frontier = 0, 1
    while stack:
        s = stack.pop()
        if problem.is_goal(s):
            return _found(parents, s, expanded, max_frontier)
        expanded += 1
        children = []
        for a in list(problem.actions(s)):
            s2, c = _step(problem, s, a)
            if s2 not in parents:
                parents[s2] = (s, a, c)
                children.append(s2)
        stack.extend(reversed(children))  # first action is explored first
        max_frontier = max(max_frontier, len(stack))
    return _not_found(expanded, max_frontier)


@register(
    domain="ai",
    tasks=("search", "uninformed-search"),
    complexity=Complexity(time="O(b^d)", space="O(b d)", notes=_NOTATION),
    references=(_KORF, _AIMA + ", §3.4.4"),
)
def iterative_deepening(problem: SearchProblem, max_depth: int | None = None) -> SearchResult:
    r"""Iterative deepening: depth-limited search with limits 0, 1, 2, ...

    Finds a path with the fewest actions using :math:`O(bd)` memory. Stops with
    ``found=False`` when a whole iteration hit no depth cutoff (the reachable space is
    exhausted) or after ``max_depth``.

    Parameters
    ----------
    problem : SearchProblem
    max_depth : int, optional
        Largest limit to try (>= 0). Without it the search runs until a goal is
        found or the space is exhausted (it never ends on an infinite space with no
        goal).

    Returns
    -------
    SearchResult
        ``nodes_expanded`` sums all iterations.

    Complexity
    ----------
    Time :math:`O(b^d)`, space :math:`O(bd)`.

    References
    ----------
    Korf (1985).

    Examples
    --------
    >>> g = {1: [2, 3], 2: [4], 3: [4], 4: []}
    >>> iterative_deepening(GraphProblem(g, 1, 4)).path
    (1, 2, 4)
    """
    _check_problem(problem)
    if max_depth is not None:
        max_depth = check_int(max_depth, "max_depth")
    total, max_frontier = 0, 0
    for limit in itertools.count():
        if max_depth is not None and limit > max_depth:
            break
        res, cutoff = _depth_limited(problem, limit)
        total += res.nodes_expanded
        max_frontier = max(max_frontier, res.max_frontier)
        if res.found:
            return SearchResult(res.path, res.cost, total, max_frontier, True, res.actions)
        if not cutoff:
            break
    return _not_found(total, max_frontier)


# ---------------------------------------------------------------------------
# best-first family
# ---------------------------------------------------------------------------
def _best_first(
    problem: Any, priority: Callable[[float, float], tuple[float, float]], h: Heuristic,
    reopen: bool,
) -> SearchResult:  # fmt: skip
    start = problem.initial_state
    g = {start: 0.0}
    parents: dict[Any, tuple[Any, Any, float] | None] = {start: None}
    counter = itertools.count()
    h0 = h(start)
    frontier = [(*priority(0.0, h0), next(counter), 0.0, start)]
    expanded, max_frontier = 0, 1
    while frontier:
        *_, gs, s = heapq.heappop(frontier)
        if gs > g[s]:
            continue  # stale entry: a cheaper path to s was found later
        if problem.is_goal(s):
            return _found(parents, s, expanded, max_frontier)
        expanded += 1
        for a in list(problem.actions(s)):
            s2, c = _step(problem, s, a)
            g2 = gs + c
            if s2 in g and (not reopen or g2 >= g[s2]):
                continue
            g[s2] = g2
            parents[s2] = (s, a, c)
            heapq.heappush(frontier, (*priority(g2, h(s2)), next(counter), g2, s2))
        max_frontier = max(max_frontier, len(frontier))
    return _not_found(expanded, max_frontier)


_BF_SPACE = "O(b^(1 + C*/ε))"


@register(
    domain="ai",
    tasks=("search", "uninformed-search", "shortest-path"),
    complexity=Complexity(
        time="O(b^(1 + C*/ε) log V)",
        space=_BF_SPACE,
        notes="b = branching factor, C* = optimal cost, ε = smallest step cost; "
        "O((V + E) log V) on an explicit graph (binary heap, lazy deletion)",
    ),
    references=(_AIMA + ", §3.4.2", "Dijkstra, Numerische Mathematik 1 (1959) 269-271"),
)
def uniform_cost(problem: SearchProblem) -> SearchResult:
    r"""Uniform-cost search (Dijkstra): expands the frontier node with the least :math:`g`.

    Optimal for step costs :math:`\ge 0` (zero-cost steps allowed). Equal priorities
    are expanded first-in first-out.

    Parameters
    ----------
    problem : SearchProblem

    Returns
    -------
    SearchResult

    Complexity
    ----------
    :math:`O((V + E)\log V)` time and :math:`O(V)` space on an explicit graph.

    References
    ----------
    Dijkstra (1959); Russell & Norvig (2020), §3.4.2.

    Examples
    --------
    >>> g = {"S": {"A": 1, "B": 5}, "A": {"B": 1}, "B": {}}
    >>> res = uniform_cost(GraphProblem(g, "S", "B"))
    >>> res.path, res.cost
    (('S', 'A', 'B'), 2.0)
    """
    _check_problem(problem)
    return _best_first(problem, lambda g, h: (g, 0.0), lambda s: 0.0, reopen=True)


@register(
    domain="ai",
    tasks=("search", "informed-search"),
    complexity=Complexity(
        time="O(b^m)", space="O(b^m)", notes=_NOTATION + "; O((V + E) log V) on graphs"
    ),
    references=(_AIMA + ", §3.5.1",),
)
def greedy_best_first(problem: SearchProblem, heuristic: Heuristic | None = None) -> SearchResult:
    r"""Greedy best-first graph search: expands the node with the least :math:`h`.

    Each state enters the frontier at most once (no reopening). Fast with a good
    heuristic but not optimal.

    Parameters
    ----------
    problem : SearchProblem
    heuristic : callable, optional
        ``h(state) >= 0``; default ``problem.heuristic`` or 0.

    Returns
    -------
    SearchResult

    Complexity
    ----------
    :math:`O(b^m)` time and space in the worst case; :math:`O((V+E)\log V)` on graphs.

    References
    ----------
    Russell & Norvig (2020), §3.5.1.

    Examples
    --------
    >>> g = {"S": {"A": 1, "B": 1}, "A": {"G": 10}, "B": {"G": 1}, "G": {}}
    >>> greedy_best_first(GraphProblem(g, "S", "G", heuristic={"A": 0.5, "B": 1})).path
    ('S', 'A', 'G')
    """
    _check_problem(problem)
    h = _heuristic_fn(problem, heuristic)
    return _best_first(problem, lambda g, hv: (hv, 0.0), h, reopen=False)


@register(
    domain="ai",
    tasks=("search", "informed-search", "shortest-path"),
    complexity=Complexity(
        time="O(b^d)",
        space="O(b^d)",
        notes=_NOTATION + "; depends on heuristic quality (effective branching factor "
        "b* < b); O((V + E) log V) on graphs with a consistent heuristic",
    ),
    references=(_HART, _AIMA + ", §3.5.2"),
)
def astar(problem: SearchProblem, heuristic: Heuristic | None = None) -> SearchResult:
    r"""A* search: expands the node with the least :math:`f = g + h`.

    Optimal when :math:`h` is admissible; with a consistent heuristic no state is
    expanded twice. States are reopened when a cheaper path appears, so an admissible
    but inconsistent heuristic still gives an optimal path. :math:`f`-ties go to the
    smaller :math:`h`, then first-in first-out.

    Parameters
    ----------
    problem : SearchProblem
    heuristic : callable, optional
        ``h(state) >= 0``; default ``problem.heuristic`` or 0 (then A* is UCS).

    Returns
    -------
    SearchResult

    Complexity
    ----------
    Exponential in the worst case, :math:`O(b^d)`; :math:`O((V+E)\log V)` on an
    explicit graph with a consistent heuristic.

    References
    ----------
    Hart, Nilsson & Raphael (1968); Russell & Norvig (2020), §3.5.2.

    Examples
    --------
    >>> p = SlidingPuzzle((1, 2, 3, 4, 0, 6, 7, 5, 8))
    >>> res = astar(p)
    >>> res.actions, res.cost
    (('D', 'R'), 2.0)
    """
    _check_problem(problem)
    h = _heuristic_fn(problem, heuristic)
    return _best_first(problem, lambda g, hv: (g + hv, hv), h, reopen=True)


@register(
    domain="ai",
    tasks=("search", "informed-search"),
    complexity=Complexity(time="O(b^d)", space="O(b^d)", notes=_NOTATION),
    references=(_POHL, _AIMA + ", §3.5.4"),
)
def weighted_astar(
    problem: SearchProblem, w: float = 2.0, heuristic: Heuristic | None = None
) -> SearchResult:
    r"""Weighted A*: priority :math:`g + w\,h` with :math:`w \ge 1`.

    For an admissible :math:`h` the returned cost is at most :math:`w\,C^*`
    (bounded suboptimality); usually far fewer nodes are expanded than by A*.
    ``w = 1`` is A*.

    Parameters
    ----------
    problem : SearchProblem
    w : float, default=2.0
        Heuristic weight (>= 1).
    heuristic : callable, optional

    Returns
    -------
    SearchResult

    Complexity
    ----------
    :math:`O(b^d)` worst case.

    References
    ----------
    Pohl (1970); Russell & Norvig (2020), §3.5.4.

    Examples
    --------
    >>> p = SlidingPuzzle((4, 1, 2, 0, 8, 7, 6, 3, 5))
    >>> res = weighted_astar(p, w=1.5)
    >>> res.found, res.cost <= 1.5 * astar(p).cost
    (True, True)
    """
    _check_problem(problem)
    w = check_float(w, "w", low=1.0)
    h = _heuristic_fn(problem, heuristic)
    return _best_first(problem, lambda g, hv: (g + w * hv, hv), h, reopen=True)


@register(
    domain="ai",
    tasks=("search", "informed-search", "shortest-path"),
    complexity=Complexity(
        time="O(b^d)",
        space="O(b d)",
        notes=_NOTATION + "; time grows with the number of distinct f-thresholds",
    ),
    references=(_KORF, _AIMA + ", §3.5.5"),
)
def ida_star(
    problem: SearchProblem, heuristic: Heuristic | None = None, max_iterations: int | None = None
) -> SearchResult:
    r"""Iterative-deepening A*: depth-first contours bounded by :math:`f = g + h \le` threshold.

    The first threshold is :math:`h(s_0)`; each iteration raises it to the smallest
    :math:`f` that exceeded it. Optimal for admissible :math:`h`, with memory
    :math:`O(bd)`. States already on the current path are not revisited, so cycles
    (including zero-cost ones) cannot trap it.

    Parameters
    ----------
    problem : SearchProblem
    heuristic : callable, optional
    max_iterations : int, optional
        Maximum number of thresholds (>= 1); ``found=False`` if exceeded.

    Returns
    -------
    SearchResult
        ``nodes_expanded`` sums all iterations; ``max_frontier`` is the deepest stack.

    Complexity
    ----------
    Time :math:`O(b^d)` (times the number of thresholds in the worst case), space
    :math:`O(bd)`.

    References
    ----------
    Korf (1985); Russell & Norvig (2020), §3.5.5.

    Examples
    --------
    >>> ida_star(SlidingPuzzle((1, 2, 3, 4, 5, 6, 0, 7, 8))).cost
    2.0
    """
    _check_problem(problem)
    h = _heuristic_fn(problem, heuristic)
    if max_iterations is not None:
        max_iterations = check_int(max_iterations, "max_iterations", minimum=1)
    start = problem.initial_state
    threshold = h(start)
    total, max_frontier = 0, 1
    for _ in itertools.count() if max_iterations is None else range(max_iterations):
        next_threshold = math.inf
        if problem.is_goal(start):
            return _found({start: None}, start, total, max_frontier)
        stack = [(start, iter(list(problem.actions(start))))]
        on_path = {start}
        steps: list[tuple[Any, Any, float]] = []
        g_stack = [0.0]
        total += 1
        while stack:
            s, it = stack[-1]
            a = next(it, _SENTINEL)
            if a is _SENTINEL:
                stack.pop()
                on_path.discard(s)
                g_stack.pop()
                if steps:
                    steps.pop()
                continue
            s2, c = _step(problem, s, a)
            if s2 in on_path:
                continue
            g2 = g_stack[-1] + c
            f2 = g2 + h(s2)
            if f2 > threshold:
                next_threshold = min(next_threshold, f2)
                continue
            if problem.is_goal(s2):
                return _found_from_steps(start, [*steps, (s, a, c)], s2, total, max_frontier)
            stack.append((s2, iter(list(problem.actions(s2)))))
            on_path.add(s2)
            steps.append((s, a, c))
            g_stack.append(g2)
            total += 1
            max_frontier = max(max_frontier, len(stack))
        if next_threshold == math.inf:
            break
        threshold = next_threshold
    return _not_found(total, max_frontier)


@register(
    domain="ai",
    tasks=("search", "uninformed-search"),
    complexity=Complexity(time="O(b^(d/2))", space="O(b^(d/2))", notes=_NOTATION),
    references=(_AIMA + ", §3.4.5", _POHL),
)
def bidirectional_bfs(problem: SearchProblem) -> SearchResult:
    r"""Bidirectional breadth-first search: fewest actions, about :math:`2b^{d/2}` nodes.

    Needs a single goal: ``problem.goal_state`` and ``problem.predecessors(state)``
    returning ``(previous_state, action)`` pairs (both provided by
    :class:`GraphProblem` and :class:`SlidingPuzzle`). Whole layers are expanded,
    always on the side with the smaller frontier; when a layer touches the other
    side the shortest meeting path of that layer is returned, which is a shortest
    path overall.

    Parameters
    ----------
    problem : SearchProblem with ``goal_state`` and ``predecessors``

    Returns
    -------
    SearchResult

    Complexity
    ----------
    Time and space :math:`O(b^{d/2})`.

    References
    ----------
    Pohl (1970); Russell & Norvig (2020), §3.4.5.

    Examples
    --------
    >>> g = {i: [i + 1] for i in range(10)}
    >>> bidirectional_bfs(GraphProblem(g, 0, 10)).cost
    10.0
    """
    _check_problem(problem)
    if not hasattr(problem, "goal_state") or not callable(getattr(problem, "predecessors", None)):
        raise TypeError("bidirectional_bfs needs problem.goal_state and problem.predecessors()")
    start, goal = problem.initial_state, problem.goal_state
    if not problem.is_goal(goal):
        raise ValueError("problem.is_goal(problem.goal_state) must be True")
    fwd: dict[Any, tuple[Any, Any, float] | None] = {start: None}  # s -> (prev, a, cost)
    bwd: dict[Any, tuple[Any, Any, float] | None] = {goal: None}  # s -> (next, a, cost)
    if start == goal:
        return _found(fwd, start, 0, 1)
    dist_f, dist_b = {start: 0}, {goal: 0}
    front_f, front_b = [start], [goal]
    expanded, max_frontier = 0, 2
    while front_f and front_b:
        forward = len(front_f) <= len(front_b)
        frontier, reached, other, dist, odist = (
            (front_f, fwd, bwd, dist_f, dist_b) if forward else (front_b, bwd, fwd, dist_b, dist_f)
        )
        new, best = [], None
        for s in frontier:
            expanded += 1
            if forward:
                moves = [(a, *_step(problem, s, a)) for a in list(problem.actions(s))]
            else:
                moves = [(a, p, _step(problem, p, a)[1]) for p, a in list(problem.predecessors(s))]
            for a, s2, c in moves:
                if s2 in reached:
                    continue
                reached[s2] = (s, a, c)
                dist[s2] = dist[s] + 1
                new.append(s2)
                if s2 in other:
                    length = dist[s2] + odist[s2]
                    if best is None or length < best[0]:
                        best = (length, s2)
        if best is not None:
            return _join(fwd, bwd, best[1], expanded, max_frontier)
        if forward:
            front_f = new
        else:
            front_b = new
        max_frontier = max(max_frontier, len(front_f) + len(front_b))
    return _not_found(expanded, max_frontier)


def _join(fwd: dict, bwd: dict, meet: Any, expanded: int, frontier: int) -> SearchResult:
    head = []  # (prev, a, cost) from start to meet
    node = fwd[meet]
    states = [meet]
    while node is not None:
        prev, a, c = node
        head.append((a, c))
        states.append(prev)
        node = fwd[prev]
    states.reverse()
    head.reverse()
    tail = []
    node = bwd[meet]
    while node is not None:
        nxt, a, c = node
        tail.append((a, c))
        states.append(nxt)
        node = bwd[nxt]
    steps = head + tail
    cost = math.fsum(c for _, c in steps)
    return SearchResult(tuple(states), cost, expanded, frontier, True, tuple(a for a, _ in steps))
