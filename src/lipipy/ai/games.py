r"""Adversarial search for two-player zero-sum games.

Games
-----
A :class:`Game` has an ``initial_state``, ``to_move(s)``, ``actions(s)``,
``result(s, a)``, ``is_terminal(s)`` and ``utility(s, player)``; games with chance
nodes also define ``is_chance(s)`` and ``chance_outcomes(s)`` (a list of
``(probability, next_state)``). Utilities are zero-sum:
``utility(s, p) = -utility(s, q)`` for the two players.

All values are from the point of view of the player to move at the root
(:math:`\text{MAX}`); the opponent is :math:`\text{MIN}`.

Minimax and alpha-beta
----------------------
.. math::
    V(s) = \begin{cases}
      U(s) & s \text{ terminal}\\
      \text{eval}(s) & \text{depth limit reached}\\
      \max_a V(\text{result}(s, a)) & \text{MAX to move}\\
      \min_a V(\text{result}(s, a)) & \text{MIN to move}
    \end{cases}

Alpha-beta keeps the best values :math:`\alpha` (MAX) and :math:`\beta` (MIN)
guaranteed so far on the current path and stops expanding a node as soon as
:math:`\alpha \ge \beta`; it returns exactly the minimax value of the root. A move
ordering hook (best moves first) increases pruning.

Expectiminimax adds chance nodes whose value is the probability-weighted average
of their outcomes.

Ties: the best move is the first action (in ``actions(s)`` order, or the order
given by ``order_moves``) reaching the best value.

Monte Carlo tree search (UCT)
-----------------------------
Each iteration selects a path from the root by maximising
:math:`\bar X_j + c\sqrt{\ln N / N_j}` over children (UCB1), expands one untried
action, finishes the game with uniformly random moves and backs up the result (1 for
a win, 1/2 for a draw, 0 for a loss, from the point of view of the player who made
the move into each node). Utilities must lie in :math:`[-1, 1]`. The recommended
move is the most visited child. Chance nodes are not supported. Measured in the
tests, for every seed 0..49: the immediate win and the forced block in tic-tac-toe
(300 / 1000 iterations) and in 4 x 5 Connect-Four (300 iterations each), and the
unique winning move of Nim (1, 2, 4) (1000 iterations).

References
----------
S. Russell & P. Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed. (2020),
ch. 5. D. E. Knuth & R. W. Moore, "An analysis of alpha-beta pruning", *Artificial
Intelligence* 6 (1975) 293-326. L. Kocsis & C. Szepesvári, "Bandit based Monte-Carlo
planning", ECML 2006. C. B. Browne et al., "A survey of Monte Carlo tree search
methods", *IEEE Trans. Comput. Intell. AI Games* 4 (2012) 1-43. C. L. Bouton, "Nim, a
game with a complete mathematical theory", *Annals of Math.* 3 (1901) 35-39.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from lipipy.ai._common import check_callable, check_float, check_int, jsonable
from lipipy.core.complexity import Complexity
from lipipy.core.random import RandomState, check_random_state
from lipipy.core.registry import register

__all__ = [
    "ConnectFour",
    "Game",
    "GameResult",
    "MCTSResult",
    "Nim",
    "TicTacToe",
    "alphabeta",
    "expectiminimax",
    "mcts",
    "minimax",
]

_AIMA = "Russell & Norvig, Artificial Intelligence: A Modern Approach, 4th ed. (2020), ch. 5"
_KNUTH = "Knuth & Moore, Artificial Intelligence 6 (1975) 293-326"
_UCT = "Kocsis & Szepesvári, Bandit based Monte-Carlo planning, ECML 2006"
_BROWNE = "Browne et al., IEEE Trans. Comput. Intell. AI Games 4 (2012) 1-43"
_NOTATION = "b = branching factor, d = search depth (plies)"

Evaluate = Callable[[Any, Any], float]
OrderMoves = Callable[[Any, list[Any]], Iterable[Any]]


@runtime_checkable
class Game(Protocol):
    """Interface of a two-player zero-sum game (structural typing)."""

    initial_state: Hashable

    def to_move(self, state: Any) -> Any: ...

    def actions(self, state: Any) -> Iterable[Any]: ...

    def result(self, state: Any, action: Any) -> Any: ...

    def is_terminal(self, state: Any) -> bool: ...

    def utility(self, state: Any, player: Any) -> float: ...


@dataclass(frozen=True)
class GameResult:
    """Outcome of :func:`minimax`, :func:`alphabeta` and :func:`expectiminimax`.

    ``value`` is from the point of view of the player to move at the root;
    ``best_move`` is None at a terminal root or with ``depth=0``; ``nodes`` counts the
    states visited (including the root).
    """

    value: float
    best_move: Any
    nodes: int

    def __post_init__(self) -> None:
        if self.nodes < 1:
            raise ValueError("nodes must be >= 1 (the root is always visited)")

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "best_move": jsonable(self.best_move), "nodes": self.nodes}


@dataclass(frozen=True)
class MCTSResult:
    """Outcome of :func:`mcts`.

    Attributes
    ----------
    best_move : any
        Most visited root action (None at a terminal root).
    value : float
        Estimated utility in [-1, 1] of ``best_move`` for the player to move.
    n_iterations : int
    nodes : int
        Nodes in the search tree.
    visits : dict
        Root action -> visit count.
    """

    best_move: Any
    value: float
    n_iterations: int
    nodes: int
    visits: dict[Any, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_move": jsonable(self.best_move),
            "value": self.value,
            "n_iterations": self.n_iterations,
            "nodes": self.nodes,
            "visits": [[jsonable(k), v] for k, v in self.visits.items()],
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _check_game(game: Any) -> None:
    if not isinstance(game, Game):
        raise TypeError(
            "game must provide initial_state, to_move, actions, result, is_terminal, utility"
        )


def _is_chance(game: Any, state: Any) -> bool:
    fn = getattr(game, "is_chance", None)
    return bool(fn(state)) if callable(fn) else False


def _setup(
    game: Any, state: Any, depth: int | None, evaluate: Evaluate | None
) -> tuple[Any, int | None]:
    _check_game(game)
    root = game.initial_state if state is None else state
    if depth is not None:
        depth = check_int(depth, "depth")
        if evaluate is None:
            raise ValueError("a depth limit needs an evaluate(state, player) function")
    if evaluate is not None:
        check_callable(evaluate, "evaluate")
    return root, depth


def _utility(game: Any, state: Any, player: Any) -> float:
    u = float(game.utility(state, player))
    if not math.isfinite(u):
        raise ValueError(f"utility must be finite, got {u}")
    return u


def _evaluate(evaluate: Evaluate, state: Any, player: Any) -> float:
    v = float(evaluate(state, player))
    if not math.isfinite(v):
        raise ValueError(f"evaluate must return a finite number, got {v}")
    return v


def _ordered(game: Any, state: Any, order_moves: OrderMoves | None) -> list[Any]:
    acts = list(game.actions(state))
    if order_moves is None:
        return acts
    ordered = list(order_moves(state, acts))
    if sorted(map(repr, ordered)) != sorted(map(repr, acts)):
        raise ValueError("order_moves must return a permutation of the actions")
    return ordered


# ---------------------------------------------------------------------------
# minimax family
# ---------------------------------------------------------------------------
@register(
    domain="ai",
    tasks=("game-playing", "adversarial-search"),
    complexity=Complexity(time="O(b^d)", space="O(b d)", notes=_NOTATION),
    references=(_AIMA + ", §5.2",),
)
def minimax(
    game: Game,
    state: Any = None,
    depth: int | None = None,
    evaluate: Evaluate | None = None,
) -> GameResult:
    r"""Minimax value and best move for the player to move.

    Parameters
    ----------
    game : Game
    state : optional
        Root position (default ``game.initial_state``).
    depth : int, optional
        Search depth in plies (>= 0); needs ``evaluate``. Default: to the end.
    evaluate : callable ``evaluate(state, player) -> float``, optional
        Heuristic value of a non-terminal state for ``player`` at the depth limit.

    Returns
    -------
    GameResult

    Complexity
    ----------
    Time :math:`O(b^d)`, space :math:`O(bd)`.

    References
    ----------
    Russell & Norvig (2020), §5.2.

    Examples
    --------
    >>> minimax(Nim((1, 2))).value  # 1 XOR 2 != 0: the player to move wins
    1.0
    """
    root, depth = _setup(game, state, depth, evaluate)
    player = game.to_move(root)
    nodes = 0

    def value(s: Any, d: int | None) -> tuple[float, Any]:
        nonlocal nodes
        nodes += 1
        if game.is_terminal(s):
            return _utility(game, s, player), None
        if d == 0:
            return _evaluate(evaluate, s, player), None  # type: ignore[arg-type]
        maximize = game.to_move(s) == player
        best, best_move = (-math.inf if maximize else math.inf), None
        for a in list(game.actions(s)):
            v, _ = value(game.result(s, a), None if d is None else d - 1)
            if (v > best) if maximize else (v < best):
                best, best_move = v, a
        return best, best_move

    v, move = value(root, depth)
    return GameResult(v, move, nodes)


@register(
    domain="ai",
    tasks=("game-playing", "adversarial-search"),
    complexity=Complexity(
        time="O(b^d)",
        space="O(b d)",
        notes=_NOTATION + "; O(b^(d/2)) with perfect move ordering",
    ),
    references=(_KNUTH, _AIMA + ", §5.3"),
)
def alphabeta(
    game: Game,
    state: Any = None,
    depth: int | None = None,
    evaluate: Evaluate | None = None,
    order_moves: OrderMoves | None = None,
) -> GameResult:
    r"""Alpha-beta search: the minimax value, usually visiting far fewer nodes.

    Parameters
    ----------
    game : Game
    state : optional
        Root position (default ``game.initial_state``).
    depth : int, optional
        Depth limit in plies (needs ``evaluate``).
    evaluate : callable ``evaluate(state, player) -> float``, optional
    order_moves : callable ``order_moves(state, actions) -> iterable``, optional
        Returns the actions in the order to search them (best first prunes most).

    Returns
    -------
    GameResult
        Same value as :func:`minimax` (and the same best move when no ordering hook
        is used).

    Complexity
    ----------
    Worst case :math:`O(b^d)`; :math:`O(b^{d/2})` with perfect ordering
    (Knuth & Moore 1975). Space :math:`O(bd)`.

    References
    ----------
    Knuth & Moore (1975); Russell & Norvig (2020), §5.3.

    Examples
    --------
    >>> ttt = TicTacToe()
    >>> res = alphabeta(ttt)
    >>> res.value, res.nodes < minimax(ttt).nodes
    (0.0, True)
    """
    root, depth = _setup(game, state, depth, evaluate)
    if order_moves is not None:
        check_callable(order_moves, "order_moves")
    player = game.to_move(root)
    nodes = 0

    def value(s: Any, d: int | None, alpha: float, beta: float) -> tuple[float, Any]:
        nonlocal nodes
        nodes += 1
        if game.is_terminal(s):
            return _utility(game, s, player), None
        if d == 0:
            return _evaluate(evaluate, s, player), None  # type: ignore[arg-type]
        nd = None if d is None else d - 1
        best_move = None
        if game.to_move(s) == player:
            best = -math.inf
            for a in _ordered(game, s, order_moves):
                v, _ = value(game.result(s, a), nd, alpha, beta)
                if v > best:
                    best, best_move = v, a
                alpha = max(alpha, best)
                if alpha >= beta:
                    break
        else:
            best = math.inf
            for a in _ordered(game, s, order_moves):
                v, _ = value(game.result(s, a), nd, alpha, beta)
                if v < best:
                    best, best_move = v, a
                beta = min(beta, best)
                if alpha >= beta:
                    break
        return best, best_move

    v, move = value(root, depth, -math.inf, math.inf)
    return GameResult(v, move, nodes)


@register(
    domain="ai",
    tasks=("game-playing", "adversarial-search", "decision-making"),
    complexity=Complexity(
        time="O(b^d n^d)", space="O(b d)", notes=_NOTATION + "; n = chance outcomes per node"
    ),
    references=(_AIMA + ", §5.5",),
)
def expectiminimax(
    game: Game,
    state: Any = None,
    depth: int | None = None,
    evaluate: Evaluate | None = None,
) -> GameResult:
    r"""Expectiminimax: minimax with chance nodes valued by their expectation.

    Chance nodes (``game.is_chance(s)``) have value
    :math:`\sum_i p_i V(s_i)` over ``game.chance_outcomes(s)``; probabilities must be
    non-negative and sum to 1. A chance node does not use up depth. Without chance
    nodes it equals :func:`minimax`.

    Parameters
    ----------
    game : Game (optionally with ``is_chance`` and ``chance_outcomes``)
    state, depth, evaluate
        As in :func:`minimax`.

    Returns
    -------
    GameResult
        ``best_move`` is None when the root is a chance node.

    Complexity
    ----------
    Time :math:`O(b^d n^d)` for n outcomes per chance node; space :math:`O(bd)`.

    References
    ----------
    Russell & Norvig (2020), §5.5.

    Examples
    --------
    >>> expectiminimax(Nim((3,))).value  # no chance nodes: same as minimax
    1.0
    """
    root, depth = _setup(game, state, depth, evaluate)
    player = game.to_move(root)
    nodes = 0

    def value(s: Any, d: int | None) -> tuple[float, Any]:
        nonlocal nodes
        nodes += 1
        if game.is_terminal(s):
            return _utility(game, s, player), None
        if _is_chance(game, s):
            outcomes = list(game.chance_outcomes(s))  # type: ignore[attr-defined]
            probs = [float(p) for p, _ in outcomes]
            if not outcomes or any(p < 0 for p in probs) or abs(math.fsum(probs) - 1) > 1e-9:
                raise ValueError("chance outcome probabilities must be >= 0 and sum to 1")
            return math.fsum(
                p * value(s2, d)[0] for p, (_, s2) in zip(probs, outcomes, strict=True)
            ), None
        if d == 0:
            return _evaluate(evaluate, s, player), None  # type: ignore[arg-type]
        maximize = game.to_move(s) == player
        best, best_move = (-math.inf if maximize else math.inf), None
        for a in list(game.actions(s)):
            v, _ = value(game.result(s, a), None if d is None else d - 1)
            if (v > best) if maximize else (v < best):
                best, best_move = v, a
        return best, best_move

    v, move = value(root, depth)
    return GameResult(v, move, nodes)


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------
class _Node:
    __slots__ = ("state", "parent", "move", "children", "untried", "visits", "wins", "mover")

    def __init__(self, state: Any, parent: _Node | None, move: Any, untried: list, mover: Any):
        self.state = state
        self.parent = parent
        self.move = move
        self.children: list[_Node] = []
        self.untried = untried
        self.visits = 0
        self.wins = 0.0  # from the point of view of `mover` (who moved into this node)
        self.mover = mover


@register(
    domain="ai",
    tasks=("game-playing", "monte-carlo", "adversarial-search"),
    complexity=Complexity(
        time="O(T (L + b))",
        space="O(T)",
        notes="T = n_iterations, L = game length (selection + rollout), b = branching "
        "factor (UCB scan over children); one tree node per iteration",
    ),
    references=(_UCT, _BROWNE, _AIMA + ", §5.4"),
)
def mcts(
    game: Game,
    state: Any = None,
    n_iterations: int = 1000,
    c: float = math.sqrt(2.0),
    random_state: RandomState = None,
) -> MCTSResult:
    r"""Monte Carlo tree search with UCT selection and uniformly random rollouts.

    Parameters
    ----------
    game : Game
        Deterministic game with utilities in [-1, 1].
    state : optional
        Root position (default ``game.initial_state``).
    n_iterations : int, default=1000
        Number of select-expand-simulate-backpropagate iterations (>= 1).
    c : float, default=sqrt(2)
        Exploration constant (>= 0).
    random_state : None, int, SeedSequence or Generator

    Returns
    -------
    MCTSResult

    Complexity
    ----------
    Time :math:`O(T(L + b))` for T iterations, game length L and branching factor b;
    space :math:`O(T)` (one node per iteration).

    References
    ----------
    Kocsis & Szepesvári (2006); Browne et al. (2012).

    Examples
    --------
    >>> # X to move wins at once by completing the top row.
    >>> s = (("X", "X", None, "O", "O", None, None, None, None), "X")
    >>> mcts(TicTacToe(), s, n_iterations=300, random_state=0).best_move
    2
    """
    _check_game(game)
    root_state = game.initial_state if state is None else state
    n_iterations = check_int(n_iterations, "n_iterations", minimum=1)
    c = check_float(c, "c", low=0.0)
    rng = check_random_state(random_state)
    if _is_chance(game, root_state):
        raise ValueError("mcts does not support chance nodes")
    if game.is_terminal(root_state):
        return MCTSResult(None, _reward_to_value(game, root_state), n_iterations, 1, {})

    def new_node(s: Any, parent: _Node | None, move: Any, mover: Any) -> _Node:
        if _is_chance(game, s):
            raise ValueError("mcts does not support chance nodes")
        untried = [] if game.is_terminal(s) else list(game.actions(s))
        return _Node(s, parent, move, untried, mover)

    root = new_node(root_state, None, None, None)
    size = 1
    for _ in range(n_iterations):
        node = root
        # selection
        while not node.untried and node.children:
            log_n = math.log(node.visits)
            node = max(
                node.children,
                key=lambda ch: ch.wins / ch.visits + c * math.sqrt(log_n / ch.visits),
            )
        # expansion
        if node.untried:
            a = node.untried.pop(int(rng.integers(len(node.untried))))
            mover = game.to_move(node.state)
            child = new_node(game.result(node.state, a), node, a, mover)
            node.children.append(child)
            node = child
            size += 1
        # simulation
        s = node.state
        while not game.is_terminal(s):
            acts = list(game.actions(s))
            s = game.result(s, acts[int(rng.integers(len(acts)))])
        # backpropagation
        while node is not None:
            node.visits += 1
            if node.mover is not None:
                u = _utility(game, s, node.mover)
                if not -1.0 <= u <= 1.0:
                    raise ValueError(f"mcts needs utilities in [-1, 1], got {u}")
                node.wins += (u + 1.0) / 2.0
            node = node.parent
    best = max(root.children, key=lambda ch: ch.visits)  # first maximum: earliest expanded
    visits = {ch.move: ch.visits for ch in root.children}
    return MCTSResult(best.move, 2.0 * best.wins / best.visits - 1.0, n_iterations, size, visits)


def _reward_to_value(game: Any, state: Any) -> float:
    return _utility(game, state, game.to_move(state))


# ---------------------------------------------------------------------------
# reference games
# ---------------------------------------------------------------------------
_LINES3 = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)
]  # fmt: skip


class TicTacToe:
    """Tic-tac-toe. A state is ``(board, player)`` with ``board`` a 9-tuple of
    ``"X"``, ``"O"`` or None (row-major) and ``player`` to move; actions are cell
    indices. Utility: +1 win, -1 loss, 0 draw.

    Examples
    --------
    >>> g = TicTacToe()
    >>> g.actions(g.initial_state)[:3], g.to_move(g.initial_state)
    ([0, 1, 2], 'X')
    """

    def __init__(self) -> None:
        self.initial_state: tuple[tuple[str | None, ...], str] = ((None,) * 9, "X")

    def to_move(self, state: tuple) -> str:
        return state[1]

    def actions(self, state: tuple) -> list[int]:
        return [] if self.winner(state) else [i for i, v in enumerate(state[0]) if v is None]

    def result(self, state: tuple, action: int) -> tuple:
        board, p = state
        if board[action] is not None:
            raise ValueError(f"cell {action} is occupied")
        b = list(board)
        b[action] = p
        return tuple(b), ("O" if p == "X" else "X")

    @staticmethod
    def winner(state: tuple) -> str | None:
        b = state[0]
        for i, j, k in _LINES3:
            if b[i] is not None and b[i] == b[j] == b[k]:
                return b[i]
        return None

    def is_terminal(self, state: tuple) -> bool:
        return self.winner(state) is not None or all(v is not None for v in state[0])

    def utility(self, state: tuple, player: str) -> float:
        w = self.winner(state)
        return 0.0 if w is None else (1.0 if w == player else -1.0)


class Nim:
    """Normal-play Nim: remove any positive number of objects from one heap; the
    player who takes the last object wins. A state is ``(heaps, player)`` with
    players 0 and 1; actions are ``(heap_index, count)``.

    Bouton's theorem: the player to move wins iff the XOR of the heap sizes is
    non-zero (the Sprague-Grundy value of a heap of size k is k).

    Examples
    --------
    >>> Nim((1, 2, 3)).actions(((1, 2, 3), 0))[:3]
    [(0, 1), (1, 1), (1, 2)]
    """

    def __init__(self, heaps: Sequence[int]) -> None:
        hs = tuple(check_int(h, "heap size") for h in heaps)
        self.initial_state = (hs, 0)

    def to_move(self, state: tuple) -> int:
        return state[1]

    def actions(self, state: tuple) -> list[tuple[int, int]]:
        return [(i, k) for i, h in enumerate(state[0]) for k in range(1, h + 1)]

    def result(self, state: tuple, action: tuple[int, int]) -> tuple:
        heaps, p = state
        i, k = action
        if not 1 <= k <= heaps[i]:
            raise ValueError(f"cannot take {k} from heap {i} of size {heaps[i]}")
        hs = list(heaps)
        hs[i] -= k
        return tuple(hs), 1 - p

    def is_terminal(self, state: tuple) -> bool:
        return not any(state[0])

    def utility(self, state: tuple, player: int) -> float:
        # Terminal: the player to move faces empty heaps, so the other player took the last.
        return -1.0 if state[1] == player else 1.0


class ConnectFour:
    """Connect-k on a ``rows x cols`` board with gravity (default 4 rows x 5 columns,
    connect 4). A state is ``(board, player)`` with ``board`` a tuple of rows (row 0
    at the bottom) holding 1, 2 or 0; actions are columns that are not full.
    Utility: +1 win, -1 loss, 0 draw.

    Examples
    --------
    >>> g = ConnectFour()
    >>> g.actions(g.initial_state)
    [0, 1, 2, 3, 4]
    """

    def __init__(self, rows: int = 4, cols: int = 5, connect: int = 4) -> None:
        self.rows = check_int(rows, "rows", minimum=1)
        self.cols = check_int(cols, "cols", minimum=1)
        self.connect = check_int(connect, "connect", minimum=2)
        self.initial_state = (tuple((0,) * self.cols for _ in range(self.rows)), 1)

    def to_move(self, state: tuple) -> int:
        return state[1]

    def actions(self, state: tuple) -> list[int]:
        if self.winner(state):
            return []
        top = state[0][-1]
        return [c for c in range(self.cols) if top[c] == 0]

    def result(self, state: tuple, action: int) -> tuple:
        board, p = state
        for r in range(self.rows):
            if board[r][action] == 0:
                row = list(board[r])
                row[action] = p
                new = board[:r] + (tuple(row),) + board[r + 1 :]
                return new, 3 - p
        raise ValueError(f"column {action} is full")

    def winner(self, state: tuple) -> int:
        b, k = state[0], self.connect
        for r in range(self.rows):
            for c in range(self.cols):
                p = b[r][c]
                if p == 0:
                    continue
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    rr, cc = r + (k - 1) * dr, c + (k - 1) * dc
                    if (
                        0 <= rr < self.rows
                        and 0 <= cc < self.cols
                        and all(b[r + i * dr][c + i * dc] == p for i in range(k))
                    ):
                        return p
        return 0

    def is_terminal(self, state: tuple) -> bool:
        return bool(self.winner(state)) or all(v != 0 for v in state[0][-1])

    def utility(self, state: tuple, player: int) -> float:
        w = self.winner(state)
        return 0.0 if w == 0 else (1.0 if w == player else -1.0)
