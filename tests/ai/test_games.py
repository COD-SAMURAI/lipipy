"""lipipy.ai.games: minimax, alpha-beta, expectiminimax and MCTS on reference games."""

import functools
import itertools
import math

import numpy as np
import pytest

from lipipy.ai import (
    ConnectFour,
    GameResult,
    MCTSResult,
    Nim,
    TicTacToe,
    alphabeta,
    expectiminimax,
    mcts,
    minimax,
)


# ---------------------------------------------------------------------------
# tic-tac-toe
# ---------------------------------------------------------------------------
@pytest.mark.filterwarnings("error")
def test_tictactoe_value_is_draw_and_alphabeta_prunes():
    g = TicTacToe()
    mm = minimax(g)
    ab = alphabeta(g)
    assert mm.value == 0.0 and ab.value == 0.0
    assert mm.nodes == 549_946  # every node of the full tic-tac-toe game tree
    assert ab.nodes < mm.nodes
    assert ab.best_move == mm.best_move


def random_position(game, rng, plies):
    s = game.initial_state
    for _ in range(plies):
        if game.is_terminal(s):
            break
        acts = game.actions(s)
        s = game.result(s, acts[int(rng.integers(len(acts)))])
    return s


@pytest.mark.parametrize("seed", range(40))
def test_alphabeta_equals_minimax_on_tictactoe_positions(seed):
    g = TicTacToe()
    rng = np.random.default_rng(seed)
    s = random_position(g, rng, int(rng.integers(2, 6)))
    mm, ab = minimax(g, s), alphabeta(g, s)
    assert ab.value == mm.value and ab.best_move == mm.best_move
    if not g.is_terminal(s):
        assert ab.nodes < mm.nodes  # >= 4 empty cells: pruning always happens


def c4_eval(game):
    def evaluate(state, player):
        # Count open windows: simple but not symmetric in the players' favour.
        b = state[0]
        score = 0.0
        for r in range(game.rows):
            for c in range(game.cols):
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    cells = [(r + i * dr, c + i * dc) for i in range(game.connect)]
                    if not all(0 <= rr < game.rows and 0 <= cc < game.cols for rr, cc in cells):
                        continue
                    vals = {b[rr][cc] for rr, cc in cells}
                    if vals <= {0, player}:
                        score += sum(b[rr][cc] == player for rr, cc in cells) ** 2
                    if vals <= {0, 3 - player}:
                        score -= sum(b[rr][cc] == 3 - player for rr, cc in cells) ** 2
        return score / 100.0

    return evaluate


@pytest.mark.parametrize("seed", range(10))
def test_alphabeta_equals_minimax_depth_limited_connect_four(seed):
    g = ConnectFour()
    rng = np.random.default_rng(seed)
    s = random_position(g, rng, int(rng.integers(0, 6)))
    ev = c4_eval(g)
    for depth in (1, 3, 4):
        mm = minimax(g, s, depth=depth, evaluate=ev)
        ab = alphabeta(g, s, depth=depth, evaluate=ev)
        assert ab.value == pytest.approx(mm.value) and ab.best_move == mm.best_move
        if depth >= 3 and not g.is_terminal(s):
            assert ab.nodes < mm.nodes
        ordered = alphabeta(
            g,
            s,
            depth=depth,
            evaluate=ev,
            order_moves=lambda st, acts: sorted(acts, key=lambda c: abs(c - 2)),
        )
        assert ordered.value == pytest.approx(mm.value)


def test_move_ordering_changes_pruning_not_value():
    g = TicTacToe()
    centre_first = alphabeta(g, order_moves=lambda s, acts: sorted(acts, key=lambda a: a != 4))
    worst_first = alphabeta(g, order_moves=lambda s, acts: list(reversed(acts)))
    assert centre_first.value == worst_first.value == 0.0
    assert centre_first.nodes != worst_first.nodes
    with pytest.raises(ValueError):
        alphabeta(g, order_moves=lambda s, acts: acts[:-1])


def test_connect_four_tactics_exact():
    g = ConnectFour()
    win_now = (((1, 1, 1, 0, 0), (2, 2, 2, 0, 0), (0,) * 5, (0,) * 5), 1)
    res = alphabeta(g, win_now, depth=2, evaluate=c4_eval(g))
    assert res.best_move == 3 and res.value == 1.0
    must_block = (((1, 1, 1, 0, 0), (2, 2, 0, 0, 0), (0,) * 5, (0,) * 5), 2)
    res = alphabeta(g, must_block, depth=2, evaluate=c4_eval(g))
    assert res.best_move == 3


# ---------------------------------------------------------------------------
# Nim
# ---------------------------------------------------------------------------
HEAPS = [h for k in (1, 2, 3) for h in itertools.product(range(5), repeat=k)]


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("heaps", HEAPS)
def test_nim_values_follow_bouton_xor_rule(heaps):
    expected = 1.0 if functools.reduce(lambda a, b: a ^ b, heaps) else -1.0
    g = Nim(heaps)
    ab = alphabeta(g)
    assert ab.value == expected
    if expected == 1.0:  # a winning move leaves XOR = 0
        i, k = ab.best_move
        after = list(heaps)
        after[i] -= k
        assert functools.reduce(lambda a, b: a ^ b, after) == 0
    if sum(heaps) <= 6:
        assert minimax(g).value == expected
        assert expectiminimax(g).value == expected


# ---------------------------------------------------------------------------
# expectiminimax
# ---------------------------------------------------------------------------
class TreeGame:
    """An explicit game tree: nodes are ints; kinds 'max', 'min', 'chance', 'leaf'."""

    def __init__(self, rng, depth, branching=3):
        self.kind, self.children, self.probs, self.leaf = {}, {}, {}, {}
        self.n = 0
        self.initial_state = self._build(rng, depth, branching, "max")

    def _build(self, rng, depth, b, kind):
        node = self.n
        self.n += 1
        if depth == 0:
            self.kind[node] = "leaf"
            self.leaf[node] = float(rng.normal())
            return node
        self.kind[node] = kind
        nxt = {"max": "chance", "chance": "min", "min": "max"}[kind]
        self.children[node] = [self._build(rng, depth - 1, b, nxt) for _ in range(b)]
        if kind == "chance":
            p = rng.dirichlet(np.ones(b))
            self.probs[node] = list(p / p.sum())
        return node

    def to_move(self, s):
        return "A" if self.kind[s] in ("max", "leaf", "chance") else "B"

    def actions(self, s):
        return list(range(len(self.children.get(s, []))))

    def result(self, s, a):
        return self.children[s][a]

    def is_terminal(self, s):
        return self.kind[s] == "leaf"

    def utility(self, s, player):
        return self.leaf[s] if player == "A" else -self.leaf[s]

    def is_chance(self, s):
        return self.kind[s] == "chance"

    def chance_outcomes(self, s):
        return list(zip(self.probs[s], self.children[s], strict=True))

    def brute(self, s):
        k = self.kind[s]
        if k == "leaf":
            return self.leaf[s]
        vals = [self.brute(c) for c in self.children[s]]
        if k == "max":
            return max(vals)
        if k == "min":
            return min(vals)
        return sum(p * v for p, v in zip(self.probs[s], vals, strict=True))


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("seed", range(30))
def test_expectiminimax_equals_brute_force_expectation(seed):
    game = TreeGame(np.random.default_rng(seed), depth=int(1 + seed % 6))
    res = expectiminimax(game)
    assert res.value == pytest.approx(game.brute(game.initial_state), abs=1e-12)
    assert res.nodes == game.n
    if game.kind[game.initial_state] == "max" and game.children:
        kids = game.children[game.initial_state]
        best = max(range(len(kids)), key=lambda a: game.brute(kids[a]))
        assert game.brute(kids[res.best_move]) == pytest.approx(game.brute(kids[best]))


def test_expectiminimax_rejects_bad_probabilities():
    game = TreeGame(np.random.default_rng(0), depth=3)
    chance = next(s for s, k in game.kind.items() if k == "chance")
    game.probs[chance] = [0.5, 0.5, 0.5]
    with pytest.raises(ValueError):
        expectiminimax(game)


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------
TACTICS = [
    # (game, state, winning / saving move, iterations)
    (TicTacToe(), (("X", "X", None, "O", "O", None, None, None, None), "X"), 2, 300),
    (TicTacToe(), (("X", "X", None, "O", None, None, None, None, None), "O"), 2, 1000),
    (ConnectFour(), (((1, 1, 1, 0, 0), (2, 2, 2, 0, 0), (0,) * 5, (0,) * 5), 1), 3, 300),
    (ConnectFour(), (((1, 1, 1, 0, 0), (2, 2, 0, 0, 0), (0,) * 5, (0,) * 5), 2), 3, 300),
]


@pytest.mark.parametrize("case", range(len(TACTICS)))
def test_mcts_finds_forced_moves_across_seeds(case):
    # Measured: the winning / blocking move is chosen for all 50 seeds 0..49.
    game, state, move, iters = TACTICS[case]
    hits = sum(
        mcts(game, state, n_iterations=iters, random_state=seed).best_move == move
        for seed in range(50)
    )
    assert hits == 50


def test_mcts_value_and_bookkeeping():
    g = TicTacToe()
    win_now = TACTICS[0][1]
    res = mcts(g, win_now, n_iterations=500, random_state=1)
    assert isinstance(res, MCTSResult)
    assert res.value > 0.9  # the winning child always backs up a win
    assert sum(res.visits.values()) == 500
    assert res.nodes <= 501
    assert mcts(g, win_now, n_iterations=500, random_state=1) == res
    d = res.to_dict()
    assert d["best_move"] == 2 and d["n_iterations"] == 500


def test_mcts_empty_board_prefers_reasonable_opening():
    res = mcts(TicTacToe(), n_iterations=3000, random_state=0)
    assert res.best_move in range(9) and abs(res.value) < 0.6


def test_mcts_nim_winning_move():
    # Nim (1, 2, 4): XOR = 7, the only winning move takes 1 from the 4-heap.
    # Measured: chosen for all 50 seeds with 1000 iterations.
    g = Nim((1, 2, 4))
    hits = sum(mcts(g, n_iterations=1000, random_state=s).best_move == (2, 1) for s in range(50))
    assert hits == 50


# ---------------------------------------------------------------------------
# edge cases and validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("search", [minimax, alphabeta, expectiminimax])
def test_terminal_root_and_depth_zero(search):
    g = TicTacToe()
    won = (("X", "X", "X", "O", "O", None, None, None, None), "O")
    res = search(g, won)
    assert res.value == -1.0 and res.best_move is None and res.nodes == 1
    res = search(g, depth=0, evaluate=lambda s, p: 0.25)
    assert res.value == 0.25 and res.best_move is None and res.nodes == 1
    nim0 = search(Nim((0, 0)))
    assert nim0.value == -1.0


def test_mcts_terminal_root():
    won = (("X", "X", "X", "O", "O", None, None, None, None), "O")
    res = mcts(TicTacToe(), won, n_iterations=10, random_state=0)
    assert res.best_move is None and res.value == -1.0 and res.nodes == 1


def test_validation():
    g = TicTacToe()
    with pytest.raises(ValueError):
        minimax(g, depth=2)  # no evaluate
    with pytest.raises(ValueError):
        alphabeta(g, depth=-1, evaluate=lambda s, p: 0.0)
    with pytest.raises(TypeError):
        minimax(object())
    with pytest.raises(ValueError):
        mcts(g, n_iterations=0)
    with pytest.raises(ValueError):
        mcts(g, c=-1.0)
    with pytest.raises(ValueError):
        mcts(TreeGame(np.random.default_rng(0), 2))  # root is fine, child is a chance node
    with pytest.raises(ValueError):
        minimax(g, depth=1, evaluate=lambda s, p: math.nan)
    with pytest.raises(ValueError):
        g.result(g.result(g.initial_state, 0), 0)
    with pytest.raises(ValueError):
        Nim((2,)).result(((2,), 0), (0, 3))
    with pytest.raises(ValueError):
        Nim((-1,))
    c4 = ConnectFour(rows=1, cols=2, connect=2)
    s = c4.result(c4.initial_state, 0)
    with pytest.raises(ValueError):
        c4.result(s, 0)
    with pytest.raises(ValueError):
        GameResult(0.0, None, 0)
    assert GameResult(1.0, (0, 1), 3).to_dict() == {"value": 1.0, "best_move": [0, 1], "nodes": 3}


def test_connect_four_small_board_solved():
    # 3 x 3 board, connect 3: the first player wins (centre column).
    g = ConnectFour(rows=3, cols=3, connect=3)
    ab = alphabeta(g)
    assert ab.value == minimax(g).value
    assert ab.nodes < minimax(g).nodes
