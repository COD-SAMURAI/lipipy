r"""Environments with the Gymnasium interface, and finite MDPs with known dynamics.

Interface
---------
Every environment follows the Gymnasium API:

* ``reset(seed=None, options=None) -> (observation, info)`` starts an episode;
  passing ``seed`` re-seeds the environment's generator.
* ``step(action) -> (observation, reward, terminated, truncated, info)``.
  ``terminated`` means a terminal state was reached (the return ends there);
  ``truncated`` means a time limit cut the episode (the state is not terminal).

Unlike Gymnasium, calling ``step`` before ``reset``, after the episode ended, or
with an action outside ``action_space`` raises (``RuntimeError`` / ``ValueError``)
and leaves the environment untouched. Observations are Python ints or fresh
arrays, never views of internal state.

Finite MDPs
-----------
A :class:`TabularMDP` stores the dynamics in Gymnasium's ``P`` format,

.. math::  P[s][a] = ((p_1, s'_1, r_1, d_1), (p_2, s'_2, r_2, d_2), \dots),

the probability :math:`p_i` of moving to :math:`s'_i` with reward :math:`r_i`, where
:math:`d_i` flags that the episode terminates on that transition. Terminal states
are absorbing: ``P[s][a] = ((1.0, s, 0.0, True),)`` for every action. Dynamic
programming uses the arrays from :meth:`TabularMDP.transition_arrays`:
:math:`\tilde P(s'|s,a) = \sum_i p_i [s'_i = s'][\lnot d_i]` (the probability of
continuing to :math:`s'`) and the expected reward
:math:`r(s,a) = \sum_i p_i r_i`, so that the Bellman equation reads
:math:`q(s,a) = r(s,a) + \gamma \sum_{s'} \tilde P(s'|s,a)\, v(s')`.

Symbols: S states, A actions, M = the largest number of listed outcomes of one
(s, a) pair.

References
----------
Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed. (2018):
§3.1-3.5 (MDPs), Example 4.1 (grid world), Example 6.2 (random walk),
Example 6.6 (cliff walking), Example 7.1 (19-state random walk), §2.3 (10-armed
testbed), Exercise 2.5 (non-stationary testbed), Example 13.1 (short corridor).
Towers et al. (2024), *Gymnasium* (the API and the FrozenLake maps).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from lipipy.core.complexity import Complexity
from lipipy.core.random import check_random_state
from lipipy.core.registry import register
from lipipy.rl._common import check_int, check_positive, check_unit, is_int, is_real

_SB = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed."
_GYM = "Towers et al. (2024), Gymnasium: a standard interface for reinforcement learning"


# ---------------------------------------------------------------------------
# Spaces
# ---------------------------------------------------------------------------
@register(
    domain="rl",
    tasks=("environment",),
    complexity=Complexity(time="O(1)", space="O(1)", notes="sample and contains"),
    references=(_GYM,),
    status="experimental",
)
class Discrete:
    """The integers ``{start, ..., start + n - 1}``.

    Parameters
    ----------
    n : int
        Number of elements, ``n >= 1``.
    start : int, default=0
    random_state : None, int, SeedSequence or Generator
        Seed of :meth:`sample`.

    Complexity
    ----------
    O(1) time and memory for ``sample`` and ``contains``.

    References
    ----------
    Gymnasium ``spaces.Discrete``.

    Examples
    --------
    >>> space = Discrete(4, random_state=0)
    >>> space.contains(3), space.contains(4), space.contains(1.5)
    (True, False, False)
    >>> 0 <= space.sample() < 4
    True
    """

    def __init__(self, n: int, start: int = 0, random_state=None):
        self.n = check_int(n, "n")
        if not is_int(start):
            raise ValueError(f"start must be an integer, got {start!r}.")
        self.start = int(start)
        self._rng = check_random_state(random_state)

    def seed(self, random_state=None) -> None:
        """Re-seed :meth:`sample`."""
        self._rng = check_random_state(random_state)

    def sample(self) -> int:
        """A uniformly random element."""
        return self.start + int(self._rng.integers(self.n))

    def contains(self, x) -> bool:
        """Whether ``x`` is an integer element of the space (bools are rejected)."""
        return is_int(x) and self.start <= x < self.start + self.n

    def __repr__(self) -> str:
        return f"Discrete({self.n}, start={self.start})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Discrete) and (self.n, self.start) == (other.n, other.start)

    __hash__ = None  # mutable (has a generator)


@register(
    domain="rl",
    tasks=("environment",),
    complexity=Complexity(
        time="O(d)", space="O(d)", notes="d = number of elements of one observation"
    ),
    references=(_GYM,),
    status="experimental",
)
class Box:
    """Real arrays of a fixed shape with element-wise bounds ``low <= x <= high``.

    ``sample`` draws uniformly on bounded coordinates, from an exponential shifted
    by the bound on half-bounded ones and from a standard normal on unbounded ones
    (Gymnasium's rule).

    Parameters
    ----------
    low, high : float or array_like
        Bounds (``-inf`` / ``inf`` allowed), broadcast to ``shape``.
    shape : tuple of int, optional
        Defaults to the broadcast shape of ``low`` and ``high``.
    random_state : None, int, SeedSequence or Generator

    Complexity
    ----------
    O(d) time and memory for ``sample`` and ``contains``.

    References
    ----------
    Gymnasium ``spaces.Box``.

    Examples
    --------
    >>> box = Box(0.0, 1.0, shape=(2,), random_state=0)
    >>> x = box.sample()
    >>> x.shape, box.contains(x), box.contains([0.5, 2.0])
    ((2,), True, False)
    """

    def __init__(self, low, high, shape: tuple[int, ...] | None = None, random_state=None):
        low_a = np.asarray(low, dtype=np.float64)
        high_a = np.asarray(high, dtype=np.float64)
        if shape is None:
            shape = np.broadcast_shapes(low_a.shape, high_a.shape)
        shape = tuple(int(s) for s in shape)
        if any(s < 1 for s in shape):
            raise ValueError(f"shape must have positive entries, got {shape}.")
        self.low = np.broadcast_to(low_a, shape).copy()
        self.high = np.broadcast_to(high_a, shape).copy()
        if np.any(np.isnan(self.low)) or np.any(np.isnan(self.high)):
            raise ValueError("low and high must not contain NaN.")
        if np.any(self.low > self.high):
            raise ValueError("low must be <= high element-wise.")
        self.low.setflags(write=False)
        self.high.setflags(write=False)
        self.shape = shape
        self._rng = check_random_state(random_state)

    def seed(self, random_state=None) -> None:
        """Re-seed :meth:`sample`."""
        self._rng = check_random_state(random_state)

    def sample(self) -> np.ndarray:
        """A random element (a new array)."""
        lo, hi = self.low, self.high
        out = self._rng.normal(size=self.shape)
        both = np.isfinite(lo) & np.isfinite(hi)
        out[both] = self._rng.uniform(lo[both], hi[both])
        only_lo = np.isfinite(lo) & ~np.isfinite(hi)
        out[only_lo] = lo[only_lo] + self._rng.exponential(size=int(only_lo.sum()))
        only_hi = ~np.isfinite(lo) & np.isfinite(hi)
        out[only_hi] = hi[only_hi] - self._rng.exponential(size=int(only_hi.sum()))
        return out

    def contains(self, x) -> bool:
        """Whether ``x`` has the right shape, is real and lies within the bounds."""
        try:
            a = np.asarray(x, dtype=np.float64)
        except (TypeError, ValueError):
            return False
        return a.shape == self.shape and bool(
            np.all(~np.isnan(a)) and np.all(a >= self.low) and np.all(a <= self.high)
        )

    def __repr__(self) -> str:
        return f"Box(shape={self.shape})"


@runtime_checkable
class Env(Protocol):
    """The Gymnasium environment protocol (structural type).

    Any object with ``observation_space``, ``action_space``, ``reset`` and ``step``
    as documented in the module docstring is an ``Env``, including Gymnasium's own
    environments.
    """

    observation_space: Any
    action_space: Any

    def reset(self, seed=None, options=None) -> tuple[Any, dict]: ...

    def step(self, action) -> tuple[Any, float, bool, bool, dict]: ...


# ---------------------------------------------------------------------------
# Finite MDPs
# ---------------------------------------------------------------------------
Transition = tuple[float, int, float, bool]


class _EpisodeState:
    """Bookkeeping shared by the environments: generator, step count, episode end."""

    def _init_episode_state(self, random_state, max_episode_steps) -> None:
        if max_episode_steps is not None:
            max_episode_steps = check_int(max_episode_steps, "max_episode_steps")
        self.max_episode_steps = max_episode_steps
        self._rng = check_random_state(random_state)
        self._state: int | None = None
        self._done = True
        self._elapsed = 0

    def _check_step(self, action) -> None:
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("The episode has ended (terminated or truncated); call reset().")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r} for {self.action_space}.")


@register(
    domain="rl",
    tasks=("environment", "markov-decision-process"),
    complexity=Complexity(
        time="O(M)",
        space="O(S A M)",
        notes=(
            "per step: sample one of the M outcomes of (s, a); storage of the model; "
            "transition_arrays() builds dense O(S^2 A) arrays. S states, A actions, "
            "M = most outcomes of one (s, a)"
        ),
    ),
    references=(_SB + ", §3.1-3.5", _GYM),
    status="experimental",
)
class TabularMDP(_EpisodeState):
    r"""A finite MDP with known dynamics, usable as an environment and by DP.

    Parameters
    ----------
    P : sequence or mapping
        ``P[s][a]`` is a sequence of ``(prob, next_state, reward, terminal)`` tuples
        for every state ``s`` in ``range(S)`` and action ``a`` in ``range(A)``.
        Probabilities must be non-negative and sum to one; a terminal state must be
        absorbing (``((1.0, s, 0.0, True),)`` for every action).
    initial_state_distribution : array_like of shape (S,)
        Start-state probabilities; terminal states must get probability zero.
    max_episode_steps : int, optional
        Truncate episodes after this many steps.
    random_state : None, int, SeedSequence or Generator

    Attributes
    ----------
    P : tuple
        The dynamics as nested tuples (immutable).
    n_states, n_actions : int
    terminal_states : ndarray of bool, shape (S,)
    observation_space, action_space : Discrete

    Raises
    ------
    ValueError
        For malformed dynamics or start distributions.

    Complexity
    ----------
    ``step``: O(M) time for M listed outcomes. Storage O(S A M).
    :meth:`transition_arrays`: O(S^2 A) time and memory.

    References
    ----------
    Sutton & Barto (2018), §3.1-3.5.

    Examples
    --------
    >>> P = [[[(0.5, 0, 0.0, False), (0.5, 1, 1.0, True)]], [[(1.0, 1, 0.0, True)]]]
    >>> mdp = TabularMDP(P, [1.0, 0.0], random_state=0)
    >>> mdp.reset()
    (0, {'prob': 1.0})
    >>> P_cont, R = mdp.transition_arrays()
    >>> P_cont[0, 0].tolist(), R.tolist()
    ([0.5, 0.0], [[0.5], [0.0]])
    """

    def __init__(
        self,
        P: Sequence | Mapping,
        initial_state_distribution,
        max_episode_steps: int | None = None,
        random_state=None,
    ):
        n_states = len(P)
        if n_states < 1:
            raise ValueError("P must describe at least one state.")
        rows = []
        n_actions = None
        for s in range(n_states):
            if isinstance(P, Mapping) and s not in P:
                raise ValueError(f"P has no entry for state {s}.")
            actions = P[s]
            if n_actions is None:
                n_actions = len(actions)
                if n_actions < 1:
                    raise ValueError("Every state needs at least one action.")
            if len(actions) != n_actions:
                raise ValueError(f"State {s} has {len(actions)} actions, state 0 has {n_actions}.")
            row = []
            for a in range(n_actions):
                outcomes = actions[a]
                if len(outcomes) < 1:
                    raise ValueError(f"P[{s}][{a}] lists no outcomes.")
                checked = []
                for item in outcomes:
                    if len(item) != 4:
                        raise ValueError(f"P[{s}][{a}] entries must be 4-tuples, got {item!r}.")
                    prob, nxt, reward, term = item
                    if not is_real(prob) or not 0.0 <= prob <= 1.0:
                        raise ValueError(f"P[{s}][{a}]: invalid probability {prob!r}.")
                    if not is_int(nxt) or not 0 <= nxt < n_states:
                        raise ValueError(f"P[{s}][{a}]: invalid next state {nxt!r}.")
                    if not is_real(reward) or not np.isfinite(reward):
                        raise ValueError(f"P[{s}][{a}]: invalid reward {reward!r}.")
                    if not isinstance(term, (bool, np.bool_)):
                        raise ValueError(f"P[{s}][{a}]: terminal flag must be a bool.")
                    checked.append((float(prob), int(nxt), float(reward), bool(term)))
                total = sum(c[0] for c in checked)
                if abs(total - 1.0) > 1e-9:
                    raise ValueError(f"P[{s}][{a}] probabilities sum to {total}, not 1.")
                row.append(tuple(checked))
            rows.append(tuple(row))
        self.P: tuple = tuple(rows)
        self.n_states = n_states
        self.n_actions = int(n_actions)

        absorbing = [
            all(out == ((1.0, s, 0.0, True),) for out in self.P[s]) for s in range(n_states)
        ]
        self.terminal_states = np.array(absorbing, dtype=bool)
        self.terminal_states.setflags(write=False)

        mu = np.asarray(initial_state_distribution, dtype=np.float64)
        if mu.shape != (n_states,):
            raise ValueError(
                f"initial_state_distribution must have shape ({n_states},), got {mu.shape}."
            )
        if np.any(~np.isfinite(mu)) or np.any(mu < 0) or abs(mu.sum() - 1.0) > 1e-9:
            raise ValueError("initial_state_distribution must be non-negative and sum to 1.")
        if np.any(mu[self.terminal_states] > 0):
            raise ValueError("initial_state_distribution puts mass on a terminal state.")
        self.initial_state_distribution = mu / mu.sum()
        self.initial_state_distribution.setflags(write=False)

        # Padded outcome tables for O(M) sampling.
        m = max(len(out) for row in self.P for out in row)
        self._cum = np.ones((n_states, self.n_actions, m))
        self._next = np.zeros((n_states, self.n_actions, m), dtype=np.intp)
        self._reward = np.zeros((n_states, self.n_actions, m))
        self._term = np.ones((n_states, self.n_actions, m), dtype=bool)
        self._prob = np.zeros((n_states, self.n_actions, m))
        for s in range(n_states):
            for a in range(self.n_actions):
                out = self.P[s][a]
                k = len(out)
                p = np.array([o[0] for o in out])
                self._prob[s, a, :k] = p
                self._cum[s, a, :k] = np.cumsum(p)
                self._next[s, a, :k] = [o[1] for o in out]
                self._reward[s, a, :k] = [o[2] for o in out]
                self._term[s, a, :k] = [o[3] for o in out]

        self.observation_space = Discrete(n_states)
        self.action_space = Discrete(self.n_actions)
        self._init_episode_state(random_state, max_episode_steps)

    def transition_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        r"""Dense ``(P_cont, R)``: continuation probabilities (S, A, S), expected rewards (S, A).

        ``P_cont[s, a, s']`` sums the probabilities of non-terminating transitions to
        ``s'``, so ``1 - P_cont[s, a].sum()`` is the probability that the episode
        ends. ``R[s, a]`` is the expected immediate reward. Fresh arrays every call.
        """
        S, A = self.n_states, self.n_actions
        P_cont = np.zeros((S, A, S))
        R = np.zeros((S, A))
        for s in range(S):
            for a in range(A):
                for prob, nxt, reward, term in self.P[s][a]:
                    R[s, a] += prob * reward
                    if not term:
                        P_cont[s, a, nxt] += prob
        return P_cont, R

    def reset(self, seed=None, options=None) -> tuple[int, dict]:
        """Start an episode from a state drawn from ``initial_state_distribution``."""
        if seed is not None:
            self._rng = check_random_state(seed)
        s = int(self._rng.choice(self.n_states, p=self.initial_state_distribution))
        self._state, self._done, self._elapsed = s, False, 0
        return s, {"prob": float(self.initial_state_distribution[s])}

    def step(self, action) -> tuple[int, float, bool, bool, dict]:
        """Sample a transition from the current state with ``action``."""
        self._check_step(action)
        s, a = self._state, int(action)
        i = int(np.searchsorted(self._cum[s, a], self._rng.random(), side="right"))
        i = min(i, len(self.P[s][a]) - 1)  # guard against cumsum rounding below 1
        nxt = int(self._next[s, a, i])
        reward = float(self._reward[s, a, i])
        terminated = bool(self._term[s, a, i])
        self._elapsed += 1
        truncated = (
            not terminated
            and self.max_episode_steps is not None
            and self._elapsed >= self.max_episode_steps
        )
        self._state, self._done = nxt, terminated or truncated
        return nxt, reward, terminated, truncated, {"prob": float(self._prob[s, a, i])}

    @property
    def state(self) -> int | None:
        """Current state (None before the first reset)."""
        return self._state


# ---------------------------------------------------------------------------
# Grid worlds
# ---------------------------------------------------------------------------
# Action order of Gymnasium's CliffWalking: up, right, down, left.
_MOVES_URDL = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _cells(cells, shape, name) -> list[tuple[int, int]]:
    out = []
    for c in cells:
        r, col = c
        if not (is_int(r) and is_int(col) and 0 <= r < shape[0] and 0 <= col < shape[1]):
            raise ValueError(f"{name}: cell {c!r} is outside the {shape[0]}x{shape[1]} grid.")
        out.append((int(r), int(col)))
    return out


@register(
    domain="rl",
    tasks=("environment", "markov-decision-process"),
    complexity=Complexity(
        time="O(1)",
        space="O(S A)",
        notes="per step (at most 4 outcomes); S = rows x cols states, A = 4 actions",
    ),
    references=(_SB + ", Example 4.1, §3.5",),
    status="experimental",
)
class GridWorld(TabularMDP):
    r"""A rectangular grid world with optional terminal cells, walls and slipping.

    States are cells numbered row by row, ``s = row * cols + col``. Actions are
    ``0`` up, ``1`` right, ``2`` down, ``3`` left. A move off the grid or into a
    wall leaves the agent in place. With probability ``1 - slip`` the intended move
    happens; with probability ``slip`` a uniformly random one of the four (so
    ``slip=1`` is a fully random walk). Each step gives ``step_reward``, or
    ``rewards[cell]`` when entering a cell listed there. Entering a terminal cell
    ends the episode.

    Parameters
    ----------
    shape : (int, int), default=(4, 4)
    terminal_cells : sequence of (row, col), default=((0, 0), (3, 3))
    start_cells : sequence of (row, col), optional
        Start uniformly on these cells; default: every non-terminal, non-wall cell.
    step_reward : float, default=-1.0
    rewards : mapping (row, col) -> float, optional
    walls : sequence of (row, col), default=()
    slip : float in [0, 1], default=0.0
    max_episode_steps : int, optional
    random_state : None, int, SeedSequence or Generator

    Complexity
    ----------
    ``step``: O(1). Model storage O(S A) with S = rows × cols, A = 4.

    References
    ----------
    Sutton & Barto (2018), Example 4.1 (the default and :meth:`example_4_1`).

    Examples
    --------
    >>> env = GridWorld.example_4_1(random_state=0)
    >>> env.n_states, env.n_actions, env.terminal_states.nonzero()[0].tolist()
    (16, 4, [0, 15])
    >>> _ = env.reset(seed=1)
    >>> obs, reward, terminated, truncated, info = env.step(0)
    >>> reward, truncated
    (-1.0, False)
    """

    def __init__(
        self,
        shape: tuple[int, int] = (4, 4),
        terminal_cells: Sequence[tuple[int, int]] = ((0, 0), (3, 3)),
        start_cells: Sequence[tuple[int, int]] | None = None,
        step_reward: float = -1.0,
        rewards: Mapping[tuple[int, int], float] | None = None,
        walls: Sequence[tuple[int, int]] = (),
        slip: float = 0.0,
        max_episode_steps: int | None = None,
        random_state=None,
    ):
        if len(shape) != 2:
            raise ValueError(f"shape must be (rows, cols), got {shape!r}.")
        rows, cols = check_int(shape[0], "rows"), check_int(shape[1], "cols")
        shape = (rows, cols)
        slip = check_unit(slip, "slip")
        if not is_real(step_reward) or not np.isfinite(step_reward):
            raise ValueError(f"step_reward must be finite, got {step_reward!r}.")
        terminals = set(_cells(terminal_cells, shape, "terminal_cells"))
        wall_set = set(_cells(walls, shape, "walls"))
        if terminals & wall_set:
            raise ValueError("A cell cannot be both a wall and terminal.")
        reward_map = {}
        for cell, r in (rewards or {}).items():
            (cell,) = _cells([cell], shape, "rewards")
            if not is_real(r) or not np.isfinite(r):
                raise ValueError(f"rewards[{cell}] must be finite, got {r!r}.")
            reward_map[cell] = float(r)
        if start_cells is None:
            starts = [
                (r, c)
                for r in range(rows)
                for c in range(cols)
                if (r, c) not in terminals and (r, c) not in wall_set
            ]
        else:
            starts = _cells(start_cells, shape, "start_cells")
            if set(starts) & wall_set:
                raise ValueError("start_cells must not contain walls.")
        if not starts:
            raise ValueError("No start cell: every cell is terminal or a wall.")

        def index(cell):
            return cell[0] * cols + cell[1]

        def move(cell, a):
            dr, dc = _MOVES_URDL[a]
            r, c = cell[0] + dr, cell[1] + dc
            if not (0 <= r < rows and 0 <= c < cols) or (r, c) in wall_set:
                return cell
            return (r, c)

        P = []
        for r in range(rows):
            for c in range(cols):
                cell = (r, c)
                if cell in terminals or cell in wall_set:
                    # walls are unreachable; make them absorbing like terminals
                    P.append([((1.0, index(cell), 0.0, True),)] * 4)
                    continue
                row = []
                for a in range(4):
                    probs: dict[tuple[int, int], float] = {}
                    for b in range(4):
                        p = (1.0 - slip) * (a == b) + slip / 4.0
                        if p > 0:
                            nxt = move(cell, b)
                            probs[nxt] = probs.get(nxt, 0.0) + p
                    row.append(
                        tuple(
                            (p, index(n), reward_map.get(n, float(step_reward)), n in terminals)
                            for n, p in sorted(probs.items())
                        )
                    )
                P.append(row)
        mu = np.zeros(rows * cols)
        mu[[index(c) for c in starts]] = 1.0 / len(starts)
        self.shape = shape
        super().__init__(P, mu, max_episode_steps, random_state)

    @classmethod
    def example_4_1(cls, max_episode_steps: int | None = None, random_state=None) -> GridWorld:
        """Sutton & Barto's 4×4 grid: terminal corners, reward -1 on every step."""
        return cls(max_episode_steps=max_episode_steps, random_state=random_state)


@register(
    domain="rl",
    tasks=("environment", "markov-decision-process"),
    complexity=Complexity(time="O(1)", space="O(S A)", notes="S = 48 cells, A = 4"),
    references=(_SB + ", Example 6.6", _GYM),
    status="experimental",
)
class CliffWalking(TabularMDP):
    r"""Cliff walking: a 4×12 grid with a cliff along the bottom edge.

    The agent starts at the bottom-left cell (state 36) and the goal is the
    bottom-right cell (state 47, terminal). Every step gives -1; stepping into the
    cliff (states 37-46) gives -100 and sends the agent back to the start without
    ending the episode. Actions: ``0`` up, ``1`` right, ``2`` down, ``3`` left
    (Gymnasium's order). The optimal path walks along the cliff edge (13 steps,
    return -13).

    Parameters
    ----------
    max_episode_steps : int, optional
    random_state : None, int, SeedSequence or Generator

    Complexity
    ----------
    ``step``: O(1); model O(S A) with S = 48, A = 4.

    References
    ----------
    Sutton & Barto (2018), Example 6.6.

    Examples
    --------
    >>> env = CliffWalking()
    >>> env.reset(seed=0)[0]
    36
    >>> env.step(1)[:4]  # right, into the cliff
    (36, -100.0, False, False)
    """

    shape = (4, 12)
    start = 36
    goal = 47

    def __init__(self, max_episode_steps: int | None = None, random_state=None):
        rows, cols = self.shape
        P = []
        for s in range(rows * cols):
            if s == self.goal:
                P.append([((1.0, s, 0.0, True),)] * 4)
                continue
            r, c = divmod(s, cols)
            row = []
            for dr, dc in _MOVES_URDL:
                nr = min(max(r + dr, 0), rows - 1)
                nc = min(max(c + dc, 0), cols - 1)
                nxt = nr * cols + nc
                if nr == rows - 1 and 0 < nc < cols - 1:  # the cliff
                    row.append(((1.0, self.start, -100.0, False),))
                else:
                    row.append(((1.0, nxt, -1.0, nxt == self.goal),))
            P.append(row)
        mu = np.zeros(rows * cols)
        mu[self.start] = 1.0
        super().__init__(P, mu, max_episode_steps, random_state)


_FROZEN_LAKE_MAPS = {
    "4x4": ("SFFF", "FHFH", "FFFH", "HFFG"),
    "8x8": (
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ),
}
# Action order of Gymnasium's FrozenLake: left, down, right, up.
_MOVES_LDRU = ((0, -1), (1, 0), (0, 1), (-1, 0))


@register(
    domain="rl",
    tasks=("environment", "markov-decision-process"),
    complexity=Complexity(time="O(1)", space="O(S A)", notes="S = 16 or 64 cells, A = 4"),
    references=(_GYM,),
    status="experimental",
)
class FrozenLake(TabularMDP):
    r"""Frozen lake (Gymnasium's maps and dynamics): reach G without falling into H.

    Cells: ``S`` start, ``F`` frozen, ``H`` hole (terminal, reward 0), ``G`` goal
    (terminal, reward 1). Actions: ``0`` left, ``1`` down, ``2`` right, ``3`` up.
    When slippery, the intended direction and each of the two perpendicular ones
    happen with probability 1/3; ``P`` lists the three outcomes separately, exactly
    as Gymnasium does.

    Parameters
    ----------
    map_name : {"4x4", "8x8"}, default="4x4"
    is_slippery : bool, default=True
    max_episode_steps : int, optional
    random_state : None, int, SeedSequence or Generator

    Complexity
    ----------
    ``step``: O(1); model O(S A).

    References
    ----------
    Gymnasium ``FrozenLake-v1``.

    Examples
    --------
    >>> env = FrozenLake(is_slippery=False)
    >>> env.reset(seed=0)[0]
    0
    >>> env.step(1)[:3]  # down: from S onto F
    (4, 0.0, False)
    """

    def __init__(
        self,
        map_name: str = "4x4",
        is_slippery: bool = True,
        max_episode_steps: int | None = None,
        random_state=None,
    ):
        if map_name not in _FROZEN_LAKE_MAPS:
            raise ValueError(
                f"map_name must be one of {sorted(_FROZEN_LAKE_MAPS)}, got {map_name!r}."
            )
        if not isinstance(is_slippery, (bool, np.bool_)):
            raise TypeError(f"is_slippery must be a bool, got {is_slippery!r}.")
        desc = _FROZEN_LAKE_MAPS[map_name]
        rows, cols = len(desc), len(desc[0])
        P = []
        for s in range(rows * cols):
            r, c = divmod(s, cols)
            if desc[r][c] in "GH":
                P.append([((1.0, s, 0.0, True),)] * 4)
                continue
            row = []
            for a in range(4):
                directions = ((a - 1) % 4, a, (a + 1) % 4) if is_slippery else (a,)
                outs = []
                for b in directions:
                    dr, dc = _MOVES_LDRU[b]
                    nr = min(max(r + dr, 0), rows - 1)
                    nc = min(max(c + dc, 0), cols - 1)
                    cell = desc[nr][nc]
                    p = 1.0 / 3.0 if is_slippery else 1.0
                    outs.append((p, nr * cols + nc, float(cell == "G"), cell in "GH"))
                row.append(tuple(outs))
            P.append(row)
        mu = np.array([ch == "S" for line in desc for ch in line], dtype=np.float64)
        self.desc = desc
        self.map_name = map_name
        self.is_slippery = bool(is_slippery)
        super().__init__(P, mu / mu.sum(), max_episode_steps, random_state)


@register(
    domain="rl",
    tasks=("environment", "markov-reward-process"),
    complexity=Complexity(time="O(1)", space="O(S)", notes="S = n_states + 2 terminals"),
    references=(_SB + ", Example 6.2, Example 7.1",),
    status="experimental",
)
class RandomWalk(TabularMDP):
    r"""The random walk Markov reward process of Examples 6.2 and 7.1.

    States ``1..n`` are non-terminal, ``0`` and ``n + 1`` are terminal. The walk
    starts in the centre and moves left or right with probability 1/2 each.
    Exiting right gives +1, exiting left gives ``left_reward`` (0 in Example 6.2,
    -1 in Example 7.1), all other rewards are 0. There is a single action ``0``.
    With :math:`\gamma = 1` the true value of state :math:`k` is
    :math:`\ell + (1 - \ell)\, k / (n + 1)` for left reward :math:`\ell`.

    Parameters
    ----------
    n_states : odd int, default=5
        Number of non-terminal states (5 or 19 in the book).
    left_reward : float, optional
        Default 0 for 5 states, -1 otherwise.
    max_episode_steps : int, optional
    random_state : None, int, SeedSequence or Generator

    Complexity
    ----------
    ``step``: O(1); model O(S).

    References
    ----------
    Sutton & Barto (2018), Example 6.2 (5 states), Example 7.1 (19 states).

    Examples
    --------
    >>> RandomWalk(5).true_values().round(4).tolist()
    [0.0, 0.1667, 0.3333, 0.5, 0.6667, 0.8333, 0.0]
    >>> RandomWalk(19).true_values()[[1, 10, 19]].round(2).tolist()
    [-0.9, 0.0, 0.9]
    """

    def __init__(
        self,
        n_states: int = 5,
        left_reward: float | None = None,
        max_episode_steps: int | None = None,
        random_state=None,
    ):
        n = check_int(n_states, "n_states")
        if n % 2 == 0:
            raise ValueError(f"n_states must be odd (the walk starts in the centre), got {n}.")
        if left_reward is None:
            left_reward = 0.0 if n == 5 else -1.0
        if not is_real(left_reward) or not np.isfinite(left_reward):
            raise ValueError(f"left_reward must be finite, got {left_reward!r}.")
        self.left_reward = float(left_reward)
        P = [[((1.0, 0, 0.0, True),)]]
        for k in range(1, n + 1):
            left = (0.5, k - 1, self.left_reward if k == 1 else 0.0, k == 1)
            right = (0.5, k + 1, 1.0 if k == n else 0.0, k == n)
            P.append([(left, right)])
        P.append([((1.0, n + 1, 0.0, True),)])
        mu = np.zeros(n + 2)
        mu[(n + 1) // 2] = 1.0
        super().__init__(P, mu, max_episode_steps, random_state)

    def true_values(self) -> np.ndarray:
        """Exact state values for gamma = 1 (terminals 0), shape (n + 2,)."""
        n = self.n_states - 2
        v = self.left_reward + (1.0 - self.left_reward) * np.arange(n + 2) / (n + 1)
        v[0] = v[-1] = 0.0
        return v


@register(
    domain="rl",
    tasks=("environment", "markov-decision-process"),
    complexity=Complexity(time="O(1)", space="O(1)", notes="S = 4 states, A = 2"),
    references=(_SB + ", Example 13.1",),
    status="experimental",
)
class ShortCorridor(TabularMDP):
    r"""Short corridor with switched actions (Example 13.1).

    States ``0, 1, 2`` are non-terminal and ``3`` is the goal; every step gives -1.
    Action ``1`` (right) moves right and ``0`` (left) moves left, except in state
    ``1`` where they are reversed; left in state ``0`` stays put. All states look
    the same to a policy with state-independent features, so the best such policy
    takes right with a probability :math:`p` that maximises :meth:`start_value`:
    :math:`p^* = 2 - \sqrt 2 \approx 0.586`, value :math:`\approx -11.66`.

    Parameters
    ----------
    max_episode_steps : int, optional
    random_state : None, int, SeedSequence or Generator

    Complexity
    ----------
    ``step``: O(1).

    References
    ----------
    Sutton & Barto (2018), Example 13.1 and Figure 13.1.

    Examples
    --------
    >>> env = ShortCorridor()
    >>> round(env.start_value(2 - 2 ** 0.5), 2), round(env.start_value(0.5), 2)
    (-11.66, -12.0)
    """

    def __init__(self, max_episode_steps: int | None = None, random_state=None):
        P = [
            [((1.0, 0, -1.0, False),), ((1.0, 1, -1.0, False),)],
            [((1.0, 2, -1.0, False),), ((1.0, 0, -1.0, False),)],  # reversed
            [((1.0, 1, -1.0, False),), ((1.0, 3, -1.0, True),)],
            [((1.0, 3, 0.0, True),)] * 2,
        ]
        super().__init__(P, [1.0, 0.0, 0.0, 0.0], max_episode_steps, random_state)

    def start_value(self, p_right: float) -> float:
        """Exact value of the start state when taking right with probability ``p_right``."""
        p = check_unit(p_right, "p_right")
        if p == 0.0:
            return float("-inf")  # never reaches the goal
        q = 1.0 - p
        # v0 = -1 + p v1 + q v0;  v1 = -1 + q v2 + p v0;  v2 = -1 + q v1 + p * 0
        A = np.array([[1.0 - q, -p, 0.0], [-p, 1.0, -q], [0.0, -q, 1.0]])
        try:
            return float(np.linalg.solve(A, -np.ones(3))[0])
        except np.linalg.LinAlgError:  # p = 1: state 1 bounces back to 0 forever
            return float("-inf")


# ---------------------------------------------------------------------------
# Bandits
# ---------------------------------------------------------------------------
@register(
    domain="rl",
    tasks=("environment", "bandit"),
    complexity=Complexity(
        time="O(1)",
        space="O(k)",
        notes="per step O(1) (O(k) for the non-stationary random walk); k arms",
    ),
    references=(_SB + ", §2.3 and Exercise 2.5",),
    status="experimental",
)
class KArmedBandit(_EpisodeState):
    r"""The k-armed testbed: Gaussian rewards around hidden action values.

    Stationary (§2.3): :math:`q_*(a) \sim \mathcal N(\mu, 1)` drawn once, rewards
    :math:`R_t \sim \mathcal N(q_*(A_t), \sigma^2)`. Non-stationary (Exercise 2.5):
    all :math:`q_*(a)` start equal to :math:`\mu` and take independent
    :math:`\mathcal N(0, \text{walk\_std}^2)` steps after every pull.

    ``reset(seed=...)`` draws a new problem; ``reset()`` without a seed restarts the
    same problem (restoring the initial values of a non-stationary one). The
    observation is always ``0``; ``info`` carries ``"optimal_action"`` and a copy of
    ``"q_star"`` *before* the pull, for measuring regret. Episodes never terminate.

    Parameters
    ----------
    k : int, default=10
    stationary : bool, default=True
    mean : float, default=0.0
        :math:`\mu`.
    reward_std : float, default=1.0
        :math:`\sigma > 0`.
    walk_std : float, default=0.01
        Random-walk step of the non-stationary testbed.
    max_episode_steps : int, optional
    random_state : None, int, SeedSequence or Generator

    Complexity
    ----------
    ``step``: O(1) stationary, O(k) non-stationary.

    References
    ----------
    Sutton & Barto (2018), §2.3 (Figure 2.2), Exercise 2.5.

    Examples
    --------
    >>> env = KArmedBandit(k=3, random_state=0)
    >>> obs, info = env.reset()
    >>> obs, info["optimal_action"] == int(env.q_star.argmax())
    (0, True)
    >>> _, reward, terminated, truncated, _ = env.step(2)
    >>> isinstance(reward, float), terminated
    (True, False)
    """

    def __init__(
        self,
        k: int = 10,
        stationary: bool = True,
        mean: float = 0.0,
        reward_std: float = 1.0,
        walk_std: float = 0.01,
        max_episode_steps: int | None = None,
        random_state=None,
    ):
        self.k = check_int(k, "k")
        if not isinstance(stationary, (bool, np.bool_)):
            raise TypeError(f"stationary must be a bool, got {stationary!r}.")
        self.stationary = bool(stationary)
        if not is_real(mean) or not np.isfinite(mean):
            raise ValueError(f"mean must be finite, got {mean!r}.")
        self.mean = float(mean)
        self.reward_std = check_positive(reward_std, "reward_std")
        self.walk_std = check_positive(walk_std, "walk_std", allow_zero=True)
        self.observation_space = Discrete(1)
        self.action_space = Discrete(self.k)
        self._init_episode_state(random_state, max_episode_steps)
        self._draw_problem()

    def _draw_problem(self) -> None:
        if self.stationary:
            self._q0 = self._rng.normal(self.mean, 1.0, size=self.k)
        else:
            self._q0 = np.full(self.k, self.mean)
        self._q = self._q0.copy()

    @property
    def q_star(self) -> np.ndarray:
        """Current true action values (a copy)."""
        return self._q.copy()

    def reset(self, seed=None, options=None) -> tuple[int, dict]:
        if seed is not None:
            self._rng = check_random_state(seed)
            self._draw_problem()
        self._q = self._q0.copy()
        self._state, self._done, self._elapsed = 0, False, 0
        return 0, {"optimal_action": int(np.argmax(self._q)), "q_star": self._q.copy()}

    def step(self, action) -> tuple[int, float, bool, bool, dict]:
        self._check_step(action)
        a = int(action)
        info = {"optimal_action": int(np.argmax(self._q)), "q_star": self._q.copy()}
        reward = float(self._rng.normal(self._q[a], self.reward_std))
        if not self.stationary and self.walk_std > 0:
            self._q = self._q + self._rng.normal(0.0, self.walk_std, size=self.k)
        self._elapsed += 1
        truncated = self.max_episode_steps is not None and self._elapsed >= self.max_episode_steps
        self._done = truncated
        return 0, reward, False, truncated, info


@register(
    domain="rl",
    tasks=("environment", "bandit"),
    complexity=Complexity(time="O(1)", space="O(k)", notes="k arms"),
    references=("Thompson (1933), Biometrika 25", _SB + ", §2.3"),
    status="experimental",
)
class BernoulliBandit(_EpisodeState):
    r"""A k-armed bandit with rewards :math:`R_t \sim \text{Bernoulli}(p_{A_t})`.

    The observation is always ``0``; ``info`` holds ``"optimal_action"`` and a copy
    of the arm means ``"q_star"``. Episodes never terminate.

    Parameters
    ----------
    probs : array_like of shape (k,)
        Success probability of each arm, in [0, 1].
    max_episode_steps : int, optional
    random_state : None, int, SeedSequence or Generator

    Complexity
    ----------
    ``step``: O(1).

    References
    ----------
    Thompson (1933); Sutton & Barto (2018), §2.

    Examples
    --------
    >>> env = BernoulliBandit([0.0, 1.0], random_state=0)
    >>> _ = env.reset()
    >>> env.step(1)[1], env.step(0)[1]
    (1.0, 0.0)
    """

    def __init__(self, probs, max_episode_steps: int | None = None, random_state=None):
        p = np.asarray(probs, dtype=np.float64)
        if p.ndim != 1 or p.size < 1 or np.any(~(p >= 0)) or np.any(~(p <= 1)):
            raise ValueError("probs must be a non-empty 1D array of probabilities in [0, 1].")
        self._p = p.copy()
        self.k = p.size
        self.observation_space = Discrete(1)
        self.action_space = Discrete(self.k)
        self._init_episode_state(random_state, max_episode_steps)

    @property
    def q_star(self) -> np.ndarray:
        """Arm means (a copy)."""
        return self._p.copy()

    def reset(self, seed=None, options=None) -> tuple[int, dict]:
        if seed is not None:
            self._rng = check_random_state(seed)
        self._state, self._done, self._elapsed = 0, False, 0
        return 0, {"optimal_action": int(np.argmax(self._p)), "q_star": self._p.copy()}

    def step(self, action) -> tuple[int, float, bool, bool, dict]:
        self._check_step(action)
        info = {"optimal_action": int(np.argmax(self._p)), "q_star": self._p.copy()}
        reward = float(self._rng.random() < self._p[int(action)])
        self._elapsed += 1
        truncated = self.max_episode_steps is not None and self._elapsed >= self.max_episode_steps
        self._done = truncated
        return 0, reward, False, truncated, info
