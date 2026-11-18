r"""Local search and metaheuristics (maximisation).

Problems
--------
A :class:`LocalSearchProblem` provides ``random_state(rng)`` (a random initial
state), ``neighbors(state)`` (a finite list), ``random_neighbor(state, rng)`` and
``value(state)`` (to **maximise**); ``is_goal(state)`` is optional. Reference
problems: :class:`NQueensProblem` (value = minus the number of attacking pairs, goal
at 0) and :class:`ContinuousProblem` (value = :math:`-f(x)` for a function to
minimise on a box, moves of :math:`\pm` ``step`` along one coordinate).

Hill climbing
-------------
* ``"steepest"``: move to the best neighbour while it is strictly better (ties
  between equally good neighbours are broken uniformly at random); stop at a local
  maximum.
* ``"first_choice"``: sample random neighbours and move to the first strictly better
  one; stop after ``max_tries`` failures in a row.
* ``restarts``: run again from new random states until a goal is reached or the
  restarts are used up; the best state over all runs is returned.

Simulated annealing
-------------------
At step :math:`t` with temperature :math:`T = \text{schedule}(t)`, propose a random
neighbour with value change :math:`\Delta`; accept it if :math:`\Delta > 0`, otherwise
with probability :math:`e^{\Delta/T}`. Stops when :math:`T \le 0`, after
``max_steps`` or at a goal; returns the best state seen.

Genetic algorithm
-----------------
Fixed-length genomes over a discrete ``gene_pool`` or a continuous box ``bounds``.
Each generation: keep the ``elitism`` best individuals unchanged, then fill the
population with children of parents chosen by **tournament selection** (best of
``tournament_size`` random individuals), combined by **one-point crossover** with
probability ``crossover_rate`` and **mutated** gene by gene with probability
``mutation_rate`` (a random pool value, or Gaussian noise of standard deviation
``mutation_scale`` times the box width, clipped to the box).

All three are reproducible: the same ``random_state`` gives the same result.

Budgets measured in the tests (every seed 0..49 unless stated): 8-Queens is solved
by steepest hill climbing with at most 100 restarts (at most 48 were used), by
first-choice hill climbing with at most 100 restarts (at most 25 used) and by
simulated annealing with ``exp_schedule(2.0, 0.999, 20000)`` (at most 3395
evaluations). Steepest hill climbing on a convex separable function stops on a
point whose coordinates are within ``step / 2`` of the minimiser (the sphere
:math:`\sum x_i^2` in 3-D with ``step = 0.01`` ends with :math:`f \le 7.5\times
10^{-5}`); simulated annealing reaches :math:`f \le 10^{-4}` on the same sphere (seeds
0..19); the genetic algorithm reaches :math:`f \le 2\times 10^{-4}` on the 5-D
sphere (seeds 0..9) and on 2-D Rastrigin for 19 of the seeds 0..19.

References
----------
S. Russell & P. Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed. (2020),
§4.1. S. Kirkpatrick, C. D. Gelatt & M. P. Vecchi, "Optimization by simulated
annealing", *Science* 220 (1983) 671-680. J. H. Holland, *Adaptation in Natural and
Artificial Systems* (1975). D. E. Goldberg, *Genetic Algorithms in Search,
Optimization and Machine Learning* (1989).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike

from lipipy.ai._common import check_callable, check_float, check_int, jsonable
from lipipy.core.complexity import Complexity
from lipipy.core.random import RandomState, check_random_state
from lipipy.core.registry import register

__all__ = [
    "ContinuousProblem",
    "GAResult",
    "LocalSearchProblem",
    "LocalSearchResult",
    "NQueensProblem",
    "exp_schedule",
    "genetic_algorithm",
    "hill_climbing",
    "simulated_annealing",
]

_AIMA = "Russell & Norvig, Artificial Intelligence: A Modern Approach, 4th ed. (2020), §4.1"
_SA = "Kirkpatrick, Gelatt & Vecchi, Science 220 (1983) 671-680"
_GA = "Holland, Adaptation in Natural and Artificial Systems (1975)"
_GOLDBERG = "Goldberg, Genetic Algorithms in Search, Optimization and Machine Learning (1989)"


@runtime_checkable
class LocalSearchProblem(Protocol):
    """Interface of a local-search problem (structural typing; ``is_goal`` optional)."""

    def random_state(self, rng: np.random.Generator) -> Any: ...

    def neighbors(self, state: Any) -> Sequence[Any]: ...

    def random_neighbor(self, state: Any, rng: np.random.Generator) -> Any: ...

    def value(self, state: Any) -> float: ...


class NQueensProblem:
    """N-Queens as local search: ``state[c]`` is the row of the queen in column ``c``.

    ``value`` is minus the number of attacking pairs (goal: 0). Neighbours move one
    queen within its column (:math:`n(n-1)` of them).

    Examples
    --------
    >>> p = NQueensProblem(4)
    >>> p.value((1, 3, 0, 2)), p.is_goal((1, 3, 0, 2))
    (0.0, True)
    """

    def __init__(self, n: int = 8) -> None:
        self.n = check_int(n, "n", minimum=1)

    def random_state(self, rng: np.random.Generator) -> tuple[int, ...]:
        return tuple(int(r) for r in rng.integers(0, self.n, size=self.n))

    def neighbors(self, state: tuple[int, ...]) -> list[tuple[int, ...]]:
        out = []
        for c in range(self.n):
            for r in range(self.n):
                if r != state[c]:
                    out.append(state[:c] + (r,) + state[c + 1 :])
        return out

    def random_neighbor(self, state: tuple[int, ...], rng: np.random.Generator) -> tuple:
        if self.n == 1:
            return state
        c = int(rng.integers(self.n))
        r = int(rng.integers(self.n - 1))
        r = r + 1 if r >= state[c] else r
        return state[:c] + (r,) + state[c + 1 :]

    def value(self, state: tuple[int, ...]) -> float:
        attacks = 0
        for a in range(self.n):
            for b in range(a + 1, self.n):
                if state[a] == state[b] or abs(state[a] - state[b]) == b - a:
                    attacks += 1
        return float(-attacks)  # int negation: no -0.0

    def is_goal(self, state: tuple[int, ...]) -> bool:
        return self.value(state) == 0.0


class ContinuousProblem:
    """Minimise ``f`` on a box by local moves; ``value(x) = -f(x)``.

    Parameters
    ----------
    f : callable ``f(x: ndarray of shape (d,)) -> float``
    bounds : array_like, shape (d, 2)
        Finite box ``low < high``; moves are clipped to it.
    step : float
        Neighbours of ``x`` are :math:`x \\pm` ``step`` along one coordinate
        (:math:`2d` of them); a random neighbour adds Gaussian noise of standard
        deviation ``step``.
    target : float, optional
        ``is_goal(x)`` is ``f(x) <= target`` (never a goal when None).

    Notes
    -----
    States are tuples of floats (hashable, exact to compare).

    Examples
    --------
    >>> p = ContinuousProblem(lambda x: float(x @ x), [(-1, 1), (-1, 1)], step=0.5)
    >>> p.value((0.5, 0.0)), len(p.neighbors((0.0, 0.0)))
    (-0.25, 4)
    """

    def __init__(
        self,
        f: Callable[[np.ndarray], float],
        bounds: ArrayLike,
        step: float = 0.1,
        target: float | None = None,
    ) -> None:
        check_callable(f, "f")
        box = np.asarray(bounds, dtype=np.float64)
        if box.ndim != 2 or box.shape[1] != 2 or box.shape[0] == 0:
            raise ValueError(f"bounds must have shape (d, 2), got {box.shape}")
        if not np.all(np.isfinite(box)) or not np.all(box[:, 0] < box[:, 1]):
            raise ValueError("bounds must be finite with low < high")
        self.f = f
        self.low, self.high = box[:, 0].copy(), box[:, 1].copy()
        self.step = check_float(step, "step", low=0.0, low_open=True)
        self.target = None if target is None else check_float(target, "target")

    def _clip(self, x: np.ndarray) -> tuple[float, ...]:
        return tuple(float(v) for v in np.clip(x, self.low, self.high))

    def random_state(self, rng: np.random.Generator) -> tuple[float, ...]:
        return self._clip(self.low + (self.high - self.low) * rng.random(self.low.size))

    def neighbors(self, state: tuple[float, ...]) -> list[tuple[float, ...]]:
        x = np.array(state)
        out = []
        for i in range(x.size):
            for sgn in (1.0, -1.0):
                y = x.copy()
                y[i] += sgn * self.step
                out.append(self._clip(y))
        return out

    def random_neighbor(self, state: tuple[float, ...], rng: np.random.Generator) -> tuple:
        return self._clip(np.array(state) + self.step * rng.standard_normal(len(state)))

    def value(self, state: tuple[float, ...]) -> float:
        v = float(self.f(np.array(state)))
        if not math.isfinite(v):
            raise ValueError(f"f returned a non-finite value at {state}")
        return -v

    def is_goal(self, state: tuple[float, ...]) -> bool:
        return self.target is not None and -self.value(state) <= self.target


@dataclass(frozen=True)
class LocalSearchResult:
    """Outcome of :func:`hill_climbing` and :func:`simulated_annealing`.

    Attributes
    ----------
    state : any
        Best state found.
    value : float
        Its value (higher is better).
    n_steps : int
        Moves made (accepted moves for annealing).
    n_evaluations : int
        Calls of ``problem.value``.
    n_restarts : int
        Restarts used (hill climbing).
    found : bool
        Whether ``state`` is a goal (False when the problem has no ``is_goal``).
    """

    state: Any
    value: float
    n_steps: int
    n_evaluations: int
    n_restarts: int
    found: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": jsonable(self.state),
            "value": self.value,
            "n_steps": self.n_steps,
            "n_evaluations": self.n_evaluations,
            "n_restarts": self.n_restarts,
            "found": self.found,
        }


@dataclass(frozen=True)
class GAResult:
    """Outcome of :func:`genetic_algorithm`.

    Attributes
    ----------
    best : tuple
        Best genome found.
    fitness : float
        Its fitness.
    history : tuple of float
        Best fitness found so far, after each generation (non-decreasing).
    n_evaluations : int
        Fitness evaluations.
    """

    best: tuple[Any, ...]
    fitness: float
    history: tuple[float, ...]
    n_evaluations: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "best": jsonable(self.best),
            "fitness": self.fitness,
            "history": list(self.history),
            "n_evaluations": self.n_evaluations,
        }


def _check_problem(problem: Any) -> None:
    if not isinstance(problem, LocalSearchProblem):
        raise TypeError("problem must provide random_state, neighbors, random_neighbor and value")


class _Counter:
    def __init__(self, problem: Any) -> None:
        self.problem = problem
        self.calls = 0

    def __call__(self, state: Any) -> float:
        self.calls += 1
        v = float(self.problem.value(state))
        if math.isnan(v):
            raise ValueError(f"value returned NaN at {state!r}")
        return v


def _is_goal(problem: Any, state: Any) -> bool:
    fn = getattr(problem, "is_goal", None)
    return bool(fn(state)) if callable(fn) else False


# ---------------------------------------------------------------------------
# hill climbing
# ---------------------------------------------------------------------------
@register(
    domain="ai",
    tasks=("local-search", "optimization"),
    complexity=Complexity(
        time="O(R T k c)",
        space="O(k)",
        notes="R = restarts + 1, T = steps per run, k = neighbours per state "
        "(max_tries for first-choice), c = cost of one evaluation of value",
    ),
    references=(_AIMA,),
)
def hill_climbing(
    problem: LocalSearchProblem,
    variant: Literal["steepest", "first_choice"] = "steepest",
    restarts: int = 0,
    max_steps: int = 1000,
    max_tries: int = 100,
    random_state: RandomState = None,
    initial_state: Any = None,
) -> LocalSearchResult:
    r"""Hill climbing (steepest-ascent or first-choice) with optional random restarts.

    Parameters
    ----------
    problem : LocalSearchProblem
    variant : {"steepest", "first_choice"}, default="steepest"
    restarts : int, default=0
        Extra runs from new random states while no goal has been reached (>= 0).
    max_steps : int, default=1000
        Maximum moves per run (>= 0).
    max_tries : int, default=100
        First-choice: consecutive non-improving samples before stopping (>= 1).
    random_state : None, int, SeedSequence or Generator
    initial_state : optional
        Start of the first run (default: ``problem.random_state(rng)``).

    Returns
    -------
    LocalSearchResult
        The best state over all runs.

    Complexity
    ----------
    Time :math:`O(RTkc)` for R runs of at most T steps with k neighbours (or
    ``max_tries`` samples) of evaluation cost c; extra space :math:`O(k)`.

    References
    ----------
    Russell & Norvig (2020), §4.1.1.

    Examples
    --------
    >>> res = hill_climbing(NQueensProblem(8), restarts=100, random_state=0)
    >>> res.found, res.value
    (True, 0.0)
    """
    _check_problem(problem)
    if variant not in ("steepest", "first_choice"):
        raise ValueError(f"variant must be 'steepest' or 'first_choice', got {variant!r}")
    restarts = check_int(restarts, "restarts")
    max_steps = check_int(max_steps, "max_steps")
    max_tries = check_int(max_tries, "max_tries", minimum=1)
    rng = check_random_state(random_state)
    value = _Counter(problem)
    best_state, best_value, total_steps, used = None, -math.inf, 0, 0
    for run in range(restarts + 1):
        used = run
        state = (
            initial_state if (run == 0 and initial_state is not None) else problem.random_state(rng)
        )
        current = value(state)
        for _ in range(max_steps):
            if _is_goal(problem, state):
                break
            if variant == "steepest":
                nbrs = list(problem.neighbors(state))
                if not nbrs:
                    break
                vals = [value(s) for s in nbrs]
                top = max(vals)
                if top <= current:
                    break
                ties = [i for i, v in enumerate(vals) if v == top]
                state, current = nbrs[ties[int(rng.integers(len(ties)))]], top
            else:
                for _ in range(max_tries):
                    cand = problem.random_neighbor(state, rng)
                    v = value(cand)
                    if v > current:
                        state, current = cand, v
                        break
                else:
                    break
            total_steps += 1
        if current > best_value:
            best_state, best_value = state, current
        if _is_goal(problem, best_state):
            break
    return LocalSearchResult(
        best_state, best_value, total_steps, value.calls, used, _is_goal(problem, best_state)
    )


# ---------------------------------------------------------------------------
# simulated annealing
# ---------------------------------------------------------------------------
def exp_schedule(
    t0: float = 1.0, decay: float = 0.995, limit: int = 10_000
) -> Callable[[int], float]:
    r"""Exponential cooling :math:`T(t) = T_0\,\text{decay}^t` for :math:`t < \text{limit}`, then 0.

    Examples
    --------
    >>> s = exp_schedule(2.0, 0.5, limit=3)
    >>> [s(t) for t in range(4)]
    [2.0, 1.0, 0.5, 0]
    """
    t0 = check_float(t0, "t0", low=0.0, low_open=True)
    decay = check_float(decay, "decay", low=0.0, high=1.0, low_open=True)
    limit = check_int(limit, "limit")

    def schedule(t: int) -> float:
        return t0 * decay**t if t < limit else 0

    return schedule


@register(
    domain="ai",
    tasks=("local-search", "optimization", "metaheuristics"),
    complexity=Complexity(
        time="O(T c)",
        space="O(1)",
        notes="T = steps until the schedule reaches 0 (or max_steps), c = cost of one "
        "value evaluation plus one random_neighbor call; space excludes the states",
    ),
    references=(_SA, _AIMA + ", §4.1.2"),
)
def simulated_annealing(
    problem: LocalSearchProblem,
    schedule: Callable[[int], float] | None = None,
    max_steps: int = 100_000,
    random_state: RandomState = None,
    initial_state: Any = None,
) -> LocalSearchResult:
    r"""Simulated annealing: accept worse moves with probability :math:`e^{\Delta/T}`.

    Parameters
    ----------
    problem : LocalSearchProblem
    schedule : callable ``schedule(t) -> T``, optional
        Temperature at step t; the run stops when it is :math:`\le 0`. Default
        ``exp_schedule(1.0, 0.995, 10000)``.
    max_steps : int, default=100000
        Hard limit on the number of steps (>= 0).
    random_state : None, int, SeedSequence or Generator
    initial_state : optional
        Default ``problem.random_state(rng)``.

    Returns
    -------
    LocalSearchResult
        The best state seen (``n_steps`` counts accepted moves, ``n_restarts`` = 0).

    Complexity
    ----------
    Time :math:`O(Tc)` for T steps; extra space :math:`O(1)` besides two states.

    References
    ----------
    Kirkpatrick, Gelatt & Vecchi (1983); Russell & Norvig (2020), §4.1.2.

    Examples
    --------
    >>> res = simulated_annealing(NQueensProblem(8), exp_schedule(2.0, 0.999, 20_000),
    ...                           random_state=0)
    >>> res.found
    True
    """
    _check_problem(problem)
    if schedule is None:
        schedule = exp_schedule()
    check_callable(schedule, "schedule")
    max_steps = check_int(max_steps, "max_steps")
    rng = check_random_state(random_state)
    value = _Counter(problem)
    state = problem.random_state(rng) if initial_state is None else initial_state
    current = value(state)
    best_state, best_value = state, current
    accepted = 0
    for t in range(max_steps):
        if _is_goal(problem, best_state):
            break
        T = float(schedule(t))
        if not T > 0:
            break
        cand = problem.random_neighbor(state, rng)
        v = value(cand)
        delta = v - current
        if delta > 0 or rng.random() < math.exp(delta / T):
            state, current = cand, v
            accepted += 1
            if current > best_value:
                best_state, best_value = state, current
    return LocalSearchResult(
        best_state, best_value, accepted, value.calls, 0, _is_goal(problem, best_state)
    )


# ---------------------------------------------------------------------------
# genetic algorithm
# ---------------------------------------------------------------------------
@register(
    domain="ai",
    tasks=("local-search", "optimization", "metaheuristics", "evolutionary"),
    complexity=Complexity(
        time="O(T n (L + c))",
        space="O(n L)",
        notes="T = n_generations, n = population_size, L = genome length, "
        "c = cost of one fitness evaluation",
    ),
    references=(_GA, _GOLDBERG, _AIMA + ", §4.1.4"),
)
def genetic_algorithm(
    fitness: Callable[[tuple[Any, ...]], float],
    length: int,
    gene_pool: Sequence[Any] | None = None,
    bounds: ArrayLike | None = None,
    population_size: int = 100,
    n_generations: int = 200,
    crossover_rate: float = 0.9,
    mutation_rate: float | None = None,
    mutation_scale: float = 0.1,
    tournament_size: int = 3,
    elitism: int = 1,
    target: float | None = None,
    random_state: RandomState = None,
) -> GAResult:
    r"""Generational genetic algorithm maximising ``fitness`` over fixed-length genomes.

    Parameters
    ----------
    fitness : callable ``fitness(genome: tuple) -> float``
        Higher is better.
    length : int
        Genome length (>= 1).
    gene_pool : sequence, optional
        Allowed gene values (discrete genomes). Exactly one of ``gene_pool`` and
        ``bounds`` must be given.
    bounds : array_like, shape (length, 2), optional
        Box for continuous genomes.
    population_size : int, default=100
        (>= 2).
    n_generations : int, default=200
        (>= 0).
    crossover_rate : float, default=0.9
        Probability that two parents are recombined (one-point crossover).
    mutation_rate : float, optional
        Per-gene mutation probability; default ``1 / length``.
    mutation_scale : float, default=0.1
        Continuous genomes: Gaussian mutation standard deviation as a fraction of the
        box width.
    tournament_size : int, default=3
        (1 .. population_size).
    elitism : int, default=1
        Best individuals copied unchanged into the next generation (0 .. population_size).
    target : float, optional
        Stop as soon as the best fitness reaches ``target``.
    random_state : None, int, SeedSequence or Generator

    Returns
    -------
    GAResult

    Complexity
    ----------
    Time :math:`O(Tn(L + c))` for T generations of n genomes of length L with fitness
    cost c; space :math:`O(nL)`.

    References
    ----------
    Holland (1975); Goldberg (1989); Russell & Norvig (2020), §4.1.4.

    Examples
    --------
    >>> res = genetic_algorithm(lambda g: float(sum(g)), length=20, gene_pool=(0, 1),
    ...                         target=20.0, random_state=0)
    >>> res.fitness, res.best == (1,) * 20
    (20.0, True)
    """
    check_callable(fitness, "fitness")
    length = check_int(length, "length", minimum=1)
    if (gene_pool is None) == (bounds is None):
        raise ValueError("give exactly one of gene_pool (discrete) and bounds (continuous)")
    if gene_pool is not None:
        pool = list(gene_pool)
        if not pool:
            raise ValueError("gene_pool must be non-empty")
        low = high = None
    else:
        box = np.asarray(bounds, dtype=np.float64)
        if (
            box.shape != (length, 2)
            or not np.all(np.isfinite(box))
            or not np.all(box[:, 0] < box[:, 1])
        ):
            raise ValueError(f"bounds must be finite with shape ({length}, 2) and low < high")
        low, high = box[:, 0], box[:, 1]
    n = check_int(population_size, "population_size", minimum=2)
    n_generations = check_int(n_generations, "n_generations")
    crossover_rate = check_float(crossover_rate, "crossover_rate", low=0.0, high=1.0)
    mutation_rate = (
        1.0 / length
        if mutation_rate is None
        else check_float(mutation_rate, "mutation_rate", low=0.0, high=1.0)
    )
    mutation_scale = check_float(mutation_scale, "mutation_scale", low=0.0)
    k = check_int(tournament_size, "tournament_size", minimum=1)
    if k > n:
        raise ValueError("tournament_size cannot exceed population_size")
    elitism = check_int(elitism, "elitism")
    if elitism > n:
        raise ValueError("elitism cannot exceed population_size")
    target = None if target is None else check_float(target, "target")
    rng = check_random_state(random_state)
    evaluations = 0

    def evaluate(genome: tuple) -> float:
        nonlocal evaluations
        evaluations += 1
        v = float(fitness(genome))
        if math.isnan(v):
            raise ValueError(f"fitness returned NaN for {genome!r}")
        return v

    def random_genome() -> tuple:
        if low is None:
            return tuple(pool[int(i)] for i in rng.integers(len(pool), size=length))
        return tuple(float(v) for v in low + (high - low) * rng.random(length))

    def mutate(genome: tuple) -> tuple:
        mask = rng.random(length) < mutation_rate
        if not mask.any():
            return genome
        g = list(genome)
        for i in np.flatnonzero(mask):
            if low is None:
                g[i] = pool[int(rng.integers(len(pool)))]
            else:
                width = high[i] - low[i]
                g[i] = float(np.clip(g[i] + mutation_scale * width * rng.standard_normal(),
                                     low[i], high[i]))  # fmt: skip
        return tuple(g)

    def tournament(scores: list[float]) -> int:
        entrants = rng.choice(n, size=k, replace=False)
        return int(max(entrants, key=lambda i: (scores[i], -i)))

    population = [random_genome() for _ in range(n)]
    scores = [evaluate(g) for g in population]
    best_i = int(np.argmax(scores))
    best, best_fit = population[best_i], scores[best_i]
    history: list[float] = []
    for _ in range(n_generations):
        if target is not None and best_fit >= target:
            break
        order = sorted(range(n), key=lambda i: -scores[i])
        children = [population[i] for i in order[:elitism]]
        child_scores = [scores[i] for i in order[:elitism]]
        while len(children) < n:
            a, b = population[tournament(scores)], population[tournament(scores)]
            if length > 1 and rng.random() < crossover_rate:
                cut = int(rng.integers(1, length))
                a, b = a[:cut] + b[cut:], b[:cut] + a[cut:]
            for child in (mutate(a), mutate(b)):
                if len(children) < n:
                    children.append(child)
                    child_scores.append(evaluate(child))
        population, scores = children, child_scores
        best_i = int(np.argmax(scores))
        if scores[best_i] > best_fit:
            best, best_fit = population[best_i], scores[best_i]
        history.append(best_fit)
    return GAResult(best, best_fit, tuple(history), evaluations)
