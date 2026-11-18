"""lipipy.ai.local: hill climbing, simulated annealing and the genetic algorithm."""

import math

import numpy as np
import pytest

from lipipy.ai import (
    ContinuousProblem,
    GAResult,
    LocalSearchResult,
    NQueensProblem,
    exp_schedule,
    genetic_algorithm,
    hill_climbing,
    simulated_annealing,
)


def sphere(x):
    return float(x @ x)


def rastrigin(x):
    return float(10 * len(x) + np.sum(x * x - 10 * np.cos(2 * np.pi * x)))


def is_queens_solution(state):
    n = len(state)
    return all(
        state[a] != state[b] and abs(state[a] - state[b]) != b - a
        for a in range(n)
        for b in range(a + 1, n)
    )


# ---------------------------------------------------------------------------
# 8-Queens
# ---------------------------------------------------------------------------
@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("variant, used", [("steepest", 48), ("first_choice", 25)])
def test_hill_climbing_with_restarts_solves_eight_queens(variant, used):
    p = NQueensProblem(8)
    worst = 0
    for seed in range(50):
        res = hill_climbing(p, variant=variant, restarts=100, random_state=seed)
        assert res.found and res.value == 0.0 and is_queens_solution(res.state)
        worst = max(worst, res.n_restarts)
    assert worst == used  # measured maximum over the 50 seeds


@pytest.mark.filterwarnings("error")
def test_simulated_annealing_solves_eight_queens():
    p = NQueensProblem(8)
    worst = 0
    for seed in range(50):
        res = simulated_annealing(p, exp_schedule(2.0, 0.999, 20_000), random_state=seed)
        assert res.found and is_queens_solution(res.state)
        worst = max(worst, res.n_evaluations)
    assert worst <= 3395


def test_single_hill_climb_often_gets_stuck():
    p = NQueensProblem(8)
    stuck = sum(not hill_climbing(p, random_state=s).found for s in range(50))
    assert stuck > 25  # a single steepest run succeeds only ~14% of the time


def test_queens_problem_basics():
    p = NQueensProblem(5)
    rng = np.random.default_rng(0)
    s = p.random_state(rng)
    assert len(s) == 5 and len(p.neighbors(s)) == 20
    for _ in range(20):
        t = p.random_neighbor(s, rng)
        assert sum(a != b for a, b in zip(s, t, strict=True)) == 1
    assert NQueensProblem(1).random_neighbor((0,), rng) == (0,)
    assert NQueensProblem(1).is_goal((0,))


# ---------------------------------------------------------------------------
# continuous test functions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(10))
def test_steepest_on_sphere_ends_within_half_a_step(seed):
    p = ContinuousProblem(sphere, [(-5, 5)] * 3, step=0.01)
    res = hill_climbing(p, max_steps=5000, random_state=seed)
    x = np.array(res.state)
    assert np.all(np.abs(x) <= 0.005 + 1e-12)
    assert -res.value <= 3 * 0.01**2 / 4


def test_simulated_annealing_on_sphere():
    p = ContinuousProblem(sphere, [(-5, 5)] * 3, step=0.05)
    for seed in range(20):
        res = simulated_annealing(p, exp_schedule(1.0, 0.998, 10_000), random_state=seed)
        assert -res.value <= 1e-4


def test_simulated_annealing_stops_at_goal_and_schedule_end():
    p = ContinuousProblem(sphere, [(-5, 5)] * 2, step=0.1, target=0.5)
    res = simulated_annealing(p, exp_schedule(1.0, 0.99, 5000), random_state=0)
    assert res.found and -res.value <= 0.5
    short = simulated_annealing(p, exp_schedule(1.0, 0.99, 3), random_state=0)
    assert short.n_evaluations == 4  # initial + 3 proposals


def test_genetic_algorithm_on_sphere():
    for seed in range(10):
        res = genetic_algorithm(
            lambda g: -sphere(np.array(g)), 5, bounds=[(-5, 5)] * 5, n_generations=200,
            random_state=seed,
        )  # fmt: skip
        assert -res.fitness <= 2e-4
        assert list(res.history) == sorted(res.history)


def test_genetic_algorithm_on_rastrigin():
    hits = 0
    for seed in range(20):
        res = genetic_algorithm(
            lambda g: -rastrigin(np.array(g)),
            2,
            bounds=[(-5.12, 5.12)] * 2,
            n_generations=150,
            mutation_rate=0.5,
            mutation_scale=0.02,
            random_state=seed,
        )
        hits += -res.fitness <= 2e-4
    assert hits >= 19


def test_genetic_algorithm_discrete_onemax_and_target():
    res = genetic_algorithm(lambda g: float(sum(g)), 30, gene_pool=(0, 1), target=30.0,
                            random_state=1)  # fmt: skip
    assert res.best == (1,) * 30 and res.fitness == 30.0
    assert len(res.history) < 200  # stopped early at the target
    assert isinstance(res, GAResult) and res.to_dict()["fitness"] == 30.0


# ---------------------------------------------------------------------------
# reproducibility and edge cases
# ---------------------------------------------------------------------------
def test_identical_seeds_identical_results():
    p = NQueensProblem(10)
    assert hill_climbing(p, restarts=5, random_state=4) == hill_climbing(
        p, restarts=5, random_state=4
    )
    assert simulated_annealing(p, random_state=4) == simulated_annealing(p, random_state=4)
    kw = dict(length=8, gene_pool=tuple(range(8)), n_generations=20, random_state=4)
    f = lambda g: NQueensProblem(8).value(g)  # noqa: E731
    assert genetic_algorithm(f, **kw) == genetic_algorithm(f, **kw)
    a = hill_climbing(p, restarts=5, random_state=np.random.default_rng(4))
    assert a == hill_climbing(p, restarts=5, random_state=4)


def test_already_solved_and_zero_budgets():
    p = NQueensProblem(4)
    sol = (1, 3, 0, 2)
    res = hill_climbing(p, initial_state=sol, random_state=0)
    assert res.found and res.n_steps == 0 and res.state == sol
    res = simulated_annealing(p, initial_state=sol, random_state=0)
    assert res.found and res.n_evaluations == 1
    res = hill_climbing(p, max_steps=0, random_state=0)
    assert res.n_steps == 0 and res.n_evaluations == 1
    res = genetic_algorithm(lambda g: 1.0, 3, gene_pool=(0,), n_generations=0, random_state=0)
    assert res.history == () and res.n_evaluations == 100


def test_unsolvable_problem_reports_not_found():
    p = NQueensProblem(3)  # no solution exists
    res = hill_climbing(p, restarts=10, random_state=0)
    assert not res.found and res.value < 0 and res.n_restarts == 10
    res = simulated_annealing(p, exp_schedule(1.0, 0.9, 200), random_state=0)
    assert not res.found


def test_result_dict():
    res = hill_climbing(NQueensProblem(4), restarts=20, random_state=0)
    d = res.to_dict()
    assert isinstance(res, LocalSearchResult) and d["found"] and len(d["state"]) == 4


def test_validation():
    p = NQueensProblem(4)
    with pytest.raises(TypeError):
        hill_climbing(object())
    with pytest.raises(ValueError):
        hill_climbing(p, variant="tabu")
    with pytest.raises(ValueError):
        hill_climbing(p, restarts=-1)
    with pytest.raises(ValueError):
        hill_climbing(p, max_tries=0)
    with pytest.raises(TypeError):
        simulated_annealing(p, schedule=3)
    with pytest.raises(ValueError):
        exp_schedule(decay=1.5)
    with pytest.raises(ValueError):
        exp_schedule(t0=0.0)
    with pytest.raises(ValueError):
        ContinuousProblem(sphere, [(1, 0)])
    with pytest.raises(ValueError):
        ContinuousProblem(sphere, [(0, 1)], step=0.0)
    with pytest.raises(ValueError):
        ContinuousProblem(lambda x: math.nan, [(0, 1)]).value((0.5,))
    f = lambda g: 0.0  # noqa: E731
    with pytest.raises(ValueError):
        genetic_algorithm(f, 3)
    with pytest.raises(ValueError):
        genetic_algorithm(f, 3, gene_pool=(0, 1), bounds=[(0, 1)] * 3)
    with pytest.raises(ValueError):
        genetic_algorithm(f, 3, gene_pool=())
    with pytest.raises(ValueError):
        genetic_algorithm(f, 3, bounds=[(0, 1)] * 2)
    with pytest.raises(ValueError):
        genetic_algorithm(f, 3, gene_pool=(0, 1), tournament_size=200)
    with pytest.raises(ValueError):
        genetic_algorithm(f, 3, gene_pool=(0, 1), elitism=101)
    with pytest.raises(ValueError):
        genetic_algorithm(f, 3, gene_pool=(0, 1), population_size=1)
    with pytest.raises(ValueError):
        genetic_algorithm(lambda g: math.nan, 3, gene_pool=(0, 1))
