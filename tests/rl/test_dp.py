"""Dynamic programming: book values, exact linear solves, PI/VI agreement, extremes."""

import numpy as np
import pytest

from lipipy.bench import estimate_complexity
from lipipy.core import ConvergenceWarning
from lipipy.rl import (
    CliffWalking,
    FrozenLake,
    GridWorld,
    RandomWalk,
    TabularMDP,
    optimal_actions,
    policy_evaluation,
    policy_iteration,
    value_iteration,
)

# Sutton & Barto, Example 4.1 / Figure 4.1: v_pi of the equiprobable random policy.
EXAMPLE_4_1_V = np.array(
    [
        [0.0, -14.0, -20.0, -22.0],
        [-14.0, -18.0, -20.0, -20.0],
        [-20.0, -20.0, -18.0, -14.0],
        [-22.0, -20.0, -14.0, 0.0],
    ]
)


def random_mdp(rng, S=6, A=3, p_terminate=0.1, n_outcomes=3):
    """A random episodic MDP: every transition ends the episode with p_terminate."""
    P = []
    for _s in range(S):
        row = []
        for _a in range(A):
            nxt = rng.choice(S, size=n_outcomes, replace=False)
            w = rng.dirichlet(np.ones(n_outcomes)) * (1 - p_terminate)
            outs = [
                (float(p), int(s2), float(rng.normal()), False)
                for p, s2 in zip(w, nxt, strict=True)
            ]
            outs.append((p_terminate, int(rng.integers(S)), float(rng.normal()), True))
            row.append(outs)
        P.append(row)
    return TabularMDP(P, np.full(S, 1.0 / S))


def linear_solve_values(mdp, pi, gamma):
    P_cont, R = mdp.transition_arrays()
    P_pi = np.einsum("sa,sat->st", pi, P_cont)
    r_pi = np.sum(pi * R, axis=1)
    return np.linalg.solve(np.eye(mdp.n_states) - gamma * P_pi, r_pi)


@pytest.mark.filterwarnings("error")
def test_example_4_1_random_policy_values():
    env = GridWorld.example_4_1()
    res = policy_evaluation(env, np.full((16, 4), 0.25), gamma=1.0, theta=1e-12)
    np.testing.assert_allclose(res.V.reshape(4, 4), EXAMPLE_4_1_V, atol=1e-9)
    assert res.converged and res.delta < 1e-12
    # q_pi(s, a) = -1 + v_pi(s') for the deterministic grid
    assert res.Q[1, 3] == pytest.approx(-1.0)  # left from state 1 ends the episode
    assert res.Q[5, 0] == pytest.approx(-1.0 + EXAMPLE_4_1_V[0, 1])


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("gamma", [0.0, 0.5, 0.9, 0.99, 1.0])
def test_iterative_evaluation_equals_linear_solve(gamma):
    """Over 10 random episodic MDPs and random stochastic policies: agreement to 1e-10."""
    for seed in range(10):
        rng = np.random.default_rng(seed)
        mdp = random_mdp(rng)
        pi = rng.dirichlet(np.ones(mdp.n_actions), size=mdp.n_states)
        res = policy_evaluation(mdp, pi, gamma=gamma, theta=1e-13)
        np.testing.assert_allclose(res.V, linear_solve_values(mdp, pi, gamma), atol=1e-10)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("gamma", [0.0, 0.7, 0.95, 1.0])
def test_policy_iteration_and_value_iteration_agree(gamma):
    """Equal optimal values (1e-8) and greedy policies (lesson e: compare masks)."""
    for seed in range(10):
        mdp = random_mdp(np.random.default_rng(seed))
        pi_res = policy_iteration(mdp, gamma=gamma, theta=1e-13)
        vi_res = value_iteration(mdp, gamma=gamma, theta=1e-13)
        assert pi_res.converged and vi_res.converged
        np.testing.assert_allclose(pi_res.V, vi_res.V, atol=1e-8)
        np.testing.assert_allclose(pi_res.Q, vi_res.Q, atol=1e-8)
        mask = optimal_actions(vi_res.Q, atol=1e-7)
        S = mdp.n_states
        assert np.all(mask[np.arange(S), pi_res.policy])
        assert np.all(mask[np.arange(S), vi_res.policy])
        # the greedy policy's own value is optimal (Bellman optimality)
        v_pol = linear_solve_values(mdp, np.eye(mdp.n_actions)[pi_res.policy], gamma)
        np.testing.assert_allclose(v_pol, vi_res.V, atol=1e-8)


def test_gamma_zero_is_the_immediate_reward():
    mdp = random_mdp(np.random.default_rng(3))
    _, R = mdp.transition_arrays()
    res = value_iteration(mdp, gamma=0.0)
    np.testing.assert_allclose(res.V, R.max(axis=1))
    assert res.n_iterations <= 2


@pytest.mark.filterwarnings("error")
def test_episodic_tasks_with_gamma_one():
    gw = policy_iteration(GridWorld.example_4_1(), gamma=1.0)
    rows, cols = np.divmod(np.arange(16), 4)
    steps_to_goal = np.minimum(rows + cols, (3 - rows) + (3 - cols))
    np.testing.assert_allclose(gw.V, -steps_to_goal.astype(float))
    cliff = value_iteration(CliffWalking(), gamma=1.0)
    assert cliff.V[36] == pytest.approx(-13.0)
    # FrozenLake with gamma = 1: V is the probability of reaching the goal
    lake = value_iteration(FrozenLake(), gamma=1.0, theta=1e-13)
    assert 0 < lake.V[0] < 1 and lake.V[15] == 0.0
    assert lake.V[0] == pytest.approx(0.8235294, abs=1e-6)
    # random walk: evaluation reproduces the analytic values
    rw = RandomWalk(19)
    res = policy_evaluation(rw, np.ones((21, 1)), gamma=1.0, theta=1e-13)
    np.testing.assert_allclose(res.V, rw.true_values(), atol=1e-9)


def test_terminal_states_have_zero_value_and_deterministic_vs_random_dynamics():
    det = value_iteration(GridWorld((3, 3), terminal_cells=[(0, 0)]), gamma=1.0)
    rand_env = GridWorld((3, 3), terminal_cells=[(0, 0)], slip=1.0)
    rand = value_iteration(rand_env, gamma=1.0, theta=1e-13)
    assert det.V[0] == rand.V[0] == 0.0
    assert det.V[8] == -4.0
    # fully random moves: every action is optimal everywhere (lesson e)
    assert np.all(optimal_actions(rand.Q, atol=1e-8))
    pi_rand = policy_iteration(rand_env, gamma=1.0, theta=1e-13)
    np.testing.assert_allclose(pi_rand.V, rand.V, atol=1e-8)


def test_ties_do_not_make_policy_iteration_cycle():
    """Example 4.1 has many tied optimal actions; PI stops after a few improvements."""
    res = policy_iteration(GridWorld.example_4_1(), gamma=1.0)
    assert res.converged and res.n_iterations <= 5
    mask = optimal_actions(res.Q)
    assert mask.sum() > 16  # genuinely non-unique optimal policies
    assert np.all(mask[np.arange(16), res.policy])


def test_non_terminating_policy_warns():
    """'Always up' never terminates from the top row; with gamma = 1 values diverge."""
    env = GridWorld.example_4_1()
    with pytest.warns(ConvergenceWarning, match="did not converge"):
        res = policy_evaluation(env, np.zeros(16, dtype=int), gamma=1.0, max_iterations=500)
    assert not res.converged
    with pytest.warns(ConvergenceWarning):
        value_iteration(env, max_iterations=2)


def test_results_are_frozen_and_serialisable():
    res = value_iteration(GridWorld.example_4_1())
    with pytest.raises(ValueError):
        res.V[0] = 1.0
    with pytest.raises(AttributeError):
        res.n_iterations = 3
    d = res.to_dict()
    assert d["V"][1] == -1.0 and isinstance(d["converged"], bool)
    assert policy_evaluation(GridWorld.example_4_1(), np.full((16, 4), 0.25)).to_dict()["Q"]


def test_bad_inputs():
    env = GridWorld.example_4_1()
    with pytest.raises(TypeError, match="TabularMDP"):
        value_iteration(object())
    for kwargs, match in [
        ({"gamma": 1.5}, "gamma"),
        ({"gamma": -0.1}, "gamma"),
        ({"theta": 0.0}, "theta"),
        ({"max_iterations": 0}, "max_iterations"),
    ]:
        with pytest.raises(ValueError, match=match):
            value_iteration(env, **kwargs)
        with pytest.raises(ValueError, match=match):
            policy_iteration(env, **kwargs)
    with pytest.raises(ValueError, match="shape"):
        policy_evaluation(env, np.full((16, 3), 1 / 3))
    with pytest.raises(ValueError, match="probability"):
        policy_evaluation(env, np.full((16, 4), 0.3))
    with pytest.raises(ValueError, match="deterministic"):
        policy_evaluation(env, np.full(16, 4))
    with pytest.raises(ValueError):
        optimal_actions([1.0, 2.0])


@pytest.mark.complexity
@pytest.mark.filterwarnings("ignore::lipipy.core.ConvergenceWarning")
def test_value_iteration_sweep_time_is_quadratic_in_states():
    def make(S, rng):
        return random_mdp(rng, S=S, A=4, n_outcomes=S // 2)

    est = estimate_complexity(
        lambda mdp: value_iteration(mdp, gamma=0.9, theta=1e-300, max_iterations=200),
        make,
        sizes=(200, 300, 450, 675),
        repeats=2,
    )
    # dense (S, A, S) @ (S,) products: O(S^2 A) per sweep; model building also O(S^2 A)
    assert 1.6 < est.time_exponent < 2.4
