"""Tabular TD and Monte Carlo: book results over many seeds, identities, errors."""

import numpy as np
import pytest

from lipipy.rl import (
    CliffWalking,
    GridWorld,
    RandomWalk,
    TabularMDP,
    double_q_learning,
    expected_sarsa,
    mc_control,
    n_step_td_prediction,
    policy_evaluation,
    q_learning,
    sarsa,
    td0_prediction,
    td_lambda_prediction,
    value_iteration,
)
from tests.rl.test_dp import random_mdp

UNIFORM_5 = np.ones((7, 1))
UNIFORM_19 = np.ones((21, 1))
CONTROL = [sarsa, expected_sarsa, q_learning, double_q_learning]


def greedy_path(Q, max_steps=60):
    """Return, cells visited and whether the goal was reached following argmax Q."""
    env = CliffWalking()
    s, _ = env.reset(seed=0)
    cells, total, term = [s], 0.0, False
    for _ in range(max_steps):
        s, r, term, _, _ = env.step(int(np.argmax(Q[s])))
        total += r
        cells.append(s)
        if term:
            break
    return total, cells, term


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
@pytest.mark.filterwarnings("error")
def test_td0_random_walk_rms_curves_match_figure_6_2():
    """Example 6.2 / Figure 6.2: V0 = 0.5, 100 runs of 100 episodes per step size.

    Measured mean RMS after episodes 1 / 10 / 100:
    alpha 0.05: 0.229 / 0.174 / 0.035; alpha 0.1: 0.222 / 0.127 / 0.053;
    alpha 0.15: 0.216 / 0.092 / 0.067. Asserted: every curve starts near the
    initial error 0.236 (within 0.03), falls by a factor > 3, larger steps learn
    faster early (episode 10) and plateau higher (episode 100).
    """
    env = RandomWalk(5)
    tv = env.true_values()
    curves = {}
    for alpha in (0.05, 0.1, 0.15):
        runs = [
            td0_prediction(
                env, UNIFORM_5, 100, alpha=alpha, initial_value=0.5, true_values=tv, random_state=s
            ).rms_error
            for s in range(100)
        ]
        curves[alpha] = np.mean(runs, axis=0)
    initial_rms = np.sqrt(np.mean((0.5 - tv[1:6]) ** 2))
    assert initial_rms == pytest.approx(0.2357, abs=1e-4)
    for c in curves.values():
        assert abs(c[0] - initial_rms) < 0.03 and c[-1] < c[0] / 3
    assert curves[0.15][9] < curves[0.1][9] < curves[0.05][9]
    assert curves[0.05][-1] < curves[0.1][-1] < curves[0.15][-1]


def test_prediction_methods_converge_with_decaying_steps():
    """5-state walk, alpha_n = N(s)^-0.6, 2000 episodes, 4 seeds: RMS over the
    non-terminal states measured <= 0.051 for TD(0), 3-step TD and TD(0.8) with
    accumulating and replacing traces; asserted < 0.08."""
    env = RandomWalk(5)
    tv = env.true_values()

    def alpha(n):
        return n**-0.6

    for seed in range(4):
        for res in (
            td0_prediction(env, UNIFORM_5, 2000, alpha=alpha, random_state=seed),
            n_step_td_prediction(env, UNIFORM_5, 2000, n=3, alpha=alpha, random_state=seed),
            td_lambda_prediction(env, UNIFORM_5, 2000, lam=0.8, alpha=alpha, random_state=seed),
            td_lambda_prediction(
                env, UNIFORM_5, 2000, lam=0.8, trace="replacing", alpha=alpha, random_state=seed
            ),
        ):
            assert np.sqrt(np.mean((res.V[1:6] - tv[1:6]) ** 2)) < 0.08
            assert res.V[0] == res.V[6] == 0.0


def test_identities_between_prediction_methods():
    """Same random_state => same episodes: n = 1 and lambda = 0 are exactly TD(0)."""
    env = RandomWalk(19)
    for seed in range(5):
        base = td0_prediction(env, UNIFORM_19, 40, alpha=0.2, random_state=seed)
        one = n_step_td_prediction(env, UNIFORM_19, 40, n=1, alpha=0.2, random_state=seed)
        lam0 = td_lambda_prediction(env, UNIFORM_19, 40, lam=0.0, alpha=0.2, random_state=seed)
        rep0 = td_lambda_prediction(
            env, UNIFORM_19, 40, lam=0.0, trace="replacing", alpha=0.2, random_state=seed
        )
        for other in (one, lam0, rep0):
            np.testing.assert_allclose(other.V, base.V, atol=1e-12)
            assert other.n_steps == base.n_steps


def test_n_step_with_large_n_is_monte_carlo():
    """n >= episode length and alpha = 1: every visited state takes the episode's return.

    With gamma = 1 and a single reward at the exit, all returns of an episode equal
    that reward G in {0, 1}; unvisited states keep the initial 0.5.
    """
    env = RandomWalk(5)
    for seed in range(10):
        res = n_step_td_prediction(
            env, UNIFORM_5, 1, n=10_000, alpha=1.0, initial_value=0.5, random_state=seed
        )
        G = res.V[3]  # the start state is always visited
        assert G in (0.0, 1.0)
        assert set(res.V[1:6]) <= {G, 0.5}


def test_gamma_zero_prediction_learns_immediate_reward():
    env = RandomWalk(5)
    res = td0_prediction(env, UNIFORM_5, 2000, alpha=lambda n: 1 / n, gamma=0.0, random_state=0)
    # only state 5 has a reward (+1 with probability 1/2 when stepping right)
    np.testing.assert_allclose(res.V[1:5], 0.0)
    assert res.V[5] == pytest.approx(0.5, abs=0.06)


def test_truncated_episodes_bootstrap():
    """A time limit is not termination: the last state's value is still used."""
    env = RandomWalk(5, max_episode_steps=1)
    res = td0_prediction(env, UNIFORM_5, 1, alpha=1.0, initial_value=0.5, random_state=0)
    assert res.n_steps == 1
    assert res.V[3] == 0.5  # target 0 + V(neighbour) = 0.5
    nres = n_step_td_prediction(
        env, UNIFORM_5, 1, n=3, alpha=1.0, initial_value=0.5, random_state=0
    )
    assert nres.V[3] == 0.5


# ---------------------------------------------------------------------------
# Control
# ---------------------------------------------------------------------------
def test_cliff_walking_sarsa_safe_path_q_learning_optimal_path():
    """Example 6.6 (alpha 0.2, epsilon 0.1, 1000 episodes, 20 seeds).

    Q-learning's greedy path is the optimal cliff-edge path (return -13) for all 20
    seeds (measured 20/20). SARSA's greedy path reaches the goal without touching
    the cliff edge (row 2, columns 1-10) for >= 18 seeds (measured 19/20; with a
    constant step size a few rarely visited states can still form a greedy loop).
    Online, SARSA's episodes are better: mean return over the last 100 episodes,
    averaged over the first 5 seeds, measured -21.3 (SARSA) vs -46.7 (Q-learning);
    asserted gap > 15.
    """
    safe, optimal = 0, 0
    online_sarsa, online_q = [], []
    for seed in range(20):
        q = q_learning(CliffWalking(), 1000, alpha=0.2, epsilon=0.1, random_state=seed)
        total, cells, term = greedy_path(q.Q)
        optimal += term and total == -13.0
        s = sarsa(CliffWalking(), 1000, alpha=0.2, epsilon=0.1, random_state=seed)
        total, cells, term = greedy_path(s.Q)
        on_edge = any(c // 12 == 2 and 1 <= c % 12 <= 10 for c in cells)
        safe += term and not on_edge and total < -13.0
        if seed < 5:
            online_sarsa.append(s.episode_returns[-100:].mean())
            online_q.append(q.episode_returns[-100:].mean())
    assert optimal == 20 and safe >= 18
    assert np.mean(online_sarsa) > np.mean(online_q) + 15


@pytest.mark.filterwarnings("error")
def test_q_learning_converges_to_value_iteration_q_star():
    """Random episodic MDPs (S=5, A=2, termination prob 0.2, gamma 0.9), uniform
    behaviour (epsilon=1), alpha_n = N(s,a)^-0.8: max |Q - Q*| / max(1, max|Q*|)
    measured <= 0.039 after 30000 episodes on 6 seeds; asserted < 0.06 on 4 seeds,
    and the error after 30000 episodes is below the error after 1000."""
    for seed in range(4):
        mdp = random_mdp(np.random.default_rng(seed), S=5, A=2, p_terminate=0.2)
        q_star = value_iteration(mdp, gamma=0.9, theta=1e-13).Q
        scale = max(1.0, np.abs(q_star).max())
        errors = []
        for episodes in (1000, 30000):
            res = q_learning(
                mdp, episodes, alpha=lambda n: n**-0.8, epsilon=1.0, gamma=0.9, random_state=seed
            )
            errors.append(np.abs(res.Q - q_star).max() / scale)
        assert errors[1] < 0.06 and errors[1] < errors[0]


def _maximization_bias_mdp():
    """Example 6.7: from A, 'right' ends with 0, 'left' goes to B; every action of B
    ends with reward -0.1 +- 1 (mean -0.1, variance 1). A has 10 actions too (the
    extra ones behave like 'right') so that every state has the same action set."""
    b_out = ((0.5, 2, 0.9, True), (0.5, 2, -1.1, True))
    P = [
        [((1.0, 1, 0.0, False),)] + [((1.0, 2, 0.0, True),)] * 9,
        [b_out] * 10,
        [((1.0, 2, 0.0, True),)] * 10,
    ]
    return TabularMDP(P, [1.0, 0.0, 0.0])


def test_double_q_learning_reduces_maximization_bias():
    """100 runs x 300 episodes (alpha 0.1, epsilon 0.1). Fraction of 'left' from A
    over the first 20 episodes: measured 0.28 (Q-learning) vs 0.11 (Double Q);
    asserted gap > 0.1. The optimal fraction under epsilon-greedy is 0.01."""
    mdp = _maximization_bias_mdp()

    def left_rate(fn):
        left = [
            fn(mdp, 300, alpha=0.1, epsilon=0.1, random_state=s).episode_lengths[:20] == 2
            for s in range(100)
        ]
        return np.mean(left)

    assert left_rate(q_learning) > left_rate(double_q_learning) + 0.1


@pytest.mark.filterwarnings("error")
def test_monte_carlo_control_approaches_the_best_epsilon_soft_policy():
    """On-policy MC control converges to the best epsilon-soft policy, not the best
    deterministic one, and its greedy actions are noisy where actions nearly tie
    (lesson e), so compare values: the exact value of the learned epsilon-greedy
    policy, averaged over the start distribution of the 4x4 grid (gamma 1,
    epsilon 0.1, 30000 episodes, 4 seeds), is within 0.3 of the optimal epsilon-soft
    value computed by value iteration (measured gaps 0.002-0.155)."""
    env = GridWorld.example_4_1()
    eps = 0.1
    P, R = env.transition_arrays()
    # epsilon-soft MDP: each action is replaced by the epsilon-greedy mixture
    P_soft = (1 - eps) * P + eps * P.mean(axis=1, keepdims=True)
    R_soft = (1 - eps) * R + eps * R.mean(axis=1, keepdims=True)
    V_soft = np.zeros(16)
    for _ in range(5000):
        V_soft = np.max(R_soft + P_soft @ V_soft, axis=1)
    mu = env.initial_state_distribution
    for seed in range(4):
        res = mc_control(env, 30000, epsilon=eps, random_state=seed)
        pi = np.full((16, 4), eps / 4)
        pi[np.arange(16), res.policy] += 1 - eps
        v = policy_evaluation(env, pi, theta=1e-12).V
        assert mu @ (V_soft - v) < 0.3
        assert np.all(v <= V_soft + 1e-9)  # nothing beats the optimal soft policy


@pytest.mark.parametrize("fn", CONTROL)
def test_control_reproducible_and_terminal_values_zero(fn):
    a = fn(CliffWalking(), 30, random_state=4)
    b = fn(CliffWalking(), 30, random_state=4)
    np.testing.assert_array_equal(a.Q, b.Q)
    np.testing.assert_array_equal(a.episode_returns, b.episode_returns)
    np.testing.assert_array_equal(a.Q[47], 0.0)
    with pytest.raises(ValueError):
        a.Q[0, 0] = 1.0
    assert a.to_dict()["episode_lengths"][0] > 0


@pytest.mark.parametrize("fn", [*CONTROL, mc_control])
def test_epsilon_and_gamma_extremes(fn):
    env = GridWorld.example_4_1()
    for kwargs in ({"epsilon": 0.0}, {"epsilon": 1.0}, {"gamma": 0.0}, {"gamma": 1.0}):
        res = fn(env, 20, random_state=0, max_steps_per_episode=200, **kwargs)
        assert np.all(np.isfinite(res.Q))


def test_time_limit_cuts_episodes():
    res = q_learning(CliffWalking(), 5, epsilon=1.0, max_steps_per_episode=7, random_state=0)
    assert np.all(res.episode_lengths <= 7)
    env = CliffWalking(max_episode_steps=4)
    res = sarsa(env, 5, epsilon=1.0, random_state=0)
    assert np.all(res.episode_lengths <= 4)


def test_bad_inputs():
    env = RandomWalk(5)
    with pytest.raises(TypeError, match="Discrete"):
        q_learning(object(), 10)
    for kwargs, match in [
        ({"alpha": 0.0}, "alpha"),
        ({"alpha": 1.5}, "alpha"),
        ({"gamma": 2.0}, "gamma"),
        ({"n_episodes": 0}, "n_episodes"),
        ({"max_steps_per_episode": 0}, "max_steps"),
        ({"initial_value": np.nan}, "initial_value"),
    ]:
        args = {"n_episodes": 5, **kwargs}
        with pytest.raises(ValueError, match=match):
            td0_prediction(env, UNIFORM_5, **args)
    with pytest.raises(ValueError, match="policy"):
        td0_prediction(env, np.ones((7, 2)), 5)
    with pytest.raises(ValueError, match="true_values"):
        td0_prediction(env, UNIFORM_5, 5, true_values=np.zeros(3))
    with pytest.raises(ValueError, match="lam"):
        td_lambda_prediction(env, UNIFORM_5, 5, lam=1.5)
    with pytest.raises(ValueError, match="trace"):
        td_lambda_prediction(env, UNIFORM_5, 5, trace="dutch")
    with pytest.raises(ValueError, match="n must"):
        n_step_td_prediction(env, UNIFORM_5, 5, n=0)
    with pytest.raises(ValueError, match="epsilon"):
        sarsa(CliffWalking(), 5, epsilon=-0.1)
