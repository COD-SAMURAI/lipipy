"""Bandits: exact updates, atomic rejection, testbed comparisons over many runs."""

from functools import partial

import numpy as np
import pytest

from lipipy.core import spawn
from lipipy.rl import (
    UCB1,
    BernoulliBandit,
    BernoulliThompson,
    EpsilonGreedy,
    GaussianThompson,
    GradientBandit,
    KArmedBandit,
    bandit_testbed,
    run_bandit,
)

AGENTS = [
    partial(EpsilonGreedy, epsilon=0.1),
    partial(EpsilonGreedy, epsilon=0.1, step_size=0.1),
    UCB1,
    BernoulliThompson,
    GaussianThompson,
    GradientBandit,
]


def snapshot(agent):
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in vars(agent).items()}


def same(a, b):
    return a.keys() == b.keys() and all(
        np.array_equal(a[k], b[k]) if isinstance(a[k], np.ndarray) else a[k] == b[k]
        for k in a
        if k != "_rng"
    )


@pytest.mark.parametrize("make", AGENTS)
def test_rejected_update_leaves_agent_unchanged(make):
    agent = make(3, random_state=0)
    agent.update(0, 1.0)
    before = snapshot(agent)
    rng_state = agent._rng.bit_generator.state
    for action, reward in [(3, 1.0), (-1, 1.0), (1.5, 1.0), (True, 1.0), (0, np.nan), (0, "1")]:
        with pytest.raises(ValueError):
            agent.update(action, reward)
    assert same(before, snapshot(agent)) and agent._rng.bit_generator.state == rng_state


@pytest.mark.parametrize("make", AGENTS)
def test_reproducible(make):
    def actions(seed):
        agent = make(4, random_state=seed)
        out = []
        for t in range(50):
            a = agent.select_action()
            agent.update(a, float(t % 2))
            out.append(a)
        return out

    assert actions(3) == actions(3)


def test_epsilon_greedy_sample_average_and_constant_step():
    agent = EpsilonGreedy(2, epsilon=0.0, random_state=0)
    rewards = [1.0, 3.0, 2.0, 6.0]
    for r in rewards:
        agent.update(1, r)
    assert agent.values[1] == pytest.approx(np.mean(rewards), rel=1e-15)
    assert agent.counts.tolist() == [0, 4]
    const = EpsilonGreedy(1, step_size=0.5, initial_value=4.0)
    const.update(0, 0.0)
    const.update(0, 0.0)
    assert const.values[0] == 1.0  # (1 - 0.5)^2 * 4
    # epsilon = 0 with ties: random tie-breaking covers every tied arm
    greedy = EpsilonGreedy(3, epsilon=0.0, random_state=1)
    assert {greedy.select_action() for _ in range(100)} == {0, 1, 2}
    # epsilon = 1 is uniform
    uniform = EpsilonGreedy(2, epsilon=1.0, initial_value=5.0, random_state=1)
    uniform.update(0, 100.0)
    picks = np.array([uniform.select_action() for _ in range(4000)])
    assert abs(picks.mean() - 0.5) < 4 * 0.5 / np.sqrt(4000)
    for kwargs in (
        {"epsilon": 1.5},
        {"step_size": 0.0},
        {"step_size": 2.0},
        {"initial_value": np.inf},
    ):
        with pytest.raises(ValueError):
            EpsilonGreedy(2, **kwargs)


def test_ucb_pulls_every_arm_first_then_uses_the_bound():
    agent = UCB1(4, c=2.0, random_state=0)
    first = []
    for _ in range(4):
        a = agent.select_action()
        first.append(a)
        agent.update(a, float(a))
    assert sorted(first) == [0, 1, 2, 3]
    bonus = agent.values + 2.0 * np.sqrt(np.log(4) / agent.counts)
    assert agent.select_action() == int(np.argmax(bonus))
    with pytest.raises(ValueError):
        UCB1(2, c=0.0)


def test_beta_bernoulli_posterior_is_exact():
    rng = np.random.default_rng(0)
    agent = BernoulliThompson(3, prior_alpha=2.0, prior_beta=0.5, random_state=0)
    succ, fail = np.zeros(3), np.zeros(3)
    for _ in range(500):
        a, r = int(rng.integers(3)), int(rng.integers(2))
        agent.update(a, r)
        succ[a] += r
        fail[a] += 1 - r
    np.testing.assert_array_equal(agent.alpha, 2.0 + succ)
    np.testing.assert_array_equal(agent.beta, 0.5 + fail)
    with pytest.raises(ValueError, match="0 or 1"):
        agent.update(0, 0.5)


def test_gaussian_posterior_matches_sequential_bayes():
    """Closed form vs one-observation-at-a-time conjugate updates."""
    rng = np.random.default_rng(1)
    agent = GaussianThompson(1, prior_mean=2.0, prior_std=3.0, noise_std=0.5)
    mean, var = 2.0, 9.0
    for r in rng.normal(1.0, 0.5, size=40):
        agent.update(0, float(r))
        post_var = 1.0 / (1.0 / var + 1.0 / 0.25)
        mean, var = post_var * (mean / var + r / 0.25), post_var
    assert agent.posterior_mean[0] == pytest.approx(mean, rel=1e-12)
    assert agent.posterior_std[0] == pytest.approx(np.sqrt(var), rel=1e-12)


def test_gradient_bandit_update_matches_equation_2_12():
    agent = GradientBandit(3, step_size=0.2, baseline=True, random_state=0)
    agent.update(1, 2.0)  # baseline = 2.0 (mean including this reward) -> no change
    np.testing.assert_allclose(agent.preferences, 0.0)
    agent.update(2, 4.0)  # baseline 3.0, advantage +1, pi uniform
    np.testing.assert_allclose(agent.preferences, [-0.2 / 3, -0.2 / 3, 0.2 * 2 / 3])
    np.testing.assert_allclose(agent.probabilities.sum(), 1.0)
    with pytest.raises(TypeError):
        GradientBandit(2, baseline=1)


@pytest.mark.filterwarnings("error")
def test_extreme_preferences_stay_finite():
    agent = GradientBandit(2, step_size=1.0, baseline=False, random_state=0)
    for _ in range(200):
        agent.update(0, 1e6)
    assert np.all(np.isfinite(agent.probabilities))
    assert agent.select_action() == 0


def test_run_bandit_bookkeeping():
    env = BernoulliBandit([0.2, 0.5, 0.9], random_state=0)
    run = run_bandit(env, EpsilonGreedy(3, epsilon=1.0, random_state=0), 300)
    gaps = 0.9 - np.array([0.2, 0.5, 0.9])[run.actions]
    np.testing.assert_allclose(run.regret, np.cumsum(gaps))
    np.testing.assert_array_equal(run.optimal, run.actions == 2)
    assert run.optimal_action_rate == pytest.approx(run.optimal.mean())
    assert run.to_dict()["total_reward"] == run.rewards.sum()
    with pytest.raises(ValueError, match="arms"):
        run_bandit(env, UCB1(4), 10)
    with pytest.raises(TypeError):
        run_bandit(env, object(), 10)
    # truncation restarts the episode without stopping the run
    short = BernoulliBandit([0.5, 0.5], max_episode_steps=7, random_state=0)
    assert run_bandit(short, UCB1(2, random_state=0), 30).rewards.shape == (30,)


# ---------------------------------------------------------------------------
# Testbed comparisons (stated margins, measured values in comments)
# ---------------------------------------------------------------------------
def _slope(regret, ts=(250, 500, 1000, 2000)):
    t = np.array(ts)
    return float(np.polyfit(np.log(t), np.log(regret[t - 1]), 1)[0])


def test_ucb_and_thompson_beat_epsilon_greedy_and_ucb_regret_is_sublinear():
    """10-armed stationary testbed, 50 runs x 2000 steps (seed 1).

    Measured: final mean regret eps-greedy(0.1) 431, UCB1 106, Gaussian Thompson 71;
    log-log slope of regret over t in {250, ..., 2000}: eps-greedy 0.68 (linear
    regret from constant exploration), UCB1 0.27, Thompson 0.23. Asserted:
    UCB1 and Thompson below half of eps-greedy, UCB1 slope < 0.45 < 0.55 < eps slope.
    """
    eps = bandit_testbed(partial(EpsilonGreedy, epsilon=0.1), 50, 2000, random_state=1)
    ucb = bandit_testbed(UCB1, 50, 2000, random_state=1)
    ts = bandit_testbed(GaussianThompson, 50, 2000, random_state=1)
    assert ucb.mean_regret[-1] < 0.5 * eps.mean_regret[-1]
    assert ts.mean_regret[-1] < 0.5 * eps.mean_regret[-1]
    assert _slope(ucb.mean_regret) < 0.45
    assert _slope(eps.mean_regret) > 0.55
    assert ucb.optimal_action_rate[-200:].mean() > 0.85


def test_gradient_bandit_baseline_helps_when_rewards_are_offset():
    """Figure 2.5 setting (q* ~ N(4, 1)), 50 runs x 500 steps.

    Measured optimal-action rate over the last 100 steps: 0.75 with the baseline,
    0.52 without; asserted gap > 0.1.
    """

    def rate(baseline):
        total = np.zeros(500)
        for ge, ga in zip(spawn(0, 50), spawn(1000, 50), strict=True):
            env = KArmedBandit(mean=4.0, random_state=ge)
            agent = GradientBandit(10, step_size=0.1, baseline=baseline, random_state=ga)
            total += run_bandit(env, agent, 500).optimal
        return (total / 50)[-100:].mean()

    assert rate(True) > rate(False) + 0.1


def test_constant_step_size_tracks_a_nonstationary_bandit():
    """Exercise 2.5: 30 runs x 5000 steps (walk std 0.01).

    Measured mean reward over the last 1000 steps: 1.00 with alpha = 0.1, 0.82 with
    sample averages; asserted gap > 0.08.
    """

    def last_reward(step_size):
        res = bandit_testbed(
            partial(EpsilonGreedy, epsilon=0.1, step_size=step_size),
            30,
            5000,
            stationary=False,
            random_state=0,
        )
        return res.mean_reward[-1000:].mean()

    assert last_reward(0.1) > last_reward(None) + 0.08


def test_testbed_reproducible_and_validated():
    a = bandit_testbed(UCB1, 3, 50, random_state=5)
    b = bandit_testbed(UCB1, 3, 50, random_state=5)
    np.testing.assert_array_equal(a.final_regret, b.final_regret)
    with pytest.raises(TypeError):
        bandit_testbed("UCB1", 3, 50)
    with pytest.raises(ValueError):
        bandit_testbed(UCB1, 0, 50)
