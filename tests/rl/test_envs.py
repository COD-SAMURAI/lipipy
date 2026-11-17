"""Environments: Gymnasium API, model validation, dynamics, atomic steps, no leaks."""

import numpy as np
import pytest

from lipipy.rl import (
    BernoulliBandit,
    Box,
    CliffWalking,
    Discrete,
    Env,
    FrozenLake,
    GridWorld,
    KArmedBandit,
    RandomWalk,
    ShortCorridor,
    TabularMDP,
)

ALL_ENVS = [
    lambda: GridWorld.example_4_1(random_state=0),
    lambda: CliffWalking(random_state=0),
    lambda: FrozenLake("4x4", random_state=0),
    lambda: FrozenLake("8x8", is_slippery=False, random_state=0),
    lambda: RandomWalk(19, random_state=0),
    lambda: ShortCorridor(random_state=0),
    lambda: KArmedBandit(random_state=0),
    lambda: BernoulliBandit([0.1, 0.9], random_state=0),
]


def rollout(env, seed, n=50):
    """(obs, reward, terminated) of a random-action episode, at most n steps."""
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    out = [obs]
    for _ in range(n):
        obs, r, term, trunc, _ = env.step(int(rng.integers(env.action_space.n)))
        out.append((obs, r, term))
        if term or trunc:
            break
    return out


# ---------------------------------------------------------------------------
# Spaces
# ---------------------------------------------------------------------------
def test_discrete_space():
    space = Discrete(3, start=-1, random_state=0)
    assert [space.contains(x) for x in (-1, 1, 2, True, 0.0, np.int64(0))] == [
        True,
        True,
        False,
        False,
        False,
        True,
    ]
    draws = [space.sample() for _ in range(300)]
    assert set(draws) == {-1, 0, 1}
    space.seed(5)
    a = [space.sample() for _ in range(10)]
    space.seed(5)
    assert a == [space.sample() for _ in range(10)]
    assert Discrete(3) == Discrete(3) and Discrete(3) != Discrete(4)
    for bad in (0, -1, 2.0, True):
        with pytest.raises(ValueError):
            Discrete(bad)


def test_box_space():
    box = Box([0.0, -np.inf, 1.0, -np.inf], [1.0, 0.0, np.inf, np.inf], random_state=0)
    for _ in range(200):
        x = box.sample()
        assert box.contains(x)
    assert not box.contains(np.zeros(3)) and not box.contains([np.nan, 0, 1, 0])
    assert not box.contains("abc")
    with pytest.raises(ValueError):
        box.low[0] = 5.0  # read-only bounds
    with pytest.raises(ValueError, match="low must be"):
        Box(1.0, 0.0, shape=(2,))
    with pytest.raises(ValueError, match="NaN"):
        Box(np.nan, 1.0, shape=(1,))


@pytest.mark.parametrize("make", ALL_ENVS)
def test_protocol_reproducibility_and_atomic_steps(make):
    env = make()
    assert isinstance(env, Env)
    with pytest.raises(RuntimeError, match="reset"):
        make().step(0)
    assert rollout(make(), 7) == rollout(make(), 7)  # same seed, same episode
    env.reset(seed=1)
    snapshot = (env._state, env._elapsed, env._rng.bit_generator.state)
    with pytest.raises(ValueError, match="Invalid action"):
        env.step(env.action_space.n)
    with pytest.raises(ValueError, match="Invalid action"):
        env.step(True)
    assert (env._state, env._elapsed, env._rng.bit_generator.state) == snapshot


@pytest.mark.parametrize(
    "make", [lambda: GridWorld.example_4_1(), lambda: FrozenLake(), lambda: RandomWalk(5)]
)
def test_step_after_termination_raises_and_leaves_state(make):
    env = make()
    rng = np.random.default_rng(0)
    env.reset(seed=0)
    while True:
        _, _, term, trunc, _ = env.step(int(rng.integers(env.action_space.n)))
        if term or trunc:
            break
    snapshot = (env._state, env._elapsed, env._rng.bit_generator.state)
    with pytest.raises(RuntimeError, match="ended"):
        env.step(0)
    assert (env._state, env._elapsed, env._rng.bit_generator.state) == snapshot
    env.reset()  # a new episode works
    env.step(0)


def test_truncation():
    env = CliffWalking(max_episode_steps=3)  # moving up never terminates
    env.reset(seed=0)
    flags = [env.step(0)[2:4] for _ in range(3)]
    assert flags == [(False, False), (False, False), (False, True)]
    with pytest.raises(RuntimeError):
        env.step(0)
    with pytest.raises(ValueError, match="max_episode_steps"):
        CliffWalking(max_episode_steps=0)


# ---------------------------------------------------------------------------
# TabularMDP
# ---------------------------------------------------------------------------
GOOD_P = [[[(0.5, 0, 1.0, False), (0.5, 1, 0.0, True)]], [[(1.0, 1, 0.0, True)]]]


def test_tabular_mdp_validation():
    TabularMDP(GOOD_P, [1.0, 0.0])
    bad_p = [
        [[[(0.5, 0, 1.0, False)]], [[(1.0, 1, 0.0, True)]]],  # sums to 0.5
        [[[(1.0, 2, 1.0, False)]], [[(1.0, 1, 0.0, True)]]],  # next state out of range
        [[[(1.0, 0, np.nan, False)]], [[(1.0, 1, 0.0, True)]]],  # NaN reward
        [[[(1.0, 0, 1.0, 1)]], [[(1.0, 1, 0.0, True)]]],  # flag not bool
        [[[(1.0, 0, 1.0)]], [[(1.0, 1, 0.0, True)]]],  # 3-tuple
        [[[(-0.5, 0, 1.0, False), (1.5, 0, 0.0, False)]], [[(1.0, 1, 0.0, True)]]],
        [[[(1.0, 0, 1.0, False)]], [[(1.0, 1, 0.0, True)], [(1.0, 1, 0.0, True)]]],
        [[[]], [[(1.0, 1, 0.0, True)]]],
        [],
    ]
    for P in bad_p:
        with pytest.raises(ValueError):
            TabularMDP(P, [1.0, 0.0] if P else [])
    with pytest.raises(ValueError, match="terminal state"):  # terminal start state
        TabularMDP(GOOD_P, [0.0, 1.0])
    for mu in ([0.5, 0.6], [1.0], [1.5, -0.5]):
        with pytest.raises(ValueError, match="initial_state_distribution"):
            TabularMDP(GOOD_P, mu)


def test_tabular_mdp_model_is_immutable_and_arrays_are_fresh():
    mdp = TabularMDP(GOOD_P, [1.0, 0.0])
    with pytest.raises(TypeError):
        mdp.P[0][0] = ()
    P1, R1 = mdp.transition_arrays()
    P1[:] = 7.0
    P2, _ = mdp.transition_arrays()
    assert P2[0, 0, 0] == 0.5
    np.testing.assert_array_equal(mdp.terminal_states, [False, True])
    with pytest.raises(ValueError):
        mdp.terminal_states[0] = True


def test_sampled_transitions_follow_P():
    """Empirical outcome frequencies within 4 standard errors of P (30000 draws)."""
    env = FrozenLake("4x4", is_slippery=True, random_state=0)
    counts = {}
    n = 30000
    for _ in range(n):
        env.reset()
        nxt = env.step(2)[0]  # right from the start
        counts[nxt] = counts.get(nxt, 0) + 1
    expected = {}
    for p, s2, _, _ in env.P[0][2]:
        expected[s2] = expected.get(s2, 0.0) + p
    assert set(counts) == set(expected)
    for s2, p in expected.items():
        se = np.sqrt(p * (1 - p) / n)
        assert abs(counts[s2] / n - p) < 4 * se


# ---------------------------------------------------------------------------
# Specific environments
# ---------------------------------------------------------------------------
def test_gridworld_example_4_1_dynamics():
    env = GridWorld.example_4_1()
    assert env.n_states == 16 and env.n_actions == 4
    assert env.P[1][3] == ((1.0, 0, -1.0, True),)  # left from state 1 enters terminal 0
    assert env.P[4][3] == ((1.0, 4, -1.0, False),)  # off-grid: stays
    assert env.P[0][1] == ((1.0, 0, 0.0, True),)  # terminal is absorbing
    np.testing.assert_allclose(env.initial_state_distribution[1:15], 1 / 14)


def test_gridworld_slip_walls_and_rewards():
    det = GridWorld((3, 3), terminal_cells=[(2, 2)], walls=[(1, 1)], rewards={(2, 2): 10.0})
    assert det.P[1][2] == ((1.0, 1, -1.0, False),)  # down into the wall: stays
    assert det.P[5][2] == ((1.0, 8, 10.0, True),)
    rand = GridWorld((3, 3), terminal_cells=[(2, 2)], slip=1.0)
    P_cont, R = rand.transition_arrays()
    for s in range(8):  # fully stochastic: every action has the same outcome law
        for a in range(1, 4):
            np.testing.assert_allclose(P_cont[s, a], P_cont[s, 0])
    np.testing.assert_allclose(rand.P[4][0], rand.P[4][2])
    for kwargs in (
        {"terminal_cells": [(4, 0)]},
        {"walls": [(0, 0)], "terminal_cells": [(0, 0)]},
        {"slip": 1.5},
        {"terminal_cells": [(r, c) for r in range(4) for c in range(4)]},
        {"rewards": {(0, 1): np.inf}},
    ):
        with pytest.raises(ValueError):
            GridWorld(**kwargs)


def test_cliff_walking_optimal_path_and_cliff():
    env = CliffWalking()
    obs, _ = env.reset(seed=0)
    assert obs == 36
    total = 0.0
    for a in [0] + [1] * 11 + [2]:  # up, along the edge, down
        obs, r, term, _, _ = env.step(a)
        total += r
    assert (obs, term, total) == (47, True, -13.0)
    env.reset()
    env.step(0)
    obs, r, term, _, _ = env.step(2)  # back down from (2, 0) is the start, not the cliff
    assert (obs, r, term) == (36, -1.0, False)
    obs, r, term, _, _ = env.step(1)
    assert (obs, r, term) == (36, -100.0, False)


def test_frozen_lake_maps_and_slipperiness():
    env = FrozenLake("8x8")
    assert env.n_states == 64 and env.desc[7][7] == "G"
    assert len(env.P[0][0]) == 3 and all(p == pytest.approx(1 / 3) for p, *_ in env.P[0][0])
    det = FrozenLake(is_slippery=False)
    assert det.P[14][2] == ((1.0, 15, 1.0, True),)
    assert det.P[4][2] == ((1.0, 5, 0.0, True),)  # into a hole
    with pytest.raises(ValueError, match="map_name"):
        FrozenLake("5x5")
    with pytest.raises(TypeError):
        FrozenLake(is_slippery=1)


def test_frozen_lake_matches_gymnasium():
    gym = pytest.importorskip("gymnasium")
    for map_name in ("4x4", "8x8"):
        for slippery in (True, False):
            ref = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=slippery).unwrapped
            ours = FrozenLake(map_name, is_slippery=slippery)
            for s in range(ours.n_states):
                for a in range(4):
                    expected = tuple(
                        (float(p), int(s2), float(r), bool(d)) for p, s2, r, d in ref.P[s][a]
                    )
                    assert ours.P[s][a] == pytest.approx(expected)


def test_random_walk():
    env = RandomWalk(5)
    assert env.n_states == 7 and env.action_space.n == 1
    assert env.reset(seed=0)[0] == 3
    np.testing.assert_allclose(RandomWalk(19).true_values()[1:20], np.arange(-9, 10) / 10)
    for bad in (4, 0):
        with pytest.raises(ValueError):
            RandomWalk(bad)


def test_short_corridor_values():
    env = ShortCorridor()
    grid = np.linspace(0.01, 0.99, 9801)
    values = np.array([env.start_value(p) for p in grid])
    best = grid[np.argmax(values)]
    assert best == pytest.approx(2 - np.sqrt(2), abs=1e-4)
    assert values.max() == pytest.approx(-11.657, abs=1e-3)
    assert env.start_value(0.0) == -np.inf and env.start_value(1.0) == -np.inf
    assert env.P[1][1] == ((1.0, 0, -1.0, False),)  # reversed in state 1


def test_k_armed_bandit_rewards_values_and_no_leaks():
    env = KArmedBandit(k=5, random_state=0)
    q = env.q_star
    q[:] = 100.0  # a copy: the environment is unaffected
    assert np.all(env.q_star < 10)
    _, info = env.reset()
    info["q_star"][:] = 100.0
    rewards = np.array([env.step(2)[1] for _ in range(20000)])
    assert abs(rewards.mean() - env.q_star[2]) < 4 / np.sqrt(20000)
    assert abs(rewards.std() - 1.0) < 0.03
    a = KArmedBandit(random_state=3).q_star
    b = KArmedBandit(random_state=3)
    b.reset(seed=4)
    assert not np.array_equal(a, b.q_star)


def test_nonstationary_bandit_walks_and_reset_restores():
    env = KArmedBandit(k=4, stationary=False, walk_std=0.1, random_state=0)
    env.reset()
    np.testing.assert_array_equal(env.q_star, 0.0)
    infos = [env.step(0)[4] for _ in range(200)]
    assert np.std(infos[-1]["q_star"]) > 0
    assert infos[0]["optimal_action"] == 0  # all equal before the first pull
    env.reset()
    np.testing.assert_array_equal(env.q_star, 0.0)
    for kwargs in ({"k": 0}, {"reward_std": 0.0}, {"walk_std": -1.0}, {"mean": np.nan}):
        with pytest.raises(ValueError):
            KArmedBandit(**kwargs)


def test_bernoulli_bandit():
    env = BernoulliBandit([0.25, 0.75], random_state=0)
    env.reset()
    rewards = np.array([env.step(1)[1] for _ in range(4000)])
    assert set(np.unique(rewards)) == {0.0, 1.0}
    assert abs(rewards.mean() - 0.75) < 4 * np.sqrt(0.75 * 0.25 / 4000)
    for bad in ([], [1.5], [[0.5]], [np.nan]):
        with pytest.raises(ValueError):
            BernoulliBandit(bad)
