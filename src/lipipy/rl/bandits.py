r"""Multi-armed bandit algorithms and a testbed runner.

Every agent has ``select_action() -> int`` and ``update(action, reward)``. Updates
are validated before any state changes, so a rejected update leaves the agent as
it was. Ties between equally good arms are broken uniformly at random with the
agent's own generator.

Algorithms
----------
* **epsilon-greedy** (§2.2-2.5): explore uniformly with probability
  :math:`\varepsilon`, else pick :math:`\arg\max_a Q(a)`; update
  :math:`Q(a) \leftarrow Q(a) + \alpha_n (R - Q(a))` with the sample-average step
  :math:`\alpha_n = 1/N(a)` or a constant :math:`\alpha` (for non-stationary
  problems), optionally from an optimistic initial value.
* **UCB1** (§2.7; Auer, Cesa-Bianchi & Fischer 2002): pull every arm once, then
  :math:`A_t = \arg\max_a Q(a) + c\sqrt{\ln t / N(a)}`; ``c = sqrt(2)`` is UCB1's
  constant (logarithmic regret for rewards in [0, 1]), the book uses ``c = 2``.
* **Thompson sampling** (Thompson 1933; Russo et al. 2018): sample a mean for each
  arm from its posterior and act greedily on the sample. Beta-Bernoulli:
  :math:`\text{Beta}(\alpha, \beta) \to \text{Beta}(\alpha + r, \beta + 1 - r)` for
  :math:`r \in \{0, 1\}`. Gaussian with known noise :math:`\sigma^2` and prior
  :math:`\mathcal N(\mu_0, \sigma_0^2)`: precision
  :math:`\tau_n = \sigma_0^{-2} + n\sigma^{-2}`, mean
  :math:`(\mu_0\sigma_0^{-2} + \sigma^{-2}\sum r) / \tau_n`.
* **Gradient bandit** (§2.8): preferences :math:`H`, policy
  :math:`\pi = \operatorname{softmax}(H)`, update
  :math:`H_a \leftarrow H_a + \alpha (R - \bar R)(\mathbb 1[a = A] - \pi(a))` with the
  running average reward :math:`\bar R` (including the current reward) as
  baseline, or :math:`\bar R = 0` without it.

Regret
------
:func:`run_bandit` reports the cumulative (pseudo-)regret
:math:`\sum_{t \le T} \max_a q_*(a) - q_*(A_t)`, using the true values the
environment exposes in ``info`` (taken before each pull, so it is also correct for
the non-stationary testbed).

References
----------
Sutton & Barto (2018), *Reinforcement Learning: An Introduction*, 2nd ed., ch. 2;
Auer, Cesa-Bianchi & Fischer (2002), *Finite-time analysis of the multiarmed bandit
problem*, Machine Learning 47; Thompson (1933), Biometrika 25; Russo, Van Roy,
Kazerouni, Osband & Wen (2018), *A tutorial on Thompson sampling*.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from lipipy.core.complexity import Complexity
from lipipy.core.random import check_random_state, spawn
from lipipy.core.registry import register
from lipipy.numerics import softmax
from lipipy.rl._common import (
    ResultMixin,
    argmax_random,
    check_int,
    check_positive,
    check_unit,
    freeze,
    is_int,
    is_real,
)
from lipipy.rl.envs import KArmedBandit

_SB = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed."
_AGENT_COMPLEXITY = Complexity(
    time="O(k)", space="O(k)", notes="per select_action / update; k arms"
)


class _BanditAgent:
    """Shared validation and generator handling."""

    def __init__(self, n_arms: int, random_state=None):
        self.n_arms = check_int(n_arms, "n_arms")
        self._rng = check_random_state(random_state)

    def _check_update(self, action, reward) -> tuple[int, float]:
        if not is_int(action) or not 0 <= action < self.n_arms:
            raise ValueError(f"action must be an integer in [0, {self.n_arms}), got {action!r}.")
        if not is_real(reward) or not np.isfinite(reward):
            raise ValueError(f"reward must be a finite number, got {reward!r}.")
        return int(action), float(reward)


@register(
    domain="rl",
    tasks=("bandit",),
    complexity=_AGENT_COMPLEXITY,
    references=(_SB + ", §2.2-2.6",),
    status="experimental",
)
class EpsilonGreedy(_BanditAgent):
    r"""Epsilon-greedy action values with sample-average or constant step sizes.

    Parameters
    ----------
    n_arms : int
    epsilon : float in [0, 1], default=0.1
    step_size : float in (0, 1] or None, default=None
        ``None`` for sample averages :math:`1/N(a)`, else a constant :math:`\alpha`.
    initial_value : float, default=0.0
        :math:`Q_1(a)` (optimistic initial values, §2.6).
    random_state : None, int, SeedSequence or Generator

    Attributes
    ----------
    values : ndarray of shape (k,)
        Current estimates :math:`Q(a)` (a copy).
    counts : ndarray of int, shape (k,)

    Complexity
    ----------
    O(k) per step.

    References
    ----------
    Sutton & Barto (2018), §2.2-2.6.

    Examples
    --------
    >>> agent = EpsilonGreedy(3, epsilon=0.0, random_state=0)
    >>> agent.update(1, 2.0); agent.update(1, 4.0)
    >>> agent.values.tolist(), agent.select_action()
    ([0.0, 3.0, 0.0], 1)
    """

    def __init__(
        self,
        n_arms: int,
        epsilon: float = 0.1,
        step_size: float | None = None,
        initial_value: float = 0.0,
        random_state=None,
    ):
        super().__init__(n_arms, random_state)
        self.epsilon = check_unit(epsilon, "epsilon")
        if step_size is not None:
            step_size = check_positive(step_size, "step_size")
            if step_size > 1:
                raise ValueError(f"step_size must be in (0, 1], got {step_size!r}.")
        self.step_size = step_size
        if not is_real(initial_value) or not np.isfinite(initial_value):
            raise ValueError(f"initial_value must be finite, got {initial_value!r}.")
        self.initial_value = float(initial_value)
        self._q = np.full(self.n_arms, self.initial_value)
        self._n = np.zeros(self.n_arms, dtype=np.int64)

    @property
    def values(self) -> np.ndarray:
        return self._q.copy()

    @property
    def counts(self) -> np.ndarray:
        return self._n.copy()

    def select_action(self) -> int:
        if self.epsilon > 0 and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_arms))
        return argmax_random(self._q, self._rng)

    def update(self, action: int, reward: float) -> None:
        a, r = self._check_update(action, reward)
        self._n[a] += 1
        alpha = 1.0 / self._n[a] if self.step_size is None else self.step_size
        self._q[a] += alpha * (r - self._q[a])


@register(
    domain="rl",
    tasks=("bandit",),
    complexity=_AGENT_COMPLEXITY,
    references=(
        _SB + ", §2.7",
        "Auer, Cesa-Bianchi & Fischer (2002), Finite-time analysis of the multiarmed "
        "bandit problem, Machine Learning 47",
    ),
    status="experimental",
)
class UCB1(_BanditAgent):
    r"""Upper-confidence-bound action selection.

    Untried arms are pulled first (in random order); afterwards
    :math:`A_t = \arg\max_a Q(a) + c\sqrt{\ln t / N(a)}` with :math:`t` the number of
    pulls so far, and :math:`Q` the sample averages.

    Parameters
    ----------
    n_arms : int
    c : float > 0, default=sqrt(2)
        Exploration constant (UCB1: :math:`\sqrt 2`; Sutton & Barto use 2).
    random_state : None, int, SeedSequence or Generator

    Attributes
    ----------
    values, counts : ndarray (copies)

    Complexity
    ----------
    O(k) per step.

    References
    ----------
    Auer, Cesa-Bianchi & Fischer (2002); Sutton & Barto (2018), §2.7.

    Examples
    --------
    >>> agent = UCB1(2, random_state=0)
    >>> first = agent.select_action(); agent.update(first, 0.0)
    >>> agent.select_action() == 1 - first  # the untried arm
    True
    """

    def __init__(self, n_arms: int, c: float = float(np.sqrt(2.0)), random_state=None):
        super().__init__(n_arms, random_state)
        self.c = check_positive(c, "c")
        self._q = np.zeros(self.n_arms)
        self._n = np.zeros(self.n_arms, dtype=np.int64)

    @property
    def values(self) -> np.ndarray:
        return self._q.copy()

    @property
    def counts(self) -> np.ndarray:
        return self._n.copy()

    def select_action(self) -> int:
        untried = np.flatnonzero(self._n == 0)
        if untried.size:
            return int(self._rng.choice(untried))
        t = self._n.sum()
        return argmax_random(self._q + self.c * np.sqrt(np.log(t) / self._n), self._rng)

    def update(self, action: int, reward: float) -> None:
        a, r = self._check_update(action, reward)
        self._n[a] += 1
        self._q[a] += (r - self._q[a]) / self._n[a]


@register(
    domain="rl",
    tasks=("bandit",),
    complexity=_AGENT_COMPLEXITY,
    references=(
        "Thompson (1933), On the likelihood that one unknown probability exceeds another, "
        "Biometrika 25",
        "Russo et al. (2018), A tutorial on Thompson sampling, Found. Trends Mach. Learn.",
    ),
    status="experimental",
)
class BernoulliThompson(_BanditAgent):
    r"""Thompson sampling for Bernoulli rewards with Beta priors.

    Parameters
    ----------
    n_arms : int
    prior_alpha, prior_beta : float > 0, default=1.0
        Beta prior (1, 1 is uniform).
    random_state : None, int, SeedSequence or Generator

    Attributes
    ----------
    alpha, beta : ndarray of shape (k,)
        Posterior parameters (copies): prior plus successes / failures.

    Raises
    ------
    ValueError
        In ``update`` if the reward is not 0 or 1.

    Complexity
    ----------
    O(k) per step (one Beta draw per arm).

    References
    ----------
    Thompson (1933); Russo et al. (2018), §3.

    Examples
    --------
    >>> agent = BernoulliThompson(2, random_state=0)
    >>> agent.update(0, 1); agent.update(0, 0); agent.update(0, 1)
    >>> agent.alpha.tolist(), agent.beta.tolist()
    ([3.0, 1.0], [2.0, 1.0])
    """

    def __init__(
        self, n_arms: int, prior_alpha: float = 1.0, prior_beta: float = 1.0, random_state=None
    ):
        super().__init__(n_arms, random_state)
        self.prior_alpha = check_positive(prior_alpha, "prior_alpha")
        self.prior_beta = check_positive(prior_beta, "prior_beta")
        self._a = np.full(self.n_arms, self.prior_alpha)
        self._b = np.full(self.n_arms, self.prior_beta)

    @property
    def alpha(self) -> np.ndarray:
        return self._a.copy()

    @property
    def beta(self) -> np.ndarray:
        return self._b.copy()

    def select_action(self) -> int:
        return argmax_random(self._rng.beta(self._a, self._b), self._rng)

    def update(self, action: int, reward: float) -> None:
        a, r = self._check_update(action, reward)
        if r not in (0.0, 1.0):
            raise ValueError(f"Bernoulli rewards must be 0 or 1, got {reward!r}.")
        self._a[a] += r
        self._b[a] += 1.0 - r


@register(
    domain="rl",
    tasks=("bandit",),
    complexity=_AGENT_COMPLEXITY,
    references=("Russo et al. (2018), A tutorial on Thompson sampling, §4",),
    status="experimental",
)
class GaussianThompson(_BanditAgent):
    r"""Thompson sampling for Gaussian rewards with known noise and normal priors.

    Parameters
    ----------
    n_arms : int
    prior_mean : float, default=0.0
    prior_std : float > 0, default=1.0
    noise_std : float > 0, default=1.0
        Known reward standard deviation :math:`\sigma`.
    random_state : None, int, SeedSequence or Generator

    Attributes
    ----------
    posterior_mean, posterior_std : ndarray of shape (k,) (copies)

    Complexity
    ----------
    O(k) per step.

    References
    ----------
    Russo et al. (2018), §4; Gelman et al., *Bayesian Data Analysis* 3e, §2.5.

    Examples
    --------
    >>> agent = GaussianThompson(1, prior_mean=0.0, prior_std=1.0, noise_std=1.0)
    >>> agent.update(0, 2.0)
    >>> agent.posterior_mean.tolist(), agent.posterior_std.round(4).tolist()
    ([1.0], [0.7071])
    """

    def __init__(
        self,
        n_arms: int,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
        noise_std: float = 1.0,
        random_state=None,
    ):
        super().__init__(n_arms, random_state)
        if not is_real(prior_mean) or not np.isfinite(prior_mean):
            raise ValueError(f"prior_mean must be finite, got {prior_mean!r}.")
        self.prior_mean = float(prior_mean)
        self.prior_std = check_positive(prior_std, "prior_std")
        self.noise_std = check_positive(noise_std, "noise_std")
        self._n = np.zeros(self.n_arms)
        self._sum = np.zeros(self.n_arms)

    def _posterior(self) -> tuple[np.ndarray, np.ndarray]:
        prec0, prec = self.prior_std**-2, self.noise_std**-2
        tau = prec0 + self._n * prec
        mean = (self.prior_mean * prec0 + self._sum * prec) / tau
        return mean, 1.0 / np.sqrt(tau)

    @property
    def posterior_mean(self) -> np.ndarray:
        return self._posterior()[0]

    @property
    def posterior_std(self) -> np.ndarray:
        return self._posterior()[1]

    def select_action(self) -> int:
        mean, std = self._posterior()
        return argmax_random(self._rng.normal(mean, std), self._rng)

    def update(self, action: int, reward: float) -> None:
        a, r = self._check_update(action, reward)
        self._n[a] += 1.0
        self._sum[a] += r


@register(
    domain="rl",
    tasks=("bandit",),
    complexity=_AGENT_COMPLEXITY,
    references=(_SB + ", §2.8",),
    status="experimental",
)
class GradientBandit(_BanditAgent):
    r"""Gradient bandit: softmax over preferences, stochastic gradient ascent.

    Parameters
    ----------
    n_arms : int
    step_size : float > 0, default=0.1
    baseline : bool, default=True
        Use the running average reward as baseline (else 0).
    random_state : None, int, SeedSequence or Generator

    Attributes
    ----------
    preferences : ndarray of shape (k,) (copy)
    probabilities : ndarray of shape (k,)
        :math:`\pi = \operatorname{softmax}(H)`.

    Complexity
    ----------
    O(k) per step.

    References
    ----------
    Sutton & Barto (2018), §2.8, eq. (2.12).

    Examples
    --------
    >>> agent = GradientBandit(2, step_size=0.5, baseline=False, random_state=0)
    >>> agent.update(0, 1.0)
    >>> agent.preferences.tolist()
    [0.25, -0.25]
    """

    def __init__(
        self, n_arms: int, step_size: float = 0.1, baseline: bool = True, random_state=None
    ):
        super().__init__(n_arms, random_state)
        self.step_size = check_positive(step_size, "step_size")
        if not isinstance(baseline, (bool, np.bool_)):
            raise TypeError(f"baseline must be a bool, got {baseline!r}.")
        self.baseline = bool(baseline)
        self._h = np.zeros(self.n_arms)
        self._mean_reward = 0.0
        self._t = 0

    @property
    def preferences(self) -> np.ndarray:
        return self._h.copy()

    @property
    def probabilities(self) -> np.ndarray:
        return softmax(self._h)

    def select_action(self) -> int:
        return int(self._rng.choice(self.n_arms, p=self.probabilities))

    def update(self, action: int, reward: float) -> None:
        a, r = self._check_update(action, reward)
        pi = self.probabilities
        t = self._t + 1
        mean = self._mean_reward + (r - self._mean_reward) / t
        advantage = r - (mean if self.baseline else 0.0)
        one_hot = np.zeros(self.n_arms)
        one_hot[a] = 1.0
        self._h = self._h + self.step_size * advantage * (one_hot - pi)
        self._t, self._mean_reward = t, mean


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BanditRun(ResultMixin):
    """One run of an agent on a bandit (see :func:`run_bandit`).

    Attributes
    ----------
    rewards : ndarray of shape (T,)
    actions : ndarray of int, shape (T,)
    optimal : ndarray of bool, shape (T,)
        Whether each action was optimal at the time.
    regret : ndarray of shape (T,)
        Cumulative pseudo-regret :math:`\\sum_{t' \\le t} \\max_a q_*(a) - q_*(A_{t'})`.
    optimal_action_rate : float
    total_reward : float
    """

    rewards: np.ndarray
    actions: np.ndarray
    optimal: np.ndarray
    regret: np.ndarray
    optimal_action_rate: float
    total_reward: float


def _check_agent(agent) -> None:
    if not (
        callable(getattr(agent, "select_action", None)) and callable(getattr(agent, "update", None))
    ):
        raise TypeError("agent must have select_action() and update(action, reward).")


@register(
    domain="rl",
    tasks=("bandit",),
    complexity=Complexity(time="O(T k)", space="O(T)", notes="T steps, k arms"),
    references=(_SB + ", §2.3",),
    status="experimental",
)
def run_bandit(env, agent, n_steps: int) -> BanditRun:
    r"""Run ``agent`` on bandit ``env`` for ``n_steps`` pulls.

    ``env`` must expose the true arm values as ``info["q_star"]`` (as
    :class:`~lipipy.rl.KArmedBandit` and :class:`~lipipy.rl.BernoulliBandit` do);
    the episode is started with ``env.reset()``.

    Parameters
    ----------
    env : bandit environment
    agent : object with ``select_action()`` and ``update(action, reward)``
    n_steps : int

    Returns
    -------
    BanditRun

    Complexity
    ----------
    Time O(T k), memory O(T) for T = ``n_steps``.

    References
    ----------
    Sutton & Barto (2018), §2.3.

    Examples
    --------
    >>> from lipipy.rl import BernoulliBandit, UCB1, run_bandit
    >>> run = run_bandit(BernoulliBandit([0.2, 0.8], random_state=0), UCB1(2, random_state=0), 500)
    >>> run.rewards.shape, bool(run.optimal_action_rate > 0.8)
    ((500,), True)
    """
    _check_agent(agent)
    n_steps = check_int(n_steps, "n_steps")
    k = env.action_space.n
    if getattr(agent, "n_arms", k) != k:
        raise ValueError(f"agent has {agent.n_arms} arms, the environment {k}.")
    env.reset()
    rewards = np.empty(n_steps)
    actions = np.empty(n_steps, dtype=np.int64)
    optimal = np.empty(n_steps, dtype=bool)
    gaps = np.empty(n_steps)
    for t in range(n_steps):
        a = agent.select_action()
        _, r, _, truncated, info = env.step(a)
        q = info["q_star"]
        agent.update(a, r)
        rewards[t], actions[t] = r, a
        optimal[t] = q[a] == q.max()
        gaps[t] = q.max() - q[a]
        if truncated and t + 1 < n_steps:
            env.reset()
    regret = np.cumsum(gaps)
    freeze(rewards, actions, optimal, regret)
    return BanditRun(rewards, actions, optimal, regret, float(optimal.mean()), float(rewards.sum()))


@dataclass(frozen=True)
class TestbedResult(ResultMixin):
    """Averages over the runs of :func:`bandit_testbed`.

    Attributes
    ----------
    mean_reward : ndarray of shape (T,)
        Average reward at each step.
    optimal_action_rate : ndarray of shape (T,)
        Fraction of runs that chose an optimal arm at each step.
    mean_regret : ndarray of shape (T,)
        Average cumulative regret.
    final_regret : ndarray of shape (n_runs,)
        Cumulative regret of each run after T steps.
    """

    __test__ = False  # not a pytest test class

    mean_reward: np.ndarray
    optimal_action_rate: np.ndarray
    mean_regret: np.ndarray
    final_regret: np.ndarray


@register(
    domain="rl",
    tasks=("bandit",),
    complexity=Complexity(time="O(P T k)", space="O(T + P)", notes="P runs of T steps, k arms"),
    references=(_SB + ", §2.3, Figure 2.2",),
    status="experimental",
)
def bandit_testbed(
    make_agent: Callable[..., object],
    n_runs: int = 200,
    n_steps: int = 1000,
    k: int = 10,
    stationary: bool = True,
    random_state=None,
) -> TestbedResult:
    r"""Average an agent over independent k-armed testbed problems (Figure 2.2).

    Run ``i`` uses a fresh :class:`~lipipy.rl.KArmedBandit` and a fresh agent
    ``make_agent(k, random_state=g)``; the environment and agent generators of all
    runs come from :func:`lipipy.core.spawn`, so results are reproducible.

    Parameters
    ----------
    make_agent : callable ``(n_arms, random_state=...) -> agent``
        E.g. ``EpsilonGreedy`` or ``functools.partial(UCB1, c=2.0)``.
    n_runs : int, default=200
    n_steps : int, default=1000
    k : int, default=10
    stationary : bool, default=True
    random_state : None, int, SeedSequence or Generator

    Returns
    -------
    TestbedResult

    Complexity
    ----------
    Time O(P T k) for P runs, memory O(T + P).

    References
    ----------
    Sutton & Barto (2018), §2.3.

    Examples
    --------
    >>> from lipipy.rl import EpsilonGreedy, bandit_testbed
    >>> res = bandit_testbed(EpsilonGreedy, n_runs=20, n_steps=200, random_state=0)
    >>> res.mean_reward.shape, bool(res.optimal_action_rate[-50:].mean() > 0.5)
    ((200,), True)
    """
    if not callable(make_agent):
        raise TypeError("make_agent must be callable.")
    n_runs = check_int(n_runs, "n_runs")
    n_steps = check_int(n_steps, "n_steps")
    streams = spawn(random_state, 2 * n_runs)
    reward_sum = np.zeros(n_steps)
    optimal_sum = np.zeros(n_steps)
    regret_sum = np.zeros(n_steps)
    final = np.empty(n_runs)
    for i in range(n_runs):
        env = KArmedBandit(k=k, stationary=stationary, random_state=streams[2 * i])
        agent = make_agent(k, random_state=streams[2 * i + 1])
        run = run_bandit(env, agent, n_steps)
        reward_sum += run.rewards
        optimal_sum += run.optimal
        regret_sum += run.regret
        final[i] = run.regret[-1]
    mean_reward, rate, mean_regret = reward_sum / n_runs, optimal_sum / n_runs, regret_sum / n_runs
    freeze(mean_reward, rate, mean_regret, final)
    return TestbedResult(mean_reward, rate, mean_regret, final)
