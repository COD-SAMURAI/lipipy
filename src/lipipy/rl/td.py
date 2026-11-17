r"""Tabular model-free prediction and control: Monte Carlo and temporal differences.

All functions interact with a Gymnasium-style environment whose observation and
action spaces are :class:`~lipipy.rl.Discrete` (S states, A actions). They draw
two independent streams from ``random_state`` with :func:`lipipy.core.spawn`: one
seeds the environment at the first ``reset`` and one drives the agent's choices,
so a run is exactly reproducible. Tables live in local arrays and are returned
only when the whole run finished.

Episodes end on ``terminated`` (no bootstrapping: the value after a terminal
transition is 0), on ``truncated`` or after ``max_steps_per_episode`` (a time
limit: the next state is not terminal, so targets still bootstrap from it).

Prediction (evaluate a fixed policy :math:`\pi`)
------------------------------------------------
* TD(0) (§6.1): :math:`V(S) \leftarrow V(S) + \alpha[R + \gamma V(S') - V(S)]`.
* n-step TD (§7.1): the target is
  :math:`G_{t:t+n} = \sum_{i=1}^{n}\gamma^{i-1} R_{t+i} + \gamma^n V(S_{t+n})`,
  truncated at the episode end.
* TD(λ) (§12.2, tabular backward view): with
  :math:`\delta = R + \gamma V(S') - V(S)`, traces decay
  :math:`e \leftarrow \gamma\lambda e` and the visited state gets
  :math:`e(S) \leftarrow e(S) + 1` (accumulating) or :math:`e(S) \leftarrow 1`
  (replacing); then :math:`V \leftarrow V + \alpha\delta e`.

Control (learn :math:`Q \approx q_*`)
-------------------------------------
Behaviour is :math:`\varepsilon`-greedy with respect to the current estimates, ties
broken at random. Targets:

* SARSA (§6.4): :math:`R + \gamma Q(S', A')`, with :math:`A'` the next action taken.
* Expected SARSA (§6.6): :math:`R + \gamma\sum_a \pi(a|S') Q(S', a)` under the
  :math:`\varepsilon`-greedy policy.
* Q-learning (§6.5): :math:`R + \gamma\max_a Q(S', a)`.
* Double Q-learning (§6.7): two tables; with probability 1/2 update
  :math:`Q_1(S,A) \leftarrow Q_1 + \alpha[R + \gamma Q_2(S', \arg\max_a Q_1(S', a)) - Q_1]`
  (else the roles swap); behaviour uses :math:`Q_1 + Q_2`.
* On-policy first-visit Monte Carlo control (§5.4): after each episode, every first
  visit of :math:`(s, a)` moves :math:`Q(s,a)` toward the return
  :math:`G_t = \sum_k \gamma^k R_{t+k+1}` by the sample average (step :math:`1/N`).

Step sizes ``alpha`` are a constant in (0, 1] or a callable ``n -> alpha_n`` of
the visit count :math:`n = N(s)` or :math:`N(s, a)` (e.g. ``lambda n: n ** -0.6``,
which satisfies the Robbins-Monro conditions under which Q-learning converges to
:math:`q_*`; Watkins & Dayan 1992).

References
----------
Sutton & Barto (2018), *Reinforcement Learning: An Introduction*, 2nd ed.:
§5.4, §6.1-6.7, §7.1, §12.2; Watkins & Dayan (1992), *Q-learning*, Machine
Learning 8; van Hasselt (2010), *Double Q-learning*, NeurIPS.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from lipipy.core.complexity import Complexity
from lipipy.core.random import spawn
from lipipy.core.registry import register
from lipipy.rl._common import (
    ResultMixin,
    StepSize,
    argmax_random,
    check_gamma,
    check_int,
    check_step_size,
    check_unit,
    epsilon_greedy,
    epsilon_greedy_probs,
    freeze,
    greedy_policy,
    is_real,
    seed_from,
)
from lipipy.rl.envs import Discrete

_SB = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed."
_NOTES = "S states, A actions, T total environment steps"


@dataclass(frozen=True)
class PredictionResult(ResultMixin):
    """Outcome of the prediction functions.

    Attributes
    ----------
    V : ndarray of shape (S,)
        Estimated state values.
    rms_error : ndarray of shape (n_episodes,) or None
        Root-mean-square error over the non-terminal states against
        ``true_values`` after each episode (only when ``true_values`` is given).
    n_steps : int
        Total environment steps.
    """

    V: np.ndarray
    rms_error: np.ndarray | None
    n_steps: int


@dataclass(frozen=True)
class TDControlResult(ResultMixin):
    """Outcome of the control functions.

    Attributes
    ----------
    Q : ndarray of shape (S, A)
    policy : ndarray of int, shape (S,)
        Greedy policy of ``Q`` (lowest action index on ties).
    episode_returns : ndarray of shape (n_episodes,)
        Undiscounted sum of rewards of every episode.
    episode_lengths : ndarray of int, shape (n_episodes,)
    """

    Q: np.ndarray
    policy: np.ndarray
    episode_returns: np.ndarray
    episode_lengths: np.ndarray


# ---------------------------------------------------------------------------
# Shared checks
# ---------------------------------------------------------------------------
def _spaces(env) -> tuple[int, int]:
    obs, act = getattr(env, "observation_space", None), getattr(env, "action_space", None)
    if not (isinstance(obs, Discrete) or type(obs).__name__ == "Discrete") or not (
        isinstance(act, Discrete) or type(act).__name__ == "Discrete"
    ):
        raise TypeError("env must have Discrete observation and action spaces (tabular).")
    return int(obs.n), int(act.n)


def _terminal_mask(env, S: int) -> np.ndarray:
    mask = getattr(env, "terminal_states", None)
    return np.zeros(S, dtype=bool) if mask is None else np.asarray(mask, dtype=bool).copy()


def _common(env, n_episodes, gamma, max_steps, random_state, initial_value):
    S, A = _spaces(env)
    n_episodes = check_int(n_episodes, "n_episodes")
    gamma = check_gamma(gamma)
    max_steps = check_int(max_steps, "max_steps_per_episode")
    if not is_real(initial_value) or not np.isfinite(initial_value):
        raise ValueError(f"initial_value must be finite, got {initial_value!r}.")
    env_rng, rng = spawn(random_state, 2)
    return S, A, n_episodes, gamma, max_steps, env_rng, rng


def _check_policy(policy, S: int, A: int) -> np.ndarray:
    pi = np.asarray(policy, dtype=np.float64)
    if pi.shape != (S, A):
        raise ValueError(f"policy must have shape ({S}, {A}), got {pi.shape}.")
    if np.any(~np.isfinite(pi)) or np.any(pi < 0) or np.any(np.abs(pi.sum(axis=1) - 1) > 1e-9):
        raise ValueError("Every row of policy must be a probability distribution.")
    return np.cumsum(pi, axis=1)


def _check_true_values(true_values, S: int) -> np.ndarray | None:
    if true_values is None:
        return None
    v = np.asarray(true_values, dtype=np.float64)
    if v.shape != (S,) or not np.all(np.isfinite(v)):
        raise ValueError(f"true_values must be a finite array of shape ({S},).")
    return v


def _sample(cdf_row: np.ndarray, rng: np.random.Generator) -> int:
    if cdf_row.size == 1:
        return 0
    return min(int(np.searchsorted(cdf_row, rng.random(), side="right")), cdf_row.size - 1)


def _reset(env, env_rng, first: bool) -> int:
    obs, _ = env.reset(seed=seed_from(env_rng)) if first else env.reset()
    return int(obs)


def _rms(V, true_v, nonterminal) -> float:
    return float(np.sqrt(np.mean((V[nonterminal] - true_v[nonterminal]) ** 2)))


_PRED_REFS = (_SB + ", §6.1, §7.1, §12.2",)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def _prediction(
    env,
    policy,
    n_episodes,
    alpha,
    gamma,
    initial_value,
    true_values,
    max_steps_per_episode,
    random_state,
    method: str,
    n: int = 1,
    lam: float = 0.0,
    trace: str = "accumulating",
) -> PredictionResult:
    S, A, n_episodes, gamma, max_steps, env_rng, rng = _common(
        env, n_episodes, gamma, max_steps_per_episode, random_state, initial_value
    )
    alpha_fn = check_step_size(alpha)
    cdf = _check_policy(policy, S, A)
    true_v = _check_true_values(true_values, S)
    terminal = _terminal_mask(env, S)
    V = np.full(S, float(initial_value))
    V[terminal] = 0.0
    visits = np.zeros(S, dtype=np.int64)
    rms = np.empty(n_episodes) if true_v is not None else None
    total = 0
    for ep in range(n_episodes):
        s = _reset(env, env_rng, ep == 0)
        if method == "td0":
            for _ in range(max_steps):
                s2, r, terminated, truncated, _ = env.step(_sample(cdf[s], rng))
                s2 = int(s2)
                total += 1
                visits[s] += 1
                target = r + (0.0 if terminated else gamma * V[s2])
                V[s] += alpha_fn(visits[s]) * (target - V[s])
                s = s2
                if terminated or truncated:
                    break
        elif method == "td_lambda":
            e = np.zeros(S)
            for _ in range(max_steps):
                s2, r, terminated, truncated, _ = env.step(_sample(cdf[s], rng))
                s2 = int(s2)
                total += 1
                visits[s] += 1
                delta = r + (0.0 if terminated else gamma * V[s2]) - V[s]
                e *= gamma * lam
                e[s] = e[s] + 1.0 if trace == "accumulating" else 1.0
                V += alpha_fn(visits[s]) * delta * e
                s = s2
                if terminated or truncated:
                    break
        else:  # n-step TD, Sutton & Barto §7.1
            states, rewards = [s], [0.0]  # rewards[i] is R_i (R_0 unused)
            T = np.inf  # time step at which the episode ended
            ended_terminal = False  # ended by termination (no bootstrap) or a time limit
            t = 0
            while True:
                if t < T:
                    s2, r, terminated, truncated, _ = env.step(_sample(cdf[states[t]], rng))
                    total += 1
                    states.append(int(s2))
                    rewards.append(float(r))
                    if terminated or truncated or t + 1 >= max_steps:
                        T, ended_terminal = t + 1, bool(terminated)
                tau = t - n + 1  # time whose estimate is updated
                if tau >= 0:
                    end = int(min(tau + n, T))
                    G = sum(gamma ** (i - tau - 1) * rewards[i] for i in range(tau + 1, end + 1))
                    if tau + n < T or not ended_terminal:  # S_end is not terminal
                        G += gamma ** (end - tau) * V[states[end]]
                    st = states[tau]
                    visits[st] += 1
                    V[st] += alpha_fn(visits[st]) * (G - V[st])
                if tau == T - 1:
                    break
                t += 1
        if rms is not None:
            rms[ep] = _rms(V, true_v, ~terminal)
    freeze(V)
    if rms is not None:
        freeze(rms)
    return PredictionResult(V, rms, total)


_PRED_DOC_PARAMS = """
    env : environment with Discrete observation and action spaces
    policy : array_like of shape (S, A)
        Action probabilities of the policy to evaluate.
    n_episodes : int
    alpha : float in (0, 1] or callable ``n -> step``, default=0.1
        Step size; a callable gets the visit count of the updated state.
    gamma : float in [0, 1], default=1.0
    initial_value : float, default=0.0
        Initial estimate of every non-terminal state (terminal states start at 0).
    true_values : array_like of shape (S,), optional
        If given, the RMS error over non-terminal states is recorded per episode.
    max_steps_per_episode : int, default=10000
    random_state : None, int, SeedSequence or Generator
"""


@register(
    domain="rl",
    tasks=("prediction",),
    complexity=Complexity(time="O(T)", space="O(S A)", notes=f"O(1) per step; {_NOTES}"),
    references=_PRED_REFS,
    status="experimental",
)
def td0_prediction(
    env,
    policy,
    n_episodes: int,
    alpha: StepSize = 0.1,
    gamma: float = 1.0,
    initial_value: float = 0.0,
    true_values=None,
    max_steps_per_episode: int = 10_000,
    random_state=None,
) -> PredictionResult:
    r"""Tabular TD(0) policy evaluation (§6.1).

    Parameters
    ----------"""
    return _prediction(
        env,
        policy,
        n_episodes,
        alpha,
        gamma,
        initial_value,
        true_values,
        max_steps_per_episode,
        random_state,
        "td0",
    )


td0_prediction.__doc__ = (
    r"""Tabular TD(0) policy evaluation (§6.1).

    :math:`V(S) \leftarrow V(S) + \alpha[R + \gamma V(S') - V(S)]` after every step.

    Parameters
    ----------"""
    + _PRED_DOC_PARAMS
    + r"""
    Returns
    -------
    PredictionResult

    Complexity
    ----------
    O(1) time per step, O(S A) memory (the policy table).

    References
    ----------
    Sutton & Barto (2018), §6.1, Example 6.2 and Figure 6.2.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.rl import RandomWalk, td0_prediction
    >>> env = RandomWalk(5)
    >>> res = td0_prediction(env, np.ones((7, 1)), 1000, alpha=0.02, initial_value=0.5,
    ...                      true_values=env.true_values(), random_state=0)
    >>> bool(res.rms_error[-1] < 0.05)
    True
    """
)


@register(
    domain="rl",
    tasks=("prediction",),
    complexity=Complexity(
        time="O(T n)", space="O(S A + L)", notes=f"n-step sums; L = longest episode; {_NOTES}"
    ),
    references=_PRED_REFS,
    status="experimental",
)
def n_step_td_prediction(
    env,
    policy,
    n_episodes: int,
    n: int = 4,
    alpha: StepSize = 0.1,
    gamma: float = 1.0,
    initial_value: float = 0.0,
    true_values=None,
    max_steps_per_episode: int = 10_000,
    random_state=None,
) -> PredictionResult:
    """n-step TD prediction (§7.1)."""
    n = check_int(n, "n")
    return _prediction(
        env,
        policy,
        n_episodes,
        alpha,
        gamma,
        initial_value,
        true_values,
        max_steps_per_episode,
        random_state,
        "nstep",
        n=n,
    )


n_step_td_prediction.__doc__ = (
    r"""n-step TD prediction (§7.1).

    Updates :math:`V(S_\tau)` toward
    :math:`G_{\tau:\tau+n} = \sum_{i=1}^{n}\gamma^{i-1}R_{\tau+i} + \gamma^n V(S_{\tau+n})`
    (no bootstrap term past a terminal state). ``n = 1`` is TD(0); ``n`` at least
    the episode length is every-visit constant-:math:`\alpha` Monte Carlo.

    Parameters
    ----------"""
    + _PRED_DOC_PARAMS
    + r"""    n : int, default=4
        Number of rewards before bootstrapping.

    Returns
    -------
    PredictionResult

    Complexity
    ----------
    O(n) time per step for the n-step sum; memory O(S A + L) for an episode of
    length L.

    References
    ----------
    Sutton & Barto (2018), §7.1, Example 7.1.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.rl import RandomWalk, n_step_td_prediction, td0_prediction
    >>> env = RandomWalk(19)
    >>> a = n_step_td_prediction(env, np.ones((21, 1)), 50, n=1, random_state=3)
    >>> b = td0_prediction(env, np.ones((21, 1)), 50, random_state=3)
    >>> bool(np.allclose(a.V, b.V, atol=1e-12))  # n = 1 is TD(0)
    True
    """
)


@register(
    domain="rl",
    tasks=("prediction",),
    complexity=Complexity(
        time="O(T S)",
        space="O(S A)",
        notes=f"the trace vector is updated in O(S) per step; {_NOTES}",
    ),
    references=_PRED_REFS,
    status="experimental",
)
def td_lambda_prediction(
    env,
    policy,
    n_episodes: int,
    lam: float = 0.8,
    trace: str = "accumulating",
    alpha: StepSize = 0.1,
    gamma: float = 1.0,
    initial_value: float = 0.0,
    true_values=None,
    max_steps_per_episode: int = 10_000,
    random_state=None,
) -> PredictionResult:
    """Tabular TD(lambda) with eligibility traces (§12.2)."""
    lam = check_unit(lam, "lam")
    if trace not in ("accumulating", "replacing"):
        raise ValueError(f"trace must be 'accumulating' or 'replacing', got {trace!r}.")
    return _prediction(
        env,
        policy,
        n_episodes,
        alpha,
        gamma,
        initial_value,
        true_values,
        max_steps_per_episode,
        random_state,
        "td_lambda",
        lam=lam,
        trace=trace,
    )


td_lambda_prediction.__doc__ = (
    r"""Tabular TD(:math:`\lambda`) with accumulating or replacing traces (§12.2).

    Backward view: :math:`\delta = R + \gamma V(S') - V(S)`,
    :math:`e \leftarrow \gamma\lambda e`, then :math:`e(S) \mathrel{+}= 1` (accumulating)
    or :math:`e(S) = 1` (replacing), and :math:`V \leftarrow V + \alpha\delta e`. Traces
    restart at zero each episode. :math:`\lambda = 0` is TD(0); the step size is
    chosen by the visit count of the current state.

    Parameters
    ----------"""
    + _PRED_DOC_PARAMS
    + r"""    lam : float in [0, 1], default=0.8
    trace : {"accumulating", "replacing"}, default="accumulating"

    Returns
    -------
    PredictionResult

    Complexity
    ----------
    O(S) time per step (the whole trace vector decays), O(S A) memory.

    References
    ----------
    Sutton & Barto (2018), §12.2; Singh & Sutton (1996), *Reinforcement learning with
    replacing eligibility traces*, Machine Learning 22.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.rl import RandomWalk, td0_prediction, td_lambda_prediction
    >>> env = RandomWalk(5)
    >>> a = td_lambda_prediction(env, np.ones((7, 1)), 30, lam=0.0, random_state=1)
    >>> b = td0_prediction(env, np.ones((7, 1)), 30, random_state=1)
    >>> bool(np.allclose(a.V, b.V, atol=1e-12))  # lambda = 0 is TD(0)
    True
    """
)


# ---------------------------------------------------------------------------
# Control
# ---------------------------------------------------------------------------
def _control_inputs(env, n_episodes, epsilon, gamma, initial_value, max_steps, random_state):
    S, A, n_episodes, gamma, max_steps, env_rng, rng = _common(
        env, n_episodes, gamma, max_steps, random_state, initial_value
    )
    epsilon = check_unit(epsilon, "epsilon")
    Q = np.full((S, A), float(initial_value))
    Q[_terminal_mask(env, S)] = 0.0
    return S, A, n_episodes, gamma, max_steps, env_rng, rng, epsilon, Q


def _finish(Q, returns, lengths) -> TDControlResult:
    policy = greedy_policy(Q)
    freeze(Q, policy, returns, lengths)
    return TDControlResult(Q, policy, returns, lengths)


def _td_control(
    kind: str,
    env,
    n_episodes,
    alpha,
    epsilon,
    gamma,
    initial_value,
    max_steps_per_episode,
    random_state,
) -> TDControlResult:
    S, A, n_episodes, gamma, max_steps, env_rng, rng, epsilon, Q = _control_inputs(
        env, n_episodes, epsilon, gamma, initial_value, max_steps_per_episode, random_state
    )
    alpha_fn = check_step_size(alpha)
    visits = np.zeros((S, A), dtype=np.int64)
    returns = np.zeros(n_episodes)
    lengths = np.zeros(n_episodes, dtype=np.int64)
    for ep in range(n_episodes):
        s = _reset(env, env_rng, ep == 0)
        a = epsilon_greedy(Q[s], epsilon, rng)
        for step in range(max_steps):
            s2, r, terminated, truncated, _ = env.step(a)
            s2 = int(s2)
            returns[ep] += r
            lengths[ep] = step + 1
            a2 = epsilon_greedy(Q[s2], epsilon, rng)  # next action (SARSA uses it)
            if terminated:
                bootstrap = 0.0
            elif kind == "sarsa":
                bootstrap = Q[s2, a2]
            elif kind == "expected_sarsa":
                bootstrap = float(epsilon_greedy_probs(Q[s2], epsilon) @ Q[s2])
            else:  # q-learning
                bootstrap = Q[s2].max()
            visits[s, a] += 1
            Q[s, a] += alpha_fn(visits[s, a]) * (r + gamma * bootstrap - Q[s, a])
            s, a = s2, a2
            if terminated or truncated:
                break
    return _finish(Q, returns, lengths)


_CONTROL_DOC_PARAMS = """
    env : environment with Discrete observation and action spaces
    n_episodes : int
    alpha : float in (0, 1] or callable ``n -> step``, default=0.5
        Step size; a callable gets the visit count N(s, a) of the updated pair.
    epsilon : float in [0, 1], default=0.1
        Exploration rate of the epsilon-greedy behaviour policy.
    gamma : float in [0, 1], default=1.0
    initial_value : float, default=0.0
        Initial Q of non-terminal states (terminal states start at 0).
    max_steps_per_episode : int, default=10000
    random_state : None, int, SeedSequence or Generator
"""
_CONTROL_TAIL = """
    Returns
    -------
    TDControlResult

    Complexity
    ----------
    O(A) time per step (epsilon-greedy and the target), O(S A) memory.
"""


def _control_fn(kind: str, doc: str, refs: tuple[str, ...]) -> Callable[..., TDControlResult]:
    def fn(
        env,
        n_episodes: int,
        alpha: StepSize = 0.5,
        epsilon: float = 0.1,
        gamma: float = 1.0,
        initial_value: float = 0.0,
        max_steps_per_episode: int = 10_000,
        random_state=None,
    ) -> TDControlResult:
        return _td_control(
            kind,
            env,
            n_episodes,
            alpha,
            epsilon,
            gamma,
            initial_value,
            max_steps_per_episode,
            random_state,
        )

    fn.__name__ = fn.__qualname__ = kind
    fn.__module__ = __name__
    fn.__doc__ = doc
    return register(
        domain="rl",
        tasks=("control",),
        complexity=Complexity(time="O(T A)", space="O(S A)", notes=f"O(A) per step; {_NOTES}"),
        references=refs,
        status="experimental",
    )(fn)


sarsa = _control_fn(
    "sarsa",
    r"""SARSA: on-policy TD control (§6.4).

    :math:`Q(S,A) \leftarrow Q(S,A) + \alpha[R + \gamma Q(S',A') - Q(S,A)]` with
    :math:`A'` the epsilon-greedy action actually taken next.

    Parameters
    ----------"""
    + _CONTROL_DOC_PARAMS
    + _CONTROL_TAIL
    + r"""
    References
    ----------
    Sutton & Barto (2018), §6.4, Example 6.6; Rummery & Niranjan (1994).

    Examples
    --------
    >>> from lipipy.rl import CliffWalking, sarsa
    >>> res = sarsa(CliffWalking(), 300, alpha=0.5, epsilon=0.1, random_state=0)
    >>> res.Q.shape, bool(res.episode_returns[-50:].mean() > -40)
    ((48, 4), True)
    """,
    (_SB + ", §6.4", "Rummery & Niranjan (1994), On-line Q-learning using connectionist systems"),
)

expected_sarsa = _control_fn(
    "expected_sarsa",
    r"""Expected SARSA (§6.6).

    Target :math:`R + \gamma\sum_a \pi(a|S')Q(S',a)` under the epsilon-greedy policy,
    which removes the variance due to the random choice of :math:`A'`.

    Parameters
    ----------"""
    + _CONTROL_DOC_PARAMS
    + _CONTROL_TAIL
    + r"""
    References
    ----------
    Sutton & Barto (2018), §6.6; van Seijen et al. (2009), ADPRL.

    Examples
    --------
    >>> from lipipy.rl import CliffWalking, expected_sarsa
    >>> res = expected_sarsa(CliffWalking(), 300, alpha=0.9, random_state=0)
    >>> bool(res.episode_returns[-50:].mean() > -40)
    True
    """,
    (_SB + ", §6.6", "van Seijen, van Hasselt, Whiteson & Wiering (2009), ADPRL"),
)

q_learning = _control_fn(
    "q_learning",
    r"""Q-learning: off-policy TD control (§6.5).

    :math:`Q(S,A) \leftarrow Q(S,A) + \alpha[R + \gamma\max_a Q(S',a) - Q(S,A)]`. With
    every pair visited infinitely often and step sizes satisfying
    :math:`\sum\alpha_n = \infty, \sum\alpha_n^2 < \infty`, :math:`Q \to q_*`.

    Parameters
    ----------"""
    + _CONTROL_DOC_PARAMS
    + _CONTROL_TAIL
    + r"""
    References
    ----------
    Sutton & Barto (2018), §6.5; Watkins & Dayan (1992), Machine Learning 8.

    Examples
    --------
    >>> from lipipy.rl import CliffWalking, q_learning
    >>> res = q_learning(CliffWalking(), 300, alpha=0.5, random_state=0)
    >>> int(res.policy[36])  # from the start, the greedy action is up
    0
    """,
    (_SB + ", §6.5", "Watkins & Dayan (1992), Q-learning, Machine Learning 8"),
)


@register(
    domain="rl",
    tasks=("control",),
    complexity=Complexity(time="O(T A)", space="O(S A)", notes=f"O(A) per step; {_NOTES}"),
    references=(_SB + ", §6.7", "van Hasselt (2010), Double Q-learning, NeurIPS"),
    status="experimental",
)
def double_q_learning(
    env,
    n_episodes: int,
    alpha: StepSize = 0.5,
    epsilon: float = 0.1,
    gamma: float = 1.0,
    initial_value: float = 0.0,
    max_steps_per_episode: int = 10_000,
    random_state=None,
) -> TDControlResult:
    """Double Q-learning (§6.7)."""
    S, A, n_episodes, gamma, max_steps, env_rng, rng, epsilon, Q1 = _control_inputs(
        env, n_episodes, epsilon, gamma, initial_value, max_steps_per_episode, random_state
    )
    alpha_fn = check_step_size(alpha)
    Q2 = Q1.copy()
    visits = np.zeros((2, S, A), dtype=np.int64)
    returns = np.zeros(n_episodes)
    lengths = np.zeros(n_episodes, dtype=np.int64)
    for ep in range(n_episodes):
        s = _reset(env, env_rng, ep == 0)
        for step in range(max_steps):
            a = epsilon_greedy(Q1[s] + Q2[s], epsilon, rng)
            s2, r, terminated, truncated, _ = env.step(a)
            s2 = int(s2)
            returns[ep] += r
            lengths[ep] = step + 1
            first = rng.random() < 0.5
            Qa, Qb, i = (Q1, Q2, 0) if first else (Q2, Q1, 1)
            bootstrap = 0.0 if terminated else Qb[s2, argmax_random(Qa[s2], rng)]
            visits[i, s, a] += 1
            Qa[s, a] += alpha_fn(visits[i, s, a]) * (r + gamma * bootstrap - Qa[s, a])
            s = s2
            if terminated or truncated:
                break
    return _finish((Q1 + Q2) / 2.0, returns, lengths)


double_q_learning.__doc__ = (
    r"""Double Q-learning: decoupled action selection and evaluation (§6.7).

    With probability 1/2,
    :math:`Q_1(S,A) \leftarrow Q_1 + \alpha[R + \gamma Q_2(S', \arg\max_a Q_1(S',a)) - Q_1]`,
    otherwise the same with the tables swapped; behaviour is epsilon-greedy in
    :math:`Q_1 + Q_2`. This removes the maximisation bias of Q-learning
    (Example 6.7). The returned ``Q`` is :math:`(Q_1 + Q_2)/2`.

    Parameters
    ----------"""
    + _CONTROL_DOC_PARAMS
    + _CONTROL_TAIL
    + r"""
    References
    ----------
    Sutton & Barto (2018), §6.7, Example 6.7; van Hasselt (2010).

    Examples
    --------
    >>> from lipipy.rl import CliffWalking, double_q_learning
    >>> res = double_q_learning(CliffWalking(), 300, alpha=0.5, random_state=0)
    >>> res.episode_returns.shape
    (300,)
    """
)


@register(
    domain="rl",
    tasks=("control",),
    complexity=Complexity(
        time="O(T A)", space="O(S A + L)", notes=f"L = longest episode; {_NOTES}"
    ),
    references=(_SB + ", §5.4",),
    status="experimental",
)
def mc_control(
    env,
    n_episodes: int,
    epsilon: float = 0.1,
    gamma: float = 1.0,
    initial_value: float = 0.0,
    max_steps_per_episode: int = 10_000,
    random_state=None,
) -> TDControlResult:
    """On-policy first-visit Monte Carlo control with an epsilon-soft policy (§5.4)."""
    S, A, n_episodes, gamma, max_steps, env_rng, rng, epsilon, Q = _control_inputs(
        env, n_episodes, epsilon, gamma, initial_value, max_steps_per_episode, random_state
    )
    counts = np.zeros((S, A), dtype=np.int64)
    returns = np.zeros(n_episodes)
    lengths = np.zeros(n_episodes, dtype=np.int64)
    for ep in range(n_episodes):
        s = _reset(env, env_rng, ep == 0)
        episode: list[tuple[int, int, float]] = []
        for _ in range(max_steps):
            a = epsilon_greedy(Q[s], epsilon, rng)
            s2, r, terminated, truncated, _ = env.step(a)
            episode.append((s, a, float(r)))
            s = int(s2)
            if terminated or truncated:
                break
        returns[ep] = sum(r for _, _, r in episode)
        lengths[ep] = len(episode)
        first_visit = {}
        for t, (st, at, _) in enumerate(episode):
            first_visit.setdefault((st, at), t)
        G = 0.0
        for t in range(len(episode) - 1, -1, -1):
            st, at, r = episode[t]
            G = gamma * G + r
            if first_visit[(st, at)] == t:
                counts[st, at] += 1
                Q[st, at] += (G - Q[st, at]) / counts[st, at]
    return _finish(Q, returns, lengths)


mc_control.__doc__ = r"""On-policy first-visit Monte Carlo control, epsilon-soft policy (§5.4).

    Generate an episode with the epsilon-greedy policy of ``Q``; going backwards,
    :math:`G \leftarrow \gamma G + R_{t+1}`, and at the first visit of
    :math:`(S_t, A_t)` move :math:`Q(S_t, A_t)` to the average of its returns.
    An episode cut by a time limit is used as is (its returns are truncated).
    The method converges to the best epsilon-soft policy (not the best
    deterministic one), and slowly: the sample averages keep the returns of the
    early, worse policies.

    Parameters
    ----------
    env : environment with Discrete observation and action spaces
    n_episodes : int
    epsilon : float in [0, 1], default=0.1
    gamma : float in [0, 1], default=1.0
    initial_value : float, default=0.0
    max_steps_per_episode : int, default=10000
    random_state : None, int, SeedSequence or Generator

    Returns
    -------
    TDControlResult

    Complexity
    ----------
    O(A) per step plus O(L) per episode of length L; memory O(S A + L).

    References
    ----------
    Sutton & Barto (2018), §5.4.

    Examples
    --------
    >>> from lipipy.rl import GridWorld, mc_control
    >>> res = mc_control(GridWorld.example_4_1(), 300, epsilon=0.1, random_state=0)
    >>> res.policy.shape, bool(res.episode_returns[-50:].mean() > -10)
    ((16,), True)
    """
