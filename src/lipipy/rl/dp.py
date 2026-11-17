r"""Dynamic programming for finite MDPs with a known model.

Bellman equations
-----------------
With the continuation probabilities :math:`\tilde P(s'|s,a)` and expected rewards
:math:`r(s,a)` of :meth:`lipipy.rl.TabularMDP.transition_arrays` (terminating
transitions contribute their reward but no future value),

.. math::
    q_\pi(s,a) = r(s,a) + \gamma \sum_{s'} \tilde P(s'|s,a)\, v_\pi(s'), \qquad
    v_\pi(s) = \sum_a \pi(a|s)\, q_\pi(s,a),

and the optimal values satisfy :math:`v_*(s) = \max_a q_*(s,a)`.

Algorithms (Sutton & Barto ch. 4)
---------------------------------
* **Iterative policy evaluation** (§4.1): repeat the synchronous ("two-array")
  sweep :math:`v_{k+1} = r_\pi + \gamma \tilde P_\pi v_k` until
  :math:`\max_s |v_{k+1}(s) - v_k(s)| < \theta`. For :math:`\gamma < 1` the sweep is a
  :math:`\gamma`-contraction, so the final error is at most
  :math:`\gamma\theta / (1-\gamma)`; with :math:`\gamma = 1` it converges when the
  policy terminates with probability one from every state.
* **Policy iteration** (§4.3): alternate evaluation and greedy improvement
  :math:`\pi'(s) = \arg\max_a q_\pi(s,a)`, starting from the uniform random policy,
  until the policy is stable. A state keeps its current action when that action is
  still within ``1e-12`` (relative) of the best, so tied actions cannot make the
  iteration cycle.
* **Value iteration** (§4.4): :math:`v_{k+1}(s) = \max_a [r(s,a) + \gamma \sum_{s'}
  \tilde P(s'|s,a) v_k(s')]` until the change is below :math:`\theta`.

The returned greedy policy takes the lowest-index maximising action; when several
actions are optimal the optimal policy is not unique, so compare values, not
policies (``optimal_actions`` gives every maximiser).

References
----------
Sutton & Barto (2018), *Reinforcement Learning: An Introduction*, 2nd ed., §4.1-4.4;
Bellman (1957), *Dynamic Programming*; Howard (1960), *Dynamic Programming and
Markov Processes*.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from lipipy.core.complexity import Complexity
from lipipy.core.errors import ConvergenceWarning
from lipipy.core.registry import register
from lipipy.rl._common import ResultMixin, check_gamma, check_int, check_positive, freeze
from lipipy.rl.envs import TabularMDP

_REFS = ("Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., ch. 4",)
_NOTES = "S states, A actions; dense model arrays take O(S^2 A) memory"


@dataclass(frozen=True)
class PolicyEvaluationResult(ResultMixin):
    """Outcome of :func:`policy_evaluation`.

    Attributes
    ----------
    V : ndarray of shape (S,)
    Q : ndarray of shape (S, A)
    n_iterations : int
        Sweeps performed.
    delta : float
        Largest value change in the last sweep.
    converged : bool
        Whether ``delta < theta``.
    """

    V: np.ndarray
    Q: np.ndarray
    n_iterations: int
    delta: float
    converged: bool


@dataclass(frozen=True)
class ControlResult(ResultMixin):
    """Outcome of :func:`policy_iteration` and :func:`value_iteration`.

    Attributes
    ----------
    V : ndarray of shape (S,)
        Optimal state values (approximately, to the stopping tolerance).
    Q : ndarray of shape (S, A)
    policy : ndarray of int, shape (S,)
        A greedy policy (lowest action index among maximisers).
    n_iterations : int
        Improvement steps (policy iteration) or sweeps (value iteration).
    converged : bool
    """

    V: np.ndarray
    Q: np.ndarray
    policy: np.ndarray
    n_iterations: int
    converged: bool


def _model(mdp) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(mdp, TabularMDP):
        raise TypeError(f"mdp must be a TabularMDP, got {type(mdp).__name__}.")
    return mdp.transition_arrays()


def _check_policy(policy, S: int, A: int) -> np.ndarray:
    pi = np.asarray(policy, dtype=np.float64)
    if pi.shape == (S,):  # deterministic policy given as actions
        actions = np.asarray(policy)
        if actions.dtype.kind not in "iu" or np.any(actions < 0) or np.any(actions >= A):
            raise ValueError(f"A deterministic policy must hold actions in [0, {A}).")
        pi = np.zeros((S, A))
        pi[np.arange(S), actions] = 1.0
    if pi.shape != (S, A):
        raise ValueError(f"policy must have shape ({S}, {A}) or ({S},), got {pi.shape}.")
    if np.any(~np.isfinite(pi)) or np.any(pi < 0) or np.any(np.abs(pi.sum(axis=1) - 1) > 1e-9):
        raise ValueError("Every row of policy must be a probability distribution.")
    return pi


def _check_options(gamma, theta, max_iterations) -> tuple[float, float, int]:
    return (
        check_gamma(gamma),
        check_positive(theta, "theta"),
        check_int(max_iterations, "max_iterations"),
    )


def _evaluate(P_cont, R, pi, gamma, theta, max_iterations, V0=None):
    """Synchronous sweeps; returns (V, Q, n_iterations, delta, converged)."""
    S = R.shape[0]
    r_pi = np.sum(pi * R, axis=1)
    P_pi = np.einsum("sa,sat->st", pi, P_cont)
    V = np.zeros(S) if V0 is None else V0.copy()
    delta, n = np.inf, 0
    with np.errstate(over="ignore", invalid="ignore"):
        while n < max_iterations:
            V_new = r_pi + gamma * (P_pi @ V)
            delta = float(np.max(np.abs(V_new - V)))
            V, n = V_new, n + 1
            if not np.isfinite(delta) or delta < theta:
                break
    converged = bool(np.isfinite(delta) and delta < theta)
    Q = R + gamma * (P_cont @ V)
    return V, Q, n, delta, converged


@register(
    domain="rl",
    tasks=("prediction", "dynamic-programming"),
    complexity=Complexity(
        time="O(T S^2 A)",
        space="O(S^2 A)",
        notes=f"T sweeps of O(S^2) after forming P_pi in O(S^2 A); {_NOTES}",
    ),
    references=_REFS,
    status="experimental",
)
def policy_evaluation(
    mdp: TabularMDP,
    policy,
    gamma: float = 1.0,
    theta: float = 1e-10,
    max_iterations: int = 100_000,
) -> PolicyEvaluationResult:
    r"""Iterative policy evaluation: the value of a fixed policy (§4.1).

    Parameters
    ----------
    mdp : TabularMDP
    policy : array_like of shape (S, A) or int array of shape (S,)
        Action probabilities per state, or one action per state.
    gamma : float in [0, 1], default=1.0
    theta : float > 0, default=1e-10
        Stop when the largest change of a sweep is below ``theta``.
    max_iterations : int, default=100000

    Returns
    -------
    PolicyEvaluationResult

    Warns
    -----
    ConvergenceWarning
        If ``theta`` is not reached (e.g. a policy that never terminates with
        ``gamma = 1``).

    Complexity
    ----------
    Time O(S^2 A + T S^2) for T sweeps, memory O(S^2 A) for the dense model.

    References
    ----------
    Sutton & Barto (2018), §4.1, Example 4.1.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.rl import GridWorld, policy_evaluation
    >>> env = GridWorld.example_4_1()
    >>> res = policy_evaluation(env, np.full((16, 4), 0.25), gamma=1.0)
    >>> res.V.reshape(4, 4).round(1).tolist()[0]
    [0.0, -14.0, -20.0, -22.0]
    """
    P_cont, R = _model(mdp)
    gamma, theta, max_iterations = _check_options(gamma, theta, max_iterations)
    pi = _check_policy(policy, *R.shape)
    V, Q, n, delta, converged = _evaluate(P_cont, R, pi, gamma, theta, max_iterations)
    if not converged:
        warnings.warn(
            f"policy_evaluation did not converge in {n} sweeps (last change {delta:.3g}); "
            "with gamma=1 the policy may not terminate.",
            ConvergenceWarning,
            stacklevel=2,
        )
    freeze(V, Q)
    return PolicyEvaluationResult(V, Q, n, delta, converged)


@register(
    domain="rl",
    tasks=("control", "dynamic-programming"),
    complexity=Complexity(
        time="O(T S^2 A)",
        space="O(S^2 A)",
        notes=(
            f"T = total evaluation sweeps over all improvement steps; {_NOTES}; "
            "the number of improvement steps is at most A^S and small in practice"
        ),
    ),
    references=(*_REFS, "Howard (1960), Dynamic Programming and Markov Processes"),
    status="experimental",
)
def policy_iteration(
    mdp: TabularMDP,
    gamma: float = 1.0,
    theta: float = 1e-10,
    max_iterations: int = 1000,
    max_evaluation_sweeps: int = 100_000,
) -> ControlResult:
    r"""Policy iteration: evaluate, improve greedily, repeat until stable (§4.3).

    Starts from the uniform random policy (which terminates in the book's episodic
    examples, so that its values are finite even with ``gamma = 1``).

    Parameters
    ----------
    mdp : TabularMDP
    gamma : float in [0, 1], default=1.0
    theta : float > 0, default=1e-10
        Evaluation tolerance.
    max_iterations : int, default=1000
        Maximum improvement steps.
    max_evaluation_sweeps : int, default=100000
        Maximum sweeps per evaluation.

    Returns
    -------
    ControlResult

    Warns
    -----
    ConvergenceWarning
        If an evaluation or the improvement loop does not converge.

    Complexity
    ----------
    Time O(T S^2 A) for T evaluation sweeps in total, memory O(S^2 A).

    References
    ----------
    Sutton & Barto (2018), §4.3; Howard (1960).

    Examples
    --------
    >>> from lipipy.rl import GridWorld, policy_iteration
    >>> res = policy_iteration(GridWorld.example_4_1())
    >>> res.V.reshape(4, 4).tolist()[0], res.converged
    ([0.0, -1.0, -2.0, -3.0], True)
    """
    P_cont, R = _model(mdp)
    gamma, theta, max_iterations = _check_options(gamma, theta, max_iterations)
    max_evaluation_sweeps = check_int(max_evaluation_sweeps, "max_evaluation_sweeps")
    S, A = R.shape
    pi = np.full((S, A), 1.0 / A)
    actions = None
    V = None
    converged = False
    n = 0
    eval_ok = True
    while n < max_iterations:
        V, Q, _, _, ok = _evaluate(P_cont, R, pi, gamma, theta, max_evaluation_sweeps, V)
        eval_ok &= ok
        n += 1
        best = Q.max(axis=1)
        tol = 1e-12 * np.maximum(1.0, np.abs(best))
        if actions is None:
            new_actions = np.argmax(Q, axis=1)
        else:
            keep = Q[np.arange(S), actions] >= best - tol
            new_actions = np.where(keep, actions, np.argmax(Q, axis=1))
        if actions is not None and np.array_equal(new_actions, actions):
            converged = True
            break
        actions = new_actions
        pi = np.zeros((S, A))
        pi[np.arange(S), actions] = 1.0
    converged = converged and eval_ok
    if not converged:
        warnings.warn(
            f"policy_iteration did not converge in {n} improvement steps.",
            ConvergenceWarning,
            stacklevel=2,
        )
    Q = R + gamma * (P_cont @ V)
    policy = np.argmax(Q, axis=1)
    freeze(V, Q, policy)
    return ControlResult(V, Q, policy, n, converged)


@register(
    domain="rl",
    tasks=("control", "dynamic-programming"),
    complexity=Complexity(
        time="O(T S^2 A)", space="O(S^2 A)", notes=f"T sweeps of O(S^2 A); {_NOTES}"
    ),
    references=(*_REFS, "Bellman (1957), Dynamic Programming"),
    status="experimental",
)
def value_iteration(
    mdp: TabularMDP,
    gamma: float = 1.0,
    theta: float = 1e-10,
    max_iterations: int = 100_000,
) -> ControlResult:
    r"""Value iteration: repeated Bellman optimality backups (§4.4).

    Parameters
    ----------
    mdp : TabularMDP
    gamma : float in [0, 1], default=1.0
    theta : float > 0, default=1e-10
    max_iterations : int, default=100000

    Returns
    -------
    ControlResult

    Warns
    -----
    ConvergenceWarning
        If ``theta`` is not reached.

    Complexity
    ----------
    Time O(S^2 A) per sweep, memory O(S^2 A).

    References
    ----------
    Sutton & Barto (2018), §4.4; Bellman (1957).

    Examples
    --------
    >>> from lipipy.rl import FrozenLake, value_iteration
    >>> res = value_iteration(FrozenLake(), gamma=0.99, theta=1e-12)
    >>> round(float(res.V[0]), 3), res.policy[0].item()
    (0.542, 0)
    """
    P_cont, R = _model(mdp)
    gamma, theta, max_iterations = _check_options(gamma, theta, max_iterations)
    V = np.zeros(R.shape[0])
    delta, n = np.inf, 0
    with np.errstate(over="ignore", invalid="ignore"):
        while n < max_iterations:
            V_new = np.max(R + gamma * (P_cont @ V), axis=1)
            delta = float(np.max(np.abs(V_new - V)))
            V, n = V_new, n + 1
            if not np.isfinite(delta) or delta < theta:
                break
    converged = bool(np.isfinite(delta) and delta < theta)
    if not converged:
        warnings.warn(
            f"value_iteration did not converge in {n} sweeps (last change {delta:.3g}).",
            ConvergenceWarning,
            stacklevel=2,
        )
    Q = R + gamma * (P_cont @ V)
    policy = np.argmax(Q, axis=1)
    freeze(V, Q, policy)
    return ControlResult(V, Q, policy, n, converged)


@register(
    domain="rl",
    tasks=("control",),
    complexity=Complexity(time="O(S A)", space="O(S A)", notes="S states, A actions"),
    references=_REFS,
    status="experimental",
)
def optimal_actions(Q, atol: float = 1e-9) -> np.ndarray:
    """Boolean mask (S, A) of the actions within ``atol`` of the best in each state.

    Optimal policies are not unique when actions tie; a policy is optimal exactly
    when it puts all its mass on this mask.

    Complexity
    ----------
    O(S A).

    References
    ----------
    Sutton & Barto (2018), §3.6.

    Examples
    --------
    >>> optimal_actions([[1.0, 1.0, 0.0], [0.0, 2.0, 1.0]]).tolist()
    [[True, True, False], [False, True, False]]
    """
    Q = np.asarray(Q, dtype=np.float64)
    if Q.ndim != 2 or Q.size == 0 or not np.all(np.isfinite(Q)):
        raise ValueError("Q must be a finite, non-empty 2D array.")
    check_positive(atol, "atol", allow_zero=True)
    return Q >= Q.max(axis=1, keepdims=True) - atol
