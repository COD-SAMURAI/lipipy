"""Reinforcement learning.

Catalog. ``[x]`` = implemented, ``[ ]`` = planned.

Bandits
-------
[ ] Epsilon-greedy, UCB1, Thompson sampling, EXP3
[ ] Contextual bandits (LinUCB)

Dynamic programming (known model)
---------------------------------
[ ] Policy evaluation, policy iteration, value iteration

Model-free prediction and control (tabular)
-------------------------------------------
[ ] Monte Carlo prediction and control
[ ] TD(0), n-step TD, TD(lambda) with eligibility traces
[ ] SARSA, Expected SARSA
[ ] Q-learning, Double Q-learning

Value-based deep RL
-------------------
[ ] DQN (experience replay, target network)
[ ] Double DQN, Dueling DQN, prioritized replay, Rainbow
[ ] Distributional RL (C51, QR-DQN)

Policy gradient and actor-critic
--------------------------------
[ ] REINFORCE (with baseline)
[ ] Actor-critic, A2C, A3C
[ ] Generalized advantage estimation (GAE)
[ ] TRPO, PPO (clip and KL variants)
[ ] DDPG, TD3
[ ] Soft actor-critic (SAC)

Model-based RL and planning
---------------------------
[ ] Dyna-Q
[ ] Monte Carlo tree search (MCTS, UCT)
[ ] AlphaZero-style self-play
[ ] MuZero-style learned models
[ ] World models, Dreamer
[ ] Model predictive control (MPC, CEM)

Other settings
--------------
[ ] Offline RL (CQL, IQL, Decision Transformer)
[ ] Imitation learning (behaviour cloning, DAgger)
[ ] Inverse RL (max-entropy IRL), GAIL
[ ] Multi-agent RL (independent Q-learning, MADDPG, QMIX)
[ ] Hierarchical RL (options, feudal networks)
[ ] Exploration: curiosity (ICM), RND, count-based bonuses
[ ] Meta-RL (RL^2, MAML for RL)

Environments
------------
[ ] Grid worlds, cliff walking, multi-armed testbed
[ ] Gymnasium adapters
"""

from lipipy.rl.bandits import (
    UCB1,
    BanditRun,
    BernoulliThompson,
    EpsilonGreedy,
    GaussianThompson,
    GradientBandit,
    TestbedResult,
    bandit_testbed,
    run_bandit,
)
from lipipy.rl.dp import (
    ControlResult,
    PolicyEvaluationResult,
    optimal_actions,
    policy_evaluation,
    policy_iteration,
    value_iteration,
)
from lipipy.rl.envs import (
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
from lipipy.rl.td import (
    PredictionResult,
    TDControlResult,
    double_q_learning,
    expected_sarsa,
    mc_control,
    n_step_td_prediction,
    q_learning,
    sarsa,
    td0_prediction,
    td_lambda_prediction,
)

__all__ = [
    "BanditRun",
    "BernoulliBandit",
    "BernoulliThompson",
    "Box",
    "CliffWalking",
    "ControlResult",
    "Discrete",
    "Env",
    "EpsilonGreedy",
    "FrozenLake",
    "GaussianThompson",
    "GradientBandit",
    "GridWorld",
    "KArmedBandit",
    "PolicyEvaluationResult",
    "PredictionResult",
    "RandomWalk",
    "ShortCorridor",
    "TDControlResult",
    "TabularMDP",
    "TestbedResult",
    "UCB1",
    "bandit_testbed",
    "double_q_learning",
    "expected_sarsa",
    "mc_control",
    "n_step_td_prediction",
    "optimal_actions",
    "policy_evaluation",
    "policy_iteration",
    "q_learning",
    "run_bandit",
    "sarsa",
    "td0_prediction",
    "td_lambda_prediction",
    "value_iteration",
]
