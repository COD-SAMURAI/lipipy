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
