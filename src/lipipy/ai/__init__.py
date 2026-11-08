"""Classical (symbolic) artificial intelligence.

The "good old-fashioned AI" toolbox: search, logic, planning and reasoning
under uncertainty. Much of it is used inside modern agents and planners.

Catalog. ``[x]`` = implemented, ``[ ]`` = planned.

Uninformed search
-----------------
[ ] Breadth-first, depth-first, depth-limited, iterative deepening
[ ] Uniform-cost search (Dijkstra)
[ ] Bidirectional search

Informed (heuristic) search
---------------------------
[ ] Greedy best-first
[ ] A*, IDA*, weighted A*
[ ] Admissible/consistent heuristics, pattern databases

Local search and metaheuristics
-------------------------------
[ ] Hill climbing (steepest, stochastic, random restart)
[ ] Simulated annealing
[ ] Tabu search
[ ] Genetic algorithms, genetic programming
[ ] Evolution strategies (CMA-ES), differential evolution
[ ] Particle swarm optimization, ant colony optimization

Adversarial search (games)
--------------------------
[ ] Minimax, alpha-beta pruning, expectiminimax
[ ] Monte Carlo tree search (shared with lipipy.rl)
[ ] Counterfactual regret minimization (CFR) for imperfect-information games

Constraint satisfaction
-----------------------
[ ] Backtracking search with MRV / degree / LCV heuristics
[ ] Forward checking, arc consistency (AC-3)
[ ] Min-conflicts local search
[ ] SAT solving: DPLL, CDCL, WalkSAT

Logic and knowledge representation
----------------------------------
[ ] Propositional logic: truth tables, resolution, CNF conversion
[ ] First-order logic: unification, forward/backward chaining, resolution
[ ] Prolog-style logic programming
[ ] Description logics, ontologies, knowledge graphs
[ ] Knowledge-graph embeddings (TransE, RotatE)

Planning
--------
[ ] STRIPS / PDDL representation
[ ] Forward and backward state-space planning
[ ] GraphPlan
[ ] Hierarchical task networks (HTN)
[ ] Partial-order planning

Reasoning under uncertainty
---------------------------
[ ] Bayesian networks (exact and approximate inference; shared with lipipy.ml)
[ ] Markov decision processes (shared with lipipy.rl)
[ ] POMDPs (belief-state planning)
[ ] Dempster-Shafer theory, fuzzy logic and fuzzy inference

Decision theory and multi-agent systems
---------------------------------------
[ ] Utility theory, expected utility, value of information
[ ] Game theory: Nash equilibrium, mixed strategies, mechanism design basics
[ ] Auctions and voting

Causality
---------
[ ] Structural causal models, do-calculus
[ ] Causal discovery (PC algorithm, GES)
[ ] Treatment-effect estimation (propensity scores, IPW, doubly robust)
"""
