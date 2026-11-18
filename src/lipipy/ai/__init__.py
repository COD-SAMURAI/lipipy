"""Classical (symbolic) artificial intelligence.

The "good old-fashioned AI" toolbox: search, logic, planning and reasoning
under uncertainty. Much of it is used inside modern agents and planners.

Catalog. ``[x]`` = implemented, ``[ ]`` = planned.

Uninformed search
-----------------
[x] Breadth-first, depth-first, depth-limited, iterative deepening
[x] Uniform-cost search (Dijkstra)
[x] Bidirectional search (breadth-first)

Informed (heuristic) search
---------------------------
[x] Greedy best-first
[x] A*, IDA*, weighted A*
[x] Admissible/consistent heuristics (Manhattan distance for sliding puzzles)
[ ] Pattern databases

Local search and metaheuristics
-------------------------------
[x] Hill climbing (steepest, first-choice, random restart)
[x] Simulated annealing
[ ] Tabu search
[x] Genetic algorithms
[ ] Genetic programming
[ ] Evolution strategies (CMA-ES), differential evolution
[ ] Particle swarm optimization, ant colony optimization

Adversarial search (games)
--------------------------
[x] Minimax, alpha-beta pruning, expectiminimax
[x] Monte Carlo tree search (UCT; shared with lipipy.rl)
[ ] Counterfactual regret minimization (CFR) for imperfect-information games

Constraint satisfaction
-----------------------
[x] Backtracking search with MRV / degree / LCV heuristics
[x] Forward checking, arc consistency (AC-3, MAC)
[x] Min-conflicts local search
[x] SAT solving: DPLL, CDCL, WalkSAT

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

from lipipy.ai.csp import (
    CSP,
    AC3Result,
    Constraint,
    CSPResult,
    ac3,
    all_different,
    backtracking_search,
    min_conflicts,
)
from lipipy.ai.games import (
    ConnectFour,
    Game,
    GameResult,
    MCTSResult,
    Nim,
    TicTacToe,
    alphabeta,
    expectiminimax,
    mcts,
    minimax,
)
from lipipy.ai.local import (
    ContinuousProblem,
    GAResult,
    LocalSearchProblem,
    LocalSearchResult,
    NQueensProblem,
    exp_schedule,
    genetic_algorithm,
    hill_climbing,
    simulated_annealing,
)
from lipipy.ai.sat import CNF, SATResult, cdcl, dpll, parse_dimacs, walksat
from lipipy.ai.search import (
    GraphProblem,
    SearchProblem,
    SearchResult,
    SlidingPuzzle,
    astar,
    bidirectional_bfs,
    breadth_first,
    depth_first,
    greedy_best_first,
    ida_star,
    iterative_deepening,
    uniform_cost,
    weighted_astar,
)

__all__ = [
    "AC3Result",
    "CNF",
    "CSP",
    "CSPResult",
    "ConnectFour",
    "Constraint",
    "ContinuousProblem",
    "GAResult",
    "Game",
    "GameResult",
    "GraphProblem",
    "LocalSearchProblem",
    "LocalSearchResult",
    "MCTSResult",
    "NQueensProblem",
    "Nim",
    "SATResult",
    "SearchProblem",
    "SearchResult",
    "SlidingPuzzle",
    "TicTacToe",
    "ac3",
    "all_different",
    "alphabeta",
    "astar",
    "backtracking_search",
    "bidirectional_bfs",
    "breadth_first",
    "cdcl",
    "depth_first",
    "dpll",
    "exp_schedule",
    "expectiminimax",
    "genetic_algorithm",
    "greedy_best_first",
    "hill_climbing",
    "ida_star",
    "iterative_deepening",
    "mcts",
    "min_conflicts",
    "minimax",
    "parse_dimacs",
    "simulated_annealing",
    "uniform_cost",
    "walksat",
    "weighted_astar",
]
