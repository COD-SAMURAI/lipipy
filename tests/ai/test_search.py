"""lipipy.ai.search vs a reference Dijkstra / BFS and on the 8-puzzle."""

import heapq
import math
from collections import deque

import numpy as np
import pytest

from lipipy.ai import (
    GraphProblem,
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
from lipipy.bench import estimate_complexity


# ---------------------------------------------------------------------------
# references
# ---------------------------------------------------------------------------
def dijkstra(adj, src):
    dist = {src: 0.0}
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj.get(u, {}).items():
            if d + w < dist.get(v, math.inf):
                dist[v] = d + w
                heapq.heappush(pq, (d + w, v))
    return dist


def bfs_hops(adj, src):
    hops = {src: 0}
    q = deque([src])
    while q:
        u = q.popleft()
        for v in adj.get(u, {}):
            if v not in hops:
                hops[v] = hops[u] + 1
                q.append(v)
    return hops


def reverse(adj):
    rev = {u: {} for u in adj}
    for u, row in adj.items():
        for v, w in row.items():
            rev.setdefault(v, {})[u] = w
    return rev


def random_graph(rng, n, p, zero_frac=0.1, integer=False):
    adj = {i: {} for i in range(n)}
    for u in range(n):
        for v in range(n):
            if u != v and rng.random() < p:
                w = float(rng.integers(0, 5)) if integer else float(rng.uniform(0.1, 10.0))
                if rng.random() < zero_frac:
                    w = 0.0
                adj[u][v] = w
    return adj


def check_valid(problem, res):
    """The returned path is a real path from start to a goal with the reported cost."""
    assert res.path[0] == problem.initial_state and problem.is_goal(res.path[-1])
    total = 0.0
    for s, a, s2 in zip(res.path[:-1], res.actions, res.path[1:], strict=True):
        assert a in list(problem.actions(s)) and problem.result(s, a) == s2
        total += problem.step_cost(s, a, s2)
    assert res.cost == pytest.approx(total, abs=1e-9)


GRAPHS = [(seed, n, p) for seed in range(40) for (n, p) in ((8, 0.3), (25, 0.12), (60, 0.06))]


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("seed, n, p", GRAPHS)
def test_optimal_costs_equal_dijkstra(seed, n, p):
    rng = np.random.default_rng(seed)
    adj = random_graph(rng, n, p)
    start, goal = 0, n - 1
    dist = dijkstra(adj, start)
    opt = dist.get(goal, math.inf)
    to_goal = dijkstra(reverse(adj), goal)
    consistent = {v: 0.9 * d for v, d in to_goal.items()}
    inconsistent = {v: float(rng.uniform(0, 1)) * d for v, d in to_goal.items()}
    for name, h in (("zero", None), ("consistent", consistent), ("inconsistent", inconsistent)):
        prob = GraphProblem(adj, start, goal, heuristic=h)
        # IDA* is a tree search: exponential on dense cyclic graphs, so small graphs only.
        for method in (uniform_cost, astar, ida_star) if n <= 10 else (uniform_cost, astar):
            res = method(prob)
            assert res.found == (opt < math.inf), (name, method.__name__)
            if res.found:
                assert res.cost == pytest.approx(opt, rel=1e-12, abs=1e-12)
                check_valid(prob, res)
            else:
                assert res.cost == math.inf and res.path == ()
        for w in (1.0, 1.5, 3.0):
            res = weighted_astar(prob, w=w)
            assert res.found == (opt < math.inf)
            if res.found:
                assert res.cost <= w * opt + 1e-9
                check_valid(prob, res)


@pytest.mark.parametrize("seed, n, p", GRAPHS)
def test_astar_consistent_expands_no_more_than_ucs(seed, n, p):
    rng = np.random.default_rng(seed)
    adj = random_graph(rng, n, p, zero_frac=0.0)  # continuous weights: no f-ties
    goal = n - 1
    to_goal = dijkstra(reverse(adj), goal)
    h = {v: 0.9 * d for v, d in to_goal.items()}
    ucs = uniform_cost(GraphProblem(adj, 0, goal))
    ast = astar(GraphProblem(adj, 0, goal, heuristic=h))
    assert ast.nodes_expanded <= ucs.nodes_expanded
    assert ast.cost == ucs.cost or (not ast.found and not ucs.found)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("seed, n, p", GRAPHS)
def test_fewest_actions_methods_match_bfs_reference(seed, n, p):
    rng = np.random.default_rng(1000 + seed)
    adj = random_graph(rng, n, p)
    hops = bfs_hops(adj, 0).get(n - 1)
    prob = GraphProblem(adj, 0, n - 1)
    tree_ok = n <= 10  # depth-limited tree searches enumerate simple paths
    methods = (breadth_first, bidirectional_bfs) + ((iterative_deepening,) if tree_ok else ())
    for method in methods:
        res = method(prob)
        assert res.found == (hops is not None)
        if res.found:
            assert len(res.path) - 1 == hops, method.__name__
            check_valid(prob, res)
    for method in (depth_first, greedy_best_first):
        res = method(prob)
        assert res.found == (hops is not None)
        if res.found:
            check_valid(prob, res)
    if hops is not None and tree_ok:
        assert depth_first(prob, depth_limit=hops).found
        if hops > 0:
            assert not depth_first(prob, depth_limit=hops - 1).found
        assert not iterative_deepening(prob, max_depth=max(hops - 1, 0)).found or hops == 0


def test_matches_networkx_when_available():
    nx = pytest.importorskip("networkx")
    rng = np.random.default_rng(7)
    for _ in range(30):
        adj = random_graph(rng, 40, 0.08)
        G = nx.DiGraph()
        G.add_nodes_from(adj)
        for u, row in adj.items():
            for v, w in row.items():
                G.add_edge(u, v, weight=w)
        try:
            ref = nx.shortest_path_length(G, 0, 39, weight="weight")
        except nx.NetworkXNoPath:
            ref = math.inf
        assert uniform_cost(GraphProblem(adj, 0, 39)).cost == pytest.approx(ref)


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------
ALL = [
    breadth_first,
    depth_first,
    iterative_deepening,
    uniform_cost,
    greedy_best_first,
    astar,
    weighted_astar,
    ida_star,
    bidirectional_bfs,
]


@pytest.mark.parametrize("method", ALL)
def test_start_is_goal(method):
    res = method(GraphProblem({"a": ["b"]}, "a", "a"))
    assert res.found and res.path == ("a",) and res.cost == 0.0 and res.actions == ()


@pytest.mark.parametrize("method", ALL)
def test_unreachable_goal(method):
    g = {1: [2], 2: [1, 3], 3: [1], 4: [1]}
    res = method(GraphProblem(g, 1, 4))
    assert not res.found and res.path == () and res.cost == math.inf
    assert res.nodes_expanded >= 1


def test_zero_cost_cycles_terminate_and_are_optimal():
    g = {"s": {"a": 0, "b": 1}, "a": {"s": 0, "b": 0}, "b": {"a": 0, "g": 2}, "g": {}}
    for method in (uniform_cost, astar, ida_star, weighted_astar):
        res = method(GraphProblem(g, "s", "g"))
        assert res.cost == 2.0


def test_tie_breaking_is_documented_order():
    # Two equal-cost paths: FIFO / action order picks the first-listed neighbour.
    g = {"s": {"a": 1, "b": 1}, "a": {"g": 1}, "b": {"g": 1}, "g": {}}
    assert uniform_cost(GraphProblem(g, "s", "g")).path == ("s", "a", "g")
    assert breadth_first(GraphProblem(g, "s", "g")).path == ("s", "a", "g")
    g2 = {"s": {"b": 1, "a": 1}, "a": {"g": 1}, "b": {"g": 1}, "g": {}}
    assert uniform_cost(GraphProblem(g2, "s", "g")).path == ("s", "b", "g")


def test_greedy_is_not_optimal_but_valid():
    g = {"S": {"A": 1, "B": 1}, "A": {"G": 10}, "B": {"G": 1}, "G": {}}
    prob = GraphProblem(g, "S", "G", heuristic={"A": 0.5, "B": 1.0})
    assert greedy_best_first(prob).cost == 11.0
    assert astar(prob).cost == 2.0


def test_ida_star_max_iterations():
    p = SlidingPuzzle((8, 6, 7, 2, 5, 4, 3, 0, 1))
    res = ida_star(p, max_iterations=1)
    assert not res.found


# ---------------------------------------------------------------------------
# sliding puzzle
# ---------------------------------------------------------------------------
def random_walk_state(rng, steps, start=(1, 2, 3, 4, 5, 6, 7, 8, 0)):
    p = SlidingPuzzle(start)
    s = p.initial_state
    for _ in range(steps):
        acts = p.actions(s)
        s = p.result(s, acts[rng.integers(len(acts))])
    return s


@pytest.mark.parametrize("seed", range(12))
def test_eight_puzzle_optimal_across_methods(seed):
    rng = np.random.default_rng(seed)
    p = SlidingPuzzle(random_walk_state(rng, 40))
    assert p.is_solvable()
    bfs = breadth_first(p)
    a = astar(p)
    i = ida_star(p)
    b = bidirectional_bfs(p)
    assert a.cost == len(bfs.path) - 1 == i.cost == b.cost
    for res in (bfs, a, i, b):
        check_valid(p, res)
    assert a.nodes_expanded <= uniform_cost(p).nodes_expanded
    w = weighted_astar(p, w=2.0)
    assert w.cost <= 2.0 * a.cost


def test_hardest_eight_puzzle_needs_31_moves():
    # One of the two hardest 8-puzzle instances (31 moves).
    p = SlidingPuzzle((8, 6, 7, 2, 5, 4, 3, 0, 1))
    res = astar(p)
    assert res.found and res.cost == 31.0
    check_valid(p, res)


@pytest.mark.filterwarnings("error")
def test_unsolvable_eight_puzzle_detected_and_not_found():
    p = SlidingPuzzle((1, 2, 3, 4, 5, 6, 8, 7, 0))  # one swap: odd parity
    assert not p.is_solvable()
    res = astar(p)
    assert not res.found
    assert res.nodes_expanded == math.factorial(9) // 2  # the whole reachable half


@pytest.mark.parametrize("seed", range(20))
def test_parity_test_matches_reachability(seed):
    rng = np.random.default_rng(seed)
    tiles = tuple(int(t) for t in rng.permutation(9))
    q = SlidingPuzzle(random_walk_state(rng, 30, start=tiles), goal=tiles)
    assert q.is_solvable()
    swapped = list(tiles)
    i, j = [k for k in range(9) if tiles[k] != 0][:2]
    swapped[i], swapped[j] = swapped[j], swapped[i]
    assert SlidingPuzzle(tuple(swapped), goal=tiles).is_solvable() is False


def test_fifteen_puzzle_short_instance():
    goal = tuple(range(1, 16)) + (0,)
    p = SlidingPuzzle(goal)
    s = goal
    for a in ("U", "L", "U", "L", "D", "R"):
        s = p.result(s, a)
    q = SlidingPuzzle(s)
    assert q.is_solvable() and ida_star(q).cost <= 6 and astar(q).cost == ida_star(q).cost


# ---------------------------------------------------------------------------
# API and validation
# ---------------------------------------------------------------------------
def test_result_object_and_dict():
    res = astar(GraphProblem({"a": {"b": 2}}, "a", "b"))
    assert isinstance(res, SearchResult)
    d = res.to_dict()
    assert d["path"] == ["a", "b"] and d["cost"] == 2.0 and d["found"] is True
    assert astar(GraphProblem({"a": {}, "b": {}}, "a", "b")).to_dict()["cost"] == "inf"
    with pytest.raises(ValueError):
        SearchResult(("a",), math.inf, 0, 0, False)
    with pytest.raises(ValueError):
        SearchResult((), 1.0, 0, 0, False)


def test_validation():
    with pytest.raises(ValueError):
        GraphProblem({"a": {"b": -1}}, "a", "b")
    with pytest.raises(ValueError):
        GraphProblem({"a": {"b": math.inf}}, "a", "b")
    with pytest.raises(ValueError):
        GraphProblem({"a": ["b"]}, "a", "z")
    with pytest.raises(TypeError):
        GraphProblem([("a", "b")], "a", "b")
    with pytest.raises(TypeError):
        astar(object())
    with pytest.raises(ValueError):
        weighted_astar(GraphProblem({"a": ["b"]}, "a", "b"), w=0.5)
    with pytest.raises(ValueError):
        depth_first(GraphProblem({"a": ["b"]}, "a", "b"), depth_limit=-1)
    with pytest.raises(ValueError):
        astar(GraphProblem({"a": ["b"]}, "a", "b", heuristic={"a": -1.0}))
    with pytest.raises(ValueError):
        SlidingPuzzle((1, 2, 3))
    with pytest.raises(ValueError):
        SlidingPuzzle((1, 1, 2, 3))

    class Bad:
        initial_state = 0

        def actions(self, s):
            return [1]

        def result(self, s, a):
            return s + a

        def step_cost(self, s, a, s2):
            return -1.0

        def is_goal(self, s):
            return s == 3

    with pytest.raises(ValueError):
        uniform_cost(Bad())
    with pytest.raises(TypeError):
        bidirectional_bfs(Bad())


def test_graph_is_copied():
    g = {"a": {"b": 1.0}}
    prob = GraphProblem(g, "a", "b")
    g["a"]["b"] = 100.0
    assert uniform_cost(prob).cost == 1.0


def test_custom_problem_protocol_infinite_space():
    class Counter:  # infinite state space, goal at 7; only IDS-type searches need care
        initial_state = 0

        def actions(self, s):
            return [1, 2]

        def result(self, s, a):
            return s + a

        def step_cost(self, s, a, s2):
            return float(a * a)

        def is_goal(self, s):
            return s == 7

    for method in (breadth_first, uniform_cost, iterative_deepening, astar):
        res = method(Counter())
        assert res.found
    assert uniform_cost(Counter()).cost == 7.0  # seven +1 steps
    assert len(breadth_first(Counter()).path) - 1 == 4  # e.g. 2+2+2+1: fewest actions


def test_ucs_memory_linear_in_graph_size():
    def make(n, rng):
        side = int(math.isqrt(n))
        adj = {}
        for r in range(side):
            for c in range(side):
                row = {}
                for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                    if 0 <= r + dr < side and 0 <= c + dc < side:
                        row[(r + dr, c + dc)] = float(rng.uniform(1, 2))
                adj[(r, c)] = row
        return GraphProblem(adj, (0, 0), (side - 1, side - 1))

    est = estimate_complexity(uniform_cost, make, sizes=(2500, 5000, 10000, 20000), repeats=1)
    assert est.space_exponent == pytest.approx(1.0, abs=0.2)


@pytest.mark.complexity
def test_ucs_time_near_linear_on_grids():
    def make(n, rng):
        side = int(math.isqrt(n))
        adj = {
            (r, c): {
                (r + dr, c + dc): float(rng.uniform(1, 2))
                for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0))
                if 0 <= r + dr < side and 0 <= c + dc < side
            }
            for r in range(side)
            for c in range(side)
        }
        return GraphProblem(adj, (0, 0), (side - 1, side - 1))

    est = estimate_complexity(uniform_cost, make, sizes=(2500, 5000, 10000, 20000), repeats=3)
    assert est.time_exponent < 1.3  # O(V log V); a quadratic frontier would give ~2
