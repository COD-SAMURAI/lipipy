"""lipipy.ai.csp: N-Queens counts, map colouring, Sudoku, AC-3 and brute force."""

import itertools

import numpy as np
import pytest

from lipipy.ai import (
    CSP,
    AC3Result,
    Constraint,
    CSPResult,
    ac3,
    backtracking_search,
    min_conflicts,
)

INFERENCES = ["forward_checking", "mac", "none"]


def queens(n):
    cols = list(range(n))
    cons = [
        Constraint((a, b), lambda r1, r2, d=b - a: r1 != r2 and abs(r1 - r2) != d)
        for a, b in itertools.combinations(cols, 2)
    ]
    return CSP(cols, {c: list(range(n)) for c in cols}, cons)


def australia():
    regions = ["WA", "NT", "SA", "Q", "NSW", "V", "T"]
    borders = [
        ("WA", "NT"),
        ("WA", "SA"),
        ("NT", "SA"),
        ("NT", "Q"),
        ("SA", "Q"),
        ("SA", "NSW"),
        ("SA", "V"),
        ("Q", "NSW"),
        ("NSW", "V"),
    ]
    cons = [Constraint(b, lambda x, y: x != y) for b in borders]
    return CSP(regions, {r: ["red", "green", "blue"] for r in regions}, cons), borders


def sudoku(grid):
    cells = [(r, c) for r in range(9) for c in range(9)]
    doms = {}
    for r, c in cells:
        ch = grid[9 * r + c]
        doms[(r, c)] = [int(ch)] if ch not in ".0" else list(range(1, 10))
    units = [[(r, c) for c in range(9)] for r in range(9)]
    units += [[(r, c) for r in range(9)] for c in range(9)]
    units += [
        [(br + i, bc + j) for i in range(3) for j in range(3)]
        for br in (0, 3, 6)
        for bc in (0, 3, 6)
    ]
    pairs = {tuple(sorted(p)) for u in units for p in itertools.combinations(u, 2)}
    return CSP(cells, doms, [Constraint(p, lambda a, b: a != b) for p in sorted(pairs)])


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("inference", INFERENCES)
def test_n_queens_solution_counts(inference):
    expected = [1, 0, 0, 2, 10, 4, 40, 92]
    for n, count in enumerate(expected, start=1):
        res = backtracking_search(queens(n), inference=inference, all_solutions=True)
        assert res.n_solutions == count, n
        assert len({tuple(sorted(s.items())) for s in res.solutions}) == count  # distinct
        assert all(queens(n).is_solution(s) for s in res.solutions)
        assert res.found == (count > 0)


def test_mac_and_fc_prune_compared_with_plain_backtracking():
    plain = backtracking_search(queens(8), inference="none", all_solutions=True)
    fc = backtracking_search(queens(8), inference="forward_checking", all_solutions=True)
    mac = backtracking_search(queens(8), inference="mac", all_solutions=True)
    assert mac.nodes <= fc.nodes < plain.nodes


@pytest.mark.parametrize("inference", INFERENCES)
def test_australia_map_colouring(inference):
    csp, borders = australia()
    res = backtracking_search(csp, inference=inference)
    assert res.found and csp.is_solution(res.solution)
    assert all(res.solution[a] != res.solution[b] for a, b in borders)
    # 6 colourings of the mainland (SA fixes a 2-colourable ring) x 3 for Tasmania.
    assert backtracking_search(csp, inference=inference, all_solutions=True).n_solutions == 18
    assert backtracking_search(csp, all_solutions=True, max_solutions=5).n_solutions == 5


HARD = "8..........36......7..9.2...5...7.......457.....1...3...1....68..85...1..9....4.."
HARD_SOLUTION = "812753649943682175675491283154237896369845721287169534521974368438526917796318452"


@pytest.mark.parametrize("inference", ["mac", "forward_checking"])
def test_hard_sudoku(inference):
    csp = sudoku(HARD)
    res = backtracking_search(csp, inference=inference)
    assert res.found and csp.is_solution(res.solution)
    got = "".join(str(res.solution[(r, c)]) for r in range(9) for c in range(9))
    assert got == HARD_SOLUTION  # the puzzle has a unique solution
    unique = backtracking_search(csp, inference="mac", all_solutions=True, max_solutions=2)
    assert unique.n_solutions == 1


def test_mac_explores_fewer_nodes_than_forward_checking_on_sudoku():
    csp = sudoku(HARD)
    mac = backtracking_search(csp, inference="mac")
    fc = backtracking_search(csp, inference="forward_checking")
    assert mac.nodes < fc.nodes


def test_invalid_sudoku_has_no_solution():
    bad = "11" + "." * 79
    res = backtracking_search(sudoku(bad), inference="mac")
    assert not res.found and res.solution is None and res.solutions == ()
    assert not ac3(sudoku(bad)).consistent


# ---------------------------------------------------------------------------
# AC-3 vs brute-force arc consistency, search vs brute-force enumeration
# ---------------------------------------------------------------------------
def random_binary_csp(rng, n=5, k=4, density=0.6, tightness=0.4):
    vs = list(range(n))
    cons = []
    for a, b in itertools.combinations(vs, 2):
        if rng.random() < density:
            allowed = {(x, y) for x in range(k) for y in range(k) if rng.random() > tightness}
            cons.append(Constraint((a, b), lambda x, y, s=frozenset(allowed): (x, y) in s))
    doms = {v: [x for x in range(k) if rng.random() < 0.9] for v in vs}
    return CSP(vs, doms, cons)


def brute_force_ac(csp):
    doms = csp.domains
    changed = True
    while changed:
        changed = False
        for c in csp.constraints:
            a, b = c.scope
            keep_a = [x for x in doms[a] if any(c.predicate(x, y) for y in doms[b])]
            keep_b = [y for y in doms[b] if any(c.predicate(x, y) for x in doms[a])]
            if keep_a != doms[a] or keep_b != doms[b]:
                doms[a], doms[b] = keep_a, keep_b
                changed = True
    return doms


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("seed", range(300))
def test_ac3_equals_brute_force_closure(seed):
    csp = random_binary_csp(np.random.default_rng(seed))
    res = ac3(csp)
    ref = brute_force_ac(csp)
    assert res.consistent == all(ref.values())
    if res.consistent:
        assert res.domains == ref
    assert isinstance(res, AC3Result)


def brute_force_solutions(csp):
    sols = []
    for values in itertools.product(*(csp.domains[v] for v in csp.variables)):
        a = dict(zip(csp.variables, values, strict=True))
        if all(c.is_satisfied(a) for c in csp.constraints):
            sols.append(tuple(values))
    return set(sols)


@pytest.mark.parametrize("seed", range(150))
def test_all_solutions_equal_brute_force_with_nary_constraints(seed):
    rng = np.random.default_rng(10_000 + seed)
    base = random_binary_csp(rng, n=5, k=3, density=0.4, tightness=0.3)
    cons = list(base.constraints)
    target = int(rng.integers(2, 7))
    cons.append(Constraint((0, 2, 4), lambda a, b, c, t=target: a + b + c != t))
    cons.append(Constraint((1,), lambda x: x != 1))
    csp = CSP(base.variables, base.domains, cons)
    ref = brute_force_solutions(csp)
    for inference in INFERENCES:
        res = backtracking_search(csp, inference=inference, all_solutions=True)
        got = {tuple(s[v] for v in csp.variables) for s in res.solutions}
        assert got == ref and res.n_solutions == len(ref)
        first = backtracking_search(csp, inference=inference)
        assert first.found == bool(ref)
        if first.found:
            assert csp.is_solution(first.solution)


# ---------------------------------------------------------------------------
# min-conflicts
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n, seeds", [(8, 100), (20, 30), (50, 10)])
def test_min_conflicts_n_queens_across_seeds(n, seeds):
    # Measured: every seed in 0..199 solves 8- and 20-queens in < 500 steps.
    csp = queens(n)
    for seed in range(seeds):
        res = min_conflicts(csp, max_steps=1000, random_state=seed)
        assert res.found and csp.is_solution(res.solution)


def test_min_conflicts_without_random_walk_can_get_stuck():
    stuck = [
        not min_conflicts(queens(8), max_steps=3000, random_state=s, p_random=0.0).found
        for s in range(100)
    ]
    assert any(stuck)  # pure greedy repair has absorbing local minima


def test_min_conflicts_reproducible_and_budget():
    csp = queens(30)
    a = min_conflicts(csp, max_steps=500, random_state=3)
    b = min_conflicts(csp, max_steps=500, random_state=3)
    assert a == b
    unsat = queens(3)
    res = min_conflicts(unsat, max_steps=50, random_state=0)
    assert not res.found and res.nodes == 50
    assert not min_conflicts(unsat, max_steps=0, random_state=0).found


# ---------------------------------------------------------------------------
# edge cases and validation
# ---------------------------------------------------------------------------
def test_empty_domain_and_empty_csp():
    csp = CSP(["a", "b"], {"a": [], "b": [1]}, [])
    for inference in INFERENCES:
        assert not backtracking_search(csp, inference=inference).found
    assert not ac3(csp).consistent
    assert not min_conflicts(csp, random_state=0).found
    empty = CSP([], {}, [])
    res = backtracking_search(empty)
    assert res.found and res.solution == {}


def test_unconstrained_and_already_solved():
    csp = CSP(["a"], {"a": [5]}, [])
    assert backtracking_search(csp).solution == {"a": 5}
    assert min_conflicts(csp, random_state=0).solution == {"a": 5}
    assert (
        backtracking_search(
            CSP(["a", "b"], {"a": [1, 2], "b": [3]}), all_solutions=True
        ).n_solutions
        == 2
    )


def test_csp_domains_are_copied_and_not_mutated():
    doms = {"x": [1, 2, 3], "y": [1, 2, 3]}
    csp = CSP(["x", "y"], doms, [Constraint(("x", "y"), lambda x, y: x > y)])
    doms["x"].append(99)
    assert csp.domains["x"] == [1, 2, 3]
    csp.domains["x"].clear()
    assert csp.domains["x"] == [1, 2, 3]
    ac3(csp)
    backtracking_search(csp, inference="mac", all_solutions=True)
    assert csp.domains == {"x": [1, 2, 3], "y": [1, 2, 3]}
    given = {"x": [3], "y": [1, 2, 3]}
    ac3(csp, given)
    assert given == {"x": [3], "y": [1, 2, 3]}


def test_lcv_and_mrv_order():
    # x has 1 value (MRV picks it first); LCV prefers y = 3, which rules out nothing in z.
    cons = [Constraint(("y", "z"), lambda y, z: y != z)]
    csp = CSP(["y", "z", "x"], {"x": [0], "y": [1, 3], "z": [1, 2]}, cons)
    res = backtracking_search(csp)
    assert res.solution == {"x": 0, "y": 3, "z": 1} and res.backtracks == 0


def test_result_dict_and_validation():
    res = backtracking_search(australia()[0])
    d = res.to_dict()
    assert d["found"] and d["n_solutions"] == 1 and isinstance(res, CSPResult)
    with pytest.raises(ValueError):
        CSPResult(None, (), 1, 0, 0, True)
    with pytest.raises(ValueError):
        Constraint((), lambda: True)
    with pytest.raises(ValueError):
        Constraint(("a", "a"), lambda x, y: True)
    with pytest.raises(TypeError):
        Constraint(("a",), 3)
    with pytest.raises(ValueError):
        CSP(["a", "a"], {"a": [1]})
    with pytest.raises(ValueError):
        CSP(["a"], {})
    with pytest.raises(ValueError):
        CSP(["a"], {"a": [1, 1]})
    with pytest.raises(ValueError):
        CSP(["a"], {"a": [1]}, [Constraint(("a", "b"), lambda x, y: True)])
    with pytest.raises(TypeError):
        CSP(["a"], {"a": [1]}, ["not a constraint"])
    with pytest.raises(ValueError):
        backtracking_search(australia()[0], inference="ac4")
    with pytest.raises(ValueError):
        backtracking_search(australia()[0], all_solutions=True, max_solutions=0)
    with pytest.raises(TypeError):
        ac3("csp")
    with pytest.raises(ValueError):
        ac3(australia()[0], {"WA": [1]})
    with pytest.raises(ValueError):
        min_conflicts(australia()[0], max_steps=-1)
    with pytest.raises(ValueError):
        min_conflicts(australia()[0], p_random=1.5)
