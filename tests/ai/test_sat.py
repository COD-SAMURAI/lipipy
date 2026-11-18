"""lipipy.ai.sat vs brute force, pigeonhole formulas and DIMACS round trips."""

import itertools

import numpy as np
import pytest

from lipipy.ai import CNF, SATResult, cdcl, dpll, parse_dimacs, walksat
from lipipy.ai.sat import _luby


def random_3sat(rng, n, ratio=4.26):
    m = int(round(ratio * n))
    clauses = []
    for _ in range(m):
        vs = rng.choice(n, 3, replace=False) + 1
        clauses.append([int(v) if rng.random() < 0.5 else -int(v) for v in vs])
    return CNF(clauses, n)


def brute_force_sat(cnf):
    n = cnf.n_vars
    if n == 0:
        return all(len(c) > 0 for c in cnf.clauses) and not cnf.clauses
    A = np.array(list(itertools.product([False, True], repeat=n)))
    ok = np.ones(len(A), dtype=bool)
    for c in cnf.clauses:
        sat = np.zeros(len(A), dtype=bool)
        for lit in c:
            col = A[:, abs(lit) - 1]
            sat |= col if lit > 0 else ~col
        ok &= sat
    return bool(ok.any())


def pigeonhole(pigeons, holes):
    var = lambda i, j: i * holes + j + 1  # noqa: E731
    clauses = [[var(i, j) for j in range(holes)] for i in range(pigeons)]
    for j in range(holes):
        for a, b in itertools.combinations(range(pigeons), 2):
            clauses.append([-var(a, j), -var(b, j)])
    return CNF(clauses, pigeons * holes)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("block", range(4))
def test_complete_solvers_agree_with_brute_force(block):
    # 4 x 500 = 2000 random 3-SAT instances, n in 3..14, at the 4.26 threshold ratio.
    rng = np.random.default_rng(block)
    n_sat = 0
    for _ in range(500):
        n = int(rng.integers(3, 15))
        cnf = random_3sat(rng, n)
        truth = brute_force_sat(cnf)
        n_sat += truth
        for solver in (dpll, cdcl):
            res = solver(cnf)
            assert res.satisfiable == truth, (solver.__name__, cnf.clauses)
            if truth:
                assert cnf.is_satisfied_by(res.model)
            else:
                assert res.model is None
    assert 0.6 < n_sat / 500 < 0.95  # both outcomes are well represented


def test_walksat_finds_models_of_satisfiable_instances():
    rng = np.random.default_rng(42)
    tried = 0
    while tried < 300:
        cnf = random_3sat(rng, int(rng.integers(5, 15)))
        if not brute_force_sat(cnf):
            res = walksat(cnf, max_flips=200, random_state=0)
            assert res.status == "UNKNOWN" and res.flips == 200  # incomplete: never UNSAT
            continue
        tried += 1
        res = walksat(cnf, max_flips=20_000, random_state=tried)
        assert res.status == "SAT" and cnf.is_satisfied_by(res.model)


@pytest.mark.parametrize("holes", [1, 2, 3, 4, 5])
def test_pigeonhole_unsat(holes):
    php = pigeonhole(holes + 1, holes)
    assert dpll(php).status == "UNSAT"
    assert cdcl(php).status == "UNSAT"
    fits = pigeonhole(holes, holes)  # n pigeons fit in n holes
    for solver in (dpll, cdcl):
        res = solver(fits)
        assert res.status == "SAT" and fits.is_satisfied_by(res.model)


def test_pigeonhole_6_into_5_holes_cdcl():
    res = cdcl(pigeonhole(7, 6))
    assert res.status == "UNSAT" and res.learned > 0 and res.conflicts > 0


def test_cdcl_uses_fewer_decisions_than_dpll_on_hard_random_3sat():
    # Measured on 20 random 3-SAT instances with n = 75 at ratio 4.26 (seeds 100..119):
    # DPLL 5830 decisions in total, CDCL 2816; CDCL needs fewer on 16 of the 20.
    total_dpll = total_cdcl = wins = 0
    for seed in range(100, 120):
        cnf = random_3sat(np.random.default_rng(seed), 75)
        a, b = dpll(cnf), cdcl(cnf)
        assert a.status == b.status
        total_dpll += a.decisions
        total_cdcl += b.decisions
        wins += b.decisions < a.decisions
    assert total_cdcl < 0.6 * total_dpll
    assert wins >= 14


def test_cdcl_learns_and_backjumps():
    cnf = random_3sat(np.random.default_rng(5), 60)
    res = cdcl(cnf, restart_base=4)
    assert res.conflicts > 0 and res.learned > 0 and res.restarts > 0
    assert res.satisfiable == dpll(cnf).satisfiable


def test_luby_sequence():
    assert [_luby(i) for i in range(1, 16)] == [1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8]


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("solver", [dpll, cdcl, walksat])
def test_trivial_formulas(solver):
    res = solver(CNF([], n_vars=3))
    assert res.status == "SAT" and len(res.model) == 3
    assert solver(CNF([[]], n_vars=1)).status == "UNSAT"
    unit = solver([[2], [-1]])
    assert unit.status == "SAT" and unit.model == (-1, 2)
    taut = solver([[1, -1], [2]])
    assert taut.status == "SAT" and 2 in taut.model
    dup = solver([[1, 1, 1]])
    assert dup.model == (1,)


@pytest.mark.parametrize("solver", [dpll, cdcl])
def test_contradictions(solver):
    assert solver([[1], [-1]]).status == "UNSAT"
    assert solver([[1, 2], [-1, 2], [1, -2], [-1, -2]]).status == "UNSAT"
    chain = [[1]] + [[-i, i + 1] for i in range(1, 30)] + [[-30]]
    assert solver(chain).status == "UNSAT"


def test_unused_variables_get_a_value():
    res = cdcl(CNF([[1]], n_vars=4))
    assert res.model[0] == 1 and len(res.model) == 4


def test_walksat_reproducible():
    cnf = random_3sat(np.random.default_rng(1), 40, ratio=3.5)
    a = walksat(cnf, random_state=9)
    b = walksat(cnf, random_state=9)
    assert a == b and a.status == "SAT"


# ---------------------------------------------------------------------------
# DIMACS and API
# ---------------------------------------------------------------------------
def test_dimacs_round_trip_and_parsing():
    cnf = random_3sat(np.random.default_rng(2), 10)
    again = parse_dimacs(cnf.to_dimacs())
    assert again.clauses == cnf.clauses and again.n_vars == cnf.n_vars
    text = "c comment\np cnf 4 3\n1 -2\n 3 0 -4\n0\n2 4 0\n%\n0\n"
    f = parse_dimacs(text)
    assert f.clauses == ((1, -2, 3), (-4,), (2, 4)) and f.n_vars == 4


@pytest.mark.parametrize(
    "text",
    [
        "1 2 0\n",  # no header
        "p cnf 2 1\np cnf 2 1\n1 0\n",  # two headers
        "p dnf 2 1\n1 0\n",
        "p cnf x 1\n1 0\n",
        "p cnf 2 2\n1 0\n",  # clause count mismatch
        "p cnf 2 1\n1 a 0\n",
        "p cnf 1 1\n2 0\n",  # variable beyond n_vars
    ],
)
def test_dimacs_errors(text):
    with pytest.raises(ValueError):
        parse_dimacs(text)


def test_validation_and_result():
    with pytest.raises(ValueError):
        CNF([[1, 0]])
    with pytest.raises(TypeError):
        CNF([[1.5]])
    with pytest.raises(TypeError):
        CNF([[True]])
    with pytest.raises(ValueError):
        CNF([[3]], n_vars=2)
    with pytest.raises(TypeError):
        parse_dimacs(b"p cnf 1 1\n1 0\n")
    with pytest.raises(ValueError):
        walksat([[1]], p=2.0)
    with pytest.raises(ValueError):
        cdcl([[1]], restart_base=0)
    with pytest.raises(ValueError):
        SATResult("SAT", None)
    with pytest.raises(ValueError):
        SATResult("MAYBE", None)
    res = cdcl([[1, 2]])
    assert res.to_dict()["status"] == "SAT" and res.satisfiable is True
    assert walksat([[1], [-1]], max_flips=10, random_state=0).satisfiable is None
    assert CNF([[1]]).is_satisfied_by((1,)) and not CNF([[1]]).is_satisfied_by((-1,))
    assert not CNF([[1]], n_vars=2).is_satisfied_by((1,))
    assert "CNF" in repr(CNF([[1]])) and len(CNF([[1], [2]])) == 2
    np_lits = CNF([[np.int64(2), np.int32(-1)]])
    assert np_lits.clauses == ((2, -1),)
