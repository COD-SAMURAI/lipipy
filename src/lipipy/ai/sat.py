r"""Propositional satisfiability (SAT) for formulas in conjunctive normal form.

Formulas use the DIMACS convention: variables are ``1 .. n``; a literal is ``v`` or
``-v``; a clause is a list of literals (their disjunction); a formula is a list of
clauses (their conjunction). A model is a tuple of ``n`` literals, the true one for
each variable.

DPLL
----
Davis-Putnam-Logemann-Loveland: simplify the formula by **unit propagation** (a
clause with one literal forces it) and **pure-literal elimination** (a literal whose
negation never occurs can be set true), then branch on a variable and backtrack
chronologically. Branching picks the most frequent variable among the remaining
clauses, trying its more frequent polarity first.

CDCL
----
Conflict-driven clause learning (as in GRASP / Chaff / MiniSat):

* unit propagation with **two watched literals** per clause (a clause is visited
  only when one of its two watched literals becomes false);
* on a conflict, **1-UIP** analysis resolves the conflicting clause with reason
  clauses of the current decision level until one literal of that level remains,
  giving an asserting learned clause;
* **non-chronological backjumping** to the second-highest decision level in the
  learned clause, where it becomes unit;
* **VSIDS**-style activities (bumped for the variables in each learned clause, with
  geometric decay) choose the next decision variable, with phase saving;
* **restarts** after :math:`32\,u_i` conflicts, :math:`u_i` the Luby sequence
  1, 1, 2, 1, 1, 2, 4, ...

Learned clauses are never deleted (fine for the small and medium formulas this
reference implementation targets). Measured in the tests: on 20 random 3-SAT
formulas with 75 variables at the threshold ratio 4.26, CDCL needs 2816 decisions in
total against 5830 for DPLL (fewer on 16 of the 20); both agree with brute force on
2000 random formulas with up to 14 variables.

WalkSAT
-------
Local search on complete assignments: pick a random unsatisfied clause; with
probability ``p`` flip a random variable of it, otherwise flip the variable whose flip
breaks the fewest currently satisfied clauses. Incomplete: it can find models but
never prove unsatisfiability.

Every model is checked against every clause before it is returned.

References
----------
M. Davis, G. Logemann & D. Loveland, "A machine program for theorem-proving",
*Comm. ACM* 5 (1962) 394-397. J. P. Marques-Silva & K. A. Sakallah, "GRASP: a search
algorithm for propositional satisfiability", *IEEE Trans. Computers* 48 (1999)
506-521. M. W. Moskewicz et al., "Chaff: engineering an efficient SAT solver", DAC
(2001). N. Eén & N. Sörensson, "An extensible SAT-solver", SAT 2003, LNCS 2919.
M. Luby, A. Sinclair & D. Zuckerman, "Optimal speedup of Las Vegas algorithms",
*Inf. Process. Lett.* 47 (1993). B. Selman, H. Kautz & B. Cohen, "Noise strategies
for improving local search", AAAI (1994). S. Russell & P. Norvig, *Artificial
Intelligence: A Modern Approach*, 4th ed. (2020), §7.6.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from lipipy.ai._common import check_float, check_int, jsonable
from lipipy.core.complexity import Complexity
from lipipy.core.random import RandomState, check_random_state
from lipipy.core.registry import register

__all__ = ["CNF", "SATResult", "cdcl", "dpll", "parse_dimacs", "walksat"]

_DLL = "Davis, Logemann & Loveland, Comm. ACM 5 (1962) 394-397"
_GRASP = "Marques-Silva & Sakallah, IEEE Trans. Computers 48 (1999) 506-521"
_CHAFF = "Moskewicz et al., Chaff: engineering an efficient SAT solver, DAC (2001)"
_MINISAT = "Eén & Sörensson, An extensible SAT-solver, SAT 2003, LNCS 2919"
_WALKSAT = "Selman, Kautz & Cohen, Noise strategies for improving local search, AAAI (1994)"
_AIMA = "Russell & Norvig, Artificial Intelligence: A Modern Approach, 4th ed. (2020), §7.6"

Status = Literal["SAT", "UNSAT", "UNKNOWN"]


# ---------------------------------------------------------------------------
# formulas
# ---------------------------------------------------------------------------
class CNF:
    """A CNF formula in DIMACS integer-literal form (immutable after construction).

    Parameters
    ----------
    clauses : iterable of iterable of int
        Non-zero integer literals.
    n_vars : int, optional
        Number of variables (default: the largest variable mentioned).

    Notes
    -----
    Repeated literals inside a clause are merged. Clauses are otherwise kept as given
    (an empty clause makes the formula unsatisfiable).

    Examples
    --------
    >>> f = CNF([[1, -2], [2, 3], [-1, -3]])
    >>> f.n_vars, f.is_satisfied_by((1, 2, -3))
    (3, True)
    """

    def __init__(self, clauses: Iterable[Iterable[int]], n_vars: int | None = None) -> None:
        cls: list[tuple[int, ...]] = []
        top = 0
        for clause in clauses:
            lits: list[int] = []
            for lit in clause:
                if isinstance(lit, (bool, np.bool_)) or not isinstance(lit, (int, np.integer)):
                    raise TypeError(f"literals must be integers, got {lit!r}")
                if lit == 0:
                    raise ValueError("literals must be non-zero (0 terminates DIMACS clauses)")
                lit = int(lit)
                if lit not in lits:
                    lits.append(lit)
                top = max(top, abs(lit))
            cls.append(tuple(lits))
        if n_vars is None:
            n_vars = top
        n_vars = check_int(n_vars, "n_vars")
        if n_vars < top:
            raise ValueError(f"n_vars={n_vars} but clauses mention variable {top}")
        self.clauses: tuple[tuple[int, ...], ...] = tuple(cls)
        self.n_vars = n_vars

    def __len__(self) -> int:
        return len(self.clauses)

    def __repr__(self) -> str:
        return f"CNF(n_vars={self.n_vars}, n_clauses={len(self.clauses)})"

    def is_satisfied_by(self, model: Sequence[int]) -> bool:
        """Whether every clause contains a literal of ``model``."""
        true = set(model)
        if len(model) != self.n_vars or any(abs(lit) > self.n_vars for lit in true):
            return False
        if any(-lit in true for lit in true):
            return False
        return all(any(lit in true for lit in c) for c in self.clauses)

    def to_dimacs(self) -> str:
        """DIMACS text (``p cnf`` header, clauses terminated by 0)."""
        lines = [f"p cnf {self.n_vars} {len(self.clauses)}"]
        lines += [" ".join(str(lit) for lit in c) + (" 0" if c else "0") for c in self.clauses]
        return "\n".join(lines) + "\n"


def parse_dimacs(text: str) -> CNF:
    """Parse DIMACS CNF text: ``c`` comment lines, a ``p cnf <vars> <clauses>`` header,
    then literals with each clause terminated by ``0`` (clauses may span lines).

    Examples
    --------
    >>> f = parse_dimacs("c example\\np cnf 3 2\\n1 -3 0\\n2 3 -1 0\\n")
    >>> f.clauses, f.n_vars
    (((1, -3), (2, 3, -1)), 3)
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    header: tuple[int, int] | None = None
    clauses: list[list[int]] = []
    current: list[int] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("c"):
            continue
        if line.startswith("%"):  # SATLIB end marker
            break
        if line.startswith("p"):
            parts = line.split()
            if header is not None or len(parts) != 4 or parts[1] != "cnf":
                raise ValueError(f"bad DIMACS header: {line!r}")
            try:
                header = (int(parts[2]), int(parts[3]))
            except ValueError:
                raise ValueError(f"bad DIMACS header: {line!r}") from None
            continue
        if header is None:
            raise ValueError("clauses before the 'p cnf' header")
        for tok in line.split():
            try:
                lit = int(tok)
            except ValueError:
                raise ValueError(f"bad literal {tok!r}") from None
            if lit == 0:
                clauses.append(current)
                current = []
            else:
                current.append(lit)
    if header is None:
        raise ValueError("missing 'p cnf' header")
    if current:
        clauses.append(current)
    if len(clauses) != header[1]:
        raise ValueError(f"header announces {header[1]} clauses, found {len(clauses)}")
    return CNF(clauses, n_vars=header[0])


@dataclass(frozen=True)
class SATResult:
    """Outcome of a SAT solver.

    Attributes
    ----------
    status : {"SAT", "UNSAT", "UNKNOWN"}
        ``"UNKNOWN"`` only from incomplete methods (WalkSAT) that ran out of flips.
    model : tuple of int or None
        One literal per variable (``v`` true, ``-v`` false) when SAT.
    decisions : int
        Branching decisions (DPLL / CDCL).
    conflicts : int
        Conflicts (DPLL: failed branches; CDCL: analysed conflicts).
    propagations : int
        Literals assigned by unit propagation.
    learned : int
        Learned clauses (CDCL).
    restarts : int
        Restarts (CDCL).
    flips : int
        Variable flips (WalkSAT).
    """

    status: Status
    model: tuple[int, ...] | None
    decisions: int = 0
    conflicts: int = 0
    propagations: int = 0
    learned: int = 0
    restarts: int = 0
    flips: int = 0

    def __post_init__(self) -> None:
        if self.status not in ("SAT", "UNSAT", "UNKNOWN"):
            raise ValueError(f"bad status {self.status!r}")
        if (self.status == "SAT") != (self.model is not None):
            raise ValueError("a model is present exactly when status is 'SAT'")

    @property
    def satisfiable(self) -> bool | None:
        """True, False, or None when unknown."""
        return {"SAT": True, "UNSAT": False, "UNKNOWN": None}[self.status]

    def to_dict(self) -> dict[str, Any]:
        return {k: jsonable(v) for k, v in asdict(self).items()}


def _as_cnf(formula: CNF | Iterable[Iterable[int]]) -> CNF:
    return formula if isinstance(formula, CNF) else CNF(formula)


def _verified(cnf: CNF, model: tuple[int, ...]) -> tuple[int, ...]:
    if not cnf.is_satisfied_by(model):
        raise RuntimeError("internal error: solver produced a model that violates a clause")
    return model


# ---------------------------------------------------------------------------
# DPLL
# ---------------------------------------------------------------------------
class _DPLL:
    def __init__(self) -> None:
        self.decisions = self.conflicts = self.propagations = 0

    @staticmethod
    def assign(clauses: list[tuple[int, ...]], lit: int) -> list[tuple[int, ...]] | None:
        """Simplify by ``lit = True``; None if an empty clause appears."""
        out = []
        for c in clauses:
            if lit in c:
                continue
            if -lit in c:
                c = tuple(x for x in c if x != -lit)
                if not c:
                    return None
            out.append(c)
        return out

    def solve(self, clauses: list[tuple[int, ...]], model: dict[int, bool]) -> dict | None:
        model = dict(model)
        while True:  # unit propagation and pure literals until a fixpoint
            unit = next((c[0] for c in clauses if len(c) == 1), None)
            if unit is not None:
                self.propagations += 1
                model[abs(unit)] = unit > 0
                reduced = self.assign(clauses, unit)
                if reduced is None:
                    self.conflicts += 1
                    return None
                clauses = reduced
                continue
            lits = {x for c in clauses for x in c}
            pure = next((x for x in lits if -x not in lits), None)
            if pure is not None:
                model[abs(pure)] = pure > 0
                clauses = self.assign(clauses, pure) or []
                continue
            break
        if not clauses:
            return model
        counts = Counter(x for c in clauses for x in c)
        var = max(counts, key=lambda x: (counts[x] + counts[-x], -abs(x)))
        var = abs(var)
        first = var if counts[var] >= counts[-var] else -var
        for lit in (first, -first):
            self.decisions += 1
            reduced = self.assign(clauses, lit)
            if reduced is None:
                self.conflicts += 1
                continue
            found = self.solve(reduced, {**model, var: lit > 0})
            if found is not None:
                return found
        return None


@register(
    domain="ai",
    tasks=("sat", "logic"),
    complexity=Complexity(
        time="O(2^n m)",
        space="O(n m)",
        notes="n variables, m = total clause length; worst case exponential; one copy "
        "of the simplified formula per decision level",
    ),
    references=(_DLL, _AIMA),
)
def dpll(formula: CNF | Iterable[Iterable[int]]) -> SATResult:
    r"""Decide satisfiability with the DPLL procedure (unit propagation, pure literals).

    Parameters
    ----------
    formula : CNF or list of clauses

    Returns
    -------
    SATResult
        ``"SAT"`` with a verified model, or ``"UNSAT"``.

    Complexity
    ----------
    Worst-case time :math:`O(2^n m)`; space :math:`O(nm)` (a simplified copy per level).

    References
    ----------
    Davis, Logemann & Loveland (1962); Russell & Norvig (2020), §7.6.1.

    Examples
    --------
    >>> dpll([[1, 2], [-1, 2], [-2, 3]]).status
    'SAT'
    >>> dpll([[1], [-1]]).status
    'UNSAT'
    """
    cnf = _as_cnf(formula)
    solver = _DPLL()
    clauses = list(cnf.clauses)
    found = None if any(len(c) == 0 for c in clauses) else solver.solve(clauses, {})
    counters = dict(
        decisions=solver.decisions, conflicts=solver.conflicts, propagations=solver.propagations
    )
    if found is None:
        return SATResult("UNSAT", None, **counters)
    model = tuple(v if found.get(v, False) else -v for v in range(1, cnf.n_vars + 1))
    return SATResult("SAT", _verified(cnf, model), **counters)


# ---------------------------------------------------------------------------
# CDCL
# ---------------------------------------------------------------------------
def _luby(i: int) -> int:
    """i-th element (1-based) of the Luby sequence 1, 1, 2, 1, 1, 2, 4, 1, ..."""
    k = 1
    while (1 << k) - 1 < i:
        k += 1
    while True:
        if i == (1 << k) - 1:
            return 1 << (k - 1)
        i -= (1 << (k - 1)) - 1
        k = 1
        while (1 << k) - 1 < i:
            k += 1


class _CDCL:
    def __init__(self, cnf: CNF, restart_base: int) -> None:
        n = cnf.n_vars
        self.n = n
        self.value: list[int] = [0] * (n + 1)  # 0 unassigned, +1 true, -1 false
        self.level = [0] * (n + 1)
        self.reason: list[list[int] | None] = [None] * (n + 1)
        self.trail: list[int] = []
        self.trail_lim: list[int] = []
        self.qhead = 0
        self.watches: dict[int, list[list[int]]] = {}
        self.clauses: list[list[int]] = []
        self.activity = [0.0] * (n + 1)
        self.var_inc = 1.0
        self.phase = [-1] * (n + 1)  # saved polarity (false first)
        self.restart_base = restart_base
        self.decisions = self.conflicts = self.propagations = self.learned = 0
        self.restarts = 0

    # literal helpers
    def lit_value(self, lit: int) -> int:
        v = self.value[abs(lit)]
        return v if lit > 0 else -v

    def watch(self, clause: list[int]) -> None:
        self.watches.setdefault(clause[0], []).append(clause)
        self.watches.setdefault(clause[1], []).append(clause)

    def enqueue(self, lit: int, reason: list[int] | None) -> None:
        var = abs(lit)
        self.value[var] = 1 if lit > 0 else -1
        self.level[var] = len(self.trail_lim)
        self.reason[var] = reason
        self.trail.append(lit)

    def add_input_clause(self, lits: Sequence[int]) -> bool:
        """Add an original clause at level 0; False if the formula is now UNSAT."""
        if any(-x in lits for x in lits):
            return True  # tautology
        if not lits:
            return False
        if len(lits) == 1:
            val = self.lit_value(lits[0])
            if val == -1:
                return False
            if val == 0:
                self.enqueue(lits[0], None)
            return True
        clause = list(lits)
        self.clauses.append(clause)
        self.watch(clause)
        return True

    def propagate(self) -> list[int] | None:
        """Two-watched-literal unit propagation; returns a conflicting clause or None."""
        while self.qhead < len(self.trail):
            p = self.trail[self.qhead]
            self.qhead += 1
            false_lit = -p
            ws = self.watches.get(false_lit, [])
            kept: list[list[int]] = []
            i = 0
            while i < len(ws):
                c = ws[i]
                i += 1
                if c[0] == false_lit:  # keep the false watch in position 1
                    c[0], c[1] = c[1], c[0]
                if self.lit_value(c[0]) == 1:
                    kept.append(c)
                    continue
                for k in range(2, len(c)):
                    if self.lit_value(c[k]) != -1:
                        c[1], c[k] = c[k], c[1]
                        self.watches.setdefault(c[1], []).append(c)
                        break
                else:
                    kept.append(c)
                    if self.lit_value(c[0]) == -1:  # conflict
                        kept.extend(ws[i:])
                        self.watches[false_lit] = kept
                        return c
                    self.propagations += 1
                    self.enqueue(c[0], c)
            self.watches[false_lit] = kept
        return None

    def analyze(self, conflict: list[int]) -> tuple[list[int], int]:
        """1-UIP learning: (learned clause with the asserting literal first, backjump level)."""
        seen = [False] * (self.n + 1)
        learned: list[int] = [0]  # slot for the asserting literal
        counter = 0
        clause: list[int] = conflict
        p = 0
        idx = len(self.trail) - 1
        current = len(self.trail_lim)
        while True:
            for q in clause:
                if q == p:
                    continue
                v = abs(q)
                if not seen[v] and self.level[v] > 0:
                    seen[v] = True
                    self.bump(v)
                    if self.level[v] == current:
                        counter += 1
                    else:
                        learned.append(q)
            while not seen[abs(self.trail[idx])]:
                idx -= 1
            p = self.trail[idx]
            idx -= 1
            seen[abs(p)] = False
            counter -= 1
            if counter == 0:
                break
            reason = self.reason[abs(p)]
            clause = reason if reason is not None else []
        learned[0] = -p
        if len(learned) == 1:
            return learned, 0
        # Put the literal with the highest level (the backjump level) in position 1.
        best = max(range(1, len(learned)), key=lambda j: self.level[abs(learned[j])])
        learned[1], learned[best] = learned[best], learned[1]
        return learned, self.level[abs(learned[1])]

    def bump(self, v: int) -> None:
        self.activity[v] += self.var_inc
        if self.activity[v] > 1e100:
            self.activity = [a * 1e-100 for a in self.activity]
            self.var_inc *= 1e-100

    def backjump(self, level: int) -> None:
        if len(self.trail_lim) <= level:
            return
        start = self.trail_lim[level]
        for lit in self.trail[start:]:
            v = abs(lit)
            self.phase[v] = 1 if lit > 0 else -1
            self.value[v] = 0
            self.reason[v] = None
        del self.trail[start:]
        del self.trail_lim[level:]
        self.qhead = len(self.trail)

    def pick(self) -> int:
        best, best_act = 0, -1.0
        for v in range(1, self.n + 1):
            if self.value[v] == 0 and self.activity[v] > best_act:
                best, best_act = v, self.activity[v]
        return best if best == 0 else best * self.phase[best]

    def solve(self) -> bool:
        if self.propagate() is not None:
            return False
        restart_index = 1
        budget = self.restart_base * _luby(restart_index)
        since_restart = 0
        while True:
            conflict = self.propagate()
            if conflict is not None:
                self.conflicts += 1
                since_restart += 1
                if not self.trail_lim:
                    return False
                learned, level = self.analyze(conflict)
                self.backjump(level)
                self.var_inc /= 0.95
                if len(learned) == 1:
                    self.enqueue(learned[0], None)
                else:
                    self.clauses.append(learned)
                    self.watch(learned)
                    self.learned += 1
                    self.enqueue(learned[0], learned)
                continue
            if since_restart >= budget:
                self.restarts += 1
                restart_index += 1
                budget = self.restart_base * _luby(restart_index)
                since_restart = 0
                self.backjump(0)
                continue
            lit = self.pick()
            if lit == 0:
                return True
            self.decisions += 1
            self.trail_lim.append(len(self.trail))
            self.enqueue(lit, None)


@register(
    domain="ai",
    tasks=("sat", "logic"),
    complexity=Complexity(
        time="O(2^n m)",
        space="O(n + m + L)",
        notes="n variables, m = total clause length, L = total length of learned "
        "clauses; worst case exponential, each propagation visits only watched clauses",
    ),
    references=(_GRASP, _CHAFF, _MINISAT),
)
def cdcl(formula: CNF | Iterable[Iterable[int]], restart_base: int = 32) -> SATResult:
    r"""Decide satisfiability with conflict-driven clause learning.

    Two watched literals, 1-UIP learning, non-chronological backjumping, VSIDS-style
    activities with phase saving and Luby restarts (see the module docstring).

    Parameters
    ----------
    formula : CNF or list of clauses
    restart_base : int, default=32
        Conflicts per unit of the Luby restart sequence (>= 1).

    Returns
    -------
    SATResult
        ``"SAT"`` with a verified model, or ``"UNSAT"``.

    Complexity
    ----------
    Worst-case time :math:`O(2^n m)`; space :math:`O(n + m + L)` with L the total size
    of the learned clauses (never deleted).

    References
    ----------
    Marques-Silva & Sakallah (1999); Moskewicz et al. (2001); Eén & Sörensson (2003).

    Examples
    --------
    >>> res = cdcl([[1, 2], [-1, 2], [-2, 3], [-3, -1]])
    >>> res.status, res.model
    ('SAT', (-1, 2, 3))
    >>> cdcl([[1, 2], [1, -2], [-1, 2], [-1, -2]]).status
    'UNSAT'
    """
    cnf = _as_cnf(formula)
    restart_base = check_int(restart_base, "restart_base", minimum=1)
    solver = _CDCL(cnf, restart_base)
    ok = all(solver.add_input_clause(c) for c in cnf.clauses)
    sat = ok and solver.solve()
    counters = dict(
        decisions=solver.decisions,
        conflicts=solver.conflicts,
        propagations=solver.propagations,
        learned=solver.learned,
        restarts=solver.restarts,
    )
    if not sat:
        return SATResult("UNSAT", None, **counters)
    model = tuple(v if solver.value[v] == 1 else -v for v in range(1, cnf.n_vars + 1))
    return SATResult("SAT", _verified(cnf, model), **counters)


# ---------------------------------------------------------------------------
# WalkSAT
# ---------------------------------------------------------------------------
@register(
    domain="ai",
    tasks=("sat", "local-search"),
    complexity=Complexity(
        time="O(T k o)",
        space="O(n + m)",
        notes="T = max_flips, k = clause length, o = occurrences of a variable; "
        "n variables, m = total clause length",
    ),
    references=(_WALKSAT, _AIMA),
)
def walksat(
    formula: CNF | Iterable[Iterable[int]],
    p: float = 0.5,
    max_flips: int = 100_000,
    random_state: RandomState = None,
) -> SATResult:
    r"""WalkSAT local search (random-walk probability ``p``).

    Parameters
    ----------
    formula : CNF or list of clauses
    p : float, default=0.5
        Probability of a random-walk flip, in [0, 1].
    max_flips : int, default=100000
        Flip budget (>= 0).
    random_state : None, int, SeedSequence or Generator

    Returns
    -------
    SATResult
        ``"SAT"`` with a verified model, or ``"UNKNOWN"`` when the budget runs out
        (``"UNSAT"`` only for a formula with an empty clause).

    Complexity
    ----------
    Time :math:`O(Tko)` for T flips, clause length k and o occurrences per variable;
    space :math:`O(n + m)`.

    References
    ----------
    Selman, Kautz & Cohen (1994); Russell & Norvig (2020), §7.6.2.

    Examples
    --------
    >>> res = walksat([[1, -2], [2, 3], [-1, -3], [-3, 2]], random_state=0)
    >>> res.status
    'SAT'
    """
    cnf = _as_cnf(formula)
    p = check_float(p, "p", low=0.0, high=1.0)
    max_flips = check_int(max_flips, "max_flips")
    rng = check_random_state(random_state)
    if any(len(c) == 0 for c in cnf.clauses):
        return SATResult("UNSAT", None)
    clauses = [c for c in cnf.clauses if not any(-x in c for x in c)]  # drop tautologies
    n = cnf.n_vars
    value = [False] + [bool(b) for b in rng.integers(0, 2, size=n)]
    occurs: dict[int, list[int]] = {}
    for i, c in enumerate(clauses):
        for lit in c:
            occurs.setdefault(lit, []).append(i)

    def is_true(lit: int) -> bool:
        return value[abs(lit)] == (lit > 0)

    n_true = [sum(is_true(x) for x in c) for c in clauses]
    unsat = {i for i, t in enumerate(n_true) if t == 0}

    def flip(v: int) -> None:
        value[v] = not value[v]
        now_true = v if value[v] else -v
        for i in occurs.get(now_true, ()):
            n_true[i] += 1
            if n_true[i] == 1:
                unsat.discard(i)
        for i in occurs.get(-now_true, ()):
            n_true[i] -= 1
            if n_true[i] == 0:
                unsat.add(i)

    def breaks(v: int) -> int:
        true_lit = v if value[v] else -v
        return sum(1 for i in occurs.get(true_lit, ()) if n_true[i] == 1)

    flips = 0
    while unsat and flips < max_flips:
        pool = sorted(unsat)
        clause = clauses[pool[int(rng.integers(len(pool)))]]
        if rng.random() < p:
            v = abs(clause[int(rng.integers(len(clause)))])
        else:
            scores = [breaks(abs(x)) for x in clause]
            low = min(scores)
            best = [abs(x) for x, s in zip(clause, scores, strict=True) if s == low]
            v = best[int(rng.integers(len(best)))]
        flip(v)
        flips += 1
    if unsat:
        return SATResult("UNKNOWN", None, flips=flips)
    model = tuple(v if value[v] else -v for v in range(1, n + 1))
    return SATResult("SAT", _verified(cnf, model), flips=flips)
