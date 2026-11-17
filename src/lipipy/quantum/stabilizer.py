r"""Clifford circuits in polynomial time: the CHP stabilizer tableau (Aaronson & Gottesman).

A stabilizer state of n qubits is the unique state fixed by n commuting, independent
Pauli operators (its *stabilizer generators*). Clifford gates map Paulis to Paulis under
conjugation, so the state can be tracked by updating the generators instead of the
:math:`2^n` amplitudes (Gottesman-Knill theorem).

Tableau
-------
Rows :math:`0..n-1` are *destabilizers* and rows :math:`n..2n-1` *stabilizers*; row i
stores a Pauli :math:`(-1)^{r_i}\prod_j X_j^{x_{ij}} Z_j^{z_{ij}}` (with
:math:`X^1Z^1 \equiv Y`) as bit vectors :math:`x_i, z_i \in \{0,1\}^n` and a sign bit
:math:`r_i`. The initial state :math:`|0\ldots0\rangle` has destabilizers :math:`X_j` and
stabilizers :math:`Z_j`. Gate updates (column operations, O(n) each):

* H on a: :math:`r \mathrel{\oplus}= x_a z_a`; swap :math:`x_a, z_a`.
* S on a: :math:`r \mathrel{\oplus}= x_a z_a`; :math:`z_a \mathrel{\oplus}= x_a`.
* CX(a→b): :math:`r \mathrel{\oplus}= x_a z_b (x_b \oplus z_a \oplus 1)`;
  :math:`x_b \mathrel{\oplus}= x_a`; :math:`z_a \mathrel{\oplus}= z_b`.
* Paulis only flip signs: X: :math:`r \mathrel{\oplus}= z_a`, Z: :math:`r \mathrel{\oplus}= x_a`,
  Y: :math:`r \mathrel{\oplus}= x_a \oplus z_a`.

Measuring :math:`Z_a`: if some stabilizer anticommutes with it (:math:`x_{pa} = 1`) the
outcome is uniformly random and the tableau is updated with "rowsum" products;
otherwise it is determined by the product of the stabilizers selected by the
destabilizers with :math:`x_{ia} = 1`. Rowsums are vectorised over all affected rows
(random case) or done as one prefix-XOR product (deterministic case).

Costs: memory :math:`O(n^2)` bits; each gate :math:`O(n)`; each measurement
:math:`O(n^2)` in the worst case.

Reference: S. Aaronson & D. Gottesman, "Improved simulation of stabilizer circuits",
Phys. Rev. A 70, 052328 (2004), arXiv:quant-ph/0406196; Gottesman, PhD thesis (1997).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lipipy.core.complexity import Complexity
from lipipy.core.errors import ResourceLimitError
from lipipy.core.random import RandomState, check_random_state
from lipipy.core.registry import register
from lipipy.quantum import gates
from lipipy.quantum._validation import check_int, check_qubits
from lipipy.quantum.circuit import Circuit

#: Default largest tableau (2n x n bits per table: 10 000 qubits ≈ 400 MB in total).
MAX_STABILIZER_QUBITS = 10_000

_AG = (
    "Aaronson & Gottesman, Improved simulation of stabilizer circuits, "
    "Phys. Rev. A 70, 052328 (2004)"
)
_CLIFFORD_MATRICES = {
    "id": gates.I,
    "h": gates.H,
    "s": gates.S,
    "sdg": gates.SDG,
    "x": gates.X,
    "y": gates.Y,
    "z": gates.Z,
    "cx": gates.CX,
    "cz": gates.CZ,
    "swap": gates.SWAP,
}


def _g_sum(x1: np.ndarray, z1: np.ndarray, x2: np.ndarray, z2: np.ndarray) -> np.ndarray:
    """Row sums of the exponent g of i in (Pauli 1)(Pauli 2), elementwise per qubit.

    g = 0 for I; z2 - x2 for Y; z2 (2 x2 - 1) for X; x2 (1 - 2 z2) for Z.
    Inputs are int8 arrays of equal (broadcastable) shape; sums over the last axis.
    """
    g = np.where(
        (x1 == 1) & (z1 == 1),
        z2 - x2,
        np.where(x1 == 1, z2 * (2 * x2 - 1), np.where(z1 == 1, x2 * (1 - 2 * z2), 0)),
    )
    return g.sum(axis=-1, dtype=np.int64)


@dataclass(frozen=True)
class StabilizerMeasurement:
    """Outcome of :meth:`StabilizerState.measure`.

    Attributes
    ----------
    outcome : int
        0 or 1.
    deterministic : bool
        True when the outcome had probability 1 (otherwise it had probability 1/2).
    """

    outcome: int
    deterministic: bool

    def to_dict(self) -> dict[str, Any]:
        return {"outcome": self.outcome, "deterministic": self.deterministic}


@register(
    domain="quantum",
    tasks=("simulation", "clifford", "stabilizer"),
    complexity=Complexity(
        time="O(g q + m q^2)",
        space="O(q^2)",
        notes="g Clifford gates at O(q) each, m measurements at O(q^2) worst case",
    ),
    references=(_AG, "Gottesman, Stabilizer codes and quantum error correction, PhD thesis (1997)"),
)
class StabilizerState:
    r"""Stabilizer state of ``num_qubits`` qubits, starting in :math:`|0\ldots0\rangle`.

    Gate methods (``h, s, sdg, x, y, z, cx, cz, swap``) update the tableau in place and
    return ``self``; they validate their arguments before changing anything.
    :meth:`measure` collapses the state. Use :meth:`copy` to branch.

    Parameters
    ----------
    num_qubits : int
    max_qubits : int, default=10_000
        Raise :class:`~lipipy.core.ResourceLimitError` before allocating the
        O(q^2) tableau for more qubits.

    Complexity
    ----------
    Memory O(q^2) (two (2q, q) bit tables). Gates O(q). Z measurement O(q^2) worst
    case: O(q · #anticommuting rows) when random, O(q · #selected rows) when
    deterministic.

    References
    ----------
    Aaronson & Gottesman (2004).

    Examples
    --------
    >>> from lipipy.quantum import StabilizerState
    >>> ghz = StabilizerState(3).h(0).cx(0, 1).cx(1, 2)
    >>> ghz.stabilizers()
    ['+XXX', '+IZZ', '+ZZI']
    >>> first = ghz.measure(0, random_state=1)
    >>> [first.deterministic, ghz.measure(1).deterministic, ghz.measure(2).outcome == first.outcome]
    [False, True, True]
    """

    def __init__(self, num_qubits: int, *, max_qubits: int = MAX_STABILIZER_QUBITS):
        n = check_int(num_qubits, "num_qubits", minimum=1)
        limit = check_int(max_qubits, "max_qubits", minimum=1)
        if n > limit:
            raise ResourceLimitError(
                f"A stabilizer tableau of {n} qubits needs ~{2 * (2 * n) * n + 2 * n} bytes, "
                f"above max_qubits={limit}."
            )
        self._n = n
        self._x = np.zeros((2 * n, n), dtype=np.int8)
        self._z = np.zeros((2 * n, n), dtype=np.int8)
        self._r = np.zeros(2 * n, dtype=np.int8)
        idx = np.arange(n)
        self._x[idx, idx] = 1  # destabilizers X_j
        self._z[n + idx, idx] = 1  # stabilizers Z_j

    @classmethod
    def from_circuit(
        cls, circuit: Circuit, *, max_qubits: int = MAX_STABILIZER_QUBITS
    ) -> StabilizerState:
        """Run a Clifford :class:`~lipipy.quantum.Circuit` from :math:`|0\\ldots0\\rangle`.

        Supported instructions: id, h, s, sdg, x, y, z, cx, cz, swap, recognised by
        name **and** matrix (a ``unitary`` merely named ``"h"`` is rejected). Raises
        ``ValueError`` on anything else before simulating.
        """
        if not isinstance(circuit, Circuit):
            raise TypeError(f"circuit must be a Circuit, got {type(circuit).__name__}")
        ops = circuit.instructions
        for op in ops:
            ref = _CLIFFORD_MATRICES.get(op.name)
            if ref is None or op.diagonal is not None or not np.array_equal(op.matrix, ref):
                raise ValueError(
                    f"instruction {op.name!r} is not a supported Clifford gate "
                    f"({', '.join(sorted(_CLIFFORD_MATRICES))})"
                )
        state = cls(circuit.num_qubits, max_qubits=max_qubits)
        for op in ops:
            if op.name != "id":
                getattr(state, op.name)(*op.qubits)
        return state

    # -- accessors -----------------------------------------------------------------------
    @property
    def num_qubits(self) -> int:
        return self._n

    def copy(self) -> StabilizerState:
        """Independent copy (O(q^2))."""
        out = StabilizerState.__new__(StabilizerState)
        out._n = self._n
        out._x, out._z, out._r = self._x.copy(), self._z.copy(), self._r.copy()
        return out

    def _label(self, row: int) -> str:
        letters = {(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}
        body = "".join(
            letters[(int(self._x[row, j]), int(self._z[row, j]))] for j in reversed(range(self._n))
        )
        return ("-" if self._r[row] else "+") + body

    def stabilizers(self) -> list[str]:
        """Signed generator labels like ``'+XXI'`` (rightmost letter = qubit 0, Qiskit order).

        Each generator P satisfies :math:`P|\\psi\\rangle = |\\psi\\rangle`. A ``Y``
        letter means the Hermitian Pauli Y.
        """
        return [self._label(self._n + i) for i in range(self._n)]

    def destabilizers(self) -> list[str]:
        """Signed destabilizer labels (same format as :meth:`stabilizers`)."""
        return [self._label(i) for i in range(self._n)]

    def __repr__(self) -> str:
        return f"StabilizerState(num_qubits={self._n})"

    # -- gates ---------------------------------------------------------------------------
    def _q(self, *qubits: int) -> tuple[int, ...]:
        return check_qubits(list(qubits), self._n, "qubits")

    def h(self, qubit: int) -> StabilizerState:
        (a,) = self._q(qubit)
        x, z = self._x[:, a].copy(), self._z[:, a].copy()
        self._r ^= x & z
        self._x[:, a], self._z[:, a] = z, x
        return self

    def s(self, qubit: int) -> StabilizerState:
        (a,) = self._q(qubit)
        self._r ^= self._x[:, a] & self._z[:, a]
        self._z[:, a] ^= self._x[:, a]
        return self

    def sdg(self, qubit: int) -> StabilizerState:
        (a,) = self._q(qubit)
        # S^† maps X -> -Y and Y -> X
        self._r ^= self._x[:, a] & (1 - self._z[:, a])
        self._z[:, a] ^= self._x[:, a]
        return self

    def x(self, qubit: int) -> StabilizerState:
        (a,) = self._q(qubit)
        self._r ^= self._z[:, a]
        return self

    def y(self, qubit: int) -> StabilizerState:
        (a,) = self._q(qubit)
        self._r ^= self._x[:, a] ^ self._z[:, a]
        return self

    def z(self, qubit: int) -> StabilizerState:
        (a,) = self._q(qubit)
        self._r ^= self._x[:, a]
        return self

    def cx(self, control: int, target: int) -> StabilizerState:
        a, b = self._q(control, target)
        xa, xb, za, zb = self._x[:, a], self._x[:, b], self._z[:, a], self._z[:, b]
        self._r ^= xa & zb & (xb ^ za ^ 1)
        self._x[:, b] ^= xa
        self._z[:, a] ^= zb
        return self

    def cz(self, control: int, target: int) -> StabilizerState:
        a, b = self._q(control, target)
        return self.h(b).cx(a, b).h(b)

    def swap(self, qubit1: int, qubit2: int) -> StabilizerState:
        a, b = self._q(qubit1, qubit2)
        self._x[:, [a, b]] = self._x[:, [b, a]]
        self._z[:, [a, b]] = self._z[:, [b, a]]
        return self

    # -- measurement ---------------------------------------------------------------------
    def measure(self, qubit: int, random_state: RandomState = None) -> StabilizerMeasurement:
        """Measure :math:`Z` on ``qubit`` and collapse the state.

        Returns
        -------
        StabilizerMeasurement
            The outcome and whether it was deterministic.
        """
        (a,) = self._q(qubit)
        rng = check_random_state(random_state)
        n = self._n
        anti = np.flatnonzero(self._x[n:, a]) + n
        if anti.size:
            outcome = int(rng.integers(2))  # drawn before the tableau changes
            p = int(anti[0])
            rows = np.flatnonzero(self._x[:, a])
            rows = rows[rows != p]
            if rows.size:  # rowsum(h, p) for every other row h with x_ha = 1, vectorised
                xp, zp = self._x[p], self._z[p]
                xh, zh = self._x[rows], self._z[rows]
                total = (
                    2 * self._r[rows].astype(np.int64)
                    + 2 * int(self._r[p])
                    + _g_sum(xp, zp, xh, zh)
                )
                self._r[rows] = ((total % 4) // 2).astype(np.int8)
                self._x[rows] = xh ^ xp
                self._z[rows] = zh ^ zp
            self._x[p - n], self._z[p - n], self._r[p - n] = self._x[p], self._z[p], self._r[p]
            self._x[p] = 0
            self._z[p] = 0
            self._z[p, a] = 1
            self._r[p] = outcome
            return StabilizerMeasurement(outcome, False)
        # deterministic: product of stabilizers n+i for destabilizers i with x_ia = 1
        sel = np.flatnonzero(self._x[:n, a]) + n
        xs, zs = self._x[sel], self._z[sel]
        x_prev = np.bitwise_xor.accumulate(xs, axis=0)
        z_prev = np.bitwise_xor.accumulate(zs, axis=0)
        x_prev = np.vstack([np.zeros((1, n), np.int8), x_prev[:-1]])
        z_prev = np.vstack([np.zeros((1, n), np.int8), z_prev[:-1]])
        exponent = 2 * int(self._r[sel].sum()) + int(_g_sum(xs, zs, x_prev, z_prev).sum())
        return StabilizerMeasurement(int((exponent % 4) // 2), True)

    def measure_all(self, random_state: RandomState = None) -> str:
        """Measure every qubit (0 first) and return the bitstring, qubit 0 rightmost."""
        rng = check_random_state(random_state)
        bits = [self.measure(j, rng).outcome for j in range(self._n)]
        return "".join(str(b) for b in reversed(bits))


@register(
    domain="quantum",
    tasks=("simulation", "clifford", "sampling"),
    complexity=Complexity(
        time="O(g q + P q^3)",
        space="O(q^2)",
        notes="one tableau run, then P = shots copies measured on all q qubits",
    ),
    references=(_AG,),
)
def stabilizer_sample_counts(
    circuit: Circuit, shots: int = 1024, random_state: RandomState = None
) -> dict[str, int]:
    """Sample measurement counts of a Clifford circuit with the CHP simulator.

    Parameters
    ----------
    circuit : Circuit
        Only Clifford gates (see :meth:`StabilizerState.from_circuit`).
    shots : int, default=1024
    random_state : None, int, SeedSequence or Generator

    Returns
    -------
    dict of str to int
        Bitstrings (qubit 0 rightmost) to counts, sorted by bitstring.

    Complexity
    ----------
    O(g · q) for the circuit, O(q^3) worst case per shot (q measurements of O(q^2)),
    memory O(q^2).

    References
    ----------
    Aaronson & Gottesman (2004).

    Examples
    --------
    >>> from lipipy.quantum import Circuit, stabilizer_sample_counts
    >>> counts = stabilizer_sample_counts(Circuit(2).h(0).cx(0, 1), shots=100, random_state=0)
    >>> sorted(counts), sum(counts.values())
    (['00', '11'], 100)
    """
    shots = check_int(shots, "shots", minimum=1)
    base = StabilizerState.from_circuit(circuit)
    rng = check_random_state(random_state)
    counts: dict[str, int] = {}
    for _ in range(shots):
        key = base.copy().measure_all(rng)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))
