r"""A fluent quantum-circuit builder with a statevector backend.

A :class:`Circuit` is an ordered list of :class:`Instruction` objects (a gate matrix
plus the qubits it acts on). Running it applies each gate in turn with the tensor
contraction of :func:`lipipy.quantum.apply_gate`, so a circuit of ``g`` gates on
``q`` qubits costs :math:`O(g \cdot 2^q)` time (for gates of bounded size) and
:math:`O(2^q)` memory; no :math:`2^q \times 2^q` matrix is formed unless you ask
for :meth:`Circuit.to_matrix`.

Builder methods (``h``, ``cx``, ...) **append to the circuit in place and return it**,
so calls chain: ``Circuit(2).h(0).cx(0, 1)``. :meth:`Circuit.inverse` and
:meth:`Circuit.compose` return new circuits.

Qubit and matrix conventions are those of :mod:`lipipy.quantum.gates`: qubit 0 is
the least-significant bit, and ``cx(control, target)`` / ``unitary(U, qubits)``
match Qiskit's ``QuantumCircuit.cx`` / ``QuantumCircuit.unitary``.

Reference: Nielsen & Chuang, *Quantum Computation and Quantum Information*, §4.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from lipipy.core.complexity import Complexity
from lipipy.core.errors import ResourceLimitError
from lipipy.core.registry import register
from lipipy.quantum import gates
from lipipy.quantum._validation import (
    MAX_QUBITS,
    check_angle,
    check_int,
    check_num_qubits,
    check_qubits,
    check_square_matrix,
    check_state,
)
from lipipy.quantum.statevector import _apply_diagonal_tensor, _apply_tensor

#: Largest circuit for which :meth:`Circuit.to_matrix` builds the dense unitary.
MAX_MATRIX_QUBITS = 10

_SELF_INVERSE = {"id", "x", "y", "z", "h", "cx", "cz", "swap", "ccx"}
_INVERSE_NAME = {"s": "sdg", "sdg": "s", "t": "tdg", "tdg": "t", "sx": "sxdg", "sxdg": "sx"}
_NEGATE_PARAMS = {"rx", "ry", "rz", "p", "cp"}


@dataclass(frozen=True, eq=False)
class Instruction:
    """One gate in a circuit.

    Attributes
    ----------
    name : str
        Gate name (``"h"``, ``"cx"``, ``"unitary"``, ...).
    qubits : tuple of int
        Targets; ``qubits[0]`` is the gate's least-significant qubit.
    params : tuple of float
        Angles of parametrized gates (empty otherwise).
    diagonal : ndarray or None
        For diagonal gates, the diagonal only (applied in O(2^q)); else ``None``.
    """

    name: str
    qubits: tuple[int, ...]
    params: tuple[float, ...] = ()
    diagonal: np.ndarray | None = field(default=None, repr=False)
    _matrix: np.ndarray | None = field(default=None, repr=False)

    @property
    def matrix(self) -> np.ndarray:
        """Dense gate matrix, shape ``(2**k, 2**k)``."""
        if self._matrix is not None:
            return self._matrix
        assert self.diagonal is not None  # nosec B101 - invariant of the constructor
        return np.diag(self.diagonal)

    def inverse(self) -> Instruction:
        """The adjoint gate, with a matching name and parameters."""
        if self.name in _SELF_INVERSE:
            return self
        name = _INVERSE_NAME.get(self.name, self.name)
        params = self.params
        if self.name in _NEGATE_PARAMS:
            params = tuple(-p for p in params)
        elif self.name == "u":
            theta, phi, lam = params
            params = (-theta, -lam, -phi)
        if self.diagonal is not None:
            return Instruction(name, self.qubits, params, diagonal=self.diagonal.conj())
        assert self._matrix is not None  # nosec B101
        return Instruction(name, self.qubits, params, _matrix=self._matrix.conj().T)

    def __str__(self) -> str:
        args = ", ".join(f"{p:.6g}" for p in self.params)
        head = f"{self.name}({args})" if args else self.name
        return f"{head} " + ", ".join(f"q{t}" for t in self.qubits)


@register(
    domain="quantum",
    tasks=("circuit", "simulation"),
    complexity=Complexity(
        time="O(g 2^q)",
        space="O(2^q)",
        notes="run() for gates on a bounded number of qubits; to_matrix() is O(g 4^q)",
    ),
    references=("Nielsen & Chuang, Quantum Computation and Quantum Information, §4",),
)
class Circuit:
    """A quantum circuit on ``num_qubits`` qubits, built with chained gate calls.

    Parameters
    ----------
    num_qubits : int
        Number of qubits ``q >= 1``. Construction never allocates a state; the
        ``max_qubits`` guard applies in :meth:`run`.

    Attributes
    ----------
    num_qubits : int
    instructions : tuple of Instruction
        The gates in application order.
    num_gates : int

    Complexity
    ----------
    :meth:`run`: time O(g · 2^q) for g gates of bounded size (a k-qubit ``unitary``
    costs O(2^q · 2^k)), extra space O(2^q). :meth:`to_matrix`: time O(g · 4^q),
    space O(4^q), limited to 10 qubits. :meth:`depth`: O(g).

    References
    ----------
    Nielsen & Chuang, §4.

    Examples
    --------
    >>> from lipipy.quantum import Circuit
    >>> bell = Circuit(2).h(0).cx(0, 1)
    >>> bell
    Circuit(num_qubits=2, num_gates=2, depth=2)
      h q0
      cx q0, q1
    >>> bell.run().real.round(6)
    array([0.707107, 0.      , 0.      , 0.707107])
    """

    def __init__(self, num_qubits: int):
        self.num_qubits = check_int(num_qubits, "num_qubits", minimum=1)
        self._ops: list[Instruction] = []

    # -- bookkeeping -------------------------------------------------------------------
    @property
    def instructions(self) -> tuple[Instruction, ...]:
        return tuple(self._ops)

    @property
    def num_gates(self) -> int:
        return len(self._ops)

    def __len__(self) -> int:
        return len(self._ops)

    def _append(self, op: Instruction) -> Circuit:
        self._ops.append(op)
        return self

    def _gate(
        self, name: str, matrix: np.ndarray, qubits: Sequence[int], params: tuple[float, ...] = ()
    ) -> Circuit:
        tgts = check_qubits(qubits, self.num_qubits, "qubits")
        return self._append(Instruction(name, tgts, tuple(params), _matrix=matrix))

    # -- single-qubit gates ------------------------------------------------------------
    def id(self, qubit: int) -> Circuit:
        """Identity (a no-op that still counts as a gate)."""
        return self._gate("id", gates.I, [qubit])

    def h(self, qubit: int) -> Circuit:
        """Hadamard."""
        return self._gate("h", gates.H, [qubit])

    def x(self, qubit: int) -> Circuit:
        """Pauli X."""
        return self._gate("x", gates.X, [qubit])

    def y(self, qubit: int) -> Circuit:
        """Pauli Y."""
        return self._gate("y", gates.Y, [qubit])

    def z(self, qubit: int) -> Circuit:
        """Pauli Z."""
        return self._gate("z", gates.Z, [qubit])

    def s(self, qubit: int) -> Circuit:
        """S = sqrt(Z)."""
        return self._gate("s", gates.S, [qubit])

    def sdg(self, qubit: int) -> Circuit:
        """S dagger."""
        return self._gate("sdg", gates.SDG, [qubit])

    def t(self, qubit: int) -> Circuit:
        """T = Z^(1/4)."""
        return self._gate("t", gates.T, [qubit])

    def tdg(self, qubit: int) -> Circuit:
        """T dagger."""
        return self._gate("tdg", gates.TDG, [qubit])

    def sx(self, qubit: int) -> Circuit:
        """SX = sqrt(X)."""
        return self._gate("sx", gates.SX, [qubit])

    def rx(self, theta: float, qubit: int) -> Circuit:
        """Rotation exp(-i theta X / 2)."""
        theta = check_angle(theta, "theta")
        return self._gate("rx", gates.rx(theta), [qubit], (theta,))

    def ry(self, theta: float, qubit: int) -> Circuit:
        """Rotation exp(-i theta Y / 2)."""
        theta = check_angle(theta, "theta")
        return self._gate("ry", gates.ry(theta), [qubit], (theta,))

    def rz(self, theta: float, qubit: int) -> Circuit:
        """Rotation exp(-i theta Z / 2)."""
        theta = check_angle(theta, "theta")
        return self._gate("rz", gates.rz(theta), [qubit], (theta,))

    def p(self, lam: float, qubit: int) -> Circuit:
        """Phase gate diag(1, e^{i lam})."""
        lam = check_angle(lam, "lam")
        return self._gate("p", gates.phase(lam), [qubit], (lam,))

    def u(self, theta: float, phi: float, lam: float, qubit: int) -> Circuit:
        """Qiskit's general single-qubit gate U(theta, phi, lam)."""
        U = gates.u3(theta, phi, lam)
        return self._gate("u", U, [qubit], (float(theta), float(phi), float(lam)))

    # -- multi-qubit gates -------------------------------------------------------------
    def cx(self, control: int, target: int) -> Circuit:
        """Controlled-X (CNOT)."""
        return self._gate("cx", gates.CX, [control, target])

    def cz(self, control: int, target: int) -> Circuit:
        """Controlled-Z."""
        return self._gate("cz", gates.CZ, [control, target])

    def cp(self, lam: float, control: int, target: int) -> Circuit:
        """Controlled phase diag(1, 1, 1, e^{i lam}) (used by the QFT)."""
        lam = check_angle(lam, "lam")
        tgts = check_qubits([control, target], self.num_qubits, "qubits")
        diag = np.array([1, 1, 1, np.exp(1j * lam)], dtype=np.complex128)
        return self._append(Instruction("cp", tgts, (lam,), diagonal=diag))

    def swap(self, qubit1: int, qubit2: int) -> Circuit:
        """SWAP two qubits."""
        return self._gate("swap", gates.SWAP, [qubit1, qubit2])

    def ccx(self, control1: int, control2: int, target: int) -> Circuit:
        """Toffoli (doubly controlled X)."""
        return self._gate("ccx", gates.CCX, [control1, control2, target])

    def unitary(self, U: np.ndarray, qubits: Sequence[int] | int, name: str = "unitary") -> Circuit:
        """Arbitrary k-qubit unitary on ``qubits`` (``qubits[0]`` = least-significant).

        Raises ``ValueError`` if ``U`` is not unitary (atol 1e-10) or its size does not
        match ``len(qubits)``.
        """
        M, k = check_square_matrix(U, "U")
        tgts = check_qubits(qubits, self.num_qubits, "qubits")
        if len(tgts) != k:
            raise ValueError(f"a {k}-qubit unitary needs {k} qubits, got {len(tgts)}")
        if not gates.is_unitary(M):
            raise ValueError("U is not unitary (max |U^† U - I| > 1e-10)")
        return self._append(Instruction(str(name), tgts, (), _matrix=M.copy()))

    def diagonal(
        self,
        diag: Sequence[complex] | np.ndarray,
        qubits: Sequence[int] | int,
        name: str = "diagonal",
    ) -> Circuit:
        """Diagonal unitary given by its ``2**k`` diagonal entries (all of modulus 1).

        Applied in O(2^q) time whatever ``k`` is, which makes phase oracles cheap.
        """
        d, k = check_state(diag, "diag")
        tgts = check_qubits(qubits, self.num_qubits, "qubits")
        if len(tgts) != k:
            raise ValueError(f"a {k}-qubit diagonal needs {k} qubits, got {len(tgts)}")
        if np.max(np.abs(np.abs(d) - 1.0)) > 1e-10:
            raise ValueError("diagonal entries must have modulus 1 to be unitary")
        return self._append(Instruction(str(name), tgts, (), diagonal=d.copy()))

    # -- simulation --------------------------------------------------------------------
    def _run_tensor(self, tensor: np.ndarray) -> np.ndarray:
        q = self.num_qubits
        for op in self._ops:
            if op.diagonal is not None:
                tensor = _apply_diagonal_tensor(tensor, op.diagonal, op.qubits, q)
            else:
                tensor = _apply_tensor(tensor, op.matrix, op.qubits, q)
        return tensor

    def run(
        self, initial_state: np.ndarray | None = None, *, max_qubits: int = MAX_QUBITS
    ) -> np.ndarray:
        """Simulate the circuit and return the final statevector.

        Parameters
        ----------
        initial_state : array_like of shape (2**q,), optional
            Start state; defaults to :math:`|0\\ldots0\\rangle`. Not modified.
        max_qubits : int, default=26
            Raise :class:`~lipipy.core.ResourceLimitError` before allocating if the
            circuit has more qubits than this.

        Returns
        -------
        ndarray of complex128, shape (2**q,)
        """
        q = check_num_qubits(self.num_qubits, max_qubits)
        if initial_state is None:
            psi = np.zeros(2**q, dtype=np.complex128)
            psi[0] = 1.0
        else:
            psi, nq = check_state(initial_state, "initial_state")
            if nq != q:
                raise ValueError(f"initial_state has {nq} qubits, the circuit has {q}")
        return self._run_tensor(psi.reshape((2,) * q)).reshape(-1)

    def to_matrix(self) -> np.ndarray:
        """The circuit's full unitary, shape ``(2**q, 2**q)``, for q <= 10.

        Column ``j`` is ``run(basis_state(q, j))``; all columns are pushed through
        the gates at once as a batch axis. Raises
        :class:`~lipipy.core.ResourceLimitError` above 10 qubits (16·4^q bytes).
        """
        q = self.num_qubits
        if q > MAX_MATRIX_QUBITS:
            raise ResourceLimitError(
                f"to_matrix needs 16·4^{q} bytes for {q} qubits; it is limited to "
                f"{MAX_MATRIX_QUBITS} qubits. Use run() for statevectors instead."
            )
        N = 2**q
        tensor = np.eye(N, dtype=np.complex128).reshape((2,) * q + (N,))
        return self._run_tensor(tensor).reshape(N, N)

    # -- transformations ---------------------------------------------------------------
    def copy(self) -> Circuit:
        """Shallow copy (instructions are immutable)."""
        out = Circuit(self.num_qubits)
        out._ops = list(self._ops)
        return out

    def inverse(self) -> Circuit:
        """New circuit implementing the adjoint: reversed order, each gate inverted."""
        out = Circuit(self.num_qubits)
        out._ops = [op.inverse() for op in reversed(self._ops)]
        return out

    def compose(self, other: Circuit, qubits: Sequence[int] | None = None) -> Circuit:
        """New circuit: ``self`` followed by ``other``.

        Parameters
        ----------
        other : Circuit
            Circuit to append, with ``other.num_qubits <= self.num_qubits``.
        qubits : sequence of int, optional
            Where ``other``'s qubit ``i`` goes (``qubits[i]``); default identity.
        """
        if not isinstance(other, Circuit):
            raise TypeError(f"other must be a Circuit, got {type(other).__name__}")
        if qubits is None:
            if other.num_qubits > self.num_qubits:
                raise ValueError(
                    f"cannot compose a {other.num_qubits}-qubit circuit onto "
                    f"{self.num_qubits} qubits"
                )
            mapping = tuple(range(other.num_qubits))
        else:
            mapping = check_qubits(qubits, self.num_qubits, "qubits")
            if len(mapping) != other.num_qubits:
                raise ValueError(f"qubits must list {other.num_qubits} indices")
        out = self.copy()
        for op in other._ops:
            new_q = tuple(mapping[t] for t in op.qubits)
            out._ops.append(
                Instruction(op.name, new_q, op.params, diagonal=op.diagonal, _matrix=op._matrix)
            )
        return out

    # -- statistics --------------------------------------------------------------------
    def depth(self) -> int:
        """Number of layers when every gate is scheduled as early as possible."""
        level = [0] * self.num_qubits
        for op in self._ops:
            d = 1 + max(level[t] for t in op.qubits)
            for t in op.qubits:
                level[t] = d
        return max(level, default=0)

    def count_ops(self) -> dict[str, int]:
        """Gate counts by name, most frequent first (ties by name)."""
        counts = Counter(op.name for op in self._ops)
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    def __repr__(self) -> str:
        head = (
            f"Circuit(num_qubits={self.num_qubits}, num_gates={self.num_gates}, "
            f"depth={self.depth()})"
        )
        limit = 20
        lines = [f"  {op}" for op in self._ops[:limit]]
        if self.num_gates > limit:
            lines.append(f"  ... ({self.num_gates - limit} more)")
        return "\n".join([head, *lines])
