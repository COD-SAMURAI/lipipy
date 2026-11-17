r"""Pauli-sum observables and model Hamiltonians.

An observable is a real linear combination of Pauli strings,

.. math::
    H = \sum_{t=1}^{T} c_t P_t, \qquad P_t \in \{I, X, Y, Z\}^{\otimes q},\ c_t \in \mathbb{R},

which is Hermitian. Labels follow Qiskit: the **rightmost** letter acts on qubit 0.

Expectation values without matrices
-----------------------------------
Write a Pauli string as :math:`P = i^{n_Y} X^{\mathbf x} Z^{\mathbf z}` with bit masks
:math:`\mathbf x` (letters X or Y) and :math:`\mathbf z` (letters Z or Y), using
:math:`Y = iXZ`. On a basis state,

.. math::  P|j\rangle = i^{n_Y} (-1)^{|j \wedge \mathbf z|}\, |j \oplus \mathbf x\rangle,

so :math:`\langle\psi|P|\psi\rangle = i^{n_Y}\sum_j (-1)^{|j\wedge\mathbf z|}
\overline{\psi_{j\oplus\mathbf x}}\psi_j` and
:math:`\mathrm{Tr}(P\rho) = i^{n_Y}\sum_j (-1)^{|j\wedge\mathbf z|}\rho_{j, j\oplus\mathbf x}`,
both :math:`O(2^q)` per term. A whole observable costs :math:`O(T \cdot 2^q)`.

Hamiltonians
------------
* :func:`ising_chain`: transverse-field Ising model
  :math:`H = -J\sum_i Z_i Z_{i+1} - h\sum_i X_i`.
* :func:`h2_hamiltonian`: H₂ in the STO-3G basis at bond length 0.735 Å, parity
  mapping with two-qubit reduction (5 terms, coefficients from the Qiskit
  documentation, generated with Qiskit Nature + PySCF). The coefficients give the
  **electronic** energy; the nuclear repulsion :data:`H2_NUCLEAR_REPULSION` is added
  to the identity term by default, so the ground energy is the total
  −1.137306 Ha (electronic −1.857275 Ha).

Reference: Nielsen & Chuang, §10.5.1 (Pauli group); Sachdev, *Quantum Phase
Transitions* (Ising chain); Kandala et al., Nature 549, 242 (2017) and the Qiskit
VQE tutorial (H₂ operator); Bravyi, Gambetta, Mezzacapo & Temme, arXiv:1701.08213
(two-qubit reduction).
"""

from __future__ import annotations

import numbers
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from lipipy.core.complexity import Complexity
from lipipy.core.errors import ResourceLimitError
from lipipy.core.registry import register
from lipipy.quantum._validation import check_angle, check_int, check_state
from lipipy.quantum.density import DensityMatrix

#: Largest observable for which :meth:`PauliSum.to_matrix` builds the dense matrix.
MAX_OBSERVABLE_MATRIX_QUBITS = 12

#: Nuclear repulsion 1/R (Hartree) of H₂ at R = 0.735 Å, with the bohr radius
#: 0.52917721092 Å used by PySCF.
H2_NUCLEAR_REPULSION = 1.0 / (0.735 / 0.52917721092)

#: Electronic two-qubit H₂ Hamiltonian at 0.735 Å (STO-3G, parity mapping,
#: two-qubit reduction), as published in the Qiskit documentation.
H2_COEFFICIENTS = (
    ("II", -1.052373245772859),
    ("IZ", 0.39793742484318045),
    ("ZI", -0.39793742484318045),
    ("ZZ", -0.01128010425623538),
    ("XX", 0.18093119978423156),
)

_PAULI_MATS = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


def _parity(values: np.ndarray) -> np.ndarray:
    """Parity (popcount mod 2) of non-negative int64 values, by XOR folding."""
    v = values.copy()
    for shift in (32, 16, 8, 4, 2, 1):
        v ^= v >> shift
    return v & 1


def _masks(label: str) -> tuple[int, int, int]:
    """(x mask, z mask, number of Y) of a Pauli label (rightmost letter = qubit 0)."""
    x = z = ny = 0
    for i, letter in enumerate(reversed(label)):
        if letter in "XY":
            x |= 1 << i
        if letter in "ZY":
            z |= 1 << i
        ny += letter == "Y"
    return x, z, ny


def _check_label(label: object) -> str:
    if not isinstance(label, str):
        raise TypeError(f"Pauli labels must be strings, got {type(label).__name__}")
    if not label or set(label) - set("IXYZ"):
        raise ValueError(f"Pauli labels must be non-empty strings over IXYZ, got {label!r}")
    return label


def _check_coeff(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"coefficients must be real numbers, got {type(value).__name__}")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"coefficients must be finite, got {value}")
    return value


@register(
    domain="quantum",
    tasks=("observable", "expectation", "hamiltonian"),
    complexity=Complexity(
        time="O(T 2^q)",
        space="O(2^q)",
        notes="T = number of Pauli terms; expectation on a statevector or (diagonal "
        "walk of) a density matrix, never building the 2^q x 2^q operator",
    ),
    references=("Nielsen & Chuang, Quantum Computation and Quantum Information, §10.5.1",),
)
class PauliSum:
    r"""Immutable Hermitian observable :math:`\sum_t c_t P_t` with real coefficients.

    Parameters
    ----------
    terms : mapping label -> coefficient, or iterable of (label, coefficient)
        Pauli labels of equal length over ``"IXYZ"`` (rightmost letter = qubit 0) and
        real coefficients. Repeated labels are summed; terms whose coefficient
        becomes exactly 0 are kept out.

    Attributes
    ----------
    num_qubits : int
    terms : tuple of (str, float)
        In first-appearance order.

    Complexity
    ----------
    ``expectation``: time O(T · 2^q), extra space O(2^q). ``to_matrix``: O(T · 4^q),
    limited to 12 qubits. Arithmetic: O(T).

    References
    ----------
    Nielsen & Chuang, §10.5.1.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.quantum import PauliSum, bell_circuit
    >>> zz_xx = PauliSum({"ZZ": 1.0, "XX": 1.0})
    >>> round(zz_xx.expectation(bell_circuit().run()), 12)
    2.0
    >>> (2 * zz_xx - PauliSum([("ZZ", 2.0)])).terms
    (('XX', 2.0),)
    """

    def __init__(self, terms: Mapping[str, float] | Iterable[tuple[str, float]]):
        items = terms.items() if isinstance(terms, Mapping) else terms
        merged: dict[str, float] = {}
        width = None
        for item in items:
            try:
                label, coeff = item
            except (TypeError, ValueError):
                raise TypeError("terms must be (label, coefficient) pairs") from None
            label, coeff = _check_label(label), _check_coeff(coeff)
            if width is None:
                width = len(label)
            elif len(label) != width:
                raise ValueError(f"all labels must have length {width}, got {label!r}")
            merged[label] = merged.get(label, 0.0) + coeff
        if width is None:
            raise ValueError("a PauliSum needs at least one term")
        self._num_qubits = width
        self._terms = tuple((lab, c) for lab, c in merged.items() if c != 0.0)
        self._compiled = tuple((*_masks(lab), c) for lab, c in self._terms)

    # -- accessors ------------------------------------------------------------------------
    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def terms(self) -> tuple[tuple[str, float], ...]:
        return self._terms

    @property
    def num_terms(self) -> int:
        return len(self._terms)

    def __repr__(self) -> str:
        body = " + ".join(f"{c:.6g}*{lab}" for lab, c in self._terms[:8]) or "0"
        more = f" + ... ({len(self._terms) - 8} more)" if len(self._terms) > 8 else ""
        return f"PauliSum({body}{more})"

    def to_dict(self) -> dict[str, Any]:
        return {"num_qubits": self._num_qubits, "terms": [list(t) for t in self._terms]}

    # -- arithmetic -----------------------------------------------------------------------
    def _check_same(self, other: object) -> PauliSum:
        if not isinstance(other, PauliSum):
            return NotImplemented  # type: ignore[return-value]
        if other.num_qubits != self._num_qubits:
            raise ValueError(
                f"cannot combine {self._num_qubits}- and {other.num_qubits}-qubit sums"
            )
        return other

    def _from_terms(self, terms: list[tuple[str, float]]) -> PauliSum:
        out = PauliSum.__new__(PauliSum)
        merged: dict[str, float] = {}
        for lab, c in terms:
            merged[lab] = merged.get(lab, 0.0) + c
        out._num_qubits = self._num_qubits
        out._terms = tuple((lab, c) for lab, c in merged.items() if c != 0.0)
        out._compiled = tuple((*_masks(lab), c) for lab, c in out._terms)
        return out

    def __add__(self, other: PauliSum) -> PauliSum:
        other = self._check_same(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_terms(list(self._terms) + list(other.terms))

    def __sub__(self, other: PauliSum) -> PauliSum:
        other = self._check_same(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_terms(list(self._terms) + [(lab, -c) for lab, c in other.terms])

    def __mul__(self, scalar: float) -> PauliSum:
        if isinstance(scalar, bool) or not isinstance(scalar, numbers.Real):
            return NotImplemented
        s = _check_coeff(scalar)
        return self._from_terms([(lab, s * c) for lab, c in self._terms])

    __rmul__ = __mul__

    def __neg__(self) -> PauliSum:
        return self * -1.0

    # -- evaluation -----------------------------------------------------------------------
    def to_matrix(self) -> np.ndarray:
        """Dense Hermitian matrix (q <= 12): :math:`\\sum_t c_t P_t` with Kronecker products."""
        q = self._num_qubits
        if q > MAX_OBSERVABLE_MATRIX_QUBITS:
            raise ResourceLimitError(
                f"to_matrix needs 16·4^{q} bytes for {q} qubits; it is limited to "
                f"{MAX_OBSERVABLE_MATRIX_QUBITS}. Use expectation() instead."
            )
        out = np.zeros((2**q, 2**q), dtype=np.complex128)
        for label, c in self._terms:
            P = np.array([[1.0 + 0j]])
            for letter in label:
                P = np.kron(P, _PAULI_MATS[letter])
            out += c * P
        return out

    def expectation(self, state: np.ndarray | DensityMatrix) -> float:
        r""":math:`\langle\psi|H|\psi\rangle` or :math:`\mathrm{Tr}(H\rho)`, in O(T · 2^q).

        Parameters
        ----------
        state : statevector of length 2**q, or DensityMatrix
            A statevector is used as given (normalize it for a physical value).

        Returns
        -------
        float
        """
        q = self._num_qubits
        if isinstance(state, DensityMatrix):
            if state.num_qubits != q:
                raise ValueError(f"state has {state.num_qubits} qubits, the observable {q}")
            R = state.data
            j = np.arange(2**q, dtype=np.int64)
            total = 0.0
            for x, z, ny, c in self._compiled:
                sign = 1 - 2 * _parity(j & z)
                val = (1j**ny) * np.sum(sign * R[j, j ^ x])
                total += c * val.real
            return float(total)
        psi, nq = check_state(state)
        if nq != q:
            raise ValueError(f"state has {nq} qubits, the observable {q}")
        j = np.arange(2**q, dtype=np.int64)
        total = 0.0
        for x, z, ny, c in self._compiled:
            sign = 1 - 2 * _parity(j & z)
            val = (1j**ny) * np.vdot(psi[j ^ x], sign * psi)
            total += c * val.real
        return float(total)


@register(
    domain="quantum",
    tasks=("hamiltonian",),
    complexity=Complexity(time="O(q^2)", space="O(q^2)", notes="2q - 1 (or 2q) terms of length q"),
    references=(
        "Sachdev, Quantum Phase Transitions, 2nd ed., ch. 5",
        "Pfeuty, The one-dimensional Ising model with a transverse field, Ann. Phys. 57 (1970)",
    ),
)
def ising_chain(
    num_qubits: int, J: float = 1.0, h: float = 1.0, periodic: bool = False
) -> PauliSum:
    r"""Transverse-field Ising chain :math:`H = -J\sum_{i} Z_i Z_{i+1} - h\sum_i X_i`.

    Bonds :math:`(i, i+1)` for :math:`i = 0..q-2`, plus :math:`(q-1, 0)` when
    ``periodic`` (needs q >= 3). For :math:`h > 0` and an open chain the ground state is
    unique; at :math:`h = 0` and :math:`J > 0` it is doubly degenerate
    (:math:`|0\ldots0\rangle`, :math:`|1\ldots1\rangle`).

    Parameters
    ----------
    num_qubits : int
        Chain length q >= 2.
    J, h : float
        Coupling and transverse field.
    periodic : bool, default=False
        Close the chain into a ring.

    Returns
    -------
    PauliSum

    Complexity
    ----------
    O(q^2) to write the 2q-ish labels of length q.

    References
    ----------
    Pfeuty (1970); Sachdev, ch. 5.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.quantum import ising_chain
    >>> H = ising_chain(3, J=1.0, h=0.5)
    >>> H.num_terms, round(float(np.linalg.eigvalsh(H.to_matrix())[0]), 6)
    (5, -2.403212)
    """
    q = check_int(num_qubits, "num_qubits", minimum=2)
    J = check_angle(J, "J")
    h = check_angle(h, "h")
    if periodic and q < 3:
        raise ValueError("a periodic chain needs at least 3 qubits")
    bonds = [(i, i + 1) for i in range(q - 1)] + ([(q - 1, 0)] if periodic else [])

    def label(ops: dict[int, str]) -> str:
        return "".join(ops.get(q - 1 - pos, "I") for pos in range(q))

    terms = [(label({a: "Z", b: "Z"}), -J) for a, b in bonds]
    terms += [(label({i: "X"}), -h) for i in range(q)]
    return PauliSum(terms)


@register(
    domain="quantum",
    tasks=("hamiltonian", "chemistry"),
    complexity=Complexity(time="O(1)", space="O(1)"),
    references=(
        "Qiskit documentation, VQE tutorial: H2 operator at 0.735 Å (STO-3G, parity "
        "mapping, two-qubit reduction), generated with Qiskit Nature and PySCF",
        "Bravyi, Gambetta, Mezzacapo & Temme, Tapering off qubits to simulate fermionic "
        "Hamiltonians, arXiv:1701.08213",
        "Kandala et al., Hardware-efficient variational quantum eigensolver for small "
        "molecules and quantum magnets, Nature 549, 242 (2017)",
    ),
)
def h2_hamiltonian(include_nuclear_repulsion: bool = True) -> PauliSum:
    r"""Two-qubit Hamiltonian of H₂ at 0.735 Å (STO-3G, parity mapping, 2-qubit reduction).

    .. math::
        H = c_0 II + c_1 (IZ - ZI) + c_2 ZZ + c_3 XX \;(+\, E_\text{nuc}\, II)

    with the coefficients :data:`H2_COEFFICIENTS` (Hartree). They describe the
    **electronic** energy, whose exact ground value is −1.857275030 Ha. With
    ``include_nuclear_repulsion=True`` (default) the nuclear repulsion
    :math:`E_\text{nuc} = 1/R = 0.719968994` Ha is added, giving the total ground
    energy −1.137306036 Ha (the FCI/STO-3G value in this minimal basis).

    Parameters
    ----------
    include_nuclear_repulsion : bool, default=True
        Add :data:`H2_NUCLEAR_REPULSION` to the identity coefficient.

    Returns
    -------
    PauliSum
        On 2 qubits.

    Complexity
    ----------
    O(1).

    References
    ----------
    Qiskit VQE tutorial (coefficients); Bravyi et al. (2017) (two-qubit reduction);
    Kandala et al. (2017).

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.quantum import h2_hamiltonian
    >>> round(float(np.linalg.eigvalsh(h2_hamiltonian().to_matrix())[0]), 6)
    -1.137306
    >>> round(float(np.linalg.eigvalsh(h2_hamiltonian(False).to_matrix())[0]), 6)
    -1.857275
    """
    if not isinstance(include_nuclear_repulsion, (bool, np.bool_)):
        raise TypeError("include_nuclear_repulsion must be a bool")
    terms = list(H2_COEFFICIENTS)
    if include_nuclear_repulsion:
        terms.append(("II", H2_NUCLEAR_REPULSION))
    return PauliSum(terms)


@register(
    domain="quantum",
    tasks=("hamiltonian", "combinatorial-optimization"),
    complexity=Complexity(time="O(E q)", space="O(E q)", notes="E = number of edges"),
    references=(
        "Farhi, Goldstone & Gutmann, A Quantum Approximate Optimization Algorithm, arXiv:1411.4028",
    ),
)
def maxcut_hamiltonian(
    edges: Iterable[tuple[int, int]],
    num_nodes: int | None = None,
    weights: Iterable[float] | None = None,
) -> PauliSum:
    r"""Cut-value observable :math:`C = \sum_{(i,j)} w_{ij}\,\tfrac{1}{2}(I - Z_i Z_j)`.

    On a basis state :math:`|z\rangle` its value is the (weighted) number of edges
    whose endpoints differ, i.e. the cut of the partition given by the bits of z.

    Parameters
    ----------
    edges : iterable of (int, int)
        Undirected edges between distinct nodes ``0..num_nodes-1`` (no duplicates).
    num_nodes : int, optional
        Number of nodes q; default ``1 + max node``.
    weights : iterable of float, optional
        Edge weights (default 1).

    Returns
    -------
    PauliSum
        On q qubits (qubit i = node i).

    Complexity
    ----------
    O(E · q).

    References
    ----------
    Farhi, Goldstone & Gutmann (2014).

    Examples
    --------
    >>> from lipipy.quantum import maxcut_hamiltonian, basis_state
    >>> C = maxcut_hamiltonian([(0, 1), (1, 2), (0, 2)])
    >>> C.expectation(basis_state(3, 0b001))     # node 0 alone: cuts 2 edges
    2.0
    """
    edge_list = _check_edges(edges, num_nodes)
    q = num_nodes if num_nodes is not None else 1 + max(max(e) for e in edge_list)
    w = [1.0] * len(edge_list) if weights is None else [_check_coeff(x) for x in weights]
    if len(w) != len(edge_list):
        raise ValueError("weights must have one entry per edge")
    terms: list[tuple[str, float]] = []
    for (a, b), wt in zip(edge_list, w, strict=True):
        zz = "".join("Z" if pos in (a, b) else "I" for pos in reversed(range(q)))
        terms += [("I" * q, 0.5 * wt), (zz, -0.5 * wt)]
    return PauliSum(terms)


def _check_edges(edges: Iterable[tuple[int, int]], num_nodes: int | None) -> list[tuple[int, int]]:
    """Validated list of undirected edges (a, b) with a != b and no duplicates."""
    try:
        edge_list = [tuple(e) for e in edges]
    except TypeError:
        raise TypeError("edges must be an iterable of (int, int) pairs") from None
    if not edge_list:
        raise ValueError("edges must not be empty")
    out = []
    seen = set()
    for e in edge_list:
        if len(e) != 2:
            raise ValueError(f"each edge must be a pair, got {e!r}")
        a, b = (check_int(v, "node", minimum=0) for v in e)
        if a == b:
            raise ValueError(f"self-loop {e!r} is not allowed")
        key = (min(a, b), max(a, b))
        if key in seen:
            raise ValueError(f"duplicate edge {e!r}")
        seen.add(key)
        out.append((a, b))
    if num_nodes is not None:
        n = check_int(num_nodes, "num_nodes", minimum=2)
        if max(max(e) for e in out) >= n:
            raise ValueError(f"edge node out of range for num_nodes={n}")
    return out
