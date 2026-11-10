r"""Statevector simulation: gate application, measurement and state analysis.

Applying a k-qubit gate without a :math:`2^q \times 2^q` matrix
---------------------------------------------------------------
The full operator of a gate :math:`U` on qubits :math:`t_0, \ldots, t_{k-1}` is
:math:`I \otimes \cdots \otimes U \otimes \cdots \otimes I` (with the factors interleaved
according to the targets); building it costs :math:`O(4^q)` memory. Instead we view
:math:`\psi` as a tensor with one axis of length 2 per qubit,

.. math::  \psi_{b_{q-1} \ldots b_1 b_0}, \qquad \text{axis } a = q - 1 - j \text{ holds qubit } j

(NumPy's C order puts the most-significant bit on axis 0), view :math:`U` as a tensor
:math:`U_{o_{k-1}\ldots o_0,\; i_{k-1}\ldots i_0}`, and contract

.. math::
    \psi'_{\ldots o_{t} \ldots} = \sum_{i} U_{o, i}\, \psi_{\ldots i_{t} \ldots}

over the target axes only (``np.tensordot``), then move the new axes back into place.
Each of the :math:`2^q` output amplitudes is a sum of :math:`2^k` terms, so the time
is :math:`O(2^q \cdot 2^k)` and the extra memory is one new state, :math:`O(2^q)`.

Bitstrings and Pauli labels
---------------------------
Following Qiskit, a bitstring or Pauli label is written **most-significant qubit
first**: character ``i`` (from the left) of a length-``q`` string refers to qubit
``q - 1 - i``. ``"01"`` means qubit 1 = 0, qubit 0 = 1 (basis index 1), and
``expectation_pauli(psi, "XZ")`` measures X on qubit 1 and Z on qubit 0.

Entropy units
-------------
:func:`entanglement_entropy` returns bits by default (``base=2``): a Bell pair has
entropy 1. Pass ``base=np.e`` for nats (a Bell pair then has :math:`\ln 2`).

Reference: Nielsen & Chuang, *Quantum Computation and Quantum Information*, §2.2,
§2.4 (density operators, partial trace), §11.3 (entropy).
"""

from __future__ import annotations

import numbers
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from lipipy.core.complexity import Complexity
from lipipy.core.random import RandomState, check_random_state
from lipipy.core.registry import register
from lipipy.quantum import gates
from lipipy.quantum._validation import (
    check_int,
    check_qubits,
    check_square_matrix,
    check_state,
)

_NC = "Nielsen & Chuang, Quantum Computation and Quantum Information, 10th ed."
_NORM_ATOL = 1e-8
_PAULIS = {"X": gates.X, "Y": gates.Y, "Z": gates.Z}


# -- kernels (inputs already validated) ------------------------------------------------
def _apply_tensor(
    tensor: np.ndarray, gate: np.ndarray, targets: Sequence[int], num_qubits: int
) -> np.ndarray:
    """Contract ``gate`` into the qubit axes of ``tensor`` of shape ``(2,)*q + batch``.

    Returns a C-contiguous array of the same shape. Trailing (batch) axes are carried
    along, which lets :meth:`Circuit.to_matrix` push all basis states through at once.
    """
    k = len(targets)
    U = gate.reshape((2,) * (2 * k))
    # Gate axes 0..k-1 (outputs) and k..2k-1 (inputs) hold gate bits k-1..0, i.e.
    # targets t_{k-1}, ..., t_0; qubit j lives on state axis q - 1 - j.
    axes = [num_qubits - 1 - t for t in reversed(targets)]
    out = np.tensordot(U, tensor, axes=(list(range(k, 2 * k)), axes))
    return np.ascontiguousarray(np.moveaxis(out, list(range(k)), axes))


def _apply_diagonal_tensor(
    tensor: np.ndarray, diag: np.ndarray, targets: Sequence[int], num_qubits: int
) -> np.ndarray:
    """Multiply by a diagonal gate (given by its diagonal) by broadcasting: O(2^q)."""
    k = len(targets)
    axes = [num_qubits - 1 - t for t in reversed(targets)]  # axis of diag tensor dim i
    order = np.argsort(axes)
    D = diag.reshape((2,) * k).transpose(order)
    shape = [1] * tensor.ndim
    for a in axes:
        shape[a] = 2
    return tensor * D.reshape(shape)


def _measure_probs_one(psi: np.ndarray, qubit: int, num_qubits: int) -> np.ndarray:
    """Probabilities ``[p0, p1]`` of measuring ``qubit``."""
    p = (np.abs(psi) ** 2).reshape((2,) * num_qubits)
    other = tuple(a for a in range(num_qubits) if a != num_qubits - 1 - qubit)
    return p.sum(axis=other)


# -- public API ------------------------------------------------------------------------
@register(
    domain="quantum",
    tasks=("simulation",),
    complexity=Complexity(
        time="O(2^q 2^k)",
        space="O(2^q)",
        notes="k = number of target qubits; never forms a 2^q x 2^q matrix",
    ),
    references=(_NC + ", §4.2-4.3",),
)
def apply_gate(state: np.ndarray, gate: np.ndarray, targets: Sequence[int] | int) -> np.ndarray:
    """Apply a k-qubit gate to a statevector, returning the new state.

    Parameters
    ----------
    state : array_like, shape (2**q,)
        Input statevector (not modified).
    gate : array_like, shape (2**k, 2**k)
        Gate matrix in the ordering of :mod:`lipipy.quantum.gates`: ``targets[0]`` is
        the gate's least-significant qubit. Any square matrix is accepted (unitarity
        is not checked here, so projectors and Kraus operators work too).
    targets : int or sequence of int
        ``k`` distinct qubit indices.

    Returns
    -------
    ndarray of complex128, shape (2**q,)

    Complexity
    ----------
    Time O(2^q · 2^k), extra space O(2^q) (one output-sized temporary).

    References
    ----------
    Nielsen & Chuang, §4.2–4.3.

    Examples
    --------
    >>> from lipipy.quantum import apply_gate, zero_state, H, CX
    >>> psi = apply_gate(zero_state(2), H, [0])
    >>> psi = apply_gate(psi, CX, [0, 1])   # control 0, target 1
    >>> psi.real.round(6)
    array([0.707107, 0.      , 0.      , 0.707107])
    """
    psi, q = check_state(state)
    U, k = check_square_matrix(gate)
    tgts = check_qubits(targets, q)
    if len(tgts) != k:
        raise ValueError(f"a {k}-qubit gate needs {k} targets, got {len(tgts)}: {tgts}")
    return _apply_tensor(psi.reshape((2,) * q), U, tgts, q).reshape(-1)


@register(
    domain="quantum",
    tasks=("simulation",),
    complexity=Complexity(time="O(2^q)", space="O(2^q)", notes="k-qubit diagonal of size 2^k"),
    references=(_NC + ", §4.2",),
)
def apply_diagonal(
    state: np.ndarray, diagonal: np.ndarray, targets: Sequence[int] | int
) -> np.ndarray:
    """Apply a diagonal gate given only by its diagonal entries.

    Equivalent to ``apply_gate(state, np.diag(diagonal), targets)`` but costs O(2^q)
    regardless of ``k``. Used for phase oracles (Grover, Deutsch-Jozsa).

    Parameters
    ----------
    state : array_like, shape (2**q,)
        Input statevector (not modified).
    diagonal : array_like, shape (2**k,)
        Diagonal of the gate, indexed with ``targets[0]`` as least-significant bit.
    targets : int or sequence of int
        ``k`` distinct qubit indices.

    Returns
    -------
    ndarray of complex128, shape (2**q,)

    Complexity
    ----------
    Time O(2^q), extra space O(2^q).

    References
    ----------
    Nielsen & Chuang, §4.2.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.quantum import apply_diagonal
    >>> apply_diagonal(np.ones(4) / 2, [1, -1], [1]).real
    array([ 0.5,  0.5, -0.5, -0.5])
    """
    psi, q = check_state(state)
    d, k = check_state(diagonal, "diagonal")
    tgts = check_qubits(targets, q)
    if len(tgts) != k:
        raise ValueError(f"a {k}-qubit diagonal needs {k} targets, got {len(tgts)}")
    return _apply_diagonal_tensor(psi.reshape((2,) * q), d, tgts, q).reshape(-1)


@register(
    domain="quantum",
    tasks=("measurement",),
    complexity=Complexity(time="O(2^q)", space="O(1)", notes="output is 8·2^q bytes"),
    references=(_NC + ", §2.2.3 (Born rule)",),
)
def probabilities(state: np.ndarray) -> np.ndarray:
    r"""Born-rule probabilities :math:`p_i = |\psi_i|^2` of every basis state.

    Parameters
    ----------
    state : array_like, shape (2**q,)
        Statevector.

    Returns
    -------
    ndarray of float64, shape (2**q,)

    Complexity
    ----------
    Time O(2^q); no memory beyond the output.

    References
    ----------
    Nielsen & Chuang, §2.2.3.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.quantum import probabilities
    >>> probabilities(np.array([1, 1j]) / np.sqrt(2)).round(12)
    array([0.5, 0.5])
    """
    psi, _ = check_state(state)
    return np.abs(psi) ** 2


@register(
    domain="quantum",
    tasks=("measurement", "sampling"),
    complexity=Complexity(
        time="O(2^q + q min(P, 2^q))",
        space="O(2^q)",
        notes="P = shots; one multinomial draw over the 2^q outcomes",
    ),
    references=(_NC + ", §2.2.3",),
)
def sample_counts(
    state: np.ndarray, shots: int = 1024, random_state: RandomState = None
) -> dict[str, int]:
    """Measure every qubit ``shots`` times and count the outcomes.

    The counts are one multinomial draw with the Born probabilities, which is
    exactly the distribution of ``shots`` independent measurements.

    Parameters
    ----------
    state : array_like, shape (2**q,)
        Normalized statevector.
    shots : int, default=1024
        Number of measurements (>= 1).
    random_state : None, int, SeedSequence or Generator
        Seed, see :func:`lipipy.core.check_random_state`.

    Returns
    -------
    dict of str to int
        Bitstrings (most-significant qubit first, like Qiskit) to counts, only for
        observed outcomes, sorted by bitstring. Counts sum to ``shots``.

    Complexity
    ----------
    Time O(2^q + q·min(shots, 2^q)), extra space O(2^q).

    References
    ----------
    Nielsen & Chuang, §2.2.3.

    Examples
    --------
    >>> from lipipy.quantum import sample_counts, basis_state
    >>> sample_counts(basis_state(3, 1), shots=10, random_state=0)
    {'001': 10}
    """
    psi, q = check_state(state)
    shots = check_int(shots, "shots", minimum=1)
    p = np.abs(psi) ** 2
    total = p.sum()
    if abs(total - 1.0) > _NORM_ATOL:
        raise ValueError(f"state must be normalized (sum of probabilities = {total})")
    rng = check_random_state(random_state)
    counts = rng.multinomial(shots, p / total)
    return {format(int(i), f"0{q}b"): int(counts[i]) for i in np.flatnonzero(counts)}


@register(
    domain="quantum",
    tasks=("measurement", "expectation"),
    complexity=Complexity(time="O(q 2^q)", space="O(2^q)"),
    references=(_NC + ", §2.2.5 (projective measurement), §10.5.1 (Pauli group)",),
)
def expectation_pauli(state: np.ndarray, pauli: str) -> float:
    r"""Expectation value :math:`\langle\psi| P |\psi\rangle` of a Pauli string.

    Parameters
    ----------
    state : array_like, shape (2**q,)
        Statevector (normalized for a physical expectation value).
    pauli : str
        Length-``q`` string over ``"IXYZ"``, written like Qiskit's ``Pauli`` labels:
        the **rightmost** character acts on qubit 0 and character ``i`` from the
        left on qubit ``q - 1 - i``. ``"XZI"`` is X on qubit 2, Z on qubit 1.

    Returns
    -------
    float
        Real expectation value in [-1, 1] for a normalized state.

    Complexity
    ----------
    Time O(q·2^q) (one single-qubit gate per non-identity letter), extra space O(2^q).

    References
    ----------
    Nielsen & Chuang, §2.2.5.

    Examples
    --------
    >>> from lipipy.quantum import expectation_pauli, basis_state
    >>> psi = basis_state(2, 1)            # qubit 0 = 1, qubit 1 = 0
    >>> expectation_pauli(psi, "IZ"), expectation_pauli(psi, "ZI")
    (-1.0, 1.0)
    """
    psi, q = check_state(state)
    if not isinstance(pauli, str):
        raise TypeError(f"pauli must be a string, got {type(pauli).__name__}")
    if len(pauli) != q:
        raise ValueError(f"pauli string must have length {q}, got {len(pauli)}")
    if set(pauli) - set("IXYZ"):
        raise ValueError(f"pauli string may only contain I, X, Y, Z; got {pauli!r}")
    tensor = psi.reshape((2,) * q)
    for i, letter in enumerate(pauli):
        if letter != "I":
            tensor = _apply_tensor(tensor, _PAULIS[letter], (q - 1 - i,), q)
    return float(np.vdot(psi, tensor.reshape(-1)).real)


@dataclass(frozen=True)
class MeasurementResult:
    """Outcome of measuring one qubit (see :func:`measure_qubit`).

    Attributes
    ----------
    outcome : int
        0 or 1.
    probability : float
        Probability of that outcome before the measurement.
    state : ndarray of complex128
        Normalized post-measurement (collapsed) state.
    """

    outcome: int
    probability: float
    state: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict; the state becomes ``[[re, im], ...]`` pairs."""
        d = asdict(self)
        d["state"] = np.column_stack([self.state.real, self.state.imag]).tolist()
        return d


@register(
    domain="quantum",
    tasks=("measurement", "sampling"),
    complexity=Complexity(time="O(2^q)", space="O(2^q)"),
    references=(_NC + ", §2.2.3 (measurement postulate)",),
)
def measure_qubit(
    state: np.ndarray, qubit: int, random_state: RandomState = None
) -> MeasurementResult:
    r"""Projectively measure one qubit in the Z basis and collapse the state.

    With :math:`P_m = |m\rangle\langle m|` on ``qubit``, outcome :math:`m` occurs
    with probability :math:`p_m = \lVert P_m\psi\rVert^2` and the state becomes
    :math:`P_m\psi / \sqrt{p_m}`.

    Parameters
    ----------
    state : array_like, shape (2**q,)
        Normalized statevector (not modified).
    qubit : int
        Qubit to measure.
    random_state : None, int, SeedSequence or Generator
        Seed, see :func:`lipipy.core.check_random_state`.

    Returns
    -------
    MeasurementResult
        ``outcome``, its ``probability`` and the collapsed ``state``.

    Complexity
    ----------
    Time O(2^q), extra space O(2^q).

    References
    ----------
    Nielsen & Chuang, §2.2.3.

    Examples
    --------
    >>> from lipipy.quantum import measure_qubit, basis_state
    >>> r = measure_qubit(basis_state(2, 2), qubit=1, random_state=0)
    >>> r.outcome, r.probability
    (1, 1.0)
    """
    psi, q = check_state(state)
    (qubit,) = check_qubits(qubit, q, "qubit")
    probs = _measure_probs_one(psi, qubit, q)
    total = probs.sum()
    if abs(total - 1.0) > _NORM_ATOL:
        raise ValueError(f"state must be normalized (norm^2 = {total})")
    rng = check_random_state(random_state)
    outcome = int(rng.random() < probs[1] / total)
    p = float(probs[outcome] / total)
    bit = (np.arange(psi.shape[0]) >> qubit) & 1
    collapsed = np.where(bit == outcome, psi, 0.0)
    collapsed /= np.sqrt(probs[outcome])
    return MeasurementResult(outcome=outcome, probability=p, state=collapsed)


@register(
    domain="quantum",
    tasks=("state-analysis", "partial-trace"),
    complexity=Complexity(
        time="O(2^q 2^k)", space="O(2^q + 4^k)", notes="k = number of kept qubits"
    ),
    references=(_NC + ", §2.4.3 (reduced density operator)",),
)
def reduced_density_matrix(state: np.ndarray, keep: Sequence[int] | int) -> np.ndarray:
    r"""Reduced density matrix :math:`\rho_A = \mathrm{Tr}_B |\psi\rangle\langle\psi|`.

    Reshape :math:`\psi` into a :math:`2^{|A|} \times 2^{|B|}` matrix :math:`M` (rows
    indexed by the kept qubits) so that :math:`\rho_A = M M^\dagger`.

    Parameters
    ----------
    state : array_like, shape (2**q,)
        Statevector.
    keep : int or sequence of int
        Qubits of subsystem A, in the order of the output's index: ``keep[0]`` is its
        least-significant qubit (same convention as gate targets).

    Returns
    -------
    ndarray of complex128, shape (2**k, 2**k)
        Hermitian, positive semidefinite, trace :math:`\lVert\psi\rVert^2`.

    Complexity
    ----------
    Time O(2^q · 2^k), extra space O(2^q + 4^k).

    References
    ----------
    Nielsen & Chuang, §2.4.3.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.quantum import reduced_density_matrix
    >>> bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
    >>> reduced_density_matrix(bell, [0]).real.round(12)
    array([[0.5, 0. ],
           [0. , 0.5]])
    """
    psi, q = check_state(state)
    kept = check_qubits(keep, q, "keep")
    keep_axes = [q - 1 - t for t in reversed(kept)]
    rest = [a for a in range(q) if a not in keep_axes]
    M = psi.reshape((2,) * q).transpose(keep_axes + rest).reshape(2 ** len(kept), -1)
    return M @ M.conj().T


@register(
    domain="quantum",
    tasks=("state-analysis", "entanglement"),
    complexity=Complexity(
        time="O(2^q 2^min(k, q-k))",
        space="O(2^q)",
        notes="k = subsystem size; SVD of a 2^k x 2^(q-k) matrix",
    ),
    references=(_NC + ", §11.3 (von Neumann entropy), §2.5 (Schmidt decomposition)",),
)
def entanglement_entropy(
    state: np.ndarray, subsystem: Sequence[int] | int, base: float = 2.0
) -> float:
    r"""Von Neumann entropy :math:`S(\rho_A) = -\sum_i \lambda_i \log \lambda_i` of a subsystem.

    For a pure bipartite state the :math:`\lambda_i` are the squared Schmidt
    coefficients, i.e. the squared singular values of :math:`\psi` reshaped to
    :math:`2^{|A|} \times 2^{|B|}`; :math:`S(\rho_A) = S(\rho_B)`.

    Parameters
    ----------
    state : array_like, shape (2**q,)
        Normalized statevector.
    subsystem : int or sequence of int
        Qubits of subsystem A (order irrelevant).
    base : float, default=2
        Logarithm base. ``2`` gives bits (a Bell pair has entropy 1); ``np.e`` gives
        nats (a Bell pair has :math:`\ln 2`).

    Returns
    -------
    float
        Entropy in ``[0, min(|A|, |B|)]`` bits; 0 for product states.

    Complexity
    ----------
    SVD of a 2^k × 2^(q−k) matrix: time O(2^q · 2^min(k, q−k)), space O(2^q).

    References
    ----------
    Nielsen & Chuang, §2.5 and §11.3.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.quantum import entanglement_entropy
    >>> bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
    >>> round(entanglement_entropy(bell, [0]), 12)
    1.0
    """
    psi, q = check_state(state)
    sub = check_qubits(subsystem, q, "subsystem")
    if not isinstance(base, numbers.Real) or isinstance(base, bool) or not base > 1:
        raise ValueError(f"base must be a real number > 1, got {base!r}")
    norm2 = float(np.vdot(psi, psi).real)
    if abs(norm2 - 1.0) > _NORM_ATOL:
        raise ValueError(f"state must be normalized (norm^2 = {norm2})")
    if len(sub) == q:
        return 0.0
    axes = [q - 1 - t for t in sub]
    rest = [a for a in range(q) if a not in axes]
    M = psi.reshape((2,) * q).transpose(axes + rest).reshape(2 ** len(sub), -1)
    lam = np.linalg.svd(M, compute_uv=False) ** 2
    lam = lam[lam > 1e-300]
    return float(max(-np.sum(lam * np.log(lam)) / np.log(base), 0.0))
