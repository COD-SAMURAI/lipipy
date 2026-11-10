"""Layer 2: quantum computing, implemented from scratch on NumPy.

Simulation cost is explicit. A dense statevector for q qubits needs 16·2^q bytes
(complex128): 26 qubits ≈ 1 GiB, 30 qubits ≈ 16 GiB. Simulators refuse requests
above ``max_qubits`` with ``ResourceLimitError`` before allocating, and point to the
matrix-product-state simulator (memory O(q χ^2)). Parity tests against Qiskit /
PennyLane run when ``lipipy[quantum]`` is installed.

Conventions (used consistently by every module)
-----------------------------------------------
* **Qubit order.** Qubit 0 is the least-significant bit of the basis index:
  basis state ``|i>`` has qubit ``j`` equal to bit ``j`` of ``i`` (little-endian,
  as in Qiskit).
* **States** are complex128 1-D arrays of length ``2**q``.
* **Gate matrices** acting on ``targets=(t0, t1, ...)`` are written in the gate's own
  index with ``t0`` as its least-significant qubit. ``CX`` on ``(control, target)``
  and ``Circuit.cx(control, target)`` therefore match Qiskit's ``cx``, and
  ``controlled(U, m)`` puts its ``m`` controls first.
* **Bitstrings and Pauli labels** are written most-significant qubit first (like
  Qiskit): the rightmost character is qubit 0.
* **QFT sign:** ``|j> -> 2^{-q/2} sum_k exp(+2 pi i j k / 2^q) |k>`` (Qiskit's), i.e.
  ``np.fft.ifft(np.eye(2**q), axis=0, norm="ortho")``.
* **Entropy** is in bits by default (``base=2``); a Bell pair has entropy 1.

Example
-------
>>> from lipipy.quantum import Circuit, sample_counts
>>> psi = Circuit(2).h(0).cx(0, 1).run()
>>> counts = sample_counts(psi, shots=1000, random_state=0)
>>> sorted(counts), sum(counts.values())
(['00', '11'], 1000)

Catalog. ``[x]`` = implemented, ``[ ]`` = planned.

Foundations
-----------
[x] Qubit states (zero, basis, Haar-random), normalization check, Bloch-sphere coordinates
[ ] Dirac notation helpers (ket/bra parsing and pretty-printing)
[x] Gates: Pauli X/Y/Z, H, S, T, SX, rotations RX/RY/RZ, phase, U3, CNOT, CZ, SWAP,
    Toffoli, controlled(U), unitarity check
[x] Tensor products, partial trace, fidelity, entanglement entropy

Circuits and simulators
-----------------------
[x] ``Circuit`` builder (fluent API): run, to_matrix, inverse, compose, depth, count_ops
[ ] Symbolic parameters for variational circuits
[x] Statevector simulator (apply k-qubit gate in O(2^q) time without building 2^q x 2^q matrices)
[x] Measurement, sampling shots, expectation values of Pauli strings
[ ] Density-matrix simulator with noise channels (depolarizing, amplitude/phase damping)
[ ] Matrix-product-state (tensor network) simulator
[ ] Clifford / stabilizer simulator (Gottesman-Knill, polynomial time)

Algorithms
----------
[x] Bell and GHZ state preparation
[x] Deutsch-Jozsa, Bernstein-Vazirani
[ ] Simon
[x] Quantum Fourier transform, phase estimation
[x] Grover search
[ ] Amplitude amplification (general) and amplitude estimation
[ ] Shor's algorithm (small-number demonstration)
[ ] Quantum teleportation, superdense coding, BB84 key distribution

Variational and quantum machine learning
----------------------------------------
[ ] Parameter-shift gradients
[ ] VQE (variational quantum eigensolver) with Pauli Hamiltonians
[ ] QAOA for MaxCut and other QUBO problems
[ ] Quantum kernels and variational classifiers

Error correction
----------------
[ ] Bit-flip and phase-flip codes, Shor 9-qubit code, Steane code
[ ] Surface code basics (syndrome extraction, decoding toy examples)
"""

from lipipy.quantum._validation import MAX_QUBITS
from lipipy.quantum.algorithms import (
    GroverResult,
    bell_circuit,
    bernstein_vazirani,
    deutsch_jozsa,
    ghz_circuit,
    grover,
    inverse_qft,
    phase_estimation,
    qft,
)
from lipipy.quantum.circuit import MAX_MATRIX_QUBITS, Circuit, Instruction
from lipipy.quantum.gates import (
    CCX,
    CX,
    CZ,
    SDG,
    SWAP,
    SX,
    TDG,
    H,
    I,
    S,
    T,
    X,
    Y,
    Z,
    controlled,
    is_unitary,
    phase,
    rx,
    ry,
    rz,
    u3,
)
from lipipy.quantum.states import (
    basis_state,
    bloch_vector,
    fidelity,
    is_normalized,
    random_state,
    tensor,
    zero_state,
)
from lipipy.quantum.statevector import (
    MeasurementResult,
    apply_diagonal,
    apply_gate,
    entanglement_entropy,
    expectation_pauli,
    measure_qubit,
    probabilities,
    reduced_density_matrix,
    sample_counts,
)

__all__ = [
    "CCX",
    "CX",
    "CZ",
    "MAX_MATRIX_QUBITS",
    "MAX_QUBITS",
    "SDG",
    "SWAP",
    "SX",
    "TDG",
    "Circuit",
    "GroverResult",
    "H",
    "I",
    "Instruction",
    "MeasurementResult",
    "S",
    "T",
    "X",
    "Y",
    "Z",
    "apply_diagonal",
    "apply_gate",
    "basis_state",
    "bell_circuit",
    "bernstein_vazirani",
    "bloch_vector",
    "controlled",
    "deutsch_jozsa",
    "entanglement_entropy",
    "expectation_pauli",
    "fidelity",
    "ghz_circuit",
    "grover",
    "inverse_qft",
    "is_normalized",
    "is_unitary",
    "measure_qubit",
    "phase",
    "phase_estimation",
    "probabilities",
    "qft",
    "random_state",
    "reduced_density_matrix",
    "rx",
    "ry",
    "rz",
    "sample_counts",
    "tensor",
    "u3",
    "zero_state",
]
