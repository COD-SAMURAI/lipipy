r"""Textbook quantum algorithms built on :class:`~lipipy.quantum.Circuit`.

Quantum Fourier transform
-------------------------
With :math:`N = 2^q` and the little-endian basis index :math:`j` (qubit 0 = LSB),

.. math::  \mathrm{QFT}\,|j\rangle = \frac{1}{\sqrt N} \sum_{k=0}^{N-1} e^{+2\pi i jk/N} |k\rangle,

the same sign as Qiskit's ``QFT``. Its matrix :math:`F_{kj} = e^{2\pi i jk/N}/\sqrt N`
is NumPy's *inverse* DFT with orthonormal scaling,
``np.fft.ifft(np.eye(N), axis=0, norm="ortho")``. The circuit uses :math:`q(q+1)/2`
gates (H and controlled phases) plus :math:`\lfloor q/2 \rfloor` swaps that reverse the
qubit order (omit them with ``do_swaps=False``).

Grover search
-------------
For :math:`M` marked items among :math:`N = 2^q`, write
:math:`\sin\theta = \sqrt{M/N}`. Each Grover iteration (oracle, then diffuser
:math:`2|s\rangle\langle s| - I`) rotates the state by :math:`2\theta`, so after
:math:`k` iterations the success probability is :math:`\sin^2((2k+1)\theta)`; the
default :math:`k = \lfloor \frac\pi4\sqrt{N/M} \rfloor` makes it close to 1.

Phase oracles
-------------
Deutsch-Jozsa, Bernstein-Vazirani and Grover use the phase form
:math:`|x\rangle \mapsto (-1)^{f(x)}|x\rangle` of the bit oracle
:math:`|x\rangle|y\rangle \mapsto |x\rangle|y \oplus f(x)\rangle` (phase kickback with
the ancilla in :math:`|-\rangle`), applied as a diagonal gate in :math:`O(2^q)`.

References: Nielsen & Chuang, *Quantum Computation and Quantum Information*,
§1.4.4 (Deutsch-Jozsa), §5.1–5.2 (QFT, phase estimation), §6.1 (Grover);
Bernstein & Vazirani, SIAM J. Comput. 26 (1997); Boyer, Brassard, Høyer & Tapp,
Fortschr. Phys. 46 (1998) (Grover with M solutions).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from lipipy.core.complexity import Complexity
from lipipy.core.registry import register
from lipipy.quantum import gates
from lipipy.quantum._validation import (
    MAX_QUBITS,
    check_int,
    check_num_qubits,
    check_square_matrix,
    check_state,
)
from lipipy.quantum.circuit import Circuit

_NC = "Nielsen & Chuang, Quantum Computation and Quantum Information, 10th ed."


# -- QFT -------------------------------------------------------------------------------
@register(
    domain="quantum",
    tasks=("circuit", "fourier-transform"),
    complexity=Complexity(
        time="O(q^2)",
        space="O(q^2)",
        notes="builds q(q+1)/2 + floor(q/2) gates; running them costs O(q^2 2^q)",
    ),
    references=(_NC + ", §5.1",),
)
def qft(num_qubits: int, do_swaps: bool = True) -> Circuit:
    r"""Circuit for the quantum Fourier transform on ``num_qubits`` qubits.

    Implements :math:`|j\rangle \mapsto N^{-1/2}\sum_k e^{2\pi i jk/N}|k\rangle` (see
    the module docstring for the sign convention). For qubit :math:`j` from the top
    down: a Hadamard, then a controlled phase :math:`P(\pi/2^{j-m})` from every lower
    qubit :math:`m`; finally swap qubit :math:`i` with :math:`q-1-i`.

    Parameters
    ----------
    num_qubits : int
        Number of qubits ``q >= 1``.
    do_swaps : bool, default=True
        Include the final qubit-reversal swaps. Without them the output is the QFT
        with its qubits in reversed order.

    Returns
    -------
    Circuit

    Complexity
    ----------
    Building: O(q^2) gates. Simulating: O(q^2 · 2^q) time, O(2^q) memory.

    References
    ----------
    Nielsen & Chuang, §5.1, Fig. 5.1.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.quantum import qft
    >>> F = np.fft.ifft(np.eye(8), axis=0, norm="ortho")
    >>> np.allclose(qft(3).to_matrix(), F)
    True
    """
    q = check_int(num_qubits, "num_qubits", minimum=1)
    circ = Circuit(q)
    for j in reversed(range(q)):
        circ.h(j)
        for m in reversed(range(j)):
            circ.cp(math.pi / 2 ** (j - m), m, j)
    if do_swaps:
        for i in range(q // 2):
            circ.swap(i, q - 1 - i)
    return circ


@register(
    domain="quantum",
    tasks=("circuit", "fourier-transform"),
    complexity=Complexity(time="O(q^2)", space="O(q^2)", notes="see qft"),
    references=(_NC + ", §5.1",),
)
def inverse_qft(num_qubits: int, do_swaps: bool = True) -> Circuit:
    """Circuit for the inverse QFT, ``qft(num_qubits, do_swaps).inverse()``.

    Parameters
    ----------
    num_qubits : int
        Number of qubits ``q >= 1``.
    do_swaps : bool, default=True
        Whether the forward QFT includes the qubit-reversal swaps.

    Returns
    -------
    Circuit

    Complexity
    ----------
    Same as :func:`qft`.

    References
    ----------
    Nielsen & Chuang, §5.1.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.quantum import qft, inverse_qft
    >>> np.allclose(qft(3).compose(inverse_qft(3)).to_matrix(), np.eye(8))
    True
    """
    return qft(num_qubits, do_swaps).inverse()


# -- entangled states ------------------------------------------------------------------
@register(
    domain="quantum",
    tasks=("circuit", "state-preparation", "entanglement"),
    complexity=Complexity(time="O(1)", space="O(1)"),
    references=(_NC + ", §1.3.6",),
)
def bell_circuit() -> Circuit:
    r"""Two-qubit circuit preparing :math:`|\Phi^+\rangle = (|00\rangle + |11\rangle)/\sqrt2`.

    Returns
    -------
    Circuit
        ``Circuit(2).h(0).cx(0, 1)``.

    Complexity
    ----------
    O(1).

    References
    ----------
    Nielsen & Chuang, §1.3.6.

    Examples
    --------
    >>> from lipipy.quantum import bell_circuit, probabilities
    >>> probabilities(bell_circuit().run()).round(12)
    array([0.5, 0. , 0. , 0.5])
    """
    return Circuit(2).h(0).cx(0, 1)


@register(
    domain="quantum",
    tasks=("circuit", "state-preparation", "entanglement"),
    complexity=Complexity(time="O(q)", space="O(q)"),
    references=(
        "Greenberger, Horne & Zeilinger, Going beyond Bell's theorem (1989), arXiv:0712.0921",
    ),
)
def ghz_circuit(num_qubits: int) -> Circuit:
    r"""Circuit preparing :math:`(|0\ldots0\rangle + |1\ldots1\rangle)/\sqrt2`.

    A Hadamard on qubit 0 followed by a CNOT chain ``cx(j-1, j)``.

    Parameters
    ----------
    num_qubits : int
        Number of qubits ``q >= 1`` (``q = 1`` gives :math:`|+\rangle`).

    Returns
    -------
    Circuit
        ``q`` gates, depth ``q``.

    Complexity
    ----------
    O(q) to build; O(q · 2^q) to run.

    References
    ----------
    Greenberger, Horne & Zeilinger (1989).

    Examples
    --------
    >>> from lipipy.quantum import ghz_circuit, sample_counts
    >>> sorted(sample_counts(ghz_circuit(3).run(), shots=100, random_state=1))
    ['000', '111']
    """
    q = check_int(num_qubits, "num_qubits", minimum=1)
    circ = Circuit(q).h(0)
    for j in range(1, q):
        circ.cx(j - 1, j)
    return circ


# -- Grover ----------------------------------------------------------------------------
@dataclass(frozen=True)
class GroverResult:
    """Output of :func:`grover`.

    Attributes
    ----------
    circuit : Circuit
        The full circuit (Hadamards, then ``iterations`` oracle + diffuser rounds).
    state : ndarray of complex128
        Final statevector.
    iterations : int
        Grover iterations applied.
    marked : tuple of int
        Marked basis indices.
    success_probability : float
        Probability that measuring all qubits gives a marked item.
    most_likely : int
        Basis index with the largest probability.
    """

    circuit: Circuit
    state: np.ndarray
    iterations: int
    marked: tuple[int, ...]
    success_probability: float
    most_likely: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly summary (the circuit and state are omitted)."""
        return {
            "num_qubits": self.circuit.num_qubits,
            "iterations": self.iterations,
            "marked": list(self.marked),
            "success_probability": self.success_probability,
            "most_likely": self.most_likely,
        }


@register(
    domain="quantum",
    tasks=("search", "amplitude-amplification"),
    complexity=Complexity(
        time="O(T 2^q)",
        space="O(2^q)",
        notes="T = iterations ≈ (π/4)·sqrt(2^q / M); oracle and diffuser are O(2^q) each",
    ),
    references=(
        "Grover, A fast quantum mechanical algorithm for database search, STOC 1996, "
        "arXiv:quant-ph/9605043",
        "Boyer, Brassard, Høyer & Tapp, Tight bounds on quantum searching, "
        "Fortschr. Phys. 46 (1998), arXiv:quant-ph/9605034",
        _NC + ", §6.1",
    ),
)
def grover(
    num_qubits: int,
    marked: int | Iterable[int],
    iterations: int | None = None,
    *,
    max_qubits: int = MAX_QUBITS,
) -> GroverResult:
    r"""Grover search for the marked basis states among :math:`N = 2^q`.

    Circuit: :math:`H^{\otimes q}`, then ``iterations`` times the oracle
    :math:`O = I - 2\sum_{m}|m\rangle\langle m|` followed by the diffuser
    :math:`D = H^{\otimes q}(2|0\rangle\langle 0| - I)H^{\otimes q} = 2|s\rangle\langle s| - I`.
    Both diagonal parts are applied with :meth:`Circuit.diagonal`.

    Parameters
    ----------
    num_qubits : int
        Search-space size :math:`N = 2^q`.
    marked : int or iterable of int
        Marked basis indices (distinct, in ``[0, N)``; at least one).
    iterations : int, optional
        Number of Grover iterations; default
        :math:`\lfloor \frac\pi4\sqrt{N/M} \rfloor`.
    max_qubits : int, default=26
        Resource guard.

    Returns
    -------
    GroverResult
        Its ``success_probability`` equals :math:`\sin^2((2k+1)\theta)` with
        :math:`\sin\theta = \sqrt{M/N}`.

    Complexity
    ----------
    Time O(T · 2^q) with T ≈ (π/4)·sqrt(N/M) iterations, extra space O(2^q).

    References
    ----------
    Grover (1996); Boyer et al. (1998); Nielsen & Chuang, §6.1.

    Examples
    --------
    >>> from lipipy.quantum import grover
    >>> r = grover(4, marked=11)
    >>> r.iterations, r.most_likely, round(r.success_probability, 4)
    (3, 11, 0.9613)
    """
    q = check_num_qubits(num_qubits, max_qubits)
    N = 2**q
    marks = (marked,) if isinstance(marked, (int, np.integer)) else tuple(marked)
    marks = tuple(check_int(m, "marked") for m in marks)
    if not marks:
        raise ValueError("marked must contain at least one index")
    if len(set(marks)) != len(marks):
        raise ValueError(f"marked contains duplicates: {marks}")
    for m in marks:
        if not 0 <= m < N:
            raise ValueError(f"marked index {m} out of range for {q} qubits")
    M = len(marks)
    if iterations is None:
        k = math.floor(math.pi / 4 * math.sqrt(N / M))
    else:
        k = check_int(iterations, "iterations", minimum=0)

    all_qubits = list(range(q))
    oracle = np.ones(N, dtype=np.complex128)
    oracle[list(marks)] = -1.0
    reflect_zero = -np.ones(N, dtype=np.complex128)  # 2|0><0| - I
    reflect_zero[0] = 1.0

    circ = Circuit(q)
    for j in all_qubits:
        circ.h(j)
    for _ in range(k):
        circ.diagonal(oracle, all_qubits, name="oracle")
        for j in all_qubits:
            circ.h(j)
        circ.diagonal(reflect_zero, all_qubits, name="reflect_zero")
        for j in all_qubits:
            circ.h(j)

    state = circ.run(max_qubits=max_qubits)
    probs = np.abs(state) ** 2
    return GroverResult(
        circuit=circ,
        state=state,
        iterations=k,
        marked=marks,
        success_probability=float(probs[list(marks)].sum()),
        most_likely=int(np.argmax(probs)),
    )


# -- oracle algorithms -----------------------------------------------------------------
def _hadamard_sandwich(q: int, oracle: Circuit) -> np.ndarray:
    """Run H^q, oracle, H^q from |0...0> and return the final state."""
    circ = Circuit(q)
    for j in range(q):
        circ.h(j)
    circ = circ.compose(oracle)
    for j in range(q):
        circ.h(j)
    return circ.run()


@register(
    domain="quantum",
    tasks=("oracle", "decision"),
    complexity=Complexity(
        time="O(q 2^q)",
        space="O(2^q)",
        notes="2^q classical calls to build the phase oracle; 1 quantum query",
    ),
    references=(
        "Deutsch & Jozsa, Rapid solution of problems by quantum computation, "
        "Proc. R. Soc. Lond. A 439 (1992)",
        _NC + ", §1.4.4",
    ),
)
def deutsch_jozsa(oracle_fn: Callable[[int], int], n: int, *, max_qubits: int = MAX_QUBITS) -> str:
    r"""Decide whether :math:`f: \{0,1\}^n \to \{0,1\}` is constant or balanced.

    After :math:`H^{\otimes n}\, O_f\, H^{\otimes n}|0\rangle` the amplitude of
    :math:`|0\ldots0\rangle` is :math:`2^{-n}\sum_x (-1)^{f(x)}`: magnitude 1 if
    ``f`` is constant, 0 if balanced. One oracle query decides with certainty.

    Parameters
    ----------
    oracle_fn : callable ``int -> {0, 1}``
        The function ``f`` on inputs ``0 .. 2**n - 1`` (input bit ``j`` = qubit ``j``).
        It is evaluated classically on every input to build the phase oracle, and
        must be constant or balanced (checked).
    n : int
        Number of input bits ``>= 1``.
    max_qubits : int, default=26
        Resource guard.

    Returns
    -------
    {"constant", "balanced"}

    Complexity
    ----------
    Time O(n · 2^n) (2^n classical evaluations to build the oracle, then O(n) gates
    on 2^n amplitudes), extra space O(2^n).

    References
    ----------
    Deutsch & Jozsa (1992); Nielsen & Chuang, §1.4.4.

    Examples
    --------
    >>> from lipipy.quantum import deutsch_jozsa
    >>> deutsch_jozsa(lambda x: 1, 3), deutsch_jozsa(lambda x: x & 1, 3)
    ('constant', 'balanced')
    """
    if not callable(oracle_fn):
        raise TypeError("oracle_fn must be callable")
    q = check_num_qubits(n, max_qubits)
    values = np.array([oracle_fn(x) for x in range(2**q)])
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("oracle_fn must return 0 or 1")
    ones = int(values.sum())
    if ones not in (0, 2**q, 2 ** (q - 1)):
        raise ValueError(
            f"oracle_fn is neither constant nor balanced ({ones} of {2**q} inputs give 1)"
        )
    phases = np.where(values == 1, -1.0, 1.0).astype(np.complex128)
    state = _hadamard_sandwich(q, Circuit(q).diagonal(phases, list(range(q)), name="oracle"))
    return "constant" if abs(state[0]) ** 2 > 0.5 else "balanced"


@register(
    domain="quantum",
    tasks=("oracle", "learning"),
    complexity=Complexity(time="O(q 2^q)", space="O(2^q)", notes="q = len(secret)"),
    references=("Bernstein & Vazirani, Quantum complexity theory, SIAM J. Comput. 26 (1997)",),
)
def bernstein_vazirani(secret: str, *, max_qubits: int = MAX_QUBITS) -> str:
    r"""Recover a hidden bitstring :math:`s` from one query to :math:`f(x) = s\cdot x \bmod 2`.

    The phase oracle :math:`(-1)^{s\cdot x} = \prod_j (-1)^{s_j x_j}` is a Z gate on
    every qubit with :math:`s_j = 1`, and
    :math:`H^{\otimes n} Z^{s} H^{\otimes n}|0\rangle = |s\rangle` exactly, so the
    measurement returns ``s`` with probability 1.

    Parameters
    ----------
    secret : str
        Bitstring over ``"01"``, most-significant qubit first (Qiskit order): the
        rightmost character is qubit 0.
    max_qubits : int, default=26
        Resource guard.

    Returns
    -------
    str
        The recovered bitstring (equal to ``secret``).

    Complexity
    ----------
    O(q) gates on 2^q amplitudes: time O(q · 2^q), extra space O(2^q).

    References
    ----------
    Bernstein & Vazirani (1997).

    Examples
    --------
    >>> from lipipy.quantum import bernstein_vazirani
    >>> bernstein_vazirani("1011")
    '1011'
    """
    if not isinstance(secret, str):
        raise TypeError(f"secret must be a bitstring str, got {type(secret).__name__}")
    if not secret or set(secret) - {"0", "1"}:
        raise ValueError(f"secret must be a non-empty string of 0/1, got {secret!r}")
    q = check_num_qubits(len(secret), max_qubits)
    oracle = Circuit(q)
    for i, bit in enumerate(secret):
        if bit == "1":
            oracle.z(q - 1 - i)
    state = _hadamard_sandwich(q, oracle)
    return format(int(np.argmax(np.abs(state))), f"0{q}b")


# -- phase estimation ------------------------------------------------------------------
@register(
    domain="quantum",
    tasks=("phase-estimation", "eigenvalue"),
    complexity=Complexity(
        time="O(t d^3 + t 2^q d + t^2 2^q)",
        space="O(2^q + d^2)",
        notes=(
            "t = num_counting, d = dimension of the unitary, q = t + log2(d) total qubits; "
            "U^(2^j) by repeated squaring"
        ),
    ),
    references=(
        "Kitaev, Quantum measurements and the Abelian stabilizer problem (1995), "
        "arXiv:quant-ph/9511026",
        _NC + ", §5.2",
    ),
)
def phase_estimation(
    unitary: np.ndarray,
    eigenstate: np.ndarray,
    num_counting: int,
    *,
    max_qubits: int = MAX_QUBITS,
) -> float:
    r"""Estimate :math:`\varphi \in [0, 1)` in :math:`U|u\rangle = e^{2\pi i\varphi}|u\rangle`.

    Counting qubits :math:`0..t-1` start in :math:`|+\rangle`; counting qubit
    :math:`j` controls :math:`U^{2^j}` on the target register (qubits
    :math:`t, t+1, \ldots`), which kicks back
    :math:`\frac{1}{\sqrt{2^t}}\sum_y e^{2\pi i\varphi y}|y\rangle
    = \mathrm{QFT}\,|2^t\varphi\rangle`.
    The inverse QFT then gives :math:`|2^t\varphi\rangle`, exactly when
    :math:`\varphi = y/2^t`; otherwise the most likely :math:`y` is the best
    :math:`t`-bit approximation with probability :math:`\ge 4/\pi^2`.

    Parameters
    ----------
    unitary : array_like, shape (d, d)
        Unitary with ``d = 2**m``.
    eigenstate : array_like, shape (d,)
        Normalized eigenvector of ``unitary`` (target register initial state).
    num_counting : int
        Number of counting qubits ``t >= 1`` (bits of precision).
    max_qubits : int, default=26
        Resource guard on ``t + m``.

    Returns
    -------
    float
        :math:`y^*/2^t` for the most likely counting-register outcome :math:`y^*`.

    Complexity
    ----------
    With q = t + m qubits and d = 2^m: time O(t·d^3) for the powers
    :math:`U^{2^j}`, O(t·2^q·d) for the controlled powers and O(t^2·2^q) for the
    inverse QFT; extra space O(2^q + d^2).

    References
    ----------
    Kitaev (1995); Nielsen & Chuang, §5.2.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.quantum import phase_estimation, phase
    >>> phase_estimation(phase(2 * np.pi * 5 / 8), np.array([0, 1]), num_counting=3)
    0.625
    """
    U, m = check_square_matrix(unitary, "unitary")
    if not gates.is_unitary(U):
        raise ValueError("unitary is not unitary (max |U^† U - I| > 1e-10)")
    u, mu = check_state(eigenstate, "eigenstate")
    if mu != m:
        raise ValueError(f"eigenstate has {mu} qubits but unitary acts on {m}")
    if abs(np.linalg.norm(u) - 1.0) > 1e-8:
        raise ValueError("eigenstate must be normalized")
    t = check_int(num_counting, "num_counting", minimum=1)
    q = check_num_qubits(t + m, max_qubits)

    circ = Circuit(q)
    targets = list(range(t, q))
    power = U
    for j in range(t):
        circ.h(j)
    for j in range(t):
        circ.unitary(gates.controlled(power), [j, *targets], name=f"c-U^{2**j}")
        power = power @ power
    circ = circ.compose(inverse_qft(t))

    initial = np.kron(u, np.eye(2**t, 1, dtype=np.complex128).ravel())  # |u> ⊗ |0..0>
    probs = np.abs(circ.run(initial, max_qubits=max_qubits)) ** 2
    counting = probs.reshape(2**m, 2**t).sum(axis=0)
    return float(int(np.argmax(counting)) / 2**t)
