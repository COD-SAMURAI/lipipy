r"""Pure qubit states as complex128 vectors.

A pure state of :math:`q` qubits is a unit vector :math:`|\psi\rangle \in \mathbb{C}^{2^q}`

.. math::  |\psi\rangle = \sum_{i=0}^{2^q-1} \psi_i |i\rangle, \qquad \sum_i |\psi_i|^2 = 1,

where the basis index :math:`i = \sum_j b_j 2^j` has qubit :math:`j` as bit :math:`j`
(qubit 0 is the least-significant bit, as in Qiskit).

Memory is :math:`16 \cdot 2^q` bytes. Every constructor here checks ``q`` against
``max_qubits`` (default :data:`MAX_QUBITS` = 26, i.e. 1 GiB) and raises
:class:`lipipy.core.ResourceLimitError` *before* allocating.

Haar-random states: if :math:`z_i = x_i + i y_i` with :math:`x_i, y_i` i.i.d.
:math:`\mathcal{N}(0, 1)`, then :math:`z / \lVert z \rVert` is uniformly
(Haar-) distributed on the unit sphere of :math:`\mathbb{C}^{2^q}`, because the complex
Gaussian vector is invariant under every unitary.

Reference: Nielsen & Chuang, *Quantum Computation and Quantum Information*,
10th anniversary ed., §1.2 and §2.2; Mezzadri, "How to generate random matrices from
the classical compact groups", Notices of the AMS 54 (2007).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from lipipy.core.complexity import Complexity
from lipipy.core.random import RandomState, check_random_state
from lipipy.core.registry import register
from lipipy.quantum._validation import MAX_QUBITS, check_int, check_num_qubits, check_state

_NC = "Nielsen & Chuang, Quantum Computation and Quantum Information, 10th ed., §2.2"


@register(
    domain="quantum",
    tasks=("state-preparation",),
    complexity=Complexity(time="O(2^q)", space="O(1)", notes="the output itself is 16·2^q bytes"),
    references=(_NC,),
)
def zero_state(num_qubits: int, *, max_qubits: int = MAX_QUBITS) -> np.ndarray:
    """The all-zeros computational basis state :math:`|0\\ldots0\\rangle`.

    Parameters
    ----------
    num_qubits : int
        Number of qubits ``q >= 1``.
    max_qubits : int, default=26
        Refuse (with :class:`~lipipy.core.ResourceLimitError`, before allocating) when
        ``num_qubits`` exceeds this.

    Returns
    -------
    ndarray of complex128, shape (2**q,)

    Complexity
    ----------
    Time O(2^q); no memory beyond the 16·2^q-byte output.

    References
    ----------
    Nielsen & Chuang, §2.2.

    Examples
    --------
    >>> from lipipy.quantum import zero_state
    >>> zero_state(2)
    array([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])
    """
    q = check_num_qubits(num_qubits, max_qubits)
    psi = np.zeros(2**q, dtype=np.complex128)
    psi[0] = 1.0
    return psi


@register(
    domain="quantum",
    tasks=("state-preparation",),
    complexity=Complexity(time="O(2^q)", space="O(1)", notes="the output itself is 16·2^q bytes"),
    references=(_NC,),
)
def basis_state(num_qubits: int, index: int, *, max_qubits: int = MAX_QUBITS) -> np.ndarray:
    """The computational basis state :math:`|\\text{index}\\rangle`.

    ``index`` uses the library's little-endian convention: bit ``j`` of ``index`` is
    the value of qubit ``j``. ``basis_state(3, 0b001)`` has qubit 0 in ``|1>``.

    Parameters
    ----------
    num_qubits : int
        Number of qubits ``q >= 1``.
    index : int
        Basis index in ``[0, 2**q)``.
    max_qubits : int, default=26
        Resource guard, see :func:`zero_state`.

    Returns
    -------
    ndarray of complex128, shape (2**q,)

    Complexity
    ----------
    Time O(2^q); no memory beyond the output.

    References
    ----------
    Nielsen & Chuang, §2.2.

    Examples
    --------
    >>> from lipipy.quantum import basis_state
    >>> basis_state(2, 2).real
    array([0., 0., 1., 0.])
    """
    q = check_num_qubits(num_qubits, max_qubits)
    index = check_int(index, "index", minimum=0)
    if index >= 2**q:
        raise ValueError(f"index must be < 2**{q} = {2**q}, got {index}")
    psi = np.zeros(2**q, dtype=np.complex128)
    psi[index] = 1.0
    return psi


@register(
    domain="quantum",
    tasks=("state-preparation", "sampling"),
    complexity=Complexity(time="O(2^q)", space="O(2^q)", notes="two Gaussian draws per amplitude"),
    references=(
        "Mezzadri, How to generate random matrices from the classical compact groups, "
        "Notices of the AMS 54 (2007)",
    ),
)
def random_state(
    num_qubits: int, random_state: RandomState = None, *, max_qubits: int = MAX_QUBITS
) -> np.ndarray:
    """A Haar-random pure state (normalized complex Gaussian vector).

    Parameters
    ----------
    num_qubits : int
        Number of qubits ``q >= 1``.
    random_state : None, int, SeedSequence or Generator
        Seed, see :func:`lipipy.core.check_random_state`.
    max_qubits : int, default=26
        Resource guard, see :func:`zero_state`.

    Returns
    -------
    ndarray of complex128, shape (2**q,)
        Unit-norm state.

    Complexity
    ----------
    Time O(2^q), extra space O(2^q) for the real and imaginary Gaussian draws.

    References
    ----------
    Mezzadri (2007), §3.

    Examples
    --------
    >>> from lipipy.quantum import random_state, is_normalized
    >>> psi = random_state(3, random_state=0)
    >>> psi.shape, is_normalized(psi)
    ((8,), True)
    """
    q = check_num_qubits(num_qubits, max_qubits)
    rng = check_random_state(random_state)
    z = rng.standard_normal(2**q) + 1j * rng.standard_normal(2**q)
    return z / np.linalg.norm(z)


@register(
    domain="quantum",
    tasks=("state-analysis",),
    complexity=Complexity(time="O(2^q)", space="O(1)"),
    references=(_NC,),
)
def is_normalized(state: np.ndarray, atol: float = 1e-10) -> bool:
    """Whether :math:`\\big|\\,\\lVert\\psi\\rVert_2 - 1\\big| \\le` ``atol``.

    Parameters
    ----------
    state : array_like, shape (2**q,)
        Statevector.
    atol : float, default=1e-10
        Absolute tolerance on the norm.

    Returns
    -------
    bool

    Complexity
    ----------
    Time O(2^q), no extra memory.

    References
    ----------
    Nielsen & Chuang, §2.2.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.quantum import is_normalized
    >>> is_normalized(np.array([1, 1]) / np.sqrt(2)), is_normalized([1, 1])
    (True, False)
    """
    psi, _ = check_state(state)
    return bool(abs(np.linalg.norm(psi) - 1.0) <= atol)


@register(
    domain="quantum",
    tasks=("state-analysis",),
    complexity=Complexity(time="O(1)", space="O(1)"),
    references=("Nielsen & Chuang, Quantum Computation and Quantum Information, §1.2, §4.2",),
)
def bloch_vector(state: np.ndarray) -> np.ndarray:
    r"""Bloch-sphere coordinates :math:`(x, y, z)` of a one-qubit pure state.

    For :math:`|\psi\rangle = \alpha|0\rangle + \beta|1\rangle`,

    .. math::
        x = \langle X\rangle = 2\,\mathrm{Re}(\bar\alpha\beta),\quad
        y = \langle Y\rangle = 2\,\mathrm{Im}(\bar\alpha\beta),\quad
        z = \langle Z\rangle = |\alpha|^2 - |\beta|^2,

    so :math:`|\psi\rangle = \cos\tfrac\theta2|0\rangle + e^{i\varphi}\sin\tfrac\theta2|1\rangle`
    maps to :math:`(\sin\theta\cos\varphi, \sin\theta\sin\varphi, \cos\theta)`.

    Parameters
    ----------
    state : array_like, shape (2,)
        Normalized one-qubit state.

    Returns
    -------
    ndarray of float64, shape (3,)
        Unit vector ``[x, y, z]``.

    Complexity
    ----------
    O(1).

    References
    ----------
    Nielsen & Chuang, §1.2.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.quantum import bloch_vector
    >>> np.round(bloch_vector(np.array([1, 1j]) / np.sqrt(2)), 12) + 0.0
    array([0., 1., 0.])
    """
    psi, q = check_state(state)
    if q != 1:
        raise ValueError(f"bloch_vector needs a one-qubit state (length 2), got {q} qubits")
    if abs(np.linalg.norm(psi) - 1.0) > 1e-10:
        raise ValueError("state must be normalized")
    a, b = psi
    ab = np.conj(a) * b
    return np.array([2 * ab.real, 2 * ab.imag, abs(a) ** 2 - abs(b) ** 2])


@register(
    domain="quantum",
    tasks=("state-preparation",),
    complexity=Complexity(time="O(2^q)", space="O(2^q)", notes="q = total qubits of the product"),
    references=(_NC,),
)
def tensor(*states: np.ndarray | Sequence[complex], max_qubits: int = MAX_QUBITS) -> np.ndarray:
    r"""Tensor (Kronecker) product :math:`|a\rangle \otimes |b\rangle \otimes \cdots`.

    Follows the written order of the math and of Qiskit's ``a.tensor(b)``: the
    **last** factor occupies the lowest-numbered qubits. ``tensor(a, b)`` with
    one-qubit ``a``, ``b`` puts ``b`` on qubit 0 and ``a`` on qubit 1.

    Parameters
    ----------
    *states : array_like
        One or more statevectors (lengths powers of two).
    max_qubits : int, default=26
        Resource guard on the total number of qubits, checked before allocating.

    Returns
    -------
    ndarray of complex128

    Complexity
    ----------
    Time O(2^q) for q total qubits; intermediate products take O(2^q) extra space.

    References
    ----------
    Nielsen & Chuang, §2.1.7.

    Examples
    --------
    >>> from lipipy.quantum import basis_state, tensor
    >>> tensor(basis_state(1, 1), basis_state(1, 0)).real   # qubit 1 = 1, qubit 0 = 0
    array([0., 0., 1., 0.])
    """
    if not states:
        raise ValueError("tensor needs at least one state")
    checked = [check_state(s, "states[i]") for s in states]
    check_num_qubits(sum(q for _, q in checked), max_qubits)
    out = checked[0][0]
    for psi, _ in checked[1:]:
        out = np.kron(out, psi)
    return out


@register(
    domain="quantum",
    tasks=("state-analysis",),
    complexity=Complexity(time="O(2^q)", space="O(1)"),
    references=("Nielsen & Chuang, Quantum Computation and Quantum Information, §9.2.2",),
)
def fidelity(state_a: np.ndarray, state_b: np.ndarray) -> float:
    r"""Fidelity :math:`F = |\langle a | b\rangle|^2` of two pure states.

    (The squared convention, as in Qiskit's ``state_fidelity``; Nielsen & Chuang use
    the square root.)

    Parameters
    ----------
    state_a, state_b : array_like, shape (2**q,)
        Normalized statevectors of the same size.

    Returns
    -------
    float in [0, 1]

    Complexity
    ----------
    Time O(2^q), no extra memory.

    References
    ----------
    Nielsen & Chuang, §9.2.2.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.quantum import fidelity
    >>> round(fidelity([1, 0], np.array([1, 1]) / np.sqrt(2)), 12)
    0.5
    """
    a, qa = check_state(state_a, "state_a")
    b, qb = check_state(state_b, "state_b")
    if qa != qb:
        raise ValueError(f"states have different sizes: {qa} and {qb} qubits")
    return float(abs(np.vdot(a, b)) ** 2)
