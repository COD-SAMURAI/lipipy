r"""Quantum gates as unitary complex128 matrices.

Matrix ordering convention
--------------------------
A k-qubit gate acts on ``targets = (t0, t1, ..., t_{k-1})``. Its matrix is written in
the basis :math:`|b_{k-1} \ldots b_1 b_0\rangle` of the gate's *own* index
:math:`i = \sum_j b_j 2^j`, where bit :math:`b_j` is the state of target ``t_j``. So
``t0`` is the least-significant qubit of the gate, exactly as in Qiskit. For example
:data:`CX` applied to ``targets=(control, target)`` is

.. math::
    \mathrm{CX} = \begin{pmatrix} 1&0&0&0\\ 0&0&0&1\\ 0&0&1&0\\ 0&1&0&0 \end{pmatrix},

which is Qiskit's ``CXGate().to_matrix()`` for ``qc.cx(control, target)``.
:func:`controlled` follows the same rule: the controls are the *first* (lowest)
targets and the base gate acts on the remaining ones.

Single-qubit gates
------------------
.. math::
    R_X(\theta) = e^{-i\theta X/2},\quad R_Y(\theta) = e^{-i\theta Y/2},\quad
    R_Z(\theta) = e^{-i\theta Z/2},\quad P(\lambda) = \mathrm{diag}(1, e^{i\lambda}),

and Qiskit's general single-qubit gate

.. math::
    U(\theta,\phi,\lambda) = \begin{pmatrix}
        \cos\frac\theta2 & -e^{i\lambda}\sin\frac\theta2 \\
        e^{i\phi}\sin\frac\theta2 & e^{i(\phi+\lambda)}\cos\frac\theta2
    \end{pmatrix}.

Reference: Nielsen & Chuang, *Quantum Computation and Quantum Information*, §4.2–4.3;
Qiskit circuit library documentation (``UGate``, ``SXGate``).
"""

from __future__ import annotations

import numpy as np

from lipipy.core.complexity import Complexity
from lipipy.core.registry import register
from lipipy.quantum._validation import check_angle, check_int, check_square_matrix

_NC = "Nielsen & Chuang, Quantum Computation and Quantum Information, 10th ed., §4.2-4.3"
_QISKIT = "Qiskit circuit library documentation (standard gates)"


def _const(rows: list[list[complex]]) -> np.ndarray:
    m = np.array(rows, dtype=np.complex128)
    m.setflags(write=False)  # shared module constants must not be mutated
    return m


_S2 = 1 / np.sqrt(2)

#: Identity.
I = _const([[1, 0], [0, 1]])  # noqa: E741
#: Pauli X (NOT).
X = _const([[0, 1], [1, 0]])
#: Pauli Y.
Y = _const([[0, -1j], [1j, 0]])
#: Pauli Z.
Z = _const([[1, 0], [0, -1]])
#: Hadamard.
H = _const([[_S2, _S2], [_S2, -_S2]])
#: Phase gate S = sqrt(Z).
S = _const([[1, 0], [0, 1j]])
#: S dagger.
SDG = _const([[1, 0], [0, -1j]])
#: T = fourth root of Z.
T = _const([[1, 0], [0, np.exp(1j * np.pi / 4)]])
#: T dagger.
TDG = _const([[1, 0], [0, np.exp(-1j * np.pi / 4)]])
#: SX = sqrt(X).
SX = _const([[(1 + 1j) / 2, (1 - 1j) / 2], [(1 - 1j) / 2, (1 + 1j) / 2]])
#: Controlled-X on ``targets=(control, target)``.
CX = _const([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
#: Controlled-Z (symmetric in its two qubits).
CZ = _const([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
#: SWAP.
SWAP = _const([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
#: Toffoli (CCX) on ``targets=(control0, control1, target)``.
CCX = _const(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
    ]
)

_ROT = Complexity(time="O(1)", space="O(1)")


@register(domain="quantum", tasks=("gate",), complexity=_ROT, references=(_NC,))
def rx(theta: float) -> np.ndarray:
    r"""Rotation about X: :math:`R_X(\theta) = \cos\frac\theta2 I - i\sin\frac\theta2 X`.

    Parameters
    ----------
    theta : float
        Angle in radians.

    Returns
    -------
    ndarray of complex128, shape (2, 2)

    Complexity
    ----------
    O(1).

    References
    ----------
    Nielsen & Chuang, eq. (4.4).

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.quantum import rx, X
    >>> np.allclose(rx(np.pi), -1j * X)
    True
    """
    t = check_angle(theta, "theta") / 2
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


@register(domain="quantum", tasks=("gate",), complexity=_ROT, references=(_NC,))
def ry(theta: float) -> np.ndarray:
    r"""Rotation about Y: :math:`R_Y(\theta) = \cos\frac\theta2 I - i\sin\frac\theta2 Y`.

    Parameters
    ----------
    theta : float
        Angle in radians.

    Returns
    -------
    ndarray of complex128, shape (2, 2)

    Complexity
    ----------
    O(1).

    References
    ----------
    Nielsen & Chuang, eq. (4.5).

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.quantum import ry
    >>> np.round(ry(np.pi / 2) @ [1, 0], 6).real
    array([0.707107, 0.707107])
    """
    t = check_angle(theta, "theta") / 2
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


@register(domain="quantum", tasks=("gate",), complexity=_ROT, references=(_NC,))
def rz(theta: float) -> np.ndarray:
    r"""Rotation about Z: :math:`R_Z(\theta) = \mathrm{diag}(e^{-i\theta/2}, e^{i\theta/2})`.

    Parameters
    ----------
    theta : float
        Angle in radians.

    Returns
    -------
    ndarray of complex128, shape (2, 2)

    Complexity
    ----------
    O(1).

    References
    ----------
    Nielsen & Chuang, eq. (4.6).

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.quantum import rz, Z
    >>> np.allclose(rz(np.pi), -1j * Z)
    True
    """
    t = check_angle(theta, "theta") / 2
    return np.array([[np.exp(-1j * t), 0], [0, np.exp(1j * t)]], dtype=np.complex128)


@register(domain="quantum", tasks=("gate",), complexity=_ROT, references=(_QISKIT,))
def phase(lam: float) -> np.ndarray:
    r"""Phase gate :math:`P(\lambda) = \mathrm{diag}(1, e^{i\lambda})` (Qiskit ``p``).

    Equal to :math:`R_Z(\lambda)` up to the global phase :math:`e^{i\lambda/2}`;
    :math:`P(\pi/2) = S`, :math:`P(\pi/4) = T`.

    Parameters
    ----------
    lam : float
        Phase in radians.

    Returns
    -------
    ndarray of complex128, shape (2, 2)

    Complexity
    ----------
    O(1).

    References
    ----------
    Qiskit ``PhaseGate``.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.quantum import phase, T
    >>> np.allclose(phase(np.pi / 4), T)
    True
    """
    lam = check_angle(lam, "lam")
    return np.array([[1, 0], [0, np.exp(1j * lam)]], dtype=np.complex128)


@register(domain="quantum", tasks=("gate",), complexity=_ROT, references=(_QISKIT, _NC))
def u3(theta: float, phi: float, lam: float) -> np.ndarray:
    r"""General single-qubit gate :math:`U(\theta, \phi, \lambda)` (Qiskit's ``U``).

    .. math::
        U = \begin{pmatrix}
            \cos\frac\theta2 & -e^{i\lambda}\sin\frac\theta2 \\
            e^{i\phi}\sin\frac\theta2 & e^{i(\phi+\lambda)}\cos\frac\theta2
        \end{pmatrix}
          = e^{i(\phi+\lambda)/2} R_Z(\phi) R_Y(\theta) R_Z(\lambda).

    Parameters
    ----------
    theta, phi, lam : float
        Euler angles in radians.

    Returns
    -------
    ndarray of complex128, shape (2, 2)

    Complexity
    ----------
    O(1).

    References
    ----------
    Qiskit ``UGate``.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.quantum import u3, H
    >>> np.allclose(u3(np.pi / 2, 0, np.pi), H)
    True
    """
    theta = check_angle(theta, "theta")
    phi = check_angle(phi, "phi")
    lam = check_angle(lam, "lam")
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array(
        [
            [c, -np.exp(1j * lam) * s],
            [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c],
        ],
        dtype=np.complex128,
    )


@register(
    domain="quantum",
    tasks=("gate",),
    complexity=Complexity(
        time="O(4^k)",
        space="O(4^k)",
        notes="k = total qubits of the controlled gate (controls + targets)",
    ),
    references=(_NC + " (§4.3, controlled operations)",),
)
def controlled(U: np.ndarray, num_controls: int = 1) -> np.ndarray:
    r"""Controlled version of a gate: apply ``U`` only when every control is ``|1>``.

    The returned matrix acts on ``targets = (c_0, ..., c_{m-1}, u_0, ..., u_{k-1})``:
    the ``m = num_controls`` controls are the lowest bits of the gate index (see the
    module docstring), so ``controlled(X)`` equals :data:`CX` and
    ``controlled(X, 2)`` equals :data:`CCX`. With gate index
    :math:`i = c + 2^m u`, the matrix is the identity except on the block
    :math:`c = 2^m - 1`, where it is ``U``.

    Parameters
    ----------
    U : array_like, shape (2**k, 2**k)
        Base gate.
    num_controls : int, default=1
        Number of control qubits ``m >= 1``.

    Returns
    -------
    ndarray of complex128, shape (2**(m+k), 2**(m+k))

    Complexity
    ----------
    Time and space O(4^(m+k)) for the dense output.

    References
    ----------
    Nielsen & Chuang, §4.3.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.quantum import controlled, X, CX, CCX
    >>> np.array_equal(controlled(X), CX), np.array_equal(controlled(X, 2), CCX)
    (True, True)
    """
    U, _ = check_square_matrix(U, "U")
    m = check_int(num_controls, "num_controls", minimum=1)
    dim_u = U.shape[0]
    stride = 2**m
    out = np.eye(stride * dim_u, dtype=np.complex128)
    block = (stride - 1) + stride * np.arange(dim_u)  # indices where all controls are 1
    out[np.ix_(block, block)] = U
    return out


@register(
    domain="quantum",
    tasks=("gate", "validation"),
    complexity=Complexity(time="O(d^3)", space="O(d^2)", notes="d = matrix dimension"),
    references=(_NC + " (§2.1.6)",),
)
def is_unitary(U: np.ndarray, atol: float = 1e-10) -> bool:
    r"""Whether :math:`\max_{ij} |(U^\dagger U - I)_{ij}| \le` ``atol``.

    Parameters
    ----------
    U : array_like, shape (d, d)
        Square matrix (any dimension).
    atol : float, default=1e-10
        Absolute tolerance.

    Returns
    -------
    bool
        ``False`` for non-square or non-2-D input.

    Complexity
    ----------
    Time O(d^3) for the product, space O(d^2).

    References
    ----------
    Nielsen & Chuang, §2.1.6.

    Examples
    --------
    >>> from lipipy.quantum import is_unitary, H
    >>> is_unitary(H), is_unitary([[1, 1], [0, 1]])
    (True, False)
    """
    M = np.asarray(U, dtype=np.complex128)
    if M.ndim != 2 or M.shape[0] != M.shape[1] or M.shape[0] == 0:
        return False
    if not np.all(np.isfinite(M)):
        return False
    return bool(np.max(np.abs(M.conj().T @ M - np.eye(M.shape[0]))) <= atol)
