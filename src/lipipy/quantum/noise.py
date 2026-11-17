r"""Quantum noise channels in Kraus form, and noise models for circuits.

A quantum channel (completely positive, trace-preserving map) acting on k qubits is
written with Kraus operators :math:`K_1, \ldots, K_r` (each :math:`2^k \times 2^k`):

.. math::
    \mathcal{E}(\rho) = \sum_{i=1}^{r} K_i \rho K_i^\dagger, \qquad
    \sum_{i=1}^{r} K_i^\dagger K_i = I \quad\text{(completeness = trace preservation)}.

Any map of this form is completely positive; equivalently its Choi matrix

.. math::
    J(\mathcal{E}) = \sum_{a,b} |a\rangle\langle b| \otimes \mathcal{E}(|a\rangle\langle b|)
                   = \sum_i \mathrm{vec}(K_i)\,\mathrm{vec}(K_i)^\dagger

is positive semidefinite (Choi's theorem). Here :math:`\mathrm{vec}` stacks columns,
so the input factor :math:`|a\rangle\langle b|` is the left (most-significant) tensor
factor, and ``choi()`` matches ``qiskit.quantum_info.Choi``.

Standard single-qubit channels (Nielsen & Chuang §8.3)
------------------------------------------------------
* bit flip :math:`(1-p)\rho + pX\rho X`; phase flip :math:`(1-p)\rho + pZ\rho Z`;
* depolarizing on n qubits :math:`(1-p)\rho + p\,\mathrm{Tr}(\rho)\, I/2^n`
  (Kraus: :math:`\sqrt{1 - p + p/4^n}\,I` and :math:`\sqrt{p/4^n}\,P` for every
  non-identity Pauli :math:`P`), the convention of Qiskit Aer's ``depolarizing_error``;
* amplitude damping :math:`K_0 = \mathrm{diag}(1, \sqrt{1-\gamma})`,
  :math:`K_1 = \sqrt\gamma\,|0\rangle\langle 1|`;
* phase damping :math:`K_0 = \mathrm{diag}(1, \sqrt{1-\lambda})`,
  :math:`K_1 = \mathrm{diag}(0, \sqrt\lambda)` (off-diagonals shrink by
  :math:`\sqrt{1-\lambda}`).

Operators of multi-qubit channels follow the gate convention of
:mod:`lipipy.quantum.gates`: the first target is the least-significant qubit.

Reference: Nielsen & Chuang, *Quantum Computation and Quantum Information*, §8.2–8.3;
Choi, Linear Algebra Appl. 10 (1975).
"""

from __future__ import annotations

import itertools
import numbers
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from lipipy.core.complexity import Complexity
from lipipy.core.registry import register
from lipipy.quantum import gates
from lipipy.quantum._validation import check_int, check_square_matrix

_NC = "Nielsen & Chuang, Quantum Computation and Quantum Information, 10th ed., §8.2-8.3"
_COMPLETENESS_ATOL = 1e-10
_PAULI_1Q = (gates.I, gates.X, gates.Y, gates.Z)


def _check_probability(value: object, name: str, upper: float = 1.0) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    value = float(value)
    if not 0.0 <= value <= upper:
        raise ValueError(f"{name} must be in [0, {upper:g}], got {value}")
    return value


@register(
    domain="quantum",
    tasks=("noise", "channel"),
    complexity=Complexity(
        time="O(r 8^k)",
        space="O(r 4^k)",
        notes="r Kraus operators on k qubits; construction checks completeness",
    ),
    references=(_NC, "Choi, Completely positive linear maps on complex matrices (1975)"),
)
class KrausChannel:
    r"""A quantum channel :math:`\rho \mapsto \sum_i K_i \rho K_i^\dagger`.

    Parameters
    ----------
    kraus_ops : sequence of array_like, each (2**k, 2**k)
        Kraus operators. Completeness :math:`\sum_i K_i^\dagger K_i = I` is checked
        (max abs deviation <= 1e-10). Operators that are exactly zero are dropped.
    name : str, default="kraus"
        Label used in ``repr``.

    Attributes
    ----------
    kraus : tuple of ndarray
        Read-only complex128 copies of the operators.
    num_qubits : int
        ``k``.

    Complexity
    ----------
    Construction O(r · 8^k) for the completeness check, memory O(r · 4^k).
    Applying it to a q-qubit density matrix costs O(r · 4^q · 2^k)
    (:meth:`lipipy.quantum.DensityMatrix.apply_channel`).

    References
    ----------
    Nielsen & Chuang, §8.2.3; Choi (1975).

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.quantum import KrausChannel, X
    >>> ch = KrausChannel([np.sqrt(0.9) * np.eye(2), np.sqrt(0.1) * X], name="flip")
    >>> ch.num_qubits, len(ch.kraus)
    (1, 2)
    >>> bool(np.linalg.eigvalsh(ch.choi()).min() > -1e-12)
    True
    """

    def __init__(self, kraus_ops: Sequence[np.ndarray], name: str = "kraus"):
        if isinstance(kraus_ops, np.ndarray) and kraus_ops.ndim == 2:
            kraus_ops = [kraus_ops]
        ops = [check_square_matrix(K, "kraus operator") for K in kraus_ops]
        if not ops:
            raise ValueError("a channel needs at least one Kraus operator")
        k = ops[0][1]
        if any(kk != k for _, kk in ops):
            raise ValueError("all Kraus operators must have the same size")
        mats = [K.copy() for K, _ in ops if np.any(K != 0)]
        if not mats:
            raise ValueError("all Kraus operators are zero")
        total = sum(K.conj().T @ K for K in mats)
        dev = float(np.max(np.abs(total - np.eye(2**k))))
        if dev > _COMPLETENESS_ATOL:
            raise ValueError(
                f"Kraus operators are not trace preserving: max |sum K^† K - I| = {dev:.3g}"
            )
        for K in mats:
            K.setflags(write=False)
        self._kraus = tuple(mats)
        self._num_qubits = k
        self._name = str(name)

    @property
    def kraus(self) -> tuple[np.ndarray, ...]:
        return self._kraus

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return (
            f"KrausChannel(name={self._name!r}, num_qubits={self._num_qubits}, "
            f"rank={len(self._kraus)})"
        )

    def apply_to_matrix(self, rho: np.ndarray) -> np.ndarray:
        r"""Dense action :math:`\sum_i K_i \rho K_i^\dagger` on a ``2**k`` square matrix."""
        R, k = check_square_matrix(rho, "rho")
        if k != self._num_qubits:
            raise ValueError(f"rho acts on {k} qubits, the channel on {self._num_qubits}")
        return sum(K @ R @ K.conj().T for K in self._kraus)

    def choi(self) -> np.ndarray:
        r"""Choi matrix :math:`\sum_i \mathrm{vec}(K_i)\mathrm{vec}(K_i)^\dagger`, ``(4**k, 4**k)``.

        Positive semidefinite for every channel; its partial trace over the output
        is the identity (trace preservation). Same convention as
        ``qiskit.quantum_info.Choi``.
        """
        vecs = np.stack([K.reshape(-1, order="F") for K in self._kraus], axis=1)
        return vecs @ vecs.conj().T

    def compose(self, other: KrausChannel) -> KrausChannel:
        r"""Channel ``other ∘ self`` (apply ``self`` first), Kraus ops :math:`B_j A_i`."""
        if not isinstance(other, KrausChannel):
            raise TypeError("other must be a KrausChannel")
        if other.num_qubits != self._num_qubits:
            raise ValueError("channels act on different numbers of qubits")
        ops = [B @ A for B in other.kraus for A in self._kraus]
        return KrausChannel(ops, name=f"{other.name}∘{self._name}")


_ONE_QUBIT = Complexity(time="O(1)", space="O(1)")


@register(domain="quantum", tasks=("noise", "channel"), complexity=_ONE_QUBIT, references=(_NC,))
def bit_flip(p: float) -> KrausChannel:
    r"""Bit-flip channel :math:`\rho \mapsto (1-p)\rho + pX\rho X`.

    Parameters
    ----------
    p : float in [0, 1]
        Flip probability.

    Returns
    -------
    KrausChannel

    Complexity
    ----------
    O(1).

    References
    ----------
    Nielsen & Chuang, §8.3.3.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.quantum import bit_flip
    >>> bit_flip(0.25).apply_to_matrix(np.diag([1.0, 0.0])).real
    array([[0.75, 0.  ],
           [0.  , 0.25]])
    """
    p = _check_probability(p, "p")
    return KrausChannel([np.sqrt(1 - p) * gates.I, np.sqrt(p) * gates.X], name="bit_flip")


@register(domain="quantum", tasks=("noise", "channel"), complexity=_ONE_QUBIT, references=(_NC,))
def phase_flip(p: float) -> KrausChannel:
    r"""Phase-flip channel :math:`\rho \mapsto (1-p)\rho + pZ\rho Z`.

    Parameters
    ----------
    p : float in [0, 1]
        Probability of a Z error.

    Returns
    -------
    KrausChannel

    Complexity
    ----------
    O(1).

    References
    ----------
    Nielsen & Chuang, §8.3.3.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.quantum import phase_flip
    >>> plus = np.full((2, 2), 0.5)
    >>> phase_flip(0.5).apply_to_matrix(plus).real   # fully dephased
    array([[0.5, 0. ],
           [0. , 0.5]])
    """
    p = _check_probability(p, "p")
    return KrausChannel([np.sqrt(1 - p) * gates.I, np.sqrt(p) * gates.Z], name="phase_flip")


@register(
    domain="quantum",
    tasks=("noise", "channel"),
    complexity=Complexity(time="O(16^n)", space="O(16^n)", notes="4^n Pauli Kraus operators"),
    references=(_NC + " (§8.3.4)", "Qiskit Aer, noise.depolarizing_error"),
)
def depolarizing(p: float, num_qubits: int = 1) -> KrausChannel:
    r"""Depolarizing channel :math:`\rho \mapsto (1-p)\rho + p\,\mathrm{Tr}(\rho)\,I/2^n`.

    Kraus operators: :math:`\sqrt{1 - p + p/4^n}\,I` and :math:`\sqrt{p/4^n}\,P` for
    each of the :math:`4^n - 1` non-identity Pauli strings :math:`P`, using
    :math:`\frac{1}{4^n}\sum_P P\rho P = \mathrm{Tr}(\rho)\, I/2^n`. ``p = 1`` maps
    every state to the maximally mixed state.

    Parameters
    ----------
    p : float in [0, 4**n / (4**n - 1)]
        Depolarizing strength (values above 1 are still CPTP, as in Qiskit Aer).
    num_qubits : int, default=1
        Number of qubits ``n``.

    Returns
    -------
    KrausChannel

    Complexity
    ----------
    O(16^n) time and memory for the 4^n operators of size 4^n.

    References
    ----------
    Nielsen & Chuang, §8.3.4; Qiskit Aer ``depolarizing_error``.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.quantum import depolarizing
    >>> depolarizing(1.0).apply_to_matrix(np.diag([1.0, 0.0])).real.round(12)
    array([[0.5, 0. ],
           [0. , 0.5]])
    """
    n = check_int(num_qubits, "num_qubits", minimum=1)
    p = _check_probability(p, "p", upper=4**n / (4**n - 1))
    ops = []
    for paulis in itertools.product(_PAULI_1Q, repeat=n):
        P = np.array([[1.0 + 0j]])
        for factor in paulis:
            P = np.kron(P, factor)
        ops.append(P)
    coef_i = np.sqrt(1 - p + p / 4**n)
    coef_p = np.sqrt(p / 4**n)
    return KrausChannel([coef_i * ops[0]] + [coef_p * P for P in ops[1:]], name="depolarizing")


@register(domain="quantum", tasks=("noise", "channel"), complexity=_ONE_QUBIT, references=(_NC,))
def amplitude_damping(gamma: float) -> KrausChannel:
    r"""Amplitude damping (energy relaxation :math:`|1\rangle \to |0\rangle` with prob. γ).

    :math:`K_0 = \begin{pmatrix}1&0\\0&\sqrt{1-\gamma}\end{pmatrix}`,
    :math:`K_1 = \begin{pmatrix}0&\sqrt\gamma\\0&0\end{pmatrix}`.

    Parameters
    ----------
    gamma : float in [0, 1]
        Decay probability (``1 - exp(-t/T1)``).

    Returns
    -------
    KrausChannel

    Complexity
    ----------
    O(1).

    References
    ----------
    Nielsen & Chuang, §8.3.5.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.quantum import amplitude_damping
    >>> amplitude_damping(1.0).apply_to_matrix(np.diag([0.0, 1.0])).real
    array([[1., 0.],
           [0., 0.]])
    """
    g = _check_probability(gamma, "gamma")
    K0 = np.array([[1, 0], [0, np.sqrt(1 - g)]], dtype=np.complex128)
    K1 = np.array([[0, np.sqrt(g)], [0, 0]], dtype=np.complex128)
    return KrausChannel([K0, K1], name="amplitude_damping")


@register(domain="quantum", tasks=("noise", "channel"), complexity=_ONE_QUBIT, references=(_NC,))
def phase_damping(lam: float) -> KrausChannel:
    r"""Phase damping (pure dephasing): off-diagonals scale by :math:`\sqrt{1-\lambda}`.

    :math:`K_0 = \mathrm{diag}(1, \sqrt{1-\lambda})`,
    :math:`K_1 = \mathrm{diag}(0, \sqrt\lambda)`. Populations are unchanged.

    Parameters
    ----------
    lam : float in [0, 1]
        Damping parameter λ.

    Returns
    -------
    KrausChannel

    Complexity
    ----------
    O(1).

    References
    ----------
    Nielsen & Chuang, §8.3.6.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.quantum import phase_damping
    >>> phase_damping(0.75).apply_to_matrix(np.full((2, 2), 0.5)).real
    array([[0.5 , 0.25],
           [0.25, 0.5 ]])
    """
    lam = _check_probability(lam, "lam")
    K0 = np.array([[1, 0], [0, np.sqrt(1 - lam)]], dtype=np.complex128)
    K1 = np.array([[0, 0], [0, np.sqrt(lam)]], dtype=np.complex128)
    return KrausChannel([K0, K1], name="phase_damping")


@dataclass(frozen=True)
class NoiseRule:
    """One rule of a :class:`NoiseModel`: ``channel`` after matching instructions.

    ``gates is None`` matches every gate name; ``qubits is None`` matches every
    placement, otherwise the instruction's qubit tuple must equal ``qubits``.
    """

    channel: KrausChannel
    gates: frozenset[str] | None
    qubits: tuple[int, ...] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel.name,
            "channel_qubits": self.channel.num_qubits,
            "gates": None if self.gates is None else sorted(self.gates),
            "qubits": None if self.qubits is None else list(self.qubits),
        }


@register(
    domain="quantum",
    tasks=("noise",),
    complexity=Complexity(time="O(R)", space="O(R)", notes="R rules scanned per instruction"),
    references=("Qiskit Aer, noise.NoiseModel (design reference)", _NC),
)
class NoiseModel:
    """Which channels follow which gates during density-matrix simulation.

    After every circuit instruction, each matching rule's channel is applied:

    * if the channel acts on as many qubits as the instruction, it acts on the
      instruction's qubits (in the same order);
    * a one-qubit channel on a multi-qubit instruction acts on each of its qubits;
    * any other size mismatch raises ``ValueError`` at simulation time.

    Rules are applied in the order they were added. :meth:`add` validates
    everything before changing the model, so a failed call leaves it unchanged.

    Complexity
    ----------
    Matching is O(R) per instruction for R rules.

    References
    ----------
    Design follows Qiskit Aer's ``NoiseModel``.

    Examples
    --------
    >>> from lipipy.quantum import NoiseModel, depolarizing, Circuit
    >>> noise = NoiseModel().add(depolarizing(0.1), gates=["h"])
    >>> rho = Circuit(1).h(0).run_density(noise_model=noise)
    >>> round(rho.purity(), 6)
    0.905
    """

    def __init__(self) -> None:
        self._rules: list[NoiseRule] = []

    @property
    def rules(self) -> tuple[NoiseRule, ...]:
        return tuple(self._rules)

    def add(
        self,
        channel: KrausChannel,
        gates: Iterable[str] | str | None = None,
        qubits: Iterable[int] | int | None = None,
    ) -> NoiseModel:
        """Add a rule and return ``self``.

        Parameters
        ----------
        channel : KrausChannel
            Applied after each matching instruction.
        gates : str or iterable of str, optional
            Instruction names to match (e.g. ``["h", "cx"]``); ``None`` = all.
        qubits : int or iterable of int, optional
            Exact qubit tuple to match; ``None`` = any placement.
        """
        if not isinstance(channel, KrausChannel):
            raise TypeError(f"channel must be a KrausChannel, got {type(channel).__name__}")
        names: frozenset[str] | None = None
        if gates is not None:
            seq = [gates] if isinstance(gates, str) else list(gates)
            if not seq or not all(isinstance(g, str) and g for g in seq):
                raise ValueError("gates must be a non-empty string or iterable of strings")
            names = frozenset(seq)
        where: tuple[int, ...] | None = None
        if qubits is not None:
            seq_q = [qubits] if isinstance(qubits, numbers.Integral) else list(qubits)
            where = tuple(check_int(t, "qubits", minimum=0) for t in seq_q)
            if not where or len(set(where)) != len(where):
                raise ValueError("qubits must be non-empty and distinct")
            if channel.num_qubits not in (1, len(where)):
                raise ValueError(
                    f"a {channel.num_qubits}-qubit channel cannot follow a gate on "
                    f"{len(where)} qubits"
                )
        self._rules.append(NoiseRule(channel, names, where))
        return self

    def channels_for(
        self, name: str, qubits: Sequence[int]
    ) -> list[tuple[KrausChannel, tuple[int, ...]]]:
        """``(channel, targets)`` pairs to apply after instruction ``name`` on ``qubits``."""
        qubits = tuple(qubits)
        out = []
        for rule in self._rules:
            if rule.gates is not None and name not in rule.gates:
                continue
            if rule.qubits is not None and rule.qubits != qubits:
                continue
            k = rule.channel.num_qubits
            if k == len(qubits):
                out.append((rule.channel, qubits))
            elif k == 1:
                out.extend((rule.channel, (t,)) for t in qubits)
            else:
                raise ValueError(
                    f"a {k}-qubit {rule.channel.name} channel cannot follow {name!r} "
                    f"on {len(qubits)} qubits"
                )
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"rules": [r.to_dict() for r in self._rules]}

    def __repr__(self) -> str:
        return f"NoiseModel(rules={len(self._rules)})"
