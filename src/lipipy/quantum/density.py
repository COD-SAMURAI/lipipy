r"""Density-matrix simulation of mixed states and noisy circuits.

A mixed state of q qubits is a :math:`2^q \times 2^q` matrix :math:`\rho` that is
Hermitian, positive semidefinite and of unit trace; a pure state is
:math:`\rho = |\psi\rangle\langle\psi|`, a mixture
:math:`\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|`.

Applying a gate without a :math:`4^q \times 4^q` superoperator
-------------------------------------------------------------
:math:`U\rho U^\dagger` is computed in two contractions of the k-qubit gate, reusing the
statevector kernel of :mod:`lipipy.quantum.statevector`:

1. treat the :math:`2^q` columns of :math:`\rho` as a batch of vectors:
   :math:`A = U\rho` (the gate touches only the row axes of its targets);
2. :math:`B = U A^\dagger = U\rho^\dagger U^\dagger`, and :math:`B^\dagger = U\rho U^\dagger`.

Each step costs :math:`O(4^q \cdot 2^k)` time and one :math:`4^q` temporary. A Kraus
channel :math:`\sum_i K_i\rho K_i^\dagger` is r such products. Memory is
:math:`16 \cdot 4^q` bytes, so the default ``max_qubits`` is 13 (1 GiB); above it,
:class:`lipipy.core.ResourceLimitError` is raised before allocating.

Conventions are those of the statevector simulator: qubit 0 is the least-significant
bit, bitstrings are printed most-significant qubit first, and :meth:`partial_trace`
orders the kept qubits like gate targets (``keep[0]`` least significant).

Reference: Nielsen & Chuang, *Quantum Computation and Quantum Information*, §2.4
(density operators), §8.2 (operator-sum representation), §9.2.2 (fidelity),
§11.3 (von Neumann entropy); Uhlmann, Rep. Math. Phys. 9 (1976).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from lipipy.core.complexity import Complexity
from lipipy.core.errors import ResourceLimitError
from lipipy.core.random import RandomState, check_random_state
from lipipy.core.registry import register
from lipipy.quantum._validation import (
    check_int,
    check_qubits,
    check_square_matrix,
    check_state,
    format_bytes,
)
from lipipy.quantum.noise import KrausChannel
from lipipy.quantum.statevector import _apply_diagonal_tensor, _apply_tensor

#: Default largest density matrix that will be allocated (16·4^13 bytes = 1 GiB).
MAX_DENSITY_QUBITS = 13

_NC = "Nielsen & Chuang, Quantum Computation and Quantum Information, 10th ed."
_VALID_ATOL = 1e-8


def _check_density_qubits(num_qubits: int, max_qubits: int) -> int:
    q = check_int(num_qubits, "num_qubits", minimum=1)
    max_qubits = check_int(max_qubits, "max_qubits", minimum=1)
    if q > max_qubits:
        raise ResourceLimitError(
            f"A density matrix of {q} qubits needs 16·4^{q} bytes = {format_bytes(16 * 4**q)} "
            f"(complex128), above max_qubits={max_qubits}. Raise max_qubits if you have the "
            "memory, or simulate pure states with the statevector simulator (16·2^q bytes)."
        )
    return q


# -- kernels on dense (N, N) arrays, inputs already validated ----------------------------
def _sandwich(R: np.ndarray, K: np.ndarray, targets: Sequence[int], q: int) -> np.ndarray:
    """K R K^† via two batched gate contractions: O(4^q · 2^k)."""
    N = 2**q
    A = _apply_tensor(R.reshape((2,) * q + (N,)), K, targets, q).reshape(N, N)
    B = _apply_tensor(np.ascontiguousarray(A.conj().T).reshape((2,) * q + (N,)), K, targets, q)
    return np.ascontiguousarray(B.reshape(N, N).conj().T)


def _diagonal_sandwich(
    R: np.ndarray, diag: np.ndarray, targets: Sequence[int], q: int
) -> np.ndarray:
    """D R D^† for a diagonal gate D: O(4^q)."""
    d = _apply_diagonal_tensor(np.ones((2,) * q, dtype=np.complex128), diag, targets, q).reshape(-1)
    return R * d[:, None] * d.conj()[None, :]


def _channel(R: np.ndarray, channel: KrausChannel, targets: Sequence[int], q: int) -> np.ndarray:
    out = np.zeros_like(R)
    for K in channel.kraus:
        out += _sandwich(R, K, targets, q)
    return out


@dataclass(frozen=True)
class DensityMeasurementResult:
    """Outcome of :meth:`DensityMatrix.measure`.

    Attributes
    ----------
    outcome : int
        0 or 1.
    probability : float
        :math:`\\mathrm{Tr}(P_m\\rho)` before the measurement.
    state : DensityMatrix
        Post-measurement state :math:`P_m\\rho P_m / p_m`.
    """

    outcome: int
    probability: float
    state: DensityMatrix

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "probability": self.probability,
            "state": self.state.to_dict(),
        }


@register(
    domain="quantum",
    tasks=("simulation", "noise", "mixed-state"),
    complexity=Complexity(
        time="O(4^q 2^k)",
        space="O(4^q)",
        notes="per k-qubit gate; a rank-r Kraus channel costs r times that",
    ),
    references=(_NC + ", §2.4, §8.2",),
)
class DensityMatrix:
    r"""An immutable q-qubit density matrix :math:`\rho`.

    Every method returns new objects; ``data`` is a read-only array, so nothing
    reachable from a ``DensityMatrix`` can break its invariants (Hermitian, PSD, unit
    trace up to floating-point rounding).

    Parameters
    ----------
    data : array_like, shape (2**q, 2**q)
        Matrix to copy. Checked to be Hermitian, of unit trace and positive
        semidefinite within ``atol`` (eigenvalue check, O(8^q)); stored symmetrized.
    atol : float, default=1e-8
        Tolerance of those checks.
    max_qubits : int, default=13
        Raise :class:`~lipipy.core.ResourceLimitError` before copying when q is larger.

    Attributes
    ----------
    num_qubits : int
    data : ndarray of complex128, read-only

    Complexity
    ----------
    Memory 16·4^q bytes. Gates: O(4^q · 2^k) time per k-qubit gate, O(4^q) extra
    memory. Purity and probabilities O(4^q); entropy, fidelity and validation O(8^q).

    References
    ----------
    Nielsen & Chuang, §2.4 and §8.2.

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.quantum import DensityMatrix, H, CX
    >>> rho = DensityMatrix.from_state([1, 0, 0, 0]).apply_gate(H, [0]).apply_gate(CX, [0, 1])
    >>> rho.probabilities().round(12)
    array([0.5, 0. , 0. , 0.5])
    >>> round(rho.purity(), 12), round(rho.partial_trace([0]).purity(), 12)
    (1.0, 0.5)
    """

    def __init__(
        self, data: np.ndarray, *, atol: float = _VALID_ATOL, max_qubits: int = MAX_DENSITY_QUBITS
    ):
        arr = np.asarray(data)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError(f"data must be a square matrix, got shape {arr.shape}")
        size = arr.shape[0]
        if size < 2 or size & (size - 1):
            raise ValueError(f"data dimension must be a power of two >= 2, got {size}")
        q = _check_density_qubits(size.bit_length() - 1, max_qubits)
        R, _ = check_square_matrix(arr, "data")
        R = np.array(R, dtype=np.complex128)  # always our own copy
        herm = float(np.max(np.abs(R - R.conj().T)))
        if herm > atol:
            raise ValueError(f"data is not Hermitian (max |rho - rho^†| = {herm:.3g})")
        R = (R + R.conj().T) / 2
        tr = float(np.trace(R).real)
        if abs(tr - 1.0) > atol:
            raise ValueError(f"data must have unit trace, got {tr}")
        lam_min = float(np.linalg.eigvalsh(R)[0])
        if lam_min < -atol:
            raise ValueError(f"data is not positive semidefinite (min eigenvalue {lam_min:.3g})")
        self._set(R, q)

    # -- construction --------------------------------------------------------------------
    def _set(self, R: np.ndarray, q: int) -> None:
        R.setflags(write=False)
        self._data = R
        self._num_qubits = q

    @classmethod
    def _trusted(cls, R: np.ndarray, q: int) -> DensityMatrix:
        """Wrap a matrix produced by a valid operation (no O(8^q) re-validation)."""
        obj = cls.__new__(cls)
        obj._set(np.ascontiguousarray(R, dtype=np.complex128), q)
        return obj

    @classmethod
    def from_state(
        cls, state: np.ndarray, *, max_qubits: int = MAX_DENSITY_QUBITS
    ) -> DensityMatrix:
        r""":math:`|\psi\rangle\langle\psi|` of a normalized statevector."""
        psi, q = check_state(state)
        _check_density_qubits(q, max_qubits)
        norm2 = float(np.vdot(psi, psi).real)
        if abs(norm2 - 1.0) > _VALID_ATOL:
            raise ValueError(f"state must be normalized (norm^2 = {norm2})")
        return cls._trusted(np.outer(psi, psi.conj()), q)

    @classmethod
    def from_mixture(
        cls,
        probabilities: Sequence[float],
        states: Sequence[np.ndarray],
        *,
        max_qubits: int = MAX_DENSITY_QUBITS,
    ) -> DensityMatrix:
        r""":math:`\sum_i p_i |\psi_i\rangle\langle\psi_i|` (normalized states, p sums to 1)."""
        p = np.asarray(probabilities, dtype=np.float64)
        if p.ndim != 1 or p.size == 0 or p.size != len(states):
            raise ValueError("probabilities and states must be non-empty and of equal length")
        if not np.all(np.isfinite(p)) or np.any(p < 0) or abs(p.sum() - 1.0) > _VALID_ATOL:
            raise ValueError("probabilities must be non-negative and sum to 1")
        pure = [cls.from_state(s, max_qubits=max_qubits) for s in states]
        if len({d.num_qubits for d in pure}) != 1:
            raise ValueError("all states must have the same number of qubits")
        R = sum(pi * d._data for pi, d in zip(p, pure, strict=True))
        return cls._trusted(R, pure[0].num_qubits)

    @classmethod
    def maximally_mixed(
        cls, num_qubits: int, *, max_qubits: int = MAX_DENSITY_QUBITS
    ) -> DensityMatrix:
        r"""The maximally mixed state :math:`I / 2^q`."""
        q = _check_density_qubits(num_qubits, max_qubits)
        return cls._trusted(np.eye(2**q, dtype=np.complex128) / 2**q, q)

    # -- accessors -----------------------------------------------------------------------
    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def data(self) -> np.ndarray:
        return self._data

    def __repr__(self) -> str:
        return f"DensityMatrix(num_qubits={self._num_qubits}, purity={self.purity():.6g})"

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict; the matrix becomes nested ``[re, im]`` pairs."""
        return {
            "num_qubits": self._num_qubits,
            "data": np.stack([self._data.real, self._data.imag], axis=-1).tolist(),
        }

    # -- evolution -----------------------------------------------------------------------
    def apply_gate(self, gate: np.ndarray, targets: Sequence[int] | int) -> DensityMatrix:
        r""":math:`U\rho U^\dagger` for a k-qubit unitary on ``targets`` (``targets[0]`` = LSB).

        Unitarity is required (atol 1e-10) so the result is again a density matrix.
        Time O(4^q · 2^k), extra memory O(4^q).
        """
        from lipipy.quantum.gates import is_unitary

        U, k = check_square_matrix(gate)
        tgts = check_qubits(targets, self._num_qubits)
        if len(tgts) != k:
            raise ValueError(f"a {k}-qubit gate needs {k} targets, got {len(tgts)}")
        if not is_unitary(U):
            raise ValueError("gate is not unitary; use apply_channel for non-unitary maps")
        return DensityMatrix._trusted(
            _sandwich(self._data, U, tgts, self._num_qubits), self._num_qubits
        )

    def apply_channel(self, channel: KrausChannel, targets: Sequence[int] | int) -> DensityMatrix:
        r""":math:`\sum_i K_i\rho K_i^\dagger` on ``targets``; time O(r · 4^q · 2^k)."""
        if not isinstance(channel, KrausChannel):
            raise TypeError(f"channel must be a KrausChannel, got {type(channel).__name__}")
        tgts = check_qubits(targets, self._num_qubits)
        if len(tgts) != channel.num_qubits:
            raise ValueError(f"a {channel.num_qubits}-qubit channel needs that many targets")
        return DensityMatrix._trusted(
            _channel(self._data, channel, tgts, self._num_qubits), self._num_qubits
        )

    def evolve(self, circuit: Any, noise_model: Any = None) -> DensityMatrix:
        """Run ``circuit`` (with optional noise) from this state; see ``Circuit.run_density``."""
        from lipipy.quantum.circuit import Circuit

        if not isinstance(circuit, Circuit):
            raise TypeError(f"circuit must be a Circuit, got {type(circuit).__name__}")
        return circuit.run_density(self, noise_model=noise_model, max_qubits=self._num_qubits)

    # -- analysis ------------------------------------------------------------------------
    def trace(self) -> float:
        """:math:`\\mathrm{Tr}\\,\\rho` (1 up to rounding)."""
        return float(np.trace(self._data).real)

    def purity(self) -> float:
        r""":math:`\mathrm{Tr}\,\rho^2 = \sum_{ij} |\rho_{ij}|^2` in [1/2^q, 1]; O(4^q)."""
        return float(np.sum(np.abs(self._data) ** 2))

    def probabilities(self) -> np.ndarray:
        r"""Born probabilities :math:`\rho_{ii}` of the basis states (clipped at 0)."""
        return np.clip(np.diagonal(self._data).real, 0.0, None)

    def sample_counts(self, shots: int = 1024, random_state: RandomState = None) -> dict[str, int]:
        """Measure all qubits ``shots`` times; bitstrings most-significant qubit first."""
        shots = check_int(shots, "shots", minimum=1)
        p = self.probabilities()
        counts = check_random_state(random_state).multinomial(shots, p / p.sum())
        q = self._num_qubits
        return {format(int(i), f"0{q}b"): int(counts[i]) for i in np.flatnonzero(counts)}

    def measure(self, qubit: int, random_state: RandomState = None) -> DensityMeasurementResult:
        r"""Projective Z measurement of one qubit: outcome m w.p. :math:`\mathrm{Tr}(P_m\rho)`."""
        q = self._num_qubits
        (qubit,) = check_qubits(qubit, q, "qubit")
        p_diag = self.probabilities()
        bit = (np.arange(2**q) >> qubit) & 1
        p1 = float(p_diag[bit == 1].sum() / p_diag.sum())
        outcome = int(check_random_state(random_state).random() < p1)
        prob = p1 if outcome else 1.0 - p1
        mask = (bit == outcome).astype(np.float64)
        R = self._data * mask[:, None] * mask[None, :] / prob
        return DensityMeasurementResult(outcome, prob, DensityMatrix._trusted(R, q))

    def partial_trace(self, keep: Sequence[int] | int) -> DensityMatrix:
        r"""Reduced state :math:`\mathrm{Tr}_B\rho` on ``keep`` (``keep[0]`` = LSB); O(4^q)."""
        q = self._num_qubits
        kept = check_qubits(keep, q, "keep")
        rows = [q - 1 - t for t in reversed(kept)]
        rest = [a for a in range(q) if a not in rows]
        K, Rn = 2 ** len(kept), 2 ** (q - len(kept))
        T = self._data.reshape((2,) * (2 * q))
        T = T.transpose(rows + rest + [a + q for a in rows] + [a + q for a in rest])
        reduced = np.einsum("ijkj->ik", T.reshape(K, Rn, K, Rn))
        return DensityMatrix._trusted(reduced, len(kept))

    def von_neumann_entropy(self, base: float = 2.0) -> float:
        r""":math:`S(\rho) = -\sum_i \lambda_i \log \lambda_i` (bits by default); O(8^q)."""
        if isinstance(base, bool) or not isinstance(base, (int, float)) or not base > 1:
            raise ValueError(f"base must be a real number > 1, got {base!r}")
        lam = np.linalg.eigvalsh(self._data)
        lam = lam[lam > 1e-15]
        return float(max(-np.sum(lam * np.log(lam)) / np.log(base), 0.0))

    def fidelity(self, other: DensityMatrix | np.ndarray) -> float:
        r"""Uhlmann fidelity :math:`F = \big(\mathrm{Tr}\sqrt{\sqrt\rho\,\sigma\sqrt\rho}\big)^2`.

        Squared convention, as :func:`lipipy.quantum.fidelity` and Qiskit's
        ``state_fidelity``. For a statevector ``other`` this is exactly
        :math:`\langle\psi|\rho|\psi\rangle` (O(4^q)); for two density matrices it uses
        eigendecompositions (O(8^q)), accurate to ~1e-7 when both are rank deficient
        (square roots of rounding-level eigenvalues).
        """
        q = self._num_qubits
        if isinstance(other, DensityMatrix):
            if other.num_qubits != q:
                raise ValueError(f"states have {q} and {other.num_qubits} qubits")
            w, V = np.linalg.eigh(self._data)
            sqrt_rho = (V * np.sqrt(np.clip(w, 0.0, None))) @ V.conj().T
            M = sqrt_rho @ other._data @ sqrt_rho
            ev = np.clip(np.linalg.eigvalsh((M + M.conj().T) / 2), 0.0, None)
            return float(min(np.sum(np.sqrt(ev)) ** 2, 1.0))
        psi, nq = check_state(other, "other")
        if nq != q:
            raise ValueError(f"states have {q} and {nq} qubits")
        if abs(np.linalg.norm(psi) - 1.0) > _VALID_ATOL:
            raise ValueError("other must be normalized")
        return float(min(max(np.vdot(psi, self._data @ psi).real, 0.0), 1.0))
