"""Independent reference constructions used by the quantum tests.

These build full 2^q x 2^q operators with Kronecker products and permutation
matrices, deliberately sharing no code with lipipy.quantum's tensor kernels.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def full_operator(gate: np.ndarray, targets: Sequence[int], num_qubits: int) -> np.ndarray:
    """P^T (I ⊗ U) P, where P relabels qubits so that targets become qubits 0..k-1."""
    k = len(targets)
    order = list(targets) + [j for j in range(num_qubits) if j not in targets]
    N = 2**num_qubits
    P = np.zeros((N, N))
    for i in range(N):
        new = sum(((i >> old) & 1) << new_pos for new_pos, old in enumerate(order))
        P[new, i] = 1.0
    return P.T @ np.kron(np.eye(2 ** (num_qubits - k)), gate) @ P


def pauli_matrix(label: str) -> np.ndarray:
    """Kronecker product in written order: the rightmost letter acts on qubit 0."""
    out = np.array([[1.0 + 0j]])
    for letter in label:
        out = np.kron(out, PAULI[letter])
    return out


def random_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random unitary via QR of a complex Gaussian matrix (Mezzadri 2007)."""
    Z = (rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))) / np.sqrt(2)
    Q, R = np.linalg.qr(Z)
    d = np.diag(R)
    return Q * (d / np.abs(d))


def random_state(num_qubits: int, rng: np.random.Generator) -> np.ndarray:
    z = rng.standard_normal(2**num_qubits) + 1j * rng.standard_normal(2**num_qubits)
    return z / np.linalg.norm(z)


def random_density(
    num_qubits: int, rng: np.random.Generator, rank: int | None = None
) -> np.ndarray:
    """Random density matrix G G^† / Tr(G G^†) with G of shape (2^q, rank)."""
    d = 2**num_qubits
    r = d if rank is None else rank
    G = rng.standard_normal((d, r)) + 1j * rng.standard_normal((d, r))
    rho = G @ G.conj().T
    return rho / np.trace(rho).real


def partial_trace_out(rho: np.ndarray, keep: Sequence[int], num_qubits: int) -> np.ndarray:
    """Reference partial trace by explicit index loops (keep[0] = least significant)."""
    rest = [j for j in range(num_qubits) if j not in keep]
    K = 2 ** len(keep)
    out = np.zeros((K, K), dtype=complex)
    N = 2**num_qubits
    for i in range(N):
        for i2 in range(N):
            if all(((i >> j) & 1) == ((i2 >> j) & 1) for j in rest):
                a = sum(((i >> t) & 1) << n for n, t in enumerate(keep))
                a2 = sum(((i2 >> t) & 1) << n for n, t in enumerate(keep))
                out[a, a2] += rho[i, i2]
    return out
