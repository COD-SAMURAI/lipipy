"""PauliSum: matrix-free expectations vs dense matrices, arithmetic, model Hamiltonians."""

import itertools

import numpy as np
import pytest

from lipipy.core import ResourceLimitError
from lipipy.quantum import (
    H2_COEFFICIENTS,
    H2_NUCLEAR_REPULSION,
    DensityMatrix,
    PauliSum,
    basis_state,
    expectation_pauli,
    h2_hamiltonian,
    ising_chain,
    maxcut_hamiltonian,
)
from tests.quantum._helpers import pauli_matrix, random_density, random_state

ATOL = 1e-12


def random_paulisum(q, terms, rng):
    labels = ["".join(rng.choice(list("IXYZ"), size=q)) for _ in range(terms)]
    return PauliSum(
        [(lab, float(c)) for lab, c in zip(labels, rng.normal(size=terms), strict=True)]
    )


@pytest.mark.parametrize("q", [1, 2, 3, 4, 5])
def test_expectation_matches_dense_matrix(q, rng):
    for _ in range(5):
        H = random_paulisum(q, 12, rng)
        M = H.to_matrix()
        np.testing.assert_allclose(M, M.conj().T, atol=ATOL)
        dense = sum(c * pauli_matrix(lab) for lab, c in H.terms)
        np.testing.assert_allclose(M, dense, atol=ATOL)
        psi = random_state(q, rng)
        assert H.expectation(psi) == pytest.approx(np.vdot(psi, M @ psi).real, abs=ATOL)
        R = random_density(q, rng)
        assert H.expectation(DensityMatrix(R)) == pytest.approx(np.trace(M @ R).real, abs=ATOL)


def test_single_terms_match_expectation_pauli(rng):
    psi = random_state(3, rng)
    for letters in itertools.product("IXYZ", repeat=3):
        label = "".join(letters)
        assert PauliSum({label: 1.0}).expectation(psi) == pytest.approx(
            expectation_pauli(psi, label), abs=ATOL
        )


def test_arithmetic(rng):
    A, B = random_paulisum(3, 6, rng), random_paulisum(3, 6, rng)
    for expr, dense in (
        (A + B, A.to_matrix() + B.to_matrix()),
        (A - B, A.to_matrix() - B.to_matrix()),
        (2.5 * A, 2.5 * A.to_matrix()),
        (A * -1.5, -1.5 * A.to_matrix()),
        (-A, -A.to_matrix()),
    ):
        np.testing.assert_allclose(expr.to_matrix(), dense, atol=ATOL)
    zero = A - A
    assert zero.num_terms == 0 and zero.num_qubits == 3
    assert zero.expectation(random_state(3, rng)) == 0.0
    merged = PauliSum([("XZ", 1.0), ("XZ", 2.0), ("II", 0.0)])
    assert merged.terms == (("XZ", 3.0),)
    assert "3*XZ" in repr(merged)
    assert merged.to_dict() == {"num_qubits": 2, "terms": [["XZ", 3.0]]}


def test_errors():
    with pytest.raises(ValueError, match="length"):
        PauliSum({"XZ": 1.0, "X": 1.0})
    with pytest.raises(ValueError, match="IXYZ"):
        PauliSum({"XA": 1.0})
    with pytest.raises(TypeError):
        PauliSum({"XZ": 1j})
    with pytest.raises(ValueError, match="finite"):
        PauliSum({"XZ": np.inf})
    with pytest.raises(ValueError, match="at least one"):
        PauliSum({})
    with pytest.raises(TypeError):
        PauliSum([("XZ",)])
    with pytest.raises(ValueError, match="combine"):
        PauliSum({"X": 1.0}) + PauliSum({"XX": 1.0})
    with pytest.raises(TypeError):
        PauliSum({"X": 1.0}) + 1.0
    with pytest.raises(ValueError, match="qubits"):
        PauliSum({"X": 1.0}).expectation(random_state(2, np.random.default_rng(0)))
    with pytest.raises(ResourceLimitError):
        PauliSum({"Z" * 13: 1.0}).to_matrix()


def test_large_observable_expectation_without_matrix():
    q = 16
    H = ising_chain(q, J=1.0, h=0.0)
    assert H.expectation(basis_state(q, 0)) == pytest.approx(-(q - 1))


def _ising_dense(q, J, h, periodic):
    X = np.array([[0, 1], [1, 0]])
    Z = np.diag([1, -1])

    def op(single):
        out = np.array([[1.0]])
        for pos in reversed(range(q)):  # leftmost factor = qubit q-1
            out = np.kron(out, single.get(pos, np.eye(2)))
        return out

    M = sum(-J * op({i: Z, (i + 1) % q: Z}) for i in range(q - 1 + int(periodic)))
    return M + sum(-h * op({i: X}) for i in range(q))


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("q", [3, 4, 5])
def test_ising_chain_matrix(q, periodic):
    J, h = 0.8, 1.3
    H = ising_chain(q, J, h, periodic=periodic)
    np.testing.assert_allclose(H.to_matrix(), _ising_dense(q, J, h, periodic), atol=ATOL)
    assert H.num_terms == q + (q if periodic else q - 1)


def test_ising_degenerate_ground_state_at_zero_field():
    w = np.linalg.eigvalsh(ising_chain(4, J=1.0, h=0.0).to_matrix())
    assert w[0] == pytest.approx(-3.0) and w[1] == pytest.approx(-3.0) and w[2] > -3.0 + 1


def test_ising_errors():
    with pytest.raises(ValueError, match="periodic"):
        ising_chain(2, periodic=True)
    with pytest.raises(ValueError):
        ising_chain(1)


def test_h2_hamiltonian_ground_energy():
    """Electronic ground energy -1.857275030 Ha; total with 1/R = -1.137306036 Ha (FCI)."""
    electronic = h2_hamiltonian(include_nuclear_repulsion=False)
    total = h2_hamiltonian()
    assert electronic.num_terms == 5 and total.num_terms == 5
    assert dict(electronic.terms) == dict(H2_COEFFICIENTS)
    e_el = np.linalg.eigvalsh(electronic.to_matrix())[0]
    e_tot = np.linalg.eigvalsh(total.to_matrix())[0]
    assert e_el == pytest.approx(-1.857275030202, abs=1e-9)
    assert e_tot == pytest.approx(-1.137306035753, abs=1e-9)
    assert e_tot - e_el == pytest.approx(H2_NUCLEAR_REPULSION, abs=1e-12)
    assert H2_NUCLEAR_REPULSION == pytest.approx(0.719968994449, abs=1e-11)
    with pytest.raises(TypeError):
        h2_hamiltonian("yes")


def test_maxcut_hamiltonian_diagonal_is_cut_value(rng):
    for _ in range(10):
        n = int(rng.integers(2, 7))
        pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
        chosen = rng.permutation(len(pairs))[: int(rng.integers(1, len(pairs) + 1))]
        edges = [pairs[i] for i in chosen]
        w = rng.uniform(0.1, 2.0, size=len(edges))
        C = maxcut_hamiltonian(edges, n, weights=w)
        M = C.to_matrix()
        np.testing.assert_allclose(M, np.diag(np.diag(M)), atol=0)
        for z in range(2**n):
            cut = sum(wt for (a, b), wt in zip(edges, w, strict=True) if ((z >> a) ^ (z >> b)) & 1)
            assert M[z, z].real == pytest.approx(cut, abs=ATOL)


def test_maxcut_errors():
    with pytest.raises(ValueError, match="self-loop"):
        maxcut_hamiltonian([(0, 0)])
    with pytest.raises(ValueError, match="duplicate"):
        maxcut_hamiltonian([(0, 1), (1, 0)])
    with pytest.raises(ValueError, match="empty"):
        maxcut_hamiltonian([])
    with pytest.raises(ValueError, match="out of range"):
        maxcut_hamiltonian([(0, 3)], num_nodes=3)
    with pytest.raises(ValueError, match="one entry per edge"):
        maxcut_hamiltonian([(0, 1)], weights=[1.0, 2.0])
    with pytest.raises(ValueError, match="pair"):
        maxcut_hamiltonian([(0, 1, 2)])
