"""Statevector kernels: gate application vs Kronecker products, measurement, entropy."""

import itertools
import math

import numpy as np
import pytest

from lipipy.bench import estimate_complexity
from lipipy.core import ResourceLimitError
from lipipy.quantum import (
    CX,
    H,
    X,
    apply_diagonal,
    apply_gate,
    basis_state,
    entanglement_entropy,
    expectation_pauli,
    measure_qubit,
    probabilities,
    reduced_density_matrix,
    sample_counts,
    zero_state,
)
from tests.quantum._helpers import full_operator, pauli_matrix, random_state, random_unitary

ATOL = 1e-12


@pytest.mark.parametrize("q", range(1, 7))
def test_apply_gate_matches_kronecker_construction(q, rng):
    for _ in range(12):
        k = int(rng.integers(1, min(q, 3) + 1))
        targets = [int(t) for t in rng.permutation(q)[:k]]
        U = random_unitary(2**k, rng)
        psi = random_state(q, rng)
        expected = full_operator(U, targets, q) @ psi
        np.testing.assert_allclose(apply_gate(psi, U, targets), expected, atol=ATOL)


def test_apply_gate_all_target_orders_and_non_unitary(rng):
    q = 4
    psi = random_state(q, rng)
    M = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))  # not unitary
    for targets in itertools.permutations(range(q), 2):
        np.testing.assert_allclose(
            apply_gate(psi, M, targets), full_operator(M, targets, q) @ psi, atol=ATOL
        )


def test_apply_gate_single_qubit_is_kron_sandwich(rng):
    q = 5
    psi = random_state(q, rng)
    U = random_unitary(2, rng)
    for j in range(q):
        op = np.kron(np.kron(np.eye(2 ** (q - 1 - j)), U), np.eye(2**j))
        np.testing.assert_allclose(apply_gate(psi, U, j), op @ psi, atol=ATOL)


def test_apply_gate_cx_convention_and_purity():
    psi = basis_state(2, 0b01)  # qubit 0 = 1
    out = apply_gate(psi, CX, [0, 1])  # control 0 -> flips qubit 1
    np.testing.assert_array_equal(out, basis_state(2, 0b11))
    np.testing.assert_array_equal(apply_gate(psi, CX, [1, 0]), psi)  # control 1 is 0
    np.testing.assert_array_equal(psi, basis_state(2, 0b01))  # input untouched


def test_apply_diagonal_matches_dense(rng):
    q = 5
    psi = random_state(q, rng)
    for k in (1, 2, 3):
        targets = [int(t) for t in rng.permutation(q)[:k]]
        d = np.exp(1j * rng.uniform(0, 2 * np.pi, 2**k))
        np.testing.assert_allclose(
            apply_diagonal(psi, d, targets), apply_gate(psi, np.diag(d), targets), atol=ATOL
        )


def test_apply_gate_bad_inputs():
    psi = zero_state(3)
    with pytest.raises(ValueError, match="duplicate"):
        apply_gate(psi, CX, [1, 1])
    with pytest.raises(ValueError, match="out of range"):
        apply_gate(psi, H, [3])
    with pytest.raises(ValueError, match="needs 2 targets"):
        apply_gate(psi, CX, [0])
    with pytest.raises(ValueError, match="power of two"):
        apply_gate(np.ones(3), H, [0])
    with pytest.raises(ValueError, match="square"):
        apply_gate(psi, np.ones((2, 4)), [0])
    with pytest.raises(TypeError):
        apply_gate(psi, H, ["0"])
    with pytest.raises(ValueError, match="empty"):
        apply_gate(psi, H, [])
    with pytest.raises(ValueError, match="needs 1 targets"):
        apply_diagonal(psi, [1, 1], [0, 1])


def test_probabilities(rng):
    psi = random_state(4, rng)
    p = probabilities(psi)
    assert p.dtype == np.float64
    np.testing.assert_allclose(p, np.abs(psi) ** 2, atol=0)
    assert p.sum() == pytest.approx(1.0, abs=ATOL)


def test_sample_counts_reproducible_and_formatted(rng):
    psi = random_state(3, rng)
    a = sample_counts(psi, shots=500, random_state=42)
    assert a == sample_counts(psi, shots=500, random_state=42)
    assert sum(a.values()) == 500
    assert all(len(k) == 3 and set(k) <= {"0", "1"} for k in a)
    assert list(a) == sorted(a)
    # bitstrings are most-significant qubit first: qubit 0 = 1 -> "001"
    assert sample_counts(basis_state(3, 1), 7, random_state=0) == {"001": 7}
    assert sample_counts(basis_state(3, 4), 7, random_state=0) == {"100": 7}


def test_sample_counts_chi_square(rng):
    q, shots = 4, 200_000
    psi = random_state(q, rng)
    p = probabilities(psi)
    counts = sample_counts(psi, shots=shots, random_state=2024)
    observed = np.array([counts.get(format(i, "04b"), 0) for i in range(2**q)])
    expected = shots * p
    chi2 = float(np.sum((observed - expected) ** 2 / expected))
    dof = 2**q - 1
    # chi2 has mean dof and sd sqrt(2 dof); 5 sd is a fixed, seed-independent bound
    assert chi2 < dof + 5 * math.sqrt(2 * dof)
    stats = pytest.importorskip("scipy.stats")
    assert stats.chisquare(observed, expected).pvalue > 1e-4


def test_sample_counts_bad_inputs():
    with pytest.raises(ValueError, match="normalized"):
        sample_counts([1, 1], 10)
    with pytest.raises(ValueError, match="shots"):
        sample_counts([1, 0], 0)


@pytest.mark.parametrize("q", [1, 2, 3, 4])
def test_expectation_pauli_matches_matrix(q, rng):
    psi = random_state(q, rng)
    for letters in itertools.islice(itertools.product("IXYZ", repeat=q), 0, None, 3):
        label = "".join(letters)
        expected = np.vdot(psi, pauli_matrix(label) @ psi).real
        assert expectation_pauli(psi, label) == pytest.approx(expected, abs=ATOL)


def test_expectation_pauli_qubit_order_and_errors():
    psi = basis_state(3, 0b001)  # only qubit 0 is 1
    assert expectation_pauli(psi, "IIZ") == -1.0
    assert expectation_pauli(psi, "ZII") == 1.0
    with pytest.raises(ValueError, match="length 3"):
        expectation_pauli(psi, "ZZ")
    with pytest.raises(ValueError, match="I, X, Y, Z"):
        expectation_pauli(psi, "ZzI")
    with pytest.raises(TypeError):
        expectation_pauli(psi, 3)


def test_measure_qubit_collapse_and_statistics(rng):
    q = 3
    psi = random_state(q, rng)
    p1 = sum(abs(psi[i]) ** 2 for i in range(2**q) if (i >> 1) & 1)
    outcomes = []
    gen = np.random.default_rng(5)
    for _ in range(4000):
        r = measure_qubit(psi, 1, random_state=gen)
        outcomes.append(r.outcome)
        expected_p = p1 if r.outcome else 1 - p1
        assert r.probability == pytest.approx(expected_p, abs=ATOL)
    freq = np.mean(outcomes)
    assert abs(freq - p1) < 5 * math.sqrt(p1 * (1 - p1) / 4000)

    r = measure_qubit(psi, 1, random_state=3)
    assert r.state.dtype == np.complex128
    assert np.linalg.norm(r.state) == pytest.approx(1.0, abs=ATOL)
    mask = ((np.arange(8) >> 1) & 1) == r.outcome
    np.testing.assert_allclose(r.state[mask], psi[mask] / math.sqrt(r.probability), atol=ATOL)
    assert np.all(r.state[~mask] == 0)
    # repeated measurement is deterministic
    assert measure_qubit(r.state, 1, random_state=99).outcome == r.outcome
    d = r.to_dict()
    assert d["outcome"] == r.outcome and len(d["state"]) == 8


def test_measure_qubit_reproducible_and_errors(rng):
    psi = random_state(2, rng)
    assert measure_qubit(psi, 0, 11).outcome == measure_qubit(psi, 0, 11).outcome
    with pytest.raises(ValueError, match="normalized"):
        measure_qubit([1, 1], 0)
    with pytest.raises(ValueError, match="out of range"):
        measure_qubit(psi, 2)


def _explicit_partial_trace(psi, keep, q):
    """rho_A[a, a'] = sum_b psi[(a, b)] conj(psi[(a', b)]) by looping over indices."""
    rest = [j for j in range(q) if j not in keep]
    rho = np.zeros((2 ** len(keep), 2 ** len(keep)), dtype=complex)
    for i in range(2**q):
        for i2 in range(2**q):
            if all(((i >> j) & 1) == ((i2 >> j) & 1) for j in rest):
                a = sum(((i >> t) & 1) << n for n, t in enumerate(keep))
                a2 = sum(((i2 >> t) & 1) << n for n, t in enumerate(keep))
                rho[a, a2] += psi[i] * np.conj(psi[i2])
    return rho


def test_reduced_density_matrix(rng):
    q = 4
    psi = random_state(q, rng)
    for keep in ([0], [2], [1, 3], [3, 1], [0, 2, 3], [2, 0, 1, 3]):
        rho = reduced_density_matrix(psi, keep)
        np.testing.assert_allclose(rho, _explicit_partial_trace(psi, keep, q), atol=ATOL)
        np.testing.assert_allclose(rho, rho.conj().T, atol=ATOL)
        assert np.trace(rho).real == pytest.approx(1.0, abs=ATOL)
        assert np.linalg.eigvalsh(rho).min() > -ATOL
    np.testing.assert_allclose(
        reduced_density_matrix(psi, [0, 1, 2, 3]), np.outer(psi, psi.conj()), atol=ATOL
    )


def test_entanglement_entropy():
    bell = np.array([1, 0, 0, 1]) / math.sqrt(2)
    assert entanglement_entropy(bell, [0]) == pytest.approx(1.0, abs=ATOL)
    assert entanglement_entropy(bell, [1], base=math.e) == pytest.approx(math.log(2), abs=ATOL)
    ghz = np.zeros(16)
    ghz[[0, 15]] = 1 / math.sqrt(2)
    for sub in ([0], [1, 2], [0, 3], [0, 1, 2]):
        assert entanglement_entropy(ghz, sub) == pytest.approx(1.0, abs=ATOL)
    assert entanglement_entropy(ghz, [0, 1, 2, 3]) == 0.0


def test_entanglement_entropy_product_and_symmetry(rng):
    a, b, c = random_state(1, rng), random_state(2, rng), random_state(1, rng)
    product = np.kron(np.kron(a, b), c)
    for sub in ([0], [3], [1, 2], [0, 3]):
        assert entanglement_entropy(product, sub) == pytest.approx(0.0, abs=1e-10)
    psi = random_state(5, rng)
    for sub in ([0], [1, 3], [0, 2, 4]):
        comp = [j for j in range(5) if j not in sub]
        s = entanglement_entropy(psi, sub)
        assert s == pytest.approx(entanglement_entropy(psi, comp), abs=1e-10)
        lam = np.linalg.eigvalsh(reduced_density_matrix(psi, sub))
        lam = lam[lam > 1e-15]
        assert s == pytest.approx(-np.sum(lam * np.log2(lam)), abs=1e-10)
        assert 0 <= s <= min(len(sub), len(comp))


def test_entropy_bad_inputs():
    with pytest.raises(ValueError, match="base"):
        entanglement_entropy([1, 0, 0, 0], [0], base=1)
    with pytest.raises(ValueError, match="normalized"):
        entanglement_entropy([1, 0, 0, 1], [0])


def test_apply_gate_memory_is_linear_in_state_size():
    """Declared extra space O(2^q): doubling the state size doubles peak memory."""
    est = estimate_complexity(
        lambda psi, U: apply_gate(psi, U, [0, 3]),
        lambda n, rng: (random_state(int(math.log2(n)), rng), np.kron(X, H)),
        sizes=(2**14, 2**15, 2**16, 2**17, 2**18),
        repeats=1,
    )
    assert est.space_exponent == pytest.approx(1.0, abs=0.15)


def test_qubit_guard_does_not_allocate_for_40_qubits():
    import tracemalloc

    tracemalloc.start()
    try:
        with pytest.raises(ResourceLimitError):
            zero_state(40)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1_000_000


@pytest.mark.complexity
def test_apply_gate_time_is_linear_in_state_size():
    est = estimate_complexity(
        lambda psi, U: apply_gate(psi, U, [1, 4]),
        lambda n, rng: (random_state(int(math.log2(n)), rng), random_unitary(4, rng)),
        sizes=(2**16, 2**17, 2**18, 2**19, 2**20),
    )
    assert est.time_exponent < 1.3
