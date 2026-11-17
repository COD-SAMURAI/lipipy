"""Density matrices: gate/channel action vs dense algebra, analysis functions, guards."""

import math
import tracemalloc

import numpy as np
import pytest

from lipipy.bench import estimate_complexity
from lipipy.core import ResourceLimitError
from lipipy.quantum import (
    CX,
    MAX_DENSITY_QUBITS,
    Circuit,
    DensityMatrix,
    H,
    NoiseModel,
    amplitude_damping,
    bit_flip,
    depolarizing,
    fidelity,
    phase_damping,
    reduced_density_matrix,
)
from tests.quantum._helpers import (
    full_operator,
    partial_trace_out,
    random_density,
    random_state,
    random_unitary,
)
from tests.quantum.test_circuit import random_circuit

ATOL = 1e-12


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("q", [1, 2, 3, 4, 5])
def test_noiseless_density_equals_statevector_outer_product(q, rng):
    for _ in range(4):
        c, _ = random_circuit(q, 30, rng)
        psi = c.run()
        np.testing.assert_allclose(c.run_density().data, np.outer(psi, psi.conj()), atol=ATOL)
        psi0 = random_state(q, rng)
        out = c.run(psi0)
        np.testing.assert_allclose(c.run_density(psi0).data, np.outer(out, out.conj()), atol=ATOL)


@pytest.mark.parametrize("q", [1, 2, 3, 4])
def test_apply_gate_and_channel_match_dense_algebra(q, rng):
    for _ in range(6):
        k = int(rng.integers(1, min(q, 2) + 1))
        targets = [int(t) for t in rng.permutation(q)[:k]]
        R = random_density(q, rng, rank=int(rng.integers(1, 2**q + 1)))
        rho = DensityMatrix(R)
        U = random_unitary(2**k, rng)
        F = full_operator(U, targets, q)
        np.testing.assert_allclose(rho.apply_gate(U, targets).data, F @ R @ F.conj().T, atol=ATOL)
        ch = depolarizing(float(rng.uniform()), k)
        expected = sum(
            full_operator(K, targets, q) @ R @ full_operator(K, targets, q).conj().T
            for K in ch.kraus
        )
        np.testing.assert_allclose(rho.apply_channel(ch, targets).data, expected, atol=ATOL)


def test_initial_density_input_and_evolve(rng):
    c, ref = random_circuit(3, 20, rng)
    R = random_density(3, rng)
    rho = DensityMatrix(R)
    out = c.run_density(rho)
    np.testing.assert_allclose(out.data, ref @ R @ ref.conj().T, atol=ATOL)
    np.testing.assert_allclose(rho.evolve(c).data, out.data, atol=0)
    np.testing.assert_array_equal(rho.data, (R + R.conj().T) / 2)  # input untouched
    with pytest.raises(TypeError):
        rho.evolve("circuit")
    with pytest.raises(ValueError, match="2 qubits, the circuit has 3"):
        c.run_density(DensityMatrix.maximally_mixed(2))
    with pytest.raises(TypeError):
        c.run_density(noise_model="noisy")


@pytest.mark.filterwarnings("error")
def test_noisy_circuits_stay_valid_states(rng):
    """Trace 1, Hermitian, PSD and purity in [1/d, 1] after random noisy circuits."""
    for seed in range(15):
        g = np.random.default_rng(seed)
        q = int(g.integers(1, 5))
        c, _ = random_circuit(q, 20, g)
        model = (
            NoiseModel()
            .add(depolarizing(float(g.uniform())))
            .add(amplitude_damping(float(g.uniform())), gates=["h", "x", "u"])
            .add(phase_damping(float(g.uniform())), gates="rz")
        )
        rho = c.run_density(noise_model=model)
        assert rho.trace() == pytest.approx(1.0, abs=1e-12)
        np.testing.assert_allclose(rho.data, rho.data.conj().T, atol=1e-12)
        assert np.linalg.eigvalsh(rho.data).min() > -1e-12
        assert 1 / 2**q - 1e-12 <= rho.purity() <= 1 + 1e-12
        assert rho.probabilities().sum() == pytest.approx(1.0, abs=1e-12)


def test_full_strength_noise_extremes():
    c = Circuit(2).h(0).cx(0, 1).x(1)
    mixed = c.run_density(noise_model=NoiseModel().add(depolarizing(1.0)))
    np.testing.assert_allclose(mixed.data, np.eye(4) / 4, atol=ATOL)
    assert mixed.purity() == pytest.approx(0.25, abs=ATOL)
    assert mixed.von_neumann_entropy() == pytest.approx(2.0, abs=1e-10)
    ground = c.run_density(noise_model=NoiseModel().add(amplitude_damping(1.0)))
    np.testing.assert_allclose(ground.data, np.diag([1.0, 0, 0, 0]), atol=ATOL)
    zero = c.run_density(noise_model=NoiseModel().add(depolarizing(0.0)))
    psi = c.run()
    np.testing.assert_allclose(zero.data, np.outer(psi, psi.conj()), atol=ATOL)


def test_constructors(rng):
    psi = random_state(2, rng)
    np.testing.assert_allclose(DensityMatrix.from_state(psi).data, np.outer(psi, psi.conj()))
    a, b = random_state(2, rng), random_state(2, rng)
    mix = DensityMatrix.from_mixture([0.25, 0.75], [a, b])
    expected = 0.25 * np.outer(a, a.conj()) + 0.75 * np.outer(b, b.conj())
    np.testing.assert_allclose(mix.data, expected, atol=ATOL)
    mm = DensityMatrix.maximally_mixed(3)
    np.testing.assert_allclose(mm.data, np.eye(8) / 8)
    assert mm.purity() == pytest.approx(1 / 8)
    one = DensityMatrix([[1, 0], [0, 0]])  # one-qubit edge case
    assert one.num_qubits == 1 and one.purity() == 1.0
    assert "DensityMatrix(num_qubits=1" in repr(one)
    d = one.to_dict()
    assert d["num_qubits"] == 1 and d["data"][0][0] == [1.0, 0.0]


def test_immutability(rng):
    R = random_density(2, rng)
    rho = DensityMatrix(R)
    R[0, 0] = 5.0
    assert rho.data[0, 0] != 5.0
    with pytest.raises(ValueError):
        rho.data[0, 0] = 1.0
    for derived in (rho.apply_gate(H, 0), rho.partial_trace([1]), rho.measure(0, 0).state):
        with pytest.raises(ValueError):
            derived.data[0, 0] = 1.0


def test_validation_errors(rng):
    with pytest.raises(ValueError, match="Hermitian"):
        DensityMatrix([[0.5, 1], [0, 0.5]])
    with pytest.raises(ValueError, match="unit trace"):
        DensityMatrix(np.eye(2))
    with pytest.raises(ValueError, match="positive semidefinite"):
        DensityMatrix([[1.5, 0], [0, -0.5]])
    with pytest.raises(ValueError, match="square"):
        DensityMatrix(np.ones((2, 4)))
    with pytest.raises(ValueError, match="power of two"):
        DensityMatrix(np.eye(3) / 3)
    with pytest.raises(ValueError, match="normalized"):
        DensityMatrix.from_state([1, 1])
    with pytest.raises(ValueError, match="sum to 1"):
        DensityMatrix.from_mixture([0.5, 0.6], [[1, 0], [0, 1]])
    with pytest.raises(ValueError, match="same number"):
        DensityMatrix.from_mixture([0.5, 0.5], [[1, 0], [1, 0, 0, 0]])
    with pytest.raises(ValueError, match=">= 1"):
        DensityMatrix.maximally_mixed(0)
    rho = DensityMatrix.maximally_mixed(2)
    with pytest.raises(ValueError, match="not unitary"):
        rho.apply_gate(np.ones((2, 2)), [0])
    with pytest.raises(ValueError, match="duplicate"):
        rho.apply_gate(CX, [0, 0])
    with pytest.raises(ValueError, match="targets"):
        rho.apply_channel(depolarizing(0.1, 2), [0])
    with pytest.raises(TypeError):
        rho.apply_channel(np.eye(2), [0])
    with pytest.raises(ValueError, match="base"):
        rho.von_neumann_entropy(base=1)


def test_resource_guard_before_allocating():
    tracemalloc.start()
    try:
        for make in (
            lambda: DensityMatrix.maximally_mixed(40),
            lambda: Circuit(40).h(0).run_density(),
        ):
            with pytest.raises(ResourceLimitError, match=r"16·4\^40 bytes"):
                make()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1_000_000
    assert MAX_DENSITY_QUBITS == 13
    with pytest.raises(ResourceLimitError, match="max_qubits=2"):
        Circuit(3).run_density(max_qubits=2)
    with pytest.raises(ResourceLimitError):
        DensityMatrix(np.eye(8) / 8, max_qubits=2)


def test_partial_trace(rng):
    q = 4
    R = random_density(q, rng)
    rho = DensityMatrix(R)
    for keep in ([0], [3], [1, 2], [2, 0], [3, 0, 1], [1, 3, 0, 2]):
        np.testing.assert_allclose(
            rho.partial_trace(keep).data, partial_trace_out(R, keep, q), atol=ATOL
        )
    psi = random_state(q, rng)
    pure = DensityMatrix.from_state(psi)
    for keep in ([0], [2, 1]):
        np.testing.assert_allclose(
            pure.partial_trace(keep).data, reduced_density_matrix(psi, keep), atol=ATOL
        )


def test_entropy_and_purity():
    bell = Circuit(2).h(0).cx(0, 1).run_density()
    assert bell.von_neumann_entropy() == pytest.approx(0.0, abs=1e-10)
    half = bell.partial_trace([0])
    assert half.von_neumann_entropy() == pytest.approx(1.0, abs=1e-12)
    assert half.von_neumann_entropy(base=math.e) == pytest.approx(math.log(2), abs=1e-12)
    assert half.purity() == pytest.approx(0.5, abs=ATOL)
    for q in (1, 3):
        assert DensityMatrix.maximally_mixed(q).von_neumann_entropy() == pytest.approx(q, abs=1e-12)


def test_fidelity(rng):
    for _ in range(10):
        a, b = random_state(2, rng), random_state(2, rng)
        ra, rb = DensityMatrix.from_state(a), DensityMatrix.from_state(b)
        assert ra.fidelity(b) == pytest.approx(fidelity(a, b), abs=ATOL)  # exact path
        assert ra.fidelity(rb) == pytest.approx(fidelity(a, b), abs=1e-7)  # rank-deficient
        R, S = random_density(2, rng), random_density(2, rng)
        rho, sigma = DensityMatrix(R), DensityMatrix(S)
        assert rho.fidelity(sigma) == pytest.approx(sigma.fidelity(rho), abs=1e-10)
        assert rho.fidelity(rho) == pytest.approx(1.0, abs=1e-10)
        assert 0 <= rho.fidelity(sigma) <= 1
        # F(rho, I/d) = (Tr sqrt(rho))^2 / d
        lam = np.linalg.eigvalsh(R)
        mm = DensityMatrix.maximally_mixed(2)
        assert rho.fidelity(mm) == pytest.approx(np.sum(np.sqrt(lam)) ** 2 / 4, abs=1e-10)
    with pytest.raises(ValueError, match="qubits"):
        rho.fidelity(DensityMatrix.maximally_mixed(1))
    with pytest.raises(ValueError, match="normalized"):
        rho.fidelity(np.ones(4))


def test_measure_and_sampling(rng):
    R = random_density(3, rng)
    rho = DensityMatrix(R)
    p_diag = np.real(np.diag(R))
    p1 = p_diag[[i for i in range(8) if (i >> 2) & 1]].sum()
    gen = np.random.default_rng(3)
    outs = [rho.measure(2, gen).outcome for _ in range(4000)]
    assert abs(np.mean(outs) - p1) < 5 * math.sqrt(p1 * (1 - p1) / 4000)
    r = rho.measure(2, random_state=1)
    mask = ((np.arange(8) >> 2) & 1) == r.outcome
    expected = R * np.outer(mask, mask) / r.probability
    np.testing.assert_allclose(r.state.data, expected, atol=ATOL)
    assert r.state.trace() == pytest.approx(1.0, abs=ATOL)
    assert r.state.measure(2, 99).outcome == r.outcome  # repeatable
    assert r.to_dict()["outcome"] == r.outcome
    counts = rho.sample_counts(1000, random_state=5)
    assert counts == rho.sample_counts(1000, random_state=5) and sum(counts.values()) == 1000
    bitflip = Circuit(1).x(0).run_density(noise_model=NoiseModel().add(bit_flip(1.0)))
    assert bitflip.sample_counts(10, random_state=0) == {"0": 10}
    with pytest.raises(ValueError, match="out of range"):
        rho.measure(3)


def test_density_gate_memory_is_linear_in_matrix_size():
    """Declared extra space O(4^q): with n = 4^q, peak memory grows like n."""
    est = estimate_complexity(
        lambda rho: rho.apply_gate(CX, [0, 2]),
        lambda n, g: DensityMatrix.maximally_mixed(int(round(math.log(n, 4)))),
        sizes=(4**6, 4**7, 4**8, 4**9),
        repeats=1,
    )
    assert est.space_exponent == pytest.approx(1.0, abs=0.15)


@pytest.mark.complexity
def test_density_gate_time_is_linear_in_matrix_size():
    est = estimate_complexity(
        lambda rho: rho.apply_gate(CX, [0, 2]),
        lambda n, g: DensityMatrix.maximally_mixed(int(round(math.log(n, 4)))),
        sizes=(4**7, 4**8, 4**9, 4**10),
    )
    assert est.time_exponent < 1.3
