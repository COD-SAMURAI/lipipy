"""States: constructors, resource guard, Bloch vectors, tensor products, fidelity."""

import tracemalloc

import numpy as np
import pytest

from lipipy.core import ResourceLimitError
from lipipy.quantum import (
    MAX_QUBITS,
    basis_state,
    bloch_vector,
    fidelity,
    is_normalized,
    random_state,
    tensor,
    zero_state,
)


def test_zero_and_basis_states():
    psi = zero_state(3)
    assert psi.dtype == np.complex128 and psi.shape == (8,)
    np.testing.assert_array_equal(psi, np.eye(8)[0])
    for i in range(8):
        np.testing.assert_array_equal(basis_state(3, i), np.eye(8)[i])
    # little-endian: index 0b001 has qubit 0 = 1
    assert basis_state(3, 1)[1] == 1


@pytest.mark.parametrize("q", [1, 3, 6])
def test_random_state_normalized_and_reproducible(q):
    a = random_state(q, random_state=7)
    b = random_state(q, random_state=7)
    assert a.dtype == np.complex128
    assert is_normalized(a)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, random_state(q, random_state=8))


def test_random_state_is_haar_moments():
    """For Haar states p = |psi_0|^2 ~ Beta(1, N-1): E p = 1/N, E p^2 = 2/(N(N+1))."""
    N, trials = 8, 40_000
    rng = np.random.default_rng(123)
    p = np.array([abs(random_state(3, rng)[0]) ** 2 for _ in range(trials)])
    se1 = np.sqrt((2 / (N * (N + 1)) - 1 / N**2) / trials)
    assert abs(p.mean() - 1 / N) < 5 * se1
    assert p.var() == pytest.approx((N - 1) / (N**2 * (N + 1)), rel=0.05)


def test_is_normalized():
    assert is_normalized(np.array([1, 1j]) / np.sqrt(2))
    assert not is_normalized([1, 1])
    assert is_normalized([1 + 1e-12, 0])
    assert not is_normalized([1 + 1e-6, 0])


def test_bloch_vector_cardinal_states():
    s = 1 / np.sqrt(2)
    cases = {
        (1, 0): [0, 0, 1],
        (0, 1): [0, 0, -1],
        (s, s): [1, 0, 0],
        (s, -s): [-1, 0, 0],
        (s, 1j * s): [0, 1, 0],
        (s, -1j * s): [0, -1, 0],
    }
    for state, expected in cases.items():
        np.testing.assert_allclose(bloch_vector(np.array(state)), expected, atol=1e-12)


def test_bloch_vector_polar_parametrization(rng):
    for _ in range(20):
        theta, phi = rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi)
        psi = np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])
        expected = [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        np.testing.assert_allclose(bloch_vector(psi), expected, atol=1e-12)


def test_tensor_and_fidelity(rng):
    a, b, c = (random_state(1, rng) for _ in range(3))
    np.testing.assert_allclose(tensor(a, b, c), np.kron(np.kron(a, b), c), atol=1e-15)
    # last factor sits on qubit 0
    np.testing.assert_array_equal(tensor(basis_state(1, 1), basis_state(1, 0)), basis_state(2, 2))
    assert fidelity(a, a) == pytest.approx(1.0, abs=1e-12)
    assert fidelity([1, 0], [0, 1]) == 0.0
    x, y = random_state(3, rng), random_state(3, rng)
    assert fidelity(x, y) == pytest.approx(abs(np.vdot(x, y)) ** 2, abs=1e-14)


def test_resource_guard_raises_before_allocating():
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        for make in (zero_state, lambda q: basis_state(q, 0), random_state):
            with pytest.raises(ResourceLimitError, match=r"16·2\^40 bytes = 16\.0 TiB"):
                make(40)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1_000_000  # a 40-qubit state would be 16 TiB
    assert MAX_QUBITS == 26
    with pytest.raises(ResourceLimitError, match="max_qubits=4"):
        zero_state(5, max_qubits=4)
    assert zero_state(5, max_qubits=5).shape == (32,)
    with pytest.raises(ResourceLimitError):
        tensor(zero_state(3), zero_state(3), max_qubits=5)


def test_bad_inputs():
    with pytest.raises(ValueError, match=">= 1"):
        zero_state(0)
    with pytest.raises(TypeError):
        zero_state(2.0)
    with pytest.raises(TypeError):
        zero_state(True)
    with pytest.raises(ValueError, match="index"):
        basis_state(2, 4)
    with pytest.raises(ValueError, match="index"):
        basis_state(2, -1)
    with pytest.raises(ValueError, match="one-qubit"):
        bloch_vector(zero_state(2))
    with pytest.raises(ValueError, match="normalized"):
        bloch_vector([1, 1])
    with pytest.raises(ValueError, match="power of two"):
        is_normalized([1, 0, 0])
    with pytest.raises(ValueError, match="1-D"):
        is_normalized(np.eye(2))
    with pytest.raises(ValueError, match="NaN"):
        is_normalized([np.nan, 0])
    with pytest.raises(ValueError, match="different sizes"):
        fidelity([1, 0], [1, 0, 0, 0])
    with pytest.raises(ValueError):
        tensor()
