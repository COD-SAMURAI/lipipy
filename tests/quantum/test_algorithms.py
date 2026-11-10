"""Algorithms: QFT vs DFT, Grover vs the analytic curve, oracles, phase estimation."""

import itertools
import math

import numpy as np
import pytest

from lipipy.quantum import (
    Circuit,
    bell_circuit,
    bernstein_vazirani,
    deutsch_jozsa,
    entanglement_entropy,
    ghz_circuit,
    grover,
    inverse_qft,
    phase,
    phase_estimation,
    qft,
    rz,
)
from tests.quantum._helpers import random_state, random_unitary

ATOL = 1e-12


def dft(q):
    """Unitary DFT with the +i sign: F[k, j] = exp(2πi jk/N) / sqrt(N)."""
    return np.fft.ifft(np.eye(2**q), axis=0, norm="ortho")


@pytest.mark.parametrize("q", range(1, 7))
def test_qft_matrix_is_dft(q):
    N = 2**q
    np.testing.assert_allclose(qft(q).to_matrix(), dft(q), atol=ATOL)
    j, k = np.meshgrid(np.arange(N), np.arange(N))
    np.testing.assert_allclose(
        qft(q).to_matrix(), np.exp(2j * np.pi * j * k / N) / math.sqrt(N), atol=ATOL
    )  # noqa: E501
    np.testing.assert_allclose(
        inverse_qft(q).to_matrix(), np.fft.fft(np.eye(N), axis=0, norm="ortho"), atol=ATOL
    )  # noqa: E501


@pytest.mark.parametrize("q", range(1, 7))
def test_qft_without_swaps_is_bit_reversed(q):
    rev = [int(format(i, f"0{q}b")[::-1], 2) for i in range(2**q)]
    np.testing.assert_allclose(qft(q, do_swaps=False).to_matrix(), dft(q)[rev], atol=ATOL)
    assert qft(q).count_ops().get("swap", 0) == q // 2
    assert qft(q).num_gates == q * (q + 1) // 2 + q // 2


def test_qft_on_state_matches_numpy_ifft(rng):
    psi = random_state(7, rng)
    np.testing.assert_allclose(qft(7).run(psi), np.fft.ifft(psi, norm="ortho"), atol=ATOL)


def test_bell_and_ghz():
    s = 1 / math.sqrt(2)
    psi = bell_circuit().run()
    np.testing.assert_allclose(psi, [s, 0, 0, s], atol=ATOL)
    assert entanglement_entropy(psi, [0]) == pytest.approx(1.0, abs=ATOL)
    assert entanglement_entropy(psi, [0], base=math.e) == pytest.approx(math.log(2), abs=ATOL)
    for q in range(1, 8):
        psi = ghz_circuit(q).run()
        expected = np.zeros(2**q)
        expected[[0, -1]] = s
        np.testing.assert_allclose(psi, expected, atol=ATOL)
        assert ghz_circuit(q).depth() == q
    product = Circuit(3).h(0).h(1).x(2).run()
    assert entanglement_entropy(product, [0]) == pytest.approx(0.0, abs=ATOL)
    assert entanglement_entropy(product, [1, 2]) == pytest.approx(0.0, abs=ATOL)


@pytest.mark.parametrize(
    ("q", "marked"),
    [(2, [3]), (3, [5]), (4, [0, 9]), (5, [1, 2, 3]), (6, [7]), (7, [100, 3, 64, 12])],
)
def test_grover_matches_analytic_success_probability(q, marked):
    N, M = 2**q, len(marked)
    theta = math.asin(math.sqrt(M / N))
    for k in range(0, 6):
        r = grover(q, marked, iterations=k)
        assert r.iterations == k
        assert r.success_probability == pytest.approx(math.sin((2 * k + 1) * theta) ** 2, abs=ATOL)
        # amplitudes stay uniform within the marked and unmarked sets
        amps = r.state.real
        unmarked = np.delete(np.arange(N), marked)
        np.testing.assert_allclose(
            amps[marked], math.sin((2 * k + 1) * theta) / math.sqrt(M), atol=ATOL
        )  # noqa: E501
        np.testing.assert_allclose(
            amps[unmarked], math.cos((2 * k + 1) * theta) / math.sqrt(N - M), atol=ATOL
        )  # noqa: E501
    default = grover(q, marked)
    assert default.iterations == math.floor(math.pi / 4 * math.sqrt(N / M))
    assert default.success_probability >= 1 - M / N - ATOL  # Boyer et al. bound
    assert default.most_likely in marked


def test_grover_result_and_errors():
    r = grover(3, 6)
    d = r.to_dict()
    assert d["marked"] == [6] and d["most_likely"] == 6 and d["iterations"] == 2
    assert r.circuit.count_ops()["oracle"] == 2
    assert grover(2, [0, 1, 2, 3]).iterations == 0
    with pytest.raises(ValueError, match="at least one"):
        grover(3, [])
    with pytest.raises(ValueError, match="out of range"):
        grover(3, 8)
    with pytest.raises(ValueError, match="duplicates"):
        grover(3, [1, 1])
    with pytest.raises(ValueError, match="iterations"):
        grover(3, 1, iterations=-1)


@pytest.mark.parametrize("n", range(1, 6))
def test_deutsch_jozsa(n, rng):
    assert deutsch_jozsa(lambda x: 0, n) == "constant"
    assert deutsch_jozsa(lambda x: 1, n) == "constant"
    assert deutsch_jozsa(lambda x: x & 1, n) == "balanced"
    assert deutsch_jozsa(lambda x: (x >> (n - 1)) & 1, n) == "balanced"
    assert deutsch_jozsa(lambda x: bin(x).count("1") % 2, n) == "balanced"
    for _ in range(5):  # random balanced functions
        ones = set(rng.permutation(2**n)[: 2 ** (n - 1)].tolist())
        assert deutsch_jozsa(lambda x, ones=ones: int(x in ones), n) == "balanced"


def test_deutsch_jozsa_errors():
    with pytest.raises(ValueError, match="neither constant nor balanced"):
        deutsch_jozsa(lambda x: int(x == 0), 3)
    with pytest.raises(ValueError, match="0 or 1"):
        deutsch_jozsa(lambda x: 2, 2)
    with pytest.raises(TypeError):
        deutsch_jozsa(3, 2)


def test_bernstein_vazirani_all_secrets_up_to_6():
    for n in range(1, 7):
        for bits in itertools.product("01", repeat=n):
            secret = "".join(bits)
            assert bernstein_vazirani(secret) == secret


def test_bernstein_vazirani_errors():
    for bad in ("", "012", "1 0"):
        with pytest.raises(ValueError):
            bernstein_vazirani(bad)
    with pytest.raises(TypeError):
        bernstein_vazirani(0b101)


@pytest.mark.parametrize("t", [1, 2, 3, 4, 5])
def test_phase_estimation_exact_for_dyadic_phases(t):
    for y in range(2**t):
        phi = y / 2**t
        assert phase_estimation(phase(2 * math.pi * phi), np.array([0, 1]), t) == phi


def test_phase_estimation_multiqubit_unitary(rng):
    """U = V diag(e^{2πi φ_j}) V†, eigenstate = V[:, 0]."""
    for t in (3, 4):
        for _ in range(4):
            phis = rng.integers(0, 2**t, size=4) / 2**t
            V = random_unitary(4, rng)
            U = V @ np.diag(np.exp(2j * np.pi * phis)) @ V.conj().T
            assert phase_estimation(U, V[:, 0], t) == phis[0]


def test_phase_estimation_non_dyadic_gives_nearest():
    phi = 1 / 3
    for t in (4, 5, 6):
        # rz(θ) has eigenvalue e^{-iθ/2} on |0>, so θ = -4πφ gives phase φ
        est = phase_estimation(rz(-4 * math.pi * phi), [1, 0], t)
        assert est == round(phi * 2**t) / 2**t


def test_phase_estimation_errors():
    with pytest.raises(ValueError, match="not unitary"):
        phase_estimation(np.ones((2, 2)), [1, 0], 2)
    with pytest.raises(ValueError, match="qubits"):
        phase_estimation(phase(0.1), [1, 0, 0, 0], 2)
    with pytest.raises(ValueError, match="normalized"):
        phase_estimation(phase(0.1), [1, 1], 2)
    with pytest.raises(ValueError, match="num_counting"):
        phase_estimation(phase(0.1), [1, 0], 0)
