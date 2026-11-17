"""Kraus channels and noise models: CPTP checks, analytic actions, rule matching."""

import numpy as np
import pytest

from lipipy.quantum import (
    CX,
    Circuit,
    KrausChannel,
    NoiseModel,
    X,
    Z,
    amplitude_damping,
    bit_flip,
    depolarizing,
    phase_damping,
    phase_flip,
)
from tests.quantum._helpers import random_density, random_unitary

ATOL = 1e-12
FACTORIES = {
    "bit_flip": bit_flip,
    "phase_flip": phase_flip,
    "depolarizing": depolarizing,
    "amplitude_damping": amplitude_damping,
    "phase_damping": phase_damping,
}


def _choi_partial_trace_output(J, d):
    """Tr over the output factor (the right / least-significant one)."""
    return np.einsum("aibi->ab", J.reshape(d, d, d, d))


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("name", FACTORIES)
def test_standard_channels_are_cptp_for_many_parameters(name, rng):
    params = [0.0, 1.0, *rng.uniform(0, 1, 25)]
    for p in params:
        ch = FACTORIES[name](p)
        total = sum(K.conj().T @ K for K in ch.kraus)
        np.testing.assert_allclose(total, np.eye(2), atol=ATOL)
        J = ch.choi()
        np.testing.assert_allclose(J, J.conj().T, atol=ATOL)
        assert np.linalg.eigvalsh(J).min() > -ATOL  # completely positive
        np.testing.assert_allclose(_choi_partial_trace_output(J, 2), np.eye(2), atol=ATOL)
        rho = random_density(1, rng)
        out = ch.apply_to_matrix(rho)
        assert np.trace(out).real == pytest.approx(1.0, abs=ATOL)
        assert np.linalg.eigvalsh(out).min() > -ATOL


def test_analytic_actions(rng):
    for _ in range(20):
        rho = random_density(1, rng)
        p = float(rng.uniform())
        close = lambda a, b: np.testing.assert_allclose(a, b, atol=ATOL)  # noqa: E731
        close(bit_flip(p).apply_to_matrix(rho), (1 - p) * rho + p * X @ rho @ X)
        close(phase_flip(p).apply_to_matrix(rho), (1 - p) * rho + p * Z @ rho @ Z)
        close(depolarizing(p).apply_to_matrix(rho), (1 - p) * rho + p * np.eye(2) / 2)
        g = p
        expected = np.array(
            [
                [rho[0, 0] + g * rho[1, 1], np.sqrt(1 - g) * rho[0, 1]],
                [np.sqrt(1 - g) * rho[1, 0], (1 - g) * rho[1, 1]],
            ]
        )
        close(amplitude_damping(g).apply_to_matrix(rho), expected)
        expected = rho.copy()
        expected[0, 1] *= np.sqrt(1 - p)
        expected[1, 0] *= np.sqrt(1 - p)
        close(phase_damping(p).apply_to_matrix(rho), expected)


@pytest.mark.parametrize("n", [1, 2])
def test_depolarizing_multi_qubit(n, rng):
    for p in (0.0, 0.3, 1.0, 4**n / (4**n - 1)):
        ch = depolarizing(p, n)
        assert ch.num_qubits == n
        rho = random_density(n, rng)
        expected = (1 - p) * rho + p * np.eye(2**n) / 2**n
        np.testing.assert_allclose(ch.apply_to_matrix(rho), expected, atol=ATOL)
        assert np.linalg.eigvalsh(ch.choi()).min() > -ATOL
    with pytest.raises(ValueError, match=r"\[0, 1.33333\]"):
        depolarizing(1.4)


def test_extremes():
    assert len(bit_flip(0.0).kraus) == 1  # the zero operator is dropped
    assert len(bit_flip(1.0).kraus) == 1
    excited = np.diag([0.0, 1.0])
    np.testing.assert_allclose(amplitude_damping(1.0).apply_to_matrix(excited), np.diag([1.0, 0.0]))
    plus = np.full((2, 2), 0.5)
    np.testing.assert_allclose(phase_damping(1.0).apply_to_matrix(plus), np.eye(2) / 2)
    np.testing.assert_allclose(depolarizing(1.0).apply_to_matrix(plus), np.eye(2) / 2, atol=ATOL)


def test_generic_kraus_and_compose(rng):
    U = random_unitary(4, rng)
    ch = KrausChannel([np.sqrt(0.3) * U, np.sqrt(0.7) * CX], name="mix")
    assert ch.num_qubits == 2 and ch.name == "mix" and "rank=2" in repr(ch)
    rho = random_density(2, rng)
    np.testing.assert_allclose(
        ch.apply_to_matrix(rho), 0.3 * U @ rho @ U.conj().T + 0.7 * CX @ rho @ CX.T, atol=ATOL
    )
    a, b = amplitude_damping(0.3), depolarizing(0.2)
    r1 = random_density(1, rng)
    np.testing.assert_allclose(
        a.compose(b).apply_to_matrix(r1), b.apply_to_matrix(a.apply_to_matrix(r1)), atol=ATOL
    )
    # single matrix accepted
    assert KrausChannel(X).num_qubits == 1


def test_kraus_is_immutable(rng):
    ops = [np.sqrt(0.5) * np.eye(2), np.sqrt(0.5) * X.copy()]
    ch = KrausChannel(ops)
    ops[0][0, 0] = 99  # caller's arrays are copied
    assert ch.kraus[0][0, 0] == pytest.approx(np.sqrt(0.5))
    with pytest.raises(ValueError):
        ch.kraus[0][0, 0] = 1


def test_kraus_errors():
    with pytest.raises(ValueError, match="trace preserving"):
        KrausChannel([np.eye(2), X])
    with pytest.raises(ValueError, match="same size"):
        KrausChannel([np.eye(2), np.eye(4)])
    with pytest.raises(ValueError, match="at least one"):
        KrausChannel([])
    with pytest.raises(ValueError, match="zero"):
        KrausChannel([np.zeros((2, 2))])
    with pytest.raises(ValueError, match="power of two"):
        KrausChannel([np.eye(3)])
    for f in FACTORIES.values():
        with pytest.raises(ValueError):
            f(-0.1)
        with pytest.raises(TypeError):
            f("0.1")
    with pytest.raises(ValueError, match="qubits"):
        bit_flip(0.1).apply_to_matrix(np.eye(4) / 4)
    with pytest.raises(ValueError):
        bit_flip(0.1).compose(depolarizing(0.1, 2))
    with pytest.raises(TypeError):
        bit_flip(0.1).compose("x")


def test_noise_model_matching():
    a, b, two = bit_flip(0.1), phase_flip(0.2), depolarizing(0.1, 2)
    model = NoiseModel().add(a, gates="h").add(b, gates=["cx"], qubits=[0, 1]).add(two, gates="cx")
    assert model.channels_for("h", (2,)) == [(a, (2,))]
    assert model.channels_for("x", (0,)) == []
    assert model.channels_for("cx", (0, 1)) == [(b, (0,)), (b, (1,)), (two, (0, 1))]
    assert model.channels_for("cx", (1, 0)) == [(two, (1, 0))]
    everything = NoiseModel().add(a)
    assert everything.channels_for("ccx", (0, 1, 2)) == [(a, (0,)), (a, (1,)), (a, (2,))]
    with pytest.raises(ValueError, match="cannot follow"):
        NoiseModel().add(two).channels_for("ccx", (0, 1, 2))
    d = model.to_dict()
    assert d["rules"][1] == {
        "channel": "phase_flip",
        "channel_qubits": 1,
        "gates": ["cx"],
        "qubits": [0, 1],
    }
    assert repr(model) == "NoiseModel(rules=3)"


def test_noise_model_add_is_atomic():
    model = NoiseModel().add(bit_flip(0.1))
    for bad in (
        lambda: model.add("not a channel"),
        lambda: model.add(bit_flip(0.1), gates=[]),
        lambda: model.add(bit_flip(0.1), gates=[3]),
        lambda: model.add(bit_flip(0.1), qubits=[0, 0]),
        lambda: model.add(depolarizing(0.1, 2), qubits=[0, 1, 2]),
        lambda: model.add(bit_flip(0.1), qubits=[-1]),
    ):
        with pytest.raises((TypeError, ValueError)):
            bad()
        assert len(model.rules) == 1


def test_noise_model_in_circuit_matches_manual_application(rng):
    from lipipy.quantum import DensityMatrix

    c = Circuit(3).h(0).cx(0, 1).rx(0.4, 2).cx(1, 2)
    model = (
        NoiseModel()
        .add(amplitude_damping(0.2), gates="cx")
        .add(depolarizing(0.05, 1), gates=["h", "rx"])
    )
    rho = c.run_density(noise_model=model)
    manual = DensityMatrix.from_state(np.eye(8)[0])
    for op in c.instructions:
        manual = manual.apply_gate(op.matrix, op.qubits)
        for ch, t in model.channels_for(op.name, op.qubits):
            manual = manual.apply_channel(ch, t)
    np.testing.assert_allclose(rho.data, manual.data, atol=ATOL)
