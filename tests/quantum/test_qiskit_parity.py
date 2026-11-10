"""Parity with Qiskit (skipped when qiskit is not installed, e.g. in CI).

Install locally with ``.venv/bin/pip install qiskit`` to run these.
"""

import numpy as np
import pytest

from lipipy.quantum import (
    Circuit,
    entanglement_entropy,
    expectation_pauli,
    qft,
    reduced_density_matrix,
)
from tests.quantum._helpers import random_state, random_unitary

qiskit = pytest.importorskip("qiskit")
qi = pytest.importorskip("qiskit.quantum_info")

ATOL = 1e-12
ONE = ["h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx"]
ROT = ["rx", "ry", "rz", "p"]


def random_pair(q, num_gates, rng):
    """The same seeded random circuit built in lipipy and in Qiskit."""
    ours, ref = Circuit(q), qiskit.QuantumCircuit(q)
    names = ONE + ROT + ["u"] + (["cx", "cz", "swap", "cp", "unitary"] if q > 1 else [])
    names += ["ccx"] if q > 2 else []
    for _ in range(num_gates):
        name = str(rng.choice(names))
        k = 3 if name == "ccx" else 2 if name in ("cx", "cz", "swap", "cp", "unitary") else 1
        qs = [int(t) for t in rng.permutation(q)[:k]]
        a = [float(x) for x in rng.uniform(-np.pi, np.pi, 3)]
        if name in ONE or name in ("cx", "cz", "swap", "ccx"):
            getattr(ours, name)(*qs)
            getattr(ref, name)(*qs)
        elif name in ROT:
            getattr(ours, name)(a[0], qs[0])
            getattr(ref, name)(a[0], qs[0])
        elif name == "u":
            ours.u(*a, qs[0])
            ref.u(*a, qs[0])
        elif name == "cp":
            ours.cp(a[0], *qs)
            ref.cp(a[0], *qs)
        else:
            U = random_unitary(4, rng)
            ours.unitary(U, qs)
            ref.unitary(U, qs)
    return ours, ref


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("q", [1, 2, 3, 5, 7])
def test_random_circuits_match_qiskit_statevector(q, seed):
    rng = np.random.default_rng(1000 * q + seed)
    ours, ref = random_pair(q, 40, rng)
    np.testing.assert_allclose(ours.run(), qi.Statevector(ref).data, atol=ATOL)
    if q <= 5:
        np.testing.assert_allclose(ours.to_matrix(), qi.Operator(ref).data, atol=ATOL)
    np.testing.assert_allclose(ours.inverse().run(), qi.Statevector(ref.inverse()).data, atol=ATOL)


@pytest.mark.parametrize("q", range(1, 7))
def test_qft_matches_qiskit(q):
    lib = pytest.importorskip("qiskit.circuit.library")
    np.testing.assert_allclose(qft(q).to_matrix(), qi.Operator(lib.QFTGate(q)).data, atol=ATOL)


def test_pauli_expectation_and_partial_trace_match_qiskit(rng):
    q = 4
    psi = random_state(q, rng)
    sv = qi.Statevector(psi)
    for label in ("ZIII", "IXYZ", "XXXX", "YIZI", "IIIY"):
        expected = sv.expectation_value(qi.Pauli(label)).real
        assert expectation_pauli(psi, label) == pytest.approx(expected, abs=ATOL)
    for keep in ([0], [1, 3], [0, 2, 3]):
        traced = [j for j in range(q) if j not in keep]
        ref = qi.partial_trace(sv, traced)
        np.testing.assert_allclose(reduced_density_matrix(psi, keep), ref.data, atol=ATOL)
        assert entanglement_entropy(psi, keep) == pytest.approx(qi.entropy(ref, base=2), abs=1e-10)
