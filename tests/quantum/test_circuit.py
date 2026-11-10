"""Circuit builder: simulation, dense matrices, inverse, compose, statistics, validation."""

import math

import numpy as np
import pytest

from lipipy.core import ResourceLimitError
from lipipy.quantum import (
    CCX,
    CX,
    CZ,
    SWAP,
    SX,
    Circuit,
    H,
    Instruction,
    S,
    T,
    X,
    Y,
    Z,
    controlled,
    phase,
    rx,
    ry,
    rz,
    u3,
)
from tests.quantum._helpers import full_operator, random_state, random_unitary

ATOL = 1e-12


def random_circuit(q, num_gates, rng):
    """A random circuit plus its reference operator built from Kronecker products."""
    c = Circuit(q)
    ref = np.eye(2**q, dtype=complex)
    one = ["h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "rx", "ry", "rz", "p", "u"]
    two = ["cx", "cz", "swap", "cp", "unitary"] if q >= 2 else []
    three = ["ccx"] if q >= 3 else []
    mats = {"h": H, "x": X, "y": Y, "z": Z, "s": S, "sdg": S.conj().T, "t": T,
            "tdg": T.conj().T, "sx": SX, "cx": CX, "cz": CZ, "swap": SWAP, "ccx": CCX}  # fmt: skip
    for _ in range(num_gates):
        name = str(rng.choice(one + two + three))
        k = 3 if name == "ccx" else 2 if name in two else 1
        qs = [int(t) for t in rng.permutation(q)[:k]]
        angles = [float(a) for a in rng.uniform(-np.pi, np.pi, 3)]
        if name in mats:
            getattr(c, name)(*qs)
            U = mats[name]
        elif name in ("rx", "ry", "rz", "p"):
            getattr(c, name)(angles[0], qs[0])
            U = {"rx": rx, "ry": ry, "rz": rz, "p": phase}[name](angles[0])
        elif name == "u":
            c.u(*angles, qs[0])
            U = u3(*angles)
        elif name == "cp":
            c.cp(angles[0], *qs)
            U = controlled(phase(angles[0]))
        else:
            U = random_unitary(4, rng)
            c.unitary(U, qs)
        ref = full_operator(U, qs, q) @ ref
    return c, ref


@pytest.mark.parametrize("q", [1, 2, 3, 4, 5])
def test_run_and_to_matrix_match_kronecker_reference(q, rng):
    for _ in range(3):
        c, ref = random_circuit(q, 25, rng)
        np.testing.assert_allclose(c.to_matrix(), ref, atol=ATOL)
        np.testing.assert_allclose(c.run(), ref[:, 0], atol=ATOL)
        psi = random_state(q, rng)
        np.testing.assert_allclose(c.run(psi), ref @ psi, atol=ATOL)


def test_bell_and_ghz_amplitudes():
    s = 1 / math.sqrt(2)
    np.testing.assert_allclose(Circuit(2).h(0).cx(0, 1).run(), [s, 0, 0, s], atol=ATOL)
    psi = Circuit(3).h(0).cx(0, 1).cx(1, 2).run()
    expected = np.zeros(8)
    expected[[0, 7]] = s
    np.testing.assert_allclose(psi, expected, atol=ATOL)


def test_inverse(rng):
    c, ref = random_circuit(4, 40, rng)
    inv = c.inverse()
    np.testing.assert_allclose(inv.to_matrix(), ref.conj().T, atol=ATOL)
    np.testing.assert_allclose(c.compose(inv).to_matrix(), np.eye(16), atol=ATOL)
    names = [
        op.name
        for op in Circuit(1).s(0).t(0).sx(0).rx(0.3, 0).u(0.1, 0.2, 0.3, 0).inverse().instructions
    ]  # noqa: E501
    assert names == ["u", "rx", "sxdg", "tdg", "sdg"]
    u_inv = Circuit(1).u(0.1, 0.2, 0.3, 0).inverse().instructions[0]
    assert u_inv.params == (-0.1, -0.3, -0.2)
    np.testing.assert_allclose(u_inv.matrix, u3(-0.1, -0.3, -0.2), atol=ATOL)
    assert Circuit(1).rx(0.3, 0).inverse().instructions[0].params == (-0.3,)


def test_compose(rng):
    a, ra = random_circuit(3, 10, rng)
    b, rb = random_circuit(2, 10, rng)
    ab = a.compose(b, qubits=[2, 0])
    perm_b = Circuit(3).unitary(rb, [2, 0]).to_matrix()
    np.testing.assert_allclose(ab.to_matrix(), perm_b @ ra, atol=ATOL)
    assert a.num_gates == 10  # compose returns a new circuit
    np.testing.assert_allclose(
        a.compose(b).to_matrix(), full_operator(rb, [0, 1], 3) @ ra, atol=ATOL
    )
    with pytest.raises(ValueError, match="compose"):
        b.compose(a)
    with pytest.raises(ValueError, match="2 indices"):
        a.compose(b, qubits=[0])
    with pytest.raises(TypeError):
        a.compose("h")


def test_depth_count_ops_repr():
    c = Circuit(3).h(0).h(1).cx(0, 1).x(2).ccx(0, 1, 2).h(2)
    assert c.depth() == 4
    assert Circuit(4).depth() == 0
    assert c.num_gates == len(c) == 6
    assert c.count_ops() == {"h": 3, "ccx": 1, "cx": 1, "x": 1}
    text = repr(c)
    assert text.startswith("Circuit(num_qubits=3, num_gates=6, depth=4)")
    assert "ccx q0, q1, q2" in text
    assert "rx(0.5) q1" in repr(Circuit(2).rx(0.5, 1))
    big = Circuit(1)
    for _ in range(30):
        big.x(0)
    assert "... (10 more)" in repr(big)


def test_chaining_returns_self_and_run_does_not_mutate(rng):
    c = Circuit(2)
    assert c.h(0) is c and c.cx(0, 1) is c
    psi = random_state(2, rng)
    before = psi.copy()
    c.run(psi)
    np.testing.assert_array_equal(psi, before)


def test_to_matrix_and_run_limits():
    with pytest.raises(ResourceLimitError, match="10 qubits"):
        Circuit(11).h(0).to_matrix()
    assert Circuit(10).h(0).to_matrix().shape == (1024, 1024)
    with pytest.raises(ResourceLimitError, match="16·2\\^40"):
        Circuit(40).h(0).run()
    with pytest.raises(ResourceLimitError):
        Circuit(6).run(max_qubits=5)


def test_diagonal_gate(rng):
    d = np.exp(1j * rng.uniform(0, 2 * np.pi, 8))
    c = Circuit(4).diagonal(d, [3, 0, 2])
    np.testing.assert_allclose(c.to_matrix(), full_operator(np.diag(d), [3, 0, 2], 4), atol=ATOL)
    np.testing.assert_allclose(c.inverse().to_matrix(), c.to_matrix().conj().T, atol=ATOL)
    with pytest.raises(ValueError, match="modulus 1"):
        Circuit(1).diagonal([1, 2], [0])


def test_validation_errors():
    c = Circuit(3)
    with pytest.raises(ValueError, match="duplicate"):
        c.cx(1, 1)
    with pytest.raises(ValueError, match="duplicate"):
        c.ccx(0, 2, 0)
    with pytest.raises(ValueError, match="out of range"):
        c.h(3)
    with pytest.raises(ValueError, match="out of range"):
        c.x(-1)
    with pytest.raises(TypeError):
        c.h(0.5)
    with pytest.raises(ValueError, match="not unitary"):
        c.unitary(np.ones((2, 2)), [0])
    with pytest.raises(ValueError, match="needs 2 qubits"):
        c.unitary(CX, [0])
    with pytest.raises(ValueError, match="finite"):
        c.rx(np.nan, 0)
    with pytest.raises(ValueError, match="2 qubits, the circuit has 3"):
        c.run(np.array([1, 0, 0, 0]))
    with pytest.raises(ValueError, match=">= 1"):
        Circuit(0)
    assert c.num_gates == 0  # failed calls append nothing


def test_instruction_invariant_enforced_at_construction():
    with pytest.raises(ValueError, match="exactly one"):
        Instruction("x", (0,))
    with pytest.raises(ValueError, match="exactly one"):
        Instruction("x", (0,), diagonal=np.ones(2), _matrix=X)
    with pytest.raises(ValueError, match=r"expected \(4, 4\)"):
        Instruction("cx", (0, 1), _matrix=X)
    with pytest.raises(ValueError, match=r"expected \(2,\)"):
        Instruction("z", (0,), diagonal=np.ones(4))
    np.testing.assert_array_equal(Instruction("z", (0,), diagonal=np.array([1, -1])).matrix, Z)
    np.testing.assert_array_equal(Instruction("x", (0,), _matrix=X).matrix, X)
