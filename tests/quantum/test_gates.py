"""Gates: unitarity, algebraic identities, conventions, and Qiskit parity."""

import numpy as np
import pytest

from lipipy.quantum import (
    CCX,
    CX,
    CZ,
    SDG,
    SWAP,
    SX,
    TDG,
    H,
    I,
    S,
    T,
    X,
    Y,
    Z,
    controlled,
    is_unitary,
    phase,
    rx,
    ry,
    rz,
    u3,
)

CONSTANTS = {
    "I": I, "X": X, "Y": Y, "Z": Z, "H": H, "S": S, "SDG": SDG, "T": T, "TDG": TDG,
    "SX": SX, "CX": CX, "CZ": CZ, "SWAP": SWAP, "CCX": CCX,
}  # fmt: skip
ANGLES = [0.0, 0.3, -1.2, np.pi / 2, np.pi, 2.7, 2 * np.pi]


@pytest.mark.parametrize("name", CONSTANTS)
def test_constant_gates_unitary_and_readonly(name):
    U = CONSTANTS[name]
    assert U.dtype == np.complex128
    assert is_unitary(U, atol=1e-14)
    with pytest.raises(ValueError):
        U[0, 0] = 2  # module constants are write-protected


@pytest.mark.parametrize("theta", ANGLES)
def test_parametrized_gates_unitary(theta):
    for U in (rx(theta), ry(theta), rz(theta), phase(theta), u3(theta, 0.4, -1.1)):
        assert is_unitary(U, atol=1e-14)


def test_identities():
    close = lambda a, b: np.testing.assert_allclose(a, b, atol=1e-14)  # noqa: E731
    close(S @ S, Z)
    close(T @ T, S)
    close(SX @ SX, X)
    close(S @ SDG, I)
    close(T @ TDG, I)
    close(H @ Z @ H, X)
    close(X @ Y, 1j * Z)
    close(H @ H, I)
    close(phase(np.pi / 2), S)
    close(phase(np.pi), Z)
    close(rx(np.pi), -1j * X)
    close(ry(np.pi), -1j * Y)
    close(u3(np.pi, 0, np.pi), X)
    close(u3(np.pi / 2, 0, np.pi), H)
    close(u3(0, 0, 0.7), phase(0.7))
    close(SX, np.exp(1j * np.pi / 4) * rx(np.pi / 2))
    close(SWAP, CX @ (SWAP @ CX @ SWAP) @ CX)  # swap = three CNOTs


@pytest.mark.parametrize("theta", ANGLES)
def test_rotations_are_exponentials(theta):
    for U, P in ((rx(theta), X), (ry(theta), Y), (rz(theta), Z)):
        expected = np.cos(theta / 2) * I - 1j * np.sin(theta / 2) * P
        np.testing.assert_allclose(U, expected, atol=1e-14)


def test_u3_euler_decomposition(rng):
    for _ in range(10):
        t, p, lam = rng.uniform(-np.pi, np.pi, size=3)
        expected = np.exp(1j * (p + lam) / 2) * rz(p) @ ry(t) @ rz(lam)
        np.testing.assert_allclose(u3(t, p, lam), expected, atol=1e-14)


def test_controlled_convention():
    np.testing.assert_array_equal(controlled(X), CX)
    np.testing.assert_array_equal(controlled(Z), CZ)
    np.testing.assert_array_equal(controlled(X, 2), CCX)
    # CX on (control, target): |control=1, target=0> = index 0b01 -> index 0b11
    assert CX[3, 1] == 1 and CX[1, 3] == 1
    C = controlled(SWAP, 1)
    assert C.shape == (8, 8) and is_unitary(C)
    # Fredkin: control on bit 0 swaps bits 1 and 2: index 0b011 -> 0b101
    assert C[5, 3] == 1


def test_is_unitary_rejects():
    assert not is_unitary([[1, 1], [0, 1]])
    assert not is_unitary(np.ones((2, 3)))
    assert not is_unitary(np.ones(4))
    assert not is_unitary([[np.nan, 0], [0, 1]])
    assert not is_unitary(1.0001 * H, atol=1e-10)
    assert is_unitary(1.0001 * H, atol=1e-3)


def test_bad_inputs():
    with pytest.raises(TypeError):
        rx("0.5")
    with pytest.raises(ValueError, match="finite"):
        ry(np.inf)
    with pytest.raises(ValueError, match="power of two"):
        controlled(np.eye(3))
    with pytest.raises(ValueError, match=">= 1"):
        controlled(X, 0)


QISKIT_GATES = {
    "I": ("IGate", ()), "X": ("XGate", ()), "Y": ("YGate", ()), "Z": ("ZGate", ()),
    "H": ("HGate", ()), "S": ("SGate", ()), "SDG": ("SdgGate", ()), "T": ("TGate", ()),
    "TDG": ("TdgGate", ()), "SX": ("SXGate", ()), "CX": ("CXGate", ()),
    "CZ": ("CZGate", ()), "SWAP": ("SwapGate", ()), "CCX": ("CCXGate", ()),
}  # fmt: skip


def test_qiskit_gate_matrices():
    lib = pytest.importorskip("qiskit.circuit.library")
    for name, (cls, args) in QISKIT_GATES.items():
        np.testing.assert_allclose(
            CONSTANTS[name], getattr(lib, cls)(*args).to_matrix(), atol=1e-14
        )
    for theta in ANGLES:
        np.testing.assert_allclose(rx(theta), lib.RXGate(theta).to_matrix(), atol=1e-14)
        np.testing.assert_allclose(ry(theta), lib.RYGate(theta).to_matrix(), atol=1e-14)
        np.testing.assert_allclose(rz(theta), lib.RZGate(theta).to_matrix(), atol=1e-14)
        np.testing.assert_allclose(phase(theta), lib.PhaseGate(theta).to_matrix(), atol=1e-14)
        np.testing.assert_allclose(
            u3(theta, 0.4, -1.1), lib.UGate(theta, 0.4, -1.1).to_matrix(), atol=1e-14
        )
    np.testing.assert_allclose(controlled(H), lib.CHGate().to_matrix(), atol=1e-14)
    qi = pytest.importorskip("qiskit.quantum_info")
    ccrx = qi.Operator(lib.RXGate(0.3).control(2, annotated=False)).data
    np.testing.assert_allclose(controlled(rx(0.3), 2), ccrx, atol=1e-14)
