r"""Variational quantum algorithms: parametrised circuits, parameter-shift gradients, VQE, QAOA.

Parametrised circuits
---------------------
A :class:`ParametricCircuit` is a circuit whose rotation angles may be affine
functions :math:`\theta_k = a_k x_{j(k)} + b_k` of symbolic :class:`Parameter` s.
:meth:`ParametricCircuit.bind` substitutes numbers and returns an ordinary
:class:`~lipipy.quantum.Circuit`.

Parameter-shift rule
--------------------
Every parametrised gate is :math:`e^{-i\theta G/2}` with :math:`G^2 = I` (RX, RY, RZ,
RZZ; and P(λ), which equals RZ(λ) up to a global phase). For such a gate the energy
:math:`E(\theta) = \langle\psi(\theta)|H|\psi(\theta)\rangle` is a trigonometric
polynomial :math:`\alpha + \beta\cos\theta + \gamma\sin\theta`, so

.. math::  \frac{\partial E}{\partial\theta} = \frac{E(\theta + \pi/2) - E(\theta - \pi/2)}{2}

**exactly** (not a finite-difference approximation). With
:math:`\theta_k = a_k x_j + b_k`, the chain rule sums over all gates that use
:math:`x_j`: :math:`\partial E/\partial x_j = \sum_{k: j(k)=j} a_k\,\partial E/\partial\theta_k`.

VQE and QAOA
------------
VQE minimises :math:`E(x)` with :func:`lipipy.numerics.minimize` and these exact
gradients; by the variational principle every value is :math:`\ge E_0` (the ground
energy). QAOA for MaxCut prepares
:math:`|\gamma,\beta\rangle = \prod_{l=1}^{p} e^{-i\beta_l B} e^{-i\gamma_l C}|+\rangle^{\otimes n}`
with the cut observable :math:`C` and mixer :math:`B = \sum_i X_i`, and maximises
:math:`\langle C\rangle`. Here :math:`e^{-i\gamma C}` is, up to global phase,
:math:`\prod_{(i,j)} R_{ZZ}(-\gamma w_{ij})` and :math:`e^{-i\beta X} = R_X(2\beta)`.

References: Mitarai, Negoro, Kitagawa & Fujii, PRA 98, 032309 (2018) and Schuld et al.,
PRA 99, 032331 (2019) (parameter shift); Peruzzo et al., Nat. Commun. 5, 4213 (2014)
(VQE); Farhi, Goldstone & Gutmann, arXiv:1411.4028 (QAOA); Wang, Hadfield, Jiang &
Rieffel, PRA 97, 022304 (2018) (analytic p = 1 QAOA).
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lipipy.core.complexity import Complexity
from lipipy.core.random import RandomState, check_random_state
from lipipy.core.registry import register
from lipipy.numerics import minimize
from lipipy.quantum import gates
from lipipy.quantum._validation import check_angle, check_int, check_qubits
from lipipy.quantum.circuit import Circuit, Instruction
from lipipy.quantum.observables import PauliSum, _check_edges, maxcut_hamiltonian

_SHIFT = math.pi / 2
_FIXED_MATRICES = {
    "h": gates.H,
    "x": gates.X,
    "y": gates.Y,
    "z": gates.Z,
    "s": gates.S,
    "sdg": gates.SDG,
    "t": gates.T,
    "tdg": gates.TDG,
    "sx": gates.SX,
    "cx": gates.CX,
    "cz": gates.CZ,
    "swap": gates.SWAP,
    "ccx": gates.CCX,
}


@dataclass(frozen=True)
class Parameter:
    """A named symbolic angle. Parameters with the same name are the same parameter.

    Supports ``a * theta``, ``theta * a``, ``-theta``, ``theta + b`` and ``theta - b``
    for real numbers ``a``, ``b`` (giving a :class:`ParameterExpression`).
    """

    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Parameter name must be a non-empty string")

    def _expr(self) -> ParameterExpression:
        return ParameterExpression(self, 1.0, 0.0)

    def __mul__(self, a: float) -> ParameterExpression:
        return self._expr() * a

    __rmul__ = __mul__

    def __neg__(self) -> ParameterExpression:
        return self._expr() * -1.0

    def __add__(self, b: float) -> ParameterExpression:
        return self._expr() + b

    __radd__ = __add__

    def __sub__(self, b: float) -> ParameterExpression:
        return self._expr() + (-_real(b, "offset"))

    def __repr__(self) -> str:
        return f"Parameter({self.name!r})"


def _real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return value


@dataclass(frozen=True)
class ParameterExpression:
    """The affine angle ``coeff * parameter + offset``."""

    parameter: Parameter
    coeff: float = 1.0
    offset: float = 0.0

    def __mul__(self, a: float) -> ParameterExpression:
        a = _real(a, "coefficient")
        return ParameterExpression(self.parameter, self.coeff * a, self.offset * a)

    __rmul__ = __mul__

    def __neg__(self) -> ParameterExpression:
        return self * -1.0

    def __add__(self, b: float) -> ParameterExpression:
        return ParameterExpression(self.parameter, self.coeff, self.offset + _real(b, "offset"))

    __radd__ = __add__

    def __sub__(self, b: float) -> ParameterExpression:
        return self + (-_real(b, "offset"))

    def value(self, x: float) -> float:
        return self.coeff * x + self.offset


Angle = float | Parameter | ParameterExpression


def _rotation(name: str, qubits: tuple[int, ...], theta: float) -> Instruction:
    """The same instruction as ``Circuit.<name>(theta, *qubits)``, built without re-validation."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    if name == "rx":
        return Instruction(
            name, qubits, (theta,), _matrix=np.array([[c, -1j * s], [-1j * s, c]], complex)
        )
    if name == "ry":
        return Instruction(name, qubits, (theta,), _matrix=np.array([[c, -s], [s, c]], complex))
    if name == "rz":
        diag = np.array([c - 1j * s, c + 1j * s])
    elif name == "p":
        diag = np.array([1.0, np.exp(1j * theta)])
    else:  # rzz
        diag = np.array([c - 1j * s, c + 1j * s, c + 1j * s, c - 1j * s])
    if name in ("rz", "p"):  # Circuit.rz / Circuit.p store dense matrices
        return Instruction(name, qubits, (theta,), _matrix=np.diag(diag).astype(np.complex128))
    return Instruction(name, qubits, (theta,), diagonal=diag.astype(np.complex128))


@dataclass(frozen=True)
class _Op:
    name: str
    qubits: tuple[int, ...]
    angle: float | ParameterExpression | None = None


@register(
    domain="quantum",
    tasks=("circuit", "variational"),
    complexity=Complexity(
        time="O(g 2^q)", space="O(2^q)", notes="bind is O(g); running the bound circuit O(g 2^q)"
    ),
    references=(
        "Peruzzo et al., A variational eigenvalue solver on a photonic quantum processor, "
        "Nat. Commun. 5, 4213 (2014)",
    ),
)
class ParametricCircuit:
    """A circuit whose rotation angles can be symbolic :class:`Parameter` expressions.

    Builder methods append in place and return ``self`` (like
    :class:`~lipipy.quantum.Circuit`); every argument is validated before anything is
    appended. Parametrisable gates: ``rx``, ``ry``, ``rz``, ``p`` and ``rzz``; fixed
    gates: ``h, x, y, z, s, sdg, t, tdg, sx, cx, cz, swap, ccx``.

    Parameters
    ----------
    num_qubits : int
        Number of qubits.

    Attributes
    ----------
    parameters : tuple of Parameter
        Distinct parameters in order of first use; :meth:`bind` takes values in this
        order.

    Complexity
    ----------
    :meth:`bind` O(g); simulation of the bound circuit O(g · 2^q).

    References
    ----------
    Peruzzo et al. (2014).

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.quantum import ParametricCircuit, Parameter
    >>> theta = Parameter("theta")
    >>> pc = ParametricCircuit(1).ry(theta, 0).rz(2 * theta + 0.1, 0)
    >>> pc.parameters
    (Parameter('theta'),)
    >>> (np.abs(pc.bind([np.pi]).run()) ** 2).round(12)   # RY(pi)|0> = |1>
    array([0., 1.])
    """

    def __init__(self, num_qubits: int):
        self.num_qubits = check_int(num_qubits, "num_qubits", minimum=1)
        self._ops: list[_Op] = []
        self._params: dict[Parameter, int] = {}

    # -- bookkeeping ---------------------------------------------------------------------
    @property
    def parameters(self) -> tuple[Parameter, ...]:
        return tuple(self._params)

    @property
    def num_parameters(self) -> int:
        return len(self._params)

    @property
    def num_gates(self) -> int:
        return len(self._ops)

    def __repr__(self) -> str:
        return (
            f"ParametricCircuit(num_qubits={self.num_qubits}, num_gates={self.num_gates}, "
            f"num_parameters={self.num_parameters})"
        )

    def _fixed(self, name: str, *qubits: int) -> ParametricCircuit:
        tgts = check_qubits(list(qubits), self.num_qubits, "qubits")
        self._ops.append(_Op(name, tgts))
        return self

    def _rot(self, name: str, angle: Angle, *qubits: int) -> ParametricCircuit:
        tgts = check_qubits(list(qubits), self.num_qubits, "qubits")
        if isinstance(angle, Parameter):
            angle = ParameterExpression(angle)
        if isinstance(angle, ParameterExpression):
            value: float | ParameterExpression = angle
        else:
            value = check_angle(angle, "angle")
        if isinstance(value, ParameterExpression) and value.parameter not in self._params:
            self._params[value.parameter] = len(self._params)
        self._ops.append(_Op(name, tgts, value))
        return self

    # -- gates ---------------------------------------------------------------------------
    def h(self, q: int) -> ParametricCircuit:
        return self._fixed("h", q)

    def x(self, q: int) -> ParametricCircuit:
        return self._fixed("x", q)

    def y(self, q: int) -> ParametricCircuit:
        return self._fixed("y", q)

    def z(self, q: int) -> ParametricCircuit:
        return self._fixed("z", q)

    def s(self, q: int) -> ParametricCircuit:
        return self._fixed("s", q)

    def sdg(self, q: int) -> ParametricCircuit:
        return self._fixed("sdg", q)

    def t(self, q: int) -> ParametricCircuit:
        return self._fixed("t", q)

    def tdg(self, q: int) -> ParametricCircuit:
        return self._fixed("tdg", q)

    def sx(self, q: int) -> ParametricCircuit:
        return self._fixed("sx", q)

    def cx(self, control: int, target: int) -> ParametricCircuit:
        return self._fixed("cx", control, target)

    def cz(self, control: int, target: int) -> ParametricCircuit:
        return self._fixed("cz", control, target)

    def swap(self, a: int, b: int) -> ParametricCircuit:
        return self._fixed("swap", a, b)

    def ccx(self, c1: int, c2: int, target: int) -> ParametricCircuit:
        return self._fixed("ccx", c1, c2, target)

    def rx(self, theta: Angle, q: int) -> ParametricCircuit:
        return self._rot("rx", theta, q)

    def ry(self, theta: Angle, q: int) -> ParametricCircuit:
        return self._rot("ry", theta, q)

    def rz(self, theta: Angle, q: int) -> ParametricCircuit:
        return self._rot("rz", theta, q)

    def p(self, lam: Angle, q: int) -> ParametricCircuit:
        return self._rot("p", lam, q)

    def rzz(self, theta: Angle, a: int, b: int) -> ParametricCircuit:
        return self._rot("rzz", theta, a, b)

    # -- binding -------------------------------------------------------------------------
    def _values(
        self, values: Sequence[float] | Mapping[Parameter | str, float] | np.ndarray
    ) -> np.ndarray:
        n = self.num_parameters
        if isinstance(values, Mapping):
            by_name = {}
            for key, v in values.items():
                name = key.name if isinstance(key, Parameter) else key
                if not isinstance(name, str):
                    raise TypeError("mapping keys must be Parameters or names")
                by_name[name] = _real(v, f"value of {name}")
            missing = [p.name for p in self._params if p.name not in by_name]
            extra = set(by_name) - {p.name for p in self._params}
            if missing or extra:
                raise ValueError(
                    f"missing parameters {missing}, unknown parameters {sorted(extra)}"
                )
            return np.array([by_name[p.name] for p in self._params], dtype=np.float64)
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 1 or x.size != n:
            raise ValueError(f"expected {n} parameter values, got shape {x.shape}")
        if not np.all(np.isfinite(x)):
            raise ValueError("parameter values must be finite")
        return x

    def _bind(self, x: np.ndarray, shift_op: int = -1, delta: float = 0.0) -> Circuit:
        # Qubits and fixed angles were validated when the gates were added, so the
        # instructions are built directly (this is the hot loop of the gradient).
        circ = Circuit(self.num_qubits)
        for k, op in enumerate(self._ops):
            if op.angle is None:
                circ._append(Instruction(op.name, op.qubits, _matrix=_FIXED_MATRICES[op.name]))
                continue
            if isinstance(op.angle, ParameterExpression):
                theta = op.angle.value(float(x[self._params[op.angle.parameter]]))
            else:
                theta = op.angle
            if k == shift_op:
                theta += delta
            circ._append(_rotation(op.name, op.qubits, theta))
        return circ

    def bind(
        self, values: Sequence[float] | Mapping[Parameter | str, float] | np.ndarray
    ) -> Circuit:
        """A numeric :class:`~lipipy.quantum.Circuit` with the parameters substituted.

        ``values`` is a sequence in the order of :attr:`parameters`, or a mapping from
        parameters (or their names) to values.
        """
        return self._bind(self._values(values))

    def _shift_terms(self) -> list[tuple[int, int, float]]:
        """(op index, parameter index, chain-rule coefficient) for each symbolic gate."""
        out = []
        for k, op in enumerate(self._ops):
            if isinstance(op.angle, ParameterExpression) and op.angle.coeff != 0.0:
                out.append((k, self._params[op.angle.parameter], op.angle.coeff))
        return out


def _check_pair(circuit: object, observable: object) -> None:
    if not isinstance(circuit, ParametricCircuit):
        raise TypeError(f"circuit must be a ParametricCircuit, got {type(circuit).__name__}")
    if not isinstance(observable, PauliSum):
        raise TypeError(f"observable must be a PauliSum, got {type(observable).__name__}")
    if circuit.num_qubits != observable.num_qubits:
        raise ValueError(
            f"circuit has {circuit.num_qubits} qubits, observable {observable.num_qubits}"
        )


@register(
    domain="quantum",
    tasks=("variational", "expectation"),
    complexity=Complexity(time="O(g 2^q + T 2^q)", space="O(2^q)", notes="T = Pauli terms"),
    references=("Peruzzo et al., Nat. Commun. 5, 4213 (2014)",),
)
def expectation_value(
    circuit: ParametricCircuit, observable: PauliSum, params: Sequence[float] | np.ndarray
) -> float:
    r"""Energy :math:`E(x) = \langle 0|U(x)^\dagger H U(x)|0\rangle` of a parametrised circuit.

    Parameters
    ----------
    circuit : ParametricCircuit
    observable : PauliSum
        On the same number of qubits.
    params : array_like of shape (num_parameters,)

    Returns
    -------
    float

    Complexity
    ----------
    O(g · 2^q) to simulate plus O(T · 2^q) for T Pauli terms; memory O(2^q).

    References
    ----------
    Peruzzo et al. (2014).

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.quantum import ParametricCircuit, Parameter, PauliSum, expectation_value
    >>> pc = ParametricCircuit(1).ry(Parameter("t"), 0)
    >>> round(expectation_value(pc, PauliSum({"Z": 1.0}), [np.pi / 3]), 12)  # cos(t)
    0.5
    """
    _check_pair(circuit, observable)
    return observable.expectation(circuit._bind(circuit._values(params)).run())


@register(
    domain="quantum",
    tasks=("variational", "gradient"),
    complexity=Complexity(
        time="O(g^2 2^q + g T 2^q)",
        space="O(2^q)",
        notes="2 circuit simulations per parametrised gate (at most g); T = Pauli terms",
    ),
    references=(
        "Mitarai, Negoro, Kitagawa & Fujii, Quantum circuit learning, PRA 98, 032309 (2018)",
        "Schuld, Bergholm, Gogolin, Izaac & Killoran, Evaluating analytic gradients on "
        "quantum hardware, PRA 99, 032331 (2019)",
    ),
)
def parameter_shift_gradient(
    circuit: ParametricCircuit, observable: PauliSum, params: Sequence[float] | np.ndarray
) -> np.ndarray:
    r"""Exact gradient :math:`\nabla_x E(x)` by the parameter-shift rule.

    For each parametrised gate k with angle :math:`a_k x_j + b_k`, evaluate the energy
    with that gate's angle shifted by :math:`\pm\pi/2` and add
    :math:`a_k\,[E_k^+ - E_k^-]/2` to component j.

    Parameters
    ----------
    circuit : ParametricCircuit
    observable : PauliSum
        On the same number of qubits.
    params : array_like of shape (num_parameters,)

    Returns
    -------
    ndarray of float64, shape (num_parameters,)

    Complexity
    ----------
    2 simulations per parametrised gate: time O(m (g + T) 2^q) for m <= g such gates,
    memory O(2^q).

    References
    ----------
    Mitarai et al. (2018); Schuld et al. (2019).

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.quantum import ParametricCircuit, Parameter, PauliSum, parameter_shift_gradient
    >>> pc = ParametricCircuit(1).ry(Parameter("t"), 0)
    >>> parameter_shift_gradient(pc, PauliSum({"Z": 1.0}), [np.pi / 2]).round(12)  # -sin(t)
    array([-1.])
    """
    _check_pair(circuit, observable)
    x = circuit._values(params)
    grad = np.zeros(circuit.num_parameters)
    for k, j, coeff in circuit._shift_terms():
        e_plus = observable.expectation(circuit._bind(x, k, _SHIFT).run())
        e_minus = observable.expectation(circuit._bind(x, k, -_SHIFT).run())
        grad[j] += coeff * (e_plus - e_minus) / 2
    return grad


@register(
    domain="quantum",
    tasks=("variational", "ansatz"),
    complexity=Complexity(time="O(q r)", space="O(q r)", notes="r = reps"),
    references=(
        "Kandala et al., Hardware-efficient variational quantum eigensolver for small "
        "molecules and quantum magnets, Nature 549, 242 (2017)",
    ),
)
def real_amplitudes(num_qubits: int, reps: int = 1) -> ParametricCircuit:
    r"""Hardware-efficient ansatz with real amplitudes: RY layers and CX chains.

    ``reps`` times [RY on every qubit, then CX(i, i+1) for i = 0..q-2], followed by a
    final RY layer: :math:`(r+1)q` parameters named ``θ[0]``, ``θ[1]``, ... Every
    prepared state has real amplitudes.

    Parameters
    ----------
    num_qubits : int
    reps : int, default=1
        Number of entangling layers (>= 0).

    Returns
    -------
    ParametricCircuit

    Complexity
    ----------
    O(q · r) gates.

    References
    ----------
    Kandala et al. (2017).

    Examples
    --------
    >>> from lipipy.quantum import real_amplitudes
    >>> real_amplitudes(3, reps=2).num_parameters
    9
    """
    q = check_int(num_qubits, "num_qubits", minimum=1)
    r = check_int(reps, "reps", minimum=0)
    pc = ParametricCircuit(q)
    count = 0
    for layer in range(r + 1):
        for i in range(q):
            pc.ry(Parameter(f"θ[{count}]"), i)
            count += 1
        if layer < r:
            for i in range(q - 1):
                pc.cx(i, i + 1)
    return pc


@dataclass(frozen=True)
class VQEResult:
    """Output of :func:`vqe`.

    Attributes
    ----------
    energy : float
        Lowest energy found (an upper bound on the ground energy).
    parameters : ndarray
        Parameters reaching it, ordered as ``ansatz.parameters``.
    history : tuple of float
        Energy at every objective evaluation, in order.
    nit, nfev, ngev : int
        Optimizer iterations and objective / gradient evaluations.
    success : bool
        Whether the gradient tolerance was met.
    message : str
    """

    energy: float
    parameters: np.ndarray = field(repr=False)
    history: tuple[float, ...] = field(repr=False)
    nit: int
    nfev: int
    ngev: int
    success: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "energy": self.energy,
            "parameters": self.parameters.tolist(),
            "history": list(self.history),
            "nit": self.nit,
            "nfev": self.nfev,
            "ngev": self.ngev,
            "success": self.success,
            "message": self.message,
        }


@register(
    domain="quantum",
    tasks=("variational", "eigenvalue", "optimization"),
    complexity=Complexity(
        time="O(K m (g + T) 2^q)",
        space="O(2^q + m^2)",
        notes="K optimizer iterations, m parameters (2m simulations per gradient), "
        "T Pauli terms; BFGS keeps an m x m matrix",
    ),
    references=(
        "Peruzzo et al., A variational eigenvalue solver on a photonic quantum processor, "
        "Nat. Commun. 5, 4213 (2014)",
        "Nocedal & Wright, Numerical Optimization, 2nd ed., ch. 6 (BFGS)",
    ),
)
def vqe(
    hamiltonian: PauliSum,
    ansatz: ParametricCircuit,
    x0: Sequence[float] | np.ndarray | None = None,
    method: str = "bfgs",
    random_state: RandomState = None,
    *,
    tol: float = 1e-6,
    maxiter: int = 1000,
) -> VQEResult:
    r"""Variational quantum eigensolver: minimise :math:`E(x)` with exact gradients.

    Uses :func:`lipipy.numerics.minimize` with ``jac`` =
    :func:`parameter_shift_gradient`. The result is a local minimum; every energy is
    :math:`\ge E_0` by the variational principle, and equals it when the ansatz can
    express a ground state and the optimizer reaches the global minimum.

    Parameters
    ----------
    hamiltonian : PauliSum
    ansatz : ParametricCircuit
        With at least one parameter, on the same number of qubits.
    x0 : array_like, optional
        Initial parameters; default uniform in :math:`[-\pi, \pi)` from ``random_state``.
    method : {"bfgs", "lbfgs", "gd"}, default="bfgs"
        Optimizer of :func:`lipipy.numerics.minimize`.
    random_state : None, int, SeedSequence or Generator
        Used only to draw ``x0``.
    tol : float, default=1e-6
        Gradient infinity-norm tolerance. Near a minimum with curvature λ the energy
        error is about tol² / (2λ) (≈1e-12 here); much smaller values run into the
        rounding floor of the energy and trigger a ConvergenceWarning.
    maxiter : int, default=1000

    Returns
    -------
    VQEResult

    Complexity
    ----------
    Each gradient needs 2m simulations for m parametrised gates: time
    O(K · m · (g + T) · 2^q) for K iterations; memory O(2^q + m^2).

    References
    ----------
    Peruzzo et al. (2014).

    Examples
    --------
    >>> import numpy as np
    >>> from lipipy.quantum import h2_hamiltonian, real_amplitudes, vqe
    >>> res = vqe(h2_hamiltonian(), real_amplitudes(2, reps=1), random_state=0)
    >>> round(res.energy, 6)
    -1.137306
    """
    if not isinstance(hamiltonian, PauliSum):
        raise TypeError(f"hamiltonian must be a PauliSum, got {type(hamiltonian).__name__}")
    _check_pair(ansatz, hamiltonian)
    if method not in ("bfgs", "lbfgs", "gd"):
        raise ValueError(f"method must be 'bfgs', 'lbfgs' or 'gd', got {method!r}")
    m = ansatz.num_parameters
    if m == 0:
        raise ValueError("the ansatz has no parameters")
    if x0 is None:
        x = check_random_state(random_state).uniform(-math.pi, math.pi, size=m)
    else:
        x = ansatz._values(x0)

    history: list[float] = []

    def energy(v: np.ndarray) -> float:
        e = expectation_value(ansatz, hamiltonian, v)
        history.append(e)
        return e

    def grad(v: np.ndarray) -> np.ndarray:
        return parameter_shift_gradient(ansatz, hamiltonian, v)

    res = minimize(energy, x, grad, method=method, tol=tol, maxiter=maxiter)
    return VQEResult(
        energy=float(res.fun),
        parameters=np.asarray(res.x, dtype=np.float64).copy(),
        history=tuple(history),
        nit=res.nit,
        nfev=res.nfev,
        ngev=res.ngev,
        success=res.success,
        message=res.message,
    )


@register(
    domain="quantum",
    tasks=("variational", "ansatz", "combinatorial-optimization"),
    complexity=Complexity(time="O(p (E + q))", space="O(p (E + q))", notes="E = edges"),
    references=("Farhi, Goldstone & Gutmann, arXiv:1411.4028",),
)
def qaoa_circuit(
    edges: Iterable[tuple[int, int]],
    p: int = 1,
    num_nodes: int | None = None,
    weights: Iterable[float] | None = None,
) -> ParametricCircuit:
    r"""QAOA ansatz for MaxCut with parameters ``γ[0..p-1]`` then ``β[0..p-1]``.

    :math:`H^{\otimes n}`, then for each layer l: :math:`R_{ZZ}(-w_{ij}\gamma_l)` on
    every edge (= :math:`e^{-i\gamma_l C}` up to global phase) and
    :math:`R_X(2\beta_l)` on every qubit (= :math:`e^{-i\beta_l B}`).

    Parameters
    ----------
    edges : iterable of (int, int)
    p : int, default=1
        Number of layers.
    num_nodes : int, optional
        Default ``1 + max node``.
    weights : iterable of float, optional
        Edge weights (default 1).

    Returns
    -------
    ParametricCircuit
        Parameters ordered ``(γ_0, ..., γ_{p-1}, β_0, ..., β_{p-1})``.

    Complexity
    ----------
    O(p (E + q)) gates.

    References
    ----------
    Farhi, Goldstone & Gutmann (2014).

    Examples
    --------
    >>> from lipipy.quantum import qaoa_circuit
    >>> qc = qaoa_circuit([(0, 1), (1, 2)], p=2)
    >>> [prm.name for prm in qc.parameters]
    ['γ[0]', 'γ[1]', 'β[0]', 'β[1]']
    """
    edge_list = _check_edges(edges, num_nodes)
    n = num_nodes if num_nodes is not None else 1 + max(max(e) for e in edge_list)
    layers = check_int(p, "p", minimum=1)
    w = [1.0] * len(edge_list) if weights is None else [_real(v, "weight") for v in weights]
    if len(w) != len(edge_list):
        raise ValueError("weights must have one entry per edge")
    gammas = [Parameter(f"γ[{i}]") for i in range(layers)]
    betas = [Parameter(f"β[{i}]") for i in range(layers)]
    pc = ParametricCircuit(n)
    # register γ before β so the parameter order is (γ..., β...)
    pc._params.update({g: i for i, g in enumerate(gammas)})
    pc._params.update({b: layers + i for i, b in enumerate(betas)})
    for i in range(n):
        pc.h(i)
    for gamma, beta in zip(gammas, betas, strict=True):
        for (a, b), wt in zip(edge_list, w, strict=True):
            pc.rzz(-wt * gamma, a, b)
        for i in range(n):
            pc.rx(2 * beta, i)
    return pc


@dataclass(frozen=True)
class QAOAResult:
    """Output of :func:`qaoa_maxcut`.

    Attributes
    ----------
    expected_cut : float
        Optimised :math:`\\langle C\\rangle`.
    gammas, betas : ndarray of shape (p,)
        Optimal angles.
    probabilities : ndarray of shape (2**n,)
        Measurement distribution of the optimal state (bit i of the index = node i).
    most_likely : str
        Most probable bitstring (node n-1 first, Qiskit order).
    most_likely_cut : float
        Cut value of ``most_likely``.
    history : tuple of float
        :math:`\\langle C\\rangle` at every objective evaluation of the best start.
    num_starts : int
    success : bool
        Whether the best start met the gradient tolerance.
    """

    expected_cut: float
    gammas: np.ndarray = field(repr=False)
    betas: np.ndarray = field(repr=False)
    probabilities: np.ndarray = field(repr=False)
    most_likely: str
    most_likely_cut: float
    history: tuple[float, ...] = field(repr=False)
    num_starts: int
    success: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_cut": self.expected_cut,
            "gammas": self.gammas.tolist(),
            "betas": self.betas.tolist(),
            "most_likely": self.most_likely,
            "most_likely_cut": self.most_likely_cut,
            "history": list(self.history),
            "num_starts": self.num_starts,
            "success": self.success,
        }


@register(
    domain="quantum",
    tasks=("variational", "combinatorial-optimization", "maxcut"),
    complexity=Complexity(
        time="O(S K p (p (E + q) + E) 2^q)",
        space="O(2^q + p^2)",
        notes="S starts, K iterations each, 4p simulations per gradient, E edges",
    ),
    references=(
        "Farhi, Goldstone & Gutmann, A Quantum Approximate Optimization Algorithm, arXiv:1411.4028",
        "Wang, Hadfield, Jiang & Rieffel, Quantum approximate optimization algorithm for "
        "MaxCut: a fermionic view, PRA 97, 022304 (2018)",
    ),
)
def qaoa_maxcut(
    edges: Iterable[tuple[int, int]],
    p: int = 1,
    num_nodes: int | None = None,
    weights: Iterable[float] | None = None,
    x0: Sequence[float] | np.ndarray | None = None,
    method: str = "bfgs",
    num_starts: int = 8,
    random_state: RandomState = None,
    *,
    tol: float = 1e-6,
    maxiter: int = 1000,
) -> QAOAResult:
    r"""QAOA for MaxCut: maximise :math:`\langle\gamma,\beta|C|\gamma,\beta\rangle`.

    Minimises :math:`-\langle C\rangle` with :func:`lipipy.numerics.minimize` and
    parameter-shift gradients, from ``x0`` or from ``num_starts`` random points
    (:math:`\gamma \in [0, \pi)`, :math:`\beta \in [0, \pi/2)`), keeping the best.

    Parameters
    ----------
    edges : iterable of (int, int)
    p : int, default=1
        QAOA depth.
    num_nodes : int, optional
    weights : iterable of float, optional
    x0 : array_like of shape (2p,), optional
        ``(γ..., β...)`` starting point; if given, ``num_starts`` is ignored.
    method : {"bfgs", "lbfgs", "gd"}, default="bfgs"
    num_starts : int, default=8
        Random restarts (the landscape has several local optima).
    random_state : None, int, SeedSequence or Generator
        Draws the random starts.
    tol : float, default=1e-6
        Gradient infinity-norm tolerance (see :func:`vqe`).
    maxiter : int, default=1000

    Returns
    -------
    QAOAResult

    Complexity
    ----------
    Per start, K iterations of 4p simulations (O(p (E + q) 2^q) each) plus O(E 2^q)
    expectations; memory O(2^q).

    References
    ----------
    Farhi et al. (2014); Wang et al. (2018).

    Examples
    --------
    >>> from lipipy.quantum import qaoa_maxcut
    >>> res = qaoa_maxcut([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)], p=1, random_state=0)
    >>> round(res.expected_cut, 6), res.most_likely_cut     # 5-ring: 3/4 per edge
    (3.75, 4.0)
    """
    edge_list = _check_edges(edges, num_nodes)
    w_list = None if weights is None else list(weights)
    ansatz = qaoa_circuit(edge_list, p, num_nodes, w_list)
    cost = maxcut_hamiltonian(edge_list, ansatz.num_qubits, w_list)
    if method not in ("bfgs", "lbfgs", "gd"):
        raise ValueError(f"method must be 'bfgs', 'lbfgs' or 'gd', got {method!r}")
    layers = ansatz.num_parameters // 2
    neg = -1.0 * cost
    if x0 is not None:
        starts = [ansatz._values(x0)]
    else:
        n_starts = check_int(num_starts, "num_starts", minimum=1)
        rng = check_random_state(random_state)
        starts = [
            np.concatenate([rng.uniform(0, math.pi, layers), rng.uniform(0, math.pi / 2, layers)])
            for _ in range(n_starts)
        ]
    runs = [vqe(neg, ansatz, start, method=method, tol=tol, maxiter=maxiter) for start in starts]
    best = min(runs, key=lambda r: r.energy)
    x = best.parameters
    probs = np.abs(ansatz.bind(x).run()) ** 2
    idx = int(np.argmax(probs))
    n = ansatz.num_qubits
    basis = np.zeros(2**n, dtype=np.complex128)
    basis[idx] = 1.0
    ml_cut = cost.expectation(basis)
    return QAOAResult(
        expected_cut=-best.energy,
        gammas=x[:layers].copy(),
        betas=x[layers:].copy(),
        probabilities=probs,
        most_likely=format(idx, f"0{n}b"),
        most_likely_cut=ml_cut,
        history=tuple(-e for e in best.history),
        num_starts=len(starts),
        success=best.success,
    )
