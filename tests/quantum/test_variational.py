"""Parametric circuits, exact parameter-shift gradients, VQE and QAOA."""

import itertools
import math

import numpy as np
import pytest

from lipipy.quantum import (
    Circuit,
    Parameter,
    ParameterExpression,
    ParametricCircuit,
    PauliSum,
    expectation_value,
    h2_hamiltonian,
    ising_chain,
    maxcut_hamiltonian,
    parameter_shift_gradient,
    qaoa_circuit,
    qaoa_maxcut,
    real_amplitudes,
    vqe,
)
from tests.quantum.test_observables import random_paulisum


def random_ansatz(q, rng, num_params=3, layers=6):
    params = [Parameter(f"x{i}") for i in range(num_params)]
    pc = ParametricCircuit(q)
    for _ in range(layers):
        for j in range(q):
            kind = str(rng.choice(["rx", "ry", "rz", "p", "h", "s", "fixed"]))
            if kind in ("h", "s"):
                getattr(pc, kind)(j)
            elif kind == "fixed":
                pc.ry(float(rng.uniform(-3, 3)), j)
            else:
                prm = params[int(rng.integers(num_params))]
                expr = float(rng.uniform(-2, 2)) * prm + float(rng.uniform(-1, 1))
                getattr(pc, kind)(expr, j)
        for j in range(q - 1):
            if rng.uniform() < 0.5:
                pc.cx(j, j + 1)
            else:
                pc.rzz(params[int(rng.integers(num_params))] * 0.7, j, j + 1)
    return pc


@pytest.mark.parametrize("seed", range(12))
def test_parameter_shift_matches_central_differences(seed):
    rng = np.random.default_rng(seed)
    q = int(rng.integers(1, 5))
    pc = random_ansatz(q, rng)
    H = random_paulisum(q, 6, rng)
    x = rng.uniform(-math.pi, math.pi, size=pc.num_parameters)
    grad = parameter_shift_gradient(pc, H, x)
    h = 1e-5
    fd = np.array(
        [
            (expectation_value(pc, H, x + h * e) - expectation_value(pc, H, x - h * e)) / (2 * h)
            for e in np.eye(pc.num_parameters)
        ]
    )
    np.testing.assert_allclose(grad, fd, atol=1e-8)


def test_parameter_shift_is_exact_for_one_rotation():
    pc = ParametricCircuit(1).rx(Parameter("t"), 0)
    Z = PauliSum({"Z": 1.0})
    for t in np.linspace(-3, 3, 13):
        assert expectation_value(pc, Z, [t]) == pytest.approx(math.cos(t), abs=1e-14)
        assert parameter_shift_gradient(pc, Z, [t])[0] == pytest.approx(-math.sin(t), abs=1e-14)


def test_zero_coefficient_and_shared_parameters():
    t = Parameter("t")
    pc = ParametricCircuit(1).ry(0 * t, 0)
    assert parameter_shift_gradient(pc, PauliSum({"Z": 1.0}), [0.3])[0] == 0.0
    shared = ParametricCircuit(1).ry(t, 0).ry(t, 0)  # = RY(2t): d<Z>/dt = -2 sin 2t
    g = parameter_shift_gradient(shared, PauliSum({"Z": 1.0}), [0.4])[0]
    assert g == pytest.approx(-2 * math.sin(0.8), abs=1e-14)


def test_bind_and_expressions(rng):
    a, b = Parameter("a"), Parameter("b")
    expr = -(2 * a) + 0.5 - 0.25
    assert expr == ParameterExpression(a, -2.0, 0.25)
    assert (a - 1.0).value(3.0) == 2.0 and (1.0 + a).offset == 1.0
    pc = ParametricCircuit(2).h(0).rx(expr, 0).rzz(b, 0, 1).cx(0, 1).p(a * 3, 1)
    assert pc.parameters == (a, b) and pc.num_parameters == 2 and pc.num_gates == 5
    bound = pc.bind([0.3, -0.7])
    ref = Circuit(2).h(0).rx(-0.6 + 0.25, 0).rzz(-0.7, 0, 1).cx(0, 1).p(0.9, 1)
    np.testing.assert_allclose(bound.run(), ref.run(), atol=1e-15)
    np.testing.assert_allclose(pc.bind({"a": 0.3, b: -0.7}).run(), ref.run(), atol=1e-15)
    assert "num_parameters=2" in repr(pc)
    with pytest.raises(ValueError, match="expected 2"):
        pc.bind([0.1])
    with pytest.raises(ValueError, match="missing"):
        pc.bind({"a": 0.1})
    with pytest.raises(ValueError, match="unknown"):
        pc.bind({"a": 0.1, "b": 0.2, "c": 0.3})
    with pytest.raises(ValueError, match="finite"):
        pc.bind([np.nan, 0.0])
    with pytest.raises(ValueError):
        Parameter("")
    with pytest.raises(TypeError):
        a * "2"
    with pytest.raises(ValueError, match="duplicate"):
        pc.rzz(a, 1, 1)
    assert pc.num_gates == 5  # failed call appended nothing


def test_fast_bind_equals_validated_builder(rng):
    """The unvalidated bind path builds exactly what the public Circuit methods build."""
    for seed in range(5):
        pc = random_ansatz(3, np.random.default_rng(seed))
        x = rng.uniform(-3, 3, pc.num_parameters)
        fast = pc.bind(x)
        slow = Circuit(3)
        for op in pc._ops:
            if op.angle is None:
                getattr(slow, op.name)(*op.qubits)
            else:
                theta = (
                    op.angle
                    if isinstance(op.angle, float)
                    else op.angle.value(float(x[pc._params[op.angle.parameter]]))
                )
                getattr(slow, op.name)(theta, *op.qubits)
        assert len(fast) == len(slow)
        for a, b in zip(fast.instructions, slow.instructions, strict=True):
            assert (a.name, a.qubits, a.params) == (b.name, b.qubits, b.params)
            assert (a.diagonal is None) == (b.diagonal is None)
            np.testing.assert_allclose(a.matrix, b.matrix, atol=1e-15)


def test_circuit_rzz_definition():
    theta = 0.83
    ZZ = np.diag([1, -1, -1, 1])
    expected = np.diag(np.exp(-0.5j * theta * np.diag(ZZ)))
    np.testing.assert_allclose(Circuit(2).rzz(theta, 0, 1).to_matrix(), expected, atol=1e-15)
    c = Circuit(3).rzz(theta, 2, 0)
    np.testing.assert_allclose(c.compose(c.inverse()).to_matrix(), np.eye(8), atol=1e-15)
    assert c.inverse().instructions[0].params == (-theta,)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("seed", range(10))
def test_vqe_h2_reaches_ground_energy(seed):
    H = h2_hamiltonian()
    exact = np.linalg.eigvalsh(H.to_matrix())[0]
    res = vqe(H, real_amplitudes(2, reps=1), random_state=seed)
    assert res.success
    assert res.energy == pytest.approx(exact, abs=1e-6)
    assert min(res.history) >= exact - 1e-12  # variational principle
    assert res.energy == pytest.approx(expectation_value(real_amplitudes(2, 1), H, res.parameters))


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(("q", "h", "seeds"), [(3, 0.7, 4), (3, 1.5, 4), (4, 0.7, 2), (4, 1.0, 2)])
def test_vqe_tfim_reaches_ground_energy(q, h, seeds):
    H = ising_chain(q, J=1.0, h=h)
    exact = np.linalg.eigvalsh(H.to_matrix())[0]
    for seed in range(seeds):
        res = vqe(H, real_amplitudes(q, reps=q - 1), random_state=seed)
        assert res.energy == pytest.approx(exact, abs=1e-6)
        assert min(res.history) >= exact - 1e-12


@pytest.mark.filterwarnings("error")
def test_vqe_degenerate_ground_state_compares_energy_only():
    H = ising_chain(3, J=1.0, h=0.0)  # ground states |000>, |111> (any superposition)
    for seed in range(5):
        res = vqe(H, real_amplitudes(3, reps=2), random_state=seed)
        assert res.energy == pytest.approx(-2.0, abs=1e-6)


def test_vqe_options_and_result():
    H = h2_hamiltonian()
    ansatz = real_amplitudes(2, 1)
    a = vqe(H, ansatz, random_state=3)
    b = vqe(H, ansatz, random_state=3)
    assert a.energy == b.energy and np.array_equal(a.parameters, b.parameters)
    lb = vqe(H, ansatz, x0=a.parameters + 0.1, method="lbfgs")
    assert lb.energy == pytest.approx(a.energy, abs=1e-6)
    d = a.to_dict()
    assert d["energy"] == a.energy and len(d["parameters"]) == 4 and d["nfev"] == len(a.history)
    with pytest.raises(ValueError, match="method"):
        vqe(H, ansatz, method="newton")
    with pytest.raises(ValueError, match="qubits"):
        vqe(ising_chain(3), ansatz)
    with pytest.raises(ValueError, match="no parameters"):
        vqe(H, ParametricCircuit(2).h(0))
    with pytest.raises(TypeError):
        vqe(H.to_matrix(), ansatz)


def wang_p1(edges, n, gamma, beta):
    """Exact <C> for p = 1 QAOA on any unweighted graph (Wang et al., PRA 97, 022304).

    Vectorised over array-valued ``gamma`` and ``beta``.
    """
    gamma, beta = np.asarray(gamma, dtype=float), np.asarray(beta, dtype=float)
    deg = [0] * n
    adj = {i: set() for i in range(n)}
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
        adj[a].add(b)
        adj[b].add(a)
    cg = np.cos(gamma)
    total = np.zeros(np.broadcast(gamma, beta).shape)
    for u, v in edges:
        du, dv, lam = deg[u], deg[v], len(adj[u] & adj[v])
        total = total + (
            0.5
            + 0.25 * np.sin(4 * beta) * np.sin(gamma) * (cg ** (du - 1) + cg ** (dv - 1))
            - 0.25
            * np.sin(2 * beta) ** 2
            * cg ** (du + dv - 2 - 2 * lam)
            * (1 - np.cos(2 * gamma) ** lam)
        )
    return total


def ring(n):
    return [(i, (i + 1) % n) for i in range(n)]


GRAPHS = {
    "triangle": ([(0, 1), (1, 2), (0, 2)], 3),
    "ring4": (ring(4), 4),
    "ring5": (ring(5), 5),
    "ring6": (ring(6), 6),
    "kite": ([(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)], 4),
}


def brute_max_cut(edges, n):
    return max(sum(((z >> a) ^ (z >> b)) & 1 for a, b in edges) for z in range(2**n))


@pytest.mark.parametrize("name", GRAPHS)
def test_qaoa_p1_expectation_matches_analytic_formula(name, rng):
    edges, n = GRAPHS[name]
    pc, C = qaoa_circuit(edges), maxcut_hamiltonian(edges)
    for _ in range(20):
        g, b = rng.uniform(-math.pi, math.pi, 2)
        assert expectation_value(pc, C, [g, b]) == pytest.approx(wang_p1(edges, n, g, b), abs=1e-12)


def _grid_max(edges, n):
    """Max of the exact p = 1 formula over a 2001 x 1001 grid of (γ in [0, π], β in [0, π/2])."""
    g, b = np.meshgrid(np.linspace(0, math.pi, 2001), np.linspace(0, math.pi / 2, 1001))
    return float(wang_p1(edges, n, g, b).max())


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("name", GRAPHS)
def test_qaoa_p1_reaches_analytic_optimum_and_a_max_cut(name):
    edges, n = GRAPHS[name]
    grid_max = _grid_max(edges, n)  # the formula has period π in γ and π/2 in β
    if name.startswith("ring"):
        assert grid_max == pytest.approx(0.75 * n, abs=1e-5)  # 3/4 per edge on rings
    max_cut = brute_max_cut(edges, n)
    for seed in range(3):
        res = qaoa_maxcut(edges, p=1, random_state=seed, num_starts=4)
        # global optimum: no grid point is better, and the value is the exact formula's
        assert res.expected_cut >= grid_max - 1e-12
        assert res.expected_cut == pytest.approx(
            float(wang_p1(edges, n, res.gammas[0], res.betas[0])), abs=1e-12
        )
        assert res.expected_cut - grid_max < 1e-4  # grid spacing ~1.6e-3 rad
        if name.startswith("ring"):
            assert res.expected_cut == pytest.approx(0.75 * n, abs=1e-10)
        assert res.expected_cut <= max_cut + 1e-12
        assert res.most_likely_cut == max_cut  # compare cut values, not bitstrings
        z = int(res.most_likely, 2)
        assert sum(((z >> a) ^ (z >> b)) & 1 for a, b in edges) == max_cut
        assert res.probabilities.sum() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.filterwarnings("error")
def test_qaoa_deeper_and_weighted():
    edges = ring(5)
    p2 = qaoa_maxcut(edges, p=2, random_state=0, num_starts=3)
    assert 3.75 - 1e-9 <= p2.expected_cut <= brute_max_cut(edges, 5) + 1e-12
    assert p2.gammas.shape == (2,) and p2.betas.shape == (2,)
    w = [1.0, 2.0, 0.5]
    res = qaoa_maxcut([(0, 1), (1, 2), (0, 2)], weights=w, random_state=1)
    C = maxcut_hamiltonian([(0, 1), (1, 2), (0, 2)], weights=w)
    pc = qaoa_circuit([(0, 1), (1, 2), (0, 2)], weights=w)
    x = np.concatenate([res.gammas, res.betas])
    assert expectation_value(pc, C, x) == pytest.approx(res.expected_cut, abs=1e-12)
    assert res.most_likely_cut == 3.0  # best weighted cut: separate node 1
    d = res.to_dict()
    assert d["num_starts"] == 8 and d["most_likely"] == res.most_likely


def test_qaoa_x0_reproducibility_and_errors():
    edges = ring(4)
    a = qaoa_maxcut(edges, random_state=7)
    b = qaoa_maxcut(edges, random_state=7)
    assert a.expected_cut == b.expected_cut and a.most_likely == b.most_likely
    one = qaoa_maxcut(edges, x0=[0.4, 0.3])
    assert one.num_starts == 1
    with pytest.raises(ValueError, match="expected 2"):
        qaoa_maxcut(edges, x0=[0.1])
    with pytest.raises(ValueError, match="num_starts"):
        qaoa_maxcut(edges, num_starts=0)
    with pytest.raises(ValueError, match="p"):
        qaoa_circuit(edges, p=0)
    with pytest.raises(ValueError, match="method"):
        qaoa_maxcut(edges, method="nelder-mead")
    names = [prm.name for prm in qaoa_circuit(edges, p=3).parameters]
    assert names == ["γ[0]", "γ[1]", "γ[2]", "β[0]", "β[1]", "β[2]"]


def test_real_amplitudes_states_are_real(rng):
    for q, reps in itertools.product((1, 2, 3), (0, 1, 2)):
        pc = real_amplitudes(q, reps)
        assert pc.num_parameters == (reps + 1) * q
        psi = pc.bind(rng.uniform(-3, 3, pc.num_parameters)).run()
        np.testing.assert_allclose(psi.imag, 0, atol=1e-15)
