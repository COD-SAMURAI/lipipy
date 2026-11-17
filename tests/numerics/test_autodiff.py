"""lipipy.numerics.autodiff: forward and reverse mode vs finite differences (and torch)."""

import math
import sys

import numpy as np
import pytest

from lipipy.bench import estimate_complexity
from lipipy.numerics.autodiff import (
    Dual,
    Tensor,
    check_grad,
    concatenate,
    derivative,
    grad,
    jvp,
    maximum,
    stack,
    value_and_grad,
    where,
)


def fd_grads(fun, arrays, eps=1e-6):
    """Central-difference gradient of fun(list of Tensors) -> scalar w.r.t. each array."""
    out = []
    for k, a in enumerate(arrays):
        g = np.zeros_like(a)
        for i in np.ndindex(a.shape):
            plus = [x.copy() for x in arrays]
            minus = [x.copy() for x in arrays]
            plus[k][i] += eps
            minus[k][i] -= eps
            fp = fun([Tensor(x) for x in plus]).item()
            fm = fun([Tensor(x) for x in minus]).item()
            g[i] = (fp - fm) / (2 * eps)
        out.append(g)
    return out


def ad_grads(fun, arrays):
    leaves = [Tensor(a, requires_grad=True) for a in arrays]
    fun(leaves).backward()
    return [leaf.grad if leaf.grad is not None else np.zeros(leaf.shape) for leaf in leaves]


def assert_grads_match(fun, arrays, tol=1e-6):
    for ad, fd in zip(ad_grads(fun, arrays), fd_grads(fun, arrays), strict=True):
        assert ad.shape == fd.shape
        scale = max(np.max(np.abs(fd), initial=0.0), 1.0)
        assert np.max(np.abs(ad - fd), initial=0.0) <= tol * scale


# ----------------------------------------------------------------------------
# forward mode
# ----------------------------------------------------------------------------
DUAL_FUNCS = [
    (lambda x: x * x * x - 2 * x + 1, lambda x: 3 * x * x - 2),
    (lambda x: 1.0 / x + x / 3.0, lambda x: -1.0 / x**2 + 1 / 3),
    (lambda x: (x**2.5) - 4.0 - x, lambda x: 2.5 * x**1.5 - 1),
    (lambda x: 2.0**x + x**x, lambda x: math.log(2) * 2**x + x**x * (math.log(x) + 1)),
    (lambda x: np.exp(x) * np.sin(x), lambda x: math.exp(x) * (math.sin(x) + math.cos(x))),
    (lambda x: np.log(x) + np.cos(x), lambda x: 1 / x - math.sin(x)),
    (lambda x: np.tan(x) - np.tanh(x), lambda x: 1 / math.cos(x) ** 2 - 1 / math.cosh(x) ** 2),
    (lambda x: np.sqrt(x) * abs(x - 2.0), None),
    (lambda x: -(+x) + 3 - x, lambda x: -2.0),
]


@pytest.mark.parametrize(("f", "df"), DUAL_FUNCS)
def test_dual_derivatives_on_grid(f, df):
    for x in np.linspace(0.3, 1.4, 23):
        got = derivative(f, float(x))
        if df is not None:
            assert got == pytest.approx(df(x), rel=1e-12, abs=1e-12)
        h = 1e-6
        fd = (f(x + h) - f(x - h)) / (2 * h)
        assert got == pytest.approx(float(fd), rel=1e-6, abs=1e-6)


def test_dual_abs_subgradient_and_comparisons():
    assert derivative(abs, 0.0) == 0.0
    assert derivative(abs, -2.0) == -1.0
    a, b = Dual(1.0, 5.0), Dual(2.0, -1.0)
    assert a < b and b > a and a <= 1.0 and a >= 1.0 and a == 1.0 and a != b
    assert 2.0 > a and np.float64(3.0) * a == Dual(3.0)
    assert (np.float64(3.0) * a).derivative == 15.0


@pytest.mark.filterwarnings("error")
def test_dual_extremes_are_ieee_not_exceptions():
    assert Dual(1000.0, 1.0).exp().value == math.inf
    assert math.isnan(Dual(-1.0, 1.0).log().value)
    assert Dual(0.0, 1.0).sqrt().derivative == math.inf
    assert Dual(0.0, 0.0).sqrt().derivative == 0.0
    assert math.isnan((Dual(-8.0, 1.0) ** (1 / 3)).value)  # not complex
    assert (Dual(0.0, 1.0) ** -1).value == math.inf
    assert (Dual(0.0, 1.0) ** 0).derivative == 0.0


def test_dual_errors():
    with pytest.raises(TypeError):
        Dual("1")
    with pytest.raises(TypeError):
        Dual(True)
    with pytest.raises(TypeError):
        Dual(1.0) + "a"
    with pytest.raises(TypeError, match="scalar"):
        derivative(lambda x: [x, x], 1.0)
    assert derivative(lambda x: 3.0, 1.0) == 0.0


def test_jvp_matches_jacobian(rng):
    def f(z):
        return [z[0] * z[1] + np.sin(z[2]), np.exp(z[0]) / z[1], z[2] ** 2]

    def jac(z):
        return np.array(
            [
                [z[1], z[0], np.cos(z[2])],
                [np.exp(z[0]) / z[1], -np.exp(z[0]) / z[1] ** 2, 0.0],
                [0.0, 0.0, 2 * z[2]],
            ]
        )

    for _ in range(50):
        x = rng.uniform(0.5, 2.0, size=3)
        v = rng.normal(size=3)
        value, tangent = jvp(f, x, v)
        np.testing.assert_allclose(value, [x[0] * x[1] + np.sin(x[2]), np.exp(x[0]) / x[1],
                                           x[2] ** 2], rtol=1e-14)  # fmt: skip
        np.testing.assert_allclose(tangent, jac(x) @ v, rtol=1e-13, atol=1e-13)
    with pytest.raises(ValueError, match="same shape"):
        jvp(f, [1.0, 2.0, 3.0], [1.0])
    with pytest.raises(TypeError, match="Dual"):
        jvp(lambda z: ["x"], [1.0], [1.0])


# ----------------------------------------------------------------------------
# reverse mode: every op against finite differences
# ----------------------------------------------------------------------------
BROADCAST_PAIRS = [((3, 4), (4,)), ((3, 1), (1, 4)), ((2, 3, 4), (3, 1)), ((), (3,)), ((5,), (5,))]


def positive(t):
    return t * t + 0.5


BINARY = {
    "add": lambda a, b: a + b,
    "radd": lambda a, b: 2.0 + a + b,
    "sub": lambda a, b: a - b,
    "rsub": lambda a, b: 1.5 - a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / positive(b),
    "rdiv": lambda a, b: 2.0 / positive(a) + b,
    "maximum": lambda a, b: maximum(a, b),
    "where": lambda a, b: where(a.data > 0.1, a * 2.0, b.tanh()),
}

UNARY = {
    "neg": lambda t: -t,
    "pow2": lambda t: t**2,
    "pow3": lambda t: t**3,
    "pow_frac": lambda t: positive(t) ** 1.7,
    "pow_neg": lambda t: positive(t) ** -0.5,
    "exp": lambda t: (t * 0.5).exp(),
    "log": lambda t: positive(t).log(),
    "tanh": lambda t: t.tanh(),
    "sigmoid": lambda t: (t * 3.0).sigmoid(),
    "relu": lambda t: t.relu(),
    "softplus": lambda t: (t * 2.0).softplus(),
    "sum_all": lambda t: t.sum() * t,
    "mean_all": lambda t: t.mean() + t,
}

SHAPE_OPS = {
    "sum_axis0": lambda t: t.sum(axis=0),
    "sum_keep": lambda t: t.sum(axis=-1, keepdims=True) * t,
    "sum_tuple": lambda t: t.sum(axis=(0, 1)),
    "mean_axis": lambda t: t.mean(axis=1),
    "mean_keep": lambda t: t.mean(axis=0, keepdims=True) - t,
    "log_softmax_last": lambda t: t.log_softmax(axis=-1),
    "log_softmax_0": lambda t: t.log_softmax(axis=0),
    "reshape": lambda t: t.reshape(-1).tanh().reshape(t.shape[::-1]),
    "transpose": lambda t: t.T * t.transpose((1, 0))[::-1],
    "slice": lambda t: t[1:, ::2],
    "int_index": lambda t: t[np.array([0, 2, 0, 0, 1])],
    "fancy_2d": lambda t: t[np.array([0, 0, 2]), np.array([1, 1, 3])],
    "bool_mask": lambda t: t[t.data > 0],
    "scalar_index": lambda t: t[1, 2] * t,
    "concatenate0": lambda t: concatenate([t, t * 2.0, t[:1]], axis=0),
    "concatenate1": lambda t: concatenate([t, t.tanh()], axis=-1),
    "stack": lambda t: stack([t, t * t], axis=1),
    "matmul_2d": lambda t: t @ t.T,
    "matmul_vec": lambda t: t @ t[0],
    "rmatmul_vec": lambda t: np.ones(3) @ t,
    "vec_vec": lambda t: t[0] @ t[1],
    "batched": lambda t: t.reshape(3, 2, 2) @ t.reshape(3, 2, 2).transpose((0, 2, 1)),
    "where_broadcast": lambda t: where(np.array([True, False, True, False]), t, 0.0),
    "max_ties_free": lambda t: maximum(t, t.sum(axis=0) * 0.1),
}


def weighted_sum(rng, fun):
    """Reduce an op output with fixed random weights so the upstream gradient is generic."""
    cache = {}

    def scalar(ts):
        out = fun(*ts)
        if out.shape not in cache:
            cache[out.shape] = rng.normal(size=out.shape)
        return (out * cache[out.shape]).sum()

    return scalar


@pytest.mark.parametrize("name", sorted(BINARY))
@pytest.mark.parametrize(("sa", "sb"), BROADCAST_PAIRS)
@pytest.mark.parametrize("seed", range(4))
def test_binary_ops_broadcasting_match_fd(name, sa, sb, seed):
    rng = np.random.default_rng(seed)
    arrays = [rng.normal(size=sa), rng.normal(size=sb)]
    f = weighted_sum(rng, BINARY[name])
    assert_grads_match(lambda ts: f(ts), arrays)


@pytest.mark.parametrize("name", sorted(UNARY))
@pytest.mark.parametrize("seed", range(6))
def test_unary_ops_match_fd(name, seed):
    rng = np.random.default_rng(100 + seed)
    shape = [(), (4,), (3, 4)][seed % 3]
    f = weighted_sum(rng, lambda t: UNARY[name](t))
    assert_grads_match(lambda ts: f(ts), [rng.normal(size=shape)])


@pytest.mark.parametrize("name", sorted(SHAPE_OPS))
@pytest.mark.parametrize("seed", range(5))
def test_shape_and_index_ops_match_fd(name, seed):
    rng = np.random.default_rng(200 + seed)
    f = weighted_sum(rng, lambda t: SHAPE_OPS[name](t))
    assert_grads_match(lambda ts: f(ts), [rng.normal(size=(3, 4))])


ALL_UNARY = list(UNARY.values()) + [
    lambda t: t.sum(axis=-1, keepdims=True) * t,
    lambda t: t.log_softmax(axis=-1),
    lambda t: t[..., ::-1],
    lambda t: maximum(t, 0.2),
    lambda t: concatenate([t, -t], axis=-1)[..., ::2],
]


@pytest.mark.parametrize("seed", range(300))
def test_random_compositions_match_fd(seed):
    """300 random chains of 2-6 ops, mixed with broadcasting binary ops."""
    rng = np.random.default_rng(1000 + seed)
    x = rng.normal(size=(2, 3))
    bias = rng.normal(size=(3,))
    n_ops = rng.integers(2, 7)
    choices = rng.integers(0, len(ALL_UNARY), size=n_ops)
    binaries = rng.integers(0, len(BINARY), size=n_ops)
    names = sorted(BINARY)

    def chain(ts):
        t, b = ts
        for c, k in zip(choices, binaries, strict=True):
            t = ALL_UNARY[c](t)
            t = BINARY[names[k]](t, b) * 0.5  # keep magnitudes moderate
        return t

    f = weighted_sum(rng, lambda *ts: chain(ts))
    vals = chain([Tensor(x), Tensor(bias)]).data
    if not np.all(np.isfinite(vals)) or np.max(np.abs(vals)) > 1e6:
        pytest.skip("composition left the well-scaled range")
    assert_grads_match(lambda ts: f(ts), [x, bias], tol=1e-5)


def test_repeated_index_gather_scatter_adds():
    x = Tensor(np.arange(5.0), requires_grad=True)
    idx = np.array([1, 1, 1, 4, 0, 4])
    (x[idx] * Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])).sum().backward()
    np.testing.assert_array_equal(x.grad, [5.0, 6.0, 0.0, 0.0, 10.0])
    idx[:] = 0  # mutating the caller's index after the op must not matter
    y = Tensor(np.arange(5.0), requires_grad=True)
    idx2 = np.array([2, 2])
    out = y[idx2].sum()
    idx2[:] = 0
    out.backward()
    np.testing.assert_array_equal(y.grad, [0.0, 0.0, 2.0, 0.0, 0.0])


def test_unbroadcast_shapes_and_values():
    a = Tensor(np.ones((3, 1)), requires_grad=True)
    b = Tensor(np.ones(4), requires_grad=True)
    c = Tensor(2.0, requires_grad=True)
    (a * b * c).sum().backward()
    assert a.grad.shape == (3, 1) and b.grad.shape == (4,) and c.grad.shape == ()
    np.testing.assert_array_equal(a.grad, np.full((3, 1), 8.0))
    np.testing.assert_array_equal(b.grad, np.full(4, 6.0))
    assert c.grad == 12.0


def test_maximum_ties_split_and_relu_subgradient():
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([2.0, 2.0, 2.0], requires_grad=True)
    maximum(a, b).sum().backward()
    np.testing.assert_array_equal(a.grad, [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(b.grad, [1.0, 0.5, 0.0])
    z = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
    z.relu().sum().backward()
    np.testing.assert_array_equal(z.grad, [0.0, 0.0, 1.0])


@pytest.mark.filterwarnings("error")
def test_log_softmax_extremes():
    x = Tensor([[1000.0, 0.0, -1000.0]], requires_grad=True)
    y = x.log_softmax(axis=-1)
    assert np.all(np.isfinite(y.data))
    y.sum().backward()
    np.testing.assert_allclose(x.grad, [[1.0 - 3.0, 1.0, 1.0]], atol=1e-12)
    s = Tensor([-800.0, 800.0], requires_grad=True)
    s.sigmoid().sum().backward()
    np.testing.assert_array_equal(s.grad, [0.0, 0.0])
    s2 = Tensor([-800.0, 800.0], requires_grad=True)
    s2.softplus().sum().backward()
    np.testing.assert_array_equal(s2.grad, [0.0, 1.0])


# ----------------------------------------------------------------------------
# graph semantics
# ----------------------------------------------------------------------------
def test_gradients_accumulate_and_zero_grad():
    x = Tensor([1.0, 2.0], requires_grad=True)
    (x * x).sum().backward()
    (x * 3.0).sum().backward()
    np.testing.assert_array_equal(x.grad, [5.0, 7.0])
    x.zero_grad()
    assert x.grad is None
    (x * x).sum().backward()
    np.testing.assert_array_equal(x.grad, [2.0, 4.0])


def test_graph_released_unless_retained():
    x = Tensor(3.0, requires_grad=True)
    y = x * x
    z = y * 2.0
    z.backward(retain_graph=True)
    z.backward()
    assert x.grad == 24.0
    with pytest.raises(RuntimeError, match="released"):
        z.backward()
    with pytest.raises(RuntimeError, match="released"):
        (y + 1.0).backward()  # y's own graph was released too
    assert x.grad == 24.0  # a refused backward changes nothing


def test_only_leaves_get_grad_and_shared_subgraphs():
    x = Tensor([1.0, -1.0], requires_grad=True)
    h = x.tanh()
    out = (h * h + h).sum()
    out.backward()
    assert h.grad is None and not h.is_leaf and x.is_leaf
    t = np.tanh([1.0, -1.0])
    np.testing.assert_allclose(x.grad, (2 * t + 1) * (1 - t**2), rtol=1e-15)


def test_no_graph_without_requires_grad():
    a = Tensor([1.0, 2.0])
    b = a * 2.0
    assert not b.requires_grad and b.is_leaf
    with pytest.raises(RuntimeError, match="does not require grad"):
        b.sum().backward()


def test_backward_seed_rules():
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x * 3.0
    with pytest.raises(RuntimeError, match="non-scalar"):
        y.backward()
    with pytest.raises(ValueError, match="shape"):
        y.backward(grad=[1.0])
    y.backward(grad=[1.0, -1.0])
    np.testing.assert_array_equal(x.grad, [3.0, -3.0])


def test_data_is_read_only_and_copied():
    src = np.array([1.0, 2.0])
    t = Tensor(src, requires_grad=True)
    src[0] = 100.0
    assert t.data[0] == 1.0
    with pytest.raises(ValueError):
        t.data[0] = 5.0
    view = t.data[:1]
    with pytest.raises(ValueError):
        view.flags.writeable = True
    n = t.numpy()
    n[0] = 7.0
    assert t.data[0] == 1.0
    (t * t).sum().backward()
    g = t.grad
    g[0] = 99.0  # the user owns .grad; it does not alias any graph array
    assert t.detach().requires_grad is False and t.detach().data[0] == 1.0


def test_tensor_input_errors():
    with pytest.raises(TypeError):
        Tensor("abc")
    with pytest.raises(TypeError):
        Tensor([1 + 2j])
    with pytest.raises(TypeError):
        Tensor([1.0], requires_grad=1)
    t = Tensor(np.ones((2, 3)))
    with pytest.raises(TypeError, match="real"):
        t ** Tensor(2.0)
    with pytest.raises(ValueError):
        t @ Tensor(np.ones((2, 3)))
    with pytest.raises(ValueError, match="out of bounds"):
        t.sum(axis=2)
    with pytest.raises(ValueError, match="permutation"):
        t.transpose((0, 0))
    with pytest.raises(TypeError, match="Tensors"):
        t[Tensor([0.0])]
    with pytest.raises(IndexError):
        t[np.array([0.5])]
    with pytest.raises(TypeError):
        where(Tensor([1.0]), 1.0, 2.0)
    with pytest.raises(TypeError, match="boolean"):
        where(np.array([1, 0]), 1.0, 2.0)
    with pytest.raises(TypeError):
        concatenate(t)
    with pytest.raises(ValueError, match="one-element"):
        t.item()
    assert float(Tensor([[2.5]])) == 2.5 and len(t) == 2 and "Tensor(" in repr(t)


def test_ten_thousand_op_chain_no_recursion_error():
    depth = 10_000
    assert depth > sys.getrecursionlimit()
    x = Tensor(1.0, requires_grad=True)
    y = x
    for _ in range(depth):
        y = y * 1.0001 + 0.0
    y.backward()
    assert x.grad == pytest.approx(1.0001**depth, rel=1e-10)
    del y  # releasing a deep graph must not recurse either


def test_numpy_scalars_and_arrays_defer_to_tensor():
    t = Tensor([1.0, 2.0], requires_grad=True)
    out = np.float64(2.0) * t + np.array([1.0, 1.0])
    assert isinstance(out, Tensor)
    out.sum().backward()
    np.testing.assert_array_equal(t.grad, [2.0, 2.0])


# ----------------------------------------------------------------------------
# functional API
# ----------------------------------------------------------------------------
def test_grad_value_and_grad_argnums(rng):
    X = rng.normal(size=(10, 3))
    y = rng.normal(size=10)
    w = rng.normal(size=3)

    def loss(w, X, y):
        r = X @ w - y
        return (r * r).mean()

    expected = 2 * X.T @ (X @ w - y) / 10
    np.testing.assert_allclose(grad(loss)(w, X, y), expected, rtol=1e-13)
    v, (gw, gX) = value_and_grad(loss, argnums=(0, 1))(w, X, y)
    assert v == pytest.approx(np.mean((X @ w - y) ** 2), rel=1e-14)
    np.testing.assert_allclose(gw, expected, rtol=1e-13)
    np.testing.assert_allclose(gX, 2 * np.outer(X @ w - y, w) / 10, rtol=1e-13)
    # An argument the output does not depend on gets a zero gradient.
    np.testing.assert_array_equal(grad(lambda a, b: (Tensor(a) * 2.0).sum(), argnums=1)(w, w), 0.0)
    with pytest.raises(ValueError, match="argnums"):
        grad(loss, argnums=3)(w, X, y)
    with pytest.raises(TypeError, match="one-element"):
        grad(lambda a: a * 2.0)(w)


def test_check_grad_detects_wrong_gradients():
    assert check_grad(lambda t: (t.exp() * t.tanh()).sum(), [0.1, -0.7, 1.3]) < 1e-8

    # A deliberately wrong VJP (factor 2 missing) must be detected.
    class Bad(Tensor):
        pass

    def wrong_square(t):
        return Tensor._result(t.data**2, (t,), lambda g: (g * t.data,))

    assert check_grad(lambda t: wrong_square(t).sum(), [0.5, 1.5]) > 0.3
    with pytest.raises(ValueError, match="eps"):
        check_grad(lambda t: t.sum(), [1.0], eps=0.0)


# ----------------------------------------------------------------------------
# torch parity (only where torch is installed)
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(20))
def test_parity_with_torch(seed):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=(3, 4)), rng.normal(size=(4,))
    idx = np.array([0, 2, 2, 1])

    def ours(x, y):
        h = (x @ y.reshape(4, 1)).tanh().sum() + maximum(x, y).log_softmax(axis=0)[idx].mean()
        return h + (x.sigmoid() * y).softplus().sum() + (x[:, ::2] ** 2).relu().mean()

    def theirs(x, y):
        h = torch.tanh(x @ y.reshape(4, 1)).sum()
        h = h + torch.log_softmax(torch.maximum(x, y), dim=0)[idx].mean()
        return (
            h
            + torch.nn.functional.softplus(torch.sigmoid(x) * y).sum()
            + torch.relu(x[:, ::2] ** 2).mean()
        )

    ta, tb = Tensor(a, requires_grad=True), Tensor(b, requires_grad=True)
    out = ours(ta, tb)
    out.backward()
    xa = torch.tensor(a, dtype=torch.float64, requires_grad=True)
    xb = torch.tensor(b, dtype=torch.float64, requires_grad=True)
    ref = theirs(xa, xb)
    ref.backward()
    assert out.item() == pytest.approx(ref.item(), rel=1e-13)
    np.testing.assert_allclose(ta.grad, xa.grad.numpy(), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(tb.grad, xb.grad.numpy(), rtol=1e-12, atol=1e-14)


# ----------------------------------------------------------------------------
# memory
# ----------------------------------------------------------------------------
def test_backward_memory_linear_in_graph_size():
    """A chain of n ops on a fixed-size vector: forward + backward memory O(n)."""

    def run(n):
        x = Tensor(np.ones(2_000), requires_grad=True)
        y = x
        for _ in range(n):
            y = y * 1.0001
        y.sum().backward()

    est = estimate_complexity(run, lambda n, rng: n, sizes=(50, 100, 200, 400), repeats=1)
    assert est.space_exponent == pytest.approx(1.0, abs=0.15)
