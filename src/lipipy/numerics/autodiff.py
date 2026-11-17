r"""Automatic differentiation: forward mode (dual numbers) and reverse mode (tensors).

Forward mode
------------
A dual number :math:`a + b\varepsilon` with :math:`\varepsilon^2 = 0` carries a value
and a directional derivative. For any smooth :math:`f`,

.. math::  f(a + b\varepsilon) = f(a) + f'(a)\,b\,\varepsilon,

so evaluating ``f`` on :class:`Dual` inputs propagates derivatives through every
operation by the chain rule: :math:`(u v)' = u'v + uv'`,
:math:`(u/v)' = (u'v - uv')/v^2`, :math:`(e^u)' = e^u u'`, and so on. Seeding input
:math:`x_i` with derivative :math:`v_i` yields the Jacobian-vector product
:math:`J_f(x)\,v` in one evaluation (:func:`jvp`). Cost: a constant factor (about
2-3) times one evaluation of ``f``.

Reverse mode
------------
A :class:`Tensor` records, for every operation, its inputs and a *vector-Jacobian
product* (VJP) rule :math:`\bar u = \bar y\,\partial y/\partial u`. ``backward``
visits the recorded graph in reverse topological order, starting from
:math:`\bar y = 1` for a scalar output, and sums the contributions of every path:

.. math::  \bar u = \sum_{y \,:\, u \to y} \bar y \,\frac{\partial y}{\partial u}.

One backward pass gives the gradient with respect to *all* inputs at a constant
multiple (about 2-3) of the forward cost, which is why deep learning uses it.

*Broadcasting.* If :math:`u` was broadcast from shape :math:`s` to the output shape,
the incoming gradient is summed over the broadcast axes ("unbroadcast") so that
:math:`\bar u` has shape :math:`s` again.

*Indexing.* ``x[idx]`` is a gather; its VJP scatters :math:`\bar y` back with
``np.add.at``, so repeated indices accumulate (:math:`\bar x_i = \sum_{k: idx_k = i}
\bar y_k`).

*Non-smooth points* (documented subgradients): ``relu'(0) = 0``, ``abs'(0) = 0``,
and ``maximum(a, b)`` splits the gradient equally between ``a`` and ``b`` where they
are equal (the PyTorch convention).

Graph mechanics follow PyTorch: only leaf tensors created with
``requires_grad=True`` receive ``.grad``; gradients **accumulate** across
``backward`` calls until :meth:`Tensor.zero_grad`; the graph is released after
``backward`` unless ``retain_graph=True``. The topological sort is iterative, so
graph depth is not limited by Python's recursion limit. Arrays held by tensors are
read-only, so a recorded graph cannot be corrupted through ``.data``.

References
----------
A. Griewank & A. Walther, *Evaluating Derivatives*, 2nd ed., SIAM (2008), ch. 3-4.
A. G. Baydin, B. A. Pearlmutter, A. A. Radul & J. M. Siskind, "Automatic
differentiation in machine learning: a survey", *JMLR* 18 (2018) 1-43.
A. Paszke et al., "Automatic differentiation in PyTorch", NIPS Autodiff Workshop (2017).
"""

from __future__ import annotations

import numbers
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from lipipy.core.complexity import Complexity
from lipipy.core.registry import register
from lipipy.numerics._common import as_real, check_callable
from lipipy.numerics.special import log_softmax as _log_softmax
from lipipy.numerics.special import sigmoid as _sigmoid
from lipipy.numerics.special import softplus as _softplus

__all__ = [
    "Dual",
    "Tensor",
    "check_grad",
    "concatenate",
    "derivative",
    "grad",
    "jvp",
    "maximum",
    "stack",
    "value_and_grad",
    "where",
]

_REFS = (
    "Griewank & Walther, Evaluating Derivatives, 2nd ed., SIAM (2008)",
    "Baydin et al., Automatic differentiation in machine learning: a survey, JMLR 18 (2018)",
)


def _is_real_number(x: Any) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, (bool, np.bool_))


# ===========================================================================
# Forward mode
# ===========================================================================
@register(
    domain="numerics",
    tasks=("automatic-differentiation",),
    complexity=Complexity(time="O(1)", space="O(1)", notes="per arithmetic operation"),
    references=_REFS,
)
class Dual:
    r"""Dual number :math:`a + b\varepsilon` (:math:`\varepsilon^2 = 0`) for forward-mode AD.

    Supports ``+ - * / **``, unary ``-``/``+``/``abs``, comparisons (on the value),
    and the methods ``exp, log, sin, cos, tan, tanh, sqrt``. Because these are
    methods, NumPy ufuncs work too: ``np.sin(Dual(1.0, 1.0))`` and ``np.exp`` on object
    arrays of duals dispatch to them.

    ``abs`` uses the subgradient 0 at 0. ``log`` and ``sqrt`` follow IEEE arithmetic
    outside their domains (NaN / inf), like the corresponding NumPy functions.

    Parameters
    ----------
    value : float
        Primal value :math:`a`.
    derivative : float, default=0.0
        Tangent :math:`b`.

    Complexity
    ----------
    O(1) time and space per operation.

    References
    ----------
    Griewank & Walther, *Evaluating Derivatives*, ch. 3.

    Examples
    --------
    >>> x = Dual(2.0, 1.0)
    >>> y = x**3 + x.sin()
    >>> round(y.value, 10), round(y.derivative, 10)  # 8 + sin 2, 12 + cos 2
    (8.9092974268, 11.5838531635)
    """

    __slots__ = ("derivative", "value")

    def __init__(self, value: float, derivative: float = 0.0) -> None:
        if not (_is_real_number(value) and _is_real_number(derivative)):
            raise TypeError("Dual value and derivative must be real numbers")
        self.value = float(value)
        self.derivative = float(derivative)

    # -- helpers ------------------------------------------------------------------
    @staticmethod
    def _lift(other: Any) -> Dual | None:
        if isinstance(other, Dual):
            return other
        if _is_real_number(other):
            return Dual(float(other), 0.0)
        return None

    def __repr__(self) -> str:
        return f"Dual(value={self.value!r}, derivative={self.derivative!r})"

    # -- arithmetic ------------------------------------------------------------------
    def __add__(self, other: Any) -> Dual:
        o = self._lift(other)
        if o is None:
            return NotImplemented
        return Dual(self.value + o.value, self.derivative + o.derivative)

    __radd__ = __add__

    def __sub__(self, other: Any) -> Dual:
        o = self._lift(other)
        if o is None:
            return NotImplemented
        return Dual(self.value - o.value, self.derivative - o.derivative)

    def __rsub__(self, other: Any) -> Dual:
        o = self._lift(other)
        if o is None:
            return NotImplemented
        return o - self

    def __mul__(self, other: Any) -> Dual:
        o = self._lift(other)
        if o is None:
            return NotImplemented
        return Dual(self.value * o.value, self.derivative * o.value + self.value * o.derivative)

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> Dual:
        o = self._lift(other)
        if o is None:
            return NotImplemented
        q = _np1(np.divide, self.value, o.value)
        return Dual(q, _div(self.derivative - q * o.derivative, o.value))

    def __rtruediv__(self, other: Any) -> Dual:
        o = self._lift(other)
        if o is None:
            return NotImplemented
        return o / self

    def __pow__(self, other: Any) -> Dual:
        if isinstance(other, Dual):
            # x^y = exp(y log x): d = x^y (y' log x + y x' / x); real only for x > 0
            v = _np1(np.power, self.value, other.value)
            return Dual(
                v,
                v
                * (
                    other.derivative * _np1(np.log, self.value)
                    + other.value * _div(self.derivative, self.value)
                ),
            )
        if not _is_real_number(other):
            return NotImplemented
        p = float(other)
        if p == 0.0:
            return Dual(1.0, 0.0)
        slope = p * _np1(np.power, self.value, p - 1.0)
        return Dual(
            _np1(np.power, self.value, p), slope * self.derivative if self.derivative else 0.0
        )

    def __rpow__(self, other: Any) -> Dual:
        if not _is_real_number(other):
            return NotImplemented
        c = float(other)
        v = _np1(np.power, c, self.value)
        return Dual(v, v * _np1(np.log, c) * self.derivative if self.derivative else 0.0)

    def __neg__(self) -> Dual:
        return Dual(-self.value, -self.derivative)

    def __pos__(self) -> Dual:
        return Dual(self.value, self.derivative)

    def __abs__(self) -> Dual:
        sign = 1.0 if self.value > 0 else -1.0 if self.value < 0 else 0.0
        return Dual(abs(self.value), sign * self.derivative)

    # -- comparisons (on the value) -------------------------------------------------
    def _cmp_value(self, other: Any) -> float | None:
        o = self._lift(other)
        return None if o is None else o.value

    def __eq__(self, other: object) -> bool:
        v = self._cmp_value(other)
        return NotImplemented if v is None else self.value == v

    def __ne__(self, other: object) -> bool:
        v = self._cmp_value(other)
        return NotImplemented if v is None else self.value != v

    def __lt__(self, other: Any) -> bool:
        v = self._cmp_value(other)
        return NotImplemented if v is None else self.value < v

    def __le__(self, other: Any) -> bool:
        v = self._cmp_value(other)
        return NotImplemented if v is None else self.value <= v

    def __gt__(self, other: Any) -> bool:
        v = self._cmp_value(other)
        return NotImplemented if v is None else self.value > v

    def __ge__(self, other: Any) -> bool:
        v = self._cmp_value(other)
        return NotImplemented if v is None else self.value >= v

    __hash__ = None  # type: ignore[assignment]  # equality is by value only

    # -- elementary functions (also reached through NumPy ufuncs) --------------------
    def exp(self) -> Dual:
        e = _np1(np.exp, self.value)
        return Dual(e, _mul(e, self.derivative))

    def log(self) -> Dual:
        return Dual(_np1(np.log, self.value), _div(self.derivative, self.value))

    def sin(self) -> Dual:
        return Dual(_np1(np.sin, self.value), _mul(_np1(np.cos, self.value), self.derivative))

    def cos(self) -> Dual:
        return Dual(_np1(np.cos, self.value), _mul(-_np1(np.sin, self.value), self.derivative))

    def tan(self) -> Dual:
        t = _np1(np.tan, self.value)
        return Dual(t, _mul(1.0 + t * t, self.derivative))

    def tanh(self) -> Dual:
        t = _np1(np.tanh, self.value)
        return Dual(t, _mul(1.0 - t * t, self.derivative))

    def sqrt(self) -> Dual:
        s = _np1(np.sqrt, self.value)
        return Dual(s, _div(self.derivative, 2.0 * s))


def _np1(func: Callable[..., Any], *args: float) -> float:
    """Apply a NumPy function to floats with IEEE results (inf/NaN) instead of exceptions."""
    with np.errstate(all="ignore"):
        return float(func(*args))


def _mul(a: float, b: float) -> float:
    """Product where a zero tangent stays zero even if the partial derivative is infinite."""
    return 0.0 if b == 0.0 else a * b


def _div(a: float, b: float) -> float:
    """Quotient a / b with IEEE semantics; a zero tangent stays zero."""
    if a == 0.0:
        return 0.0
    with np.errstate(all="ignore"):
        return float(np.float64(a) / np.float64(b))


def _to_dual_output(out: Any) -> np.ndarray:
    """Flatten a function output (Dual, number or nested sequence) to an object array."""
    arr = np.asarray(out, dtype=object)
    for item in arr.flat:
        if not (isinstance(item, Dual) or _is_real_number(item)):
            raise TypeError(f"function must return Dual numbers or reals, got {type(item)}")
    return arr


@register(
    domain="numerics",
    tasks=("automatic-differentiation",),
    complexity=Complexity(time="O(1)", space="O(1)", notes="about 2-3 evaluations of f"),
    references=_REFS,
)
def derivative(f: Callable[[Any], Any], x: float) -> float:
    r"""Exact derivative :math:`f'(x)` of a scalar function by forward-mode AD.

    Evaluates ``f(Dual(x, 1))`` and returns the tangent. ``f`` may use arithmetic,
    :class:`Dual` methods, and NumPy ufuncs (``np.sin`` etc.).

    Parameters
    ----------
    f : callable
        Scalar function of one real variable.
    x : float
        Point of evaluation.

    Returns
    -------
    float
        :math:`f'(x)` (0.0 if ``f`` returns a constant).

    Complexity
    ----------
    A constant factor (about 2-3) times one evaluation of ``f``.

    References
    ----------
    Griewank & Walther, *Evaluating Derivatives*, ch. 3.

    Examples
    --------
    >>> import numpy as np
    >>> derivative(lambda t: t * np.exp(t), 0.0)  # (1 + t) e^t at 0
    1.0
    """
    check_callable(f, "f")
    if not _is_real_number(x):
        raise TypeError(f"x must be a real number, got {type(x).__name__}")
    out = f(Dual(float(x), 1.0))
    if isinstance(out, Dual):
        return out.derivative
    if _is_real_number(out):
        return 0.0
    raise TypeError(f"f must return a scalar, got {type(out).__name__}")


@register(
    domain="numerics",
    tasks=("automatic-differentiation",),
    complexity=Complexity(
        time="O(d)", space="O(d)", notes="d inputs; about 2-3 evaluations of f, excluding f"
    ),
    references=_REFS,
)
def jvp(
    f: Callable[[np.ndarray], Any], x: ArrayLike, v: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    r"""Jacobian-vector product :math:`(f(x), J_f(x)\,v)` by forward-mode AD.

    Each :math:`x_i` is seeded as ``Dual(x_i, v_i)``; ``f`` receives a 1-D object
    array of duals and may return a dual, a sequence or an array of them.

    Parameters
    ----------
    f : callable
        Function :math:`\mathbb R^d \to \mathbb R^m` (or to a scalar).
    x : array_like of shape (d,)
        Point of evaluation.
    v : array_like of shape (d,)
        Direction.

    Returns
    -------
    value : ndarray
        :math:`f(x)`, shaped like the output of ``f``.
    tangent : ndarray
        :math:`J_f(x)\,v`, same shape.

    Complexity
    ----------
    A constant factor (about 2-3) times one evaluation of ``f``, independent of d.

    References
    ----------
    Baydin et al., *JMLR* 18 (2018), §3.1.

    Examples
    --------
    >>> import numpy as np
    >>> f = lambda z: [z[0] * z[1], np.sin(z[0])]
    >>> value, tangent = jvp(f, [1.0, 2.0], [1.0, 0.0])
    >>> value.round(6), tangent.round(6)
    (array([2.      , 0.841471]), array([2.      , 0.540302]))
    """
    check_callable(f, "f")
    x = as_real(x, "x", ndim=1)
    v = as_real(v, "v", ndim=1)
    if x.shape != v.shape:
        raise ValueError(f"x and v must have the same shape, got {x.shape} and {v.shape}")
    duals = np.empty(x.shape, dtype=object)
    for i, (xi, vi) in enumerate(zip(x, v, strict=True)):
        duals[i] = Dual(float(xi), float(vi))
    out = _to_dual_output(f(duals))
    value = np.array([o.value if isinstance(o, Dual) else float(o) for o in out.flat])
    tangent = np.array([o.derivative if isinstance(o, Dual) else 0.0 for o in out.flat])
    return value.reshape(out.shape), tangent.reshape(out.shape)


# ===========================================================================
# Reverse mode
# ===========================================================================
def _readonly(a: np.ndarray) -> np.ndarray:
    a.flags.writeable = False
    return a


def _owned(x: ArrayLike) -> np.ndarray:
    """A fresh, read-only float64 array (never aliases the caller's array)."""
    if isinstance(x, (str, bytes)):
        raise TypeError(f"Tensor data must be numeric, got {type(x).__name__}")
    if np.iscomplexobj(x):
        raise TypeError("Tensor data must be real-valued")
    try:
        arr = np.array(x, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Tensor data must be convertible to float64: {exc}") from None
    return _readonly(arr)


def _unbroadcast(g: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum ``g`` over the axes that broadcasting added or stretched, giving ``shape``."""
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and g.shape[axis] != 1:
            g = g.sum(axis=axis, keepdims=True)
    return g


def _normalize_axes(axis: int | Sequence[int] | None, ndim: int) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    axes = (axis,) if isinstance(axis, (int, np.integer)) else tuple(axis)
    out = []
    for ax in axes:
        if not isinstance(ax, (int, np.integer)):
            raise TypeError(f"axis must be an int, a sequence of ints or None, got {axis!r}")
        if not -ndim <= ax < max(ndim, 1):
            raise ValueError(f"axis {ax} is out of bounds for a tensor of dimension {ndim}")
        out.append(int(ax) % max(ndim, 1))
    if len(set(out)) != len(out):
        raise ValueError(f"repeated axis in {axis!r}")
    return tuple(out)


_VJP = Callable[[np.ndarray], tuple[np.ndarray | None, ...]]


@register(
    domain="numerics",
    tasks=("automatic-differentiation", "deep-learning"),
    complexity=Complexity(
        time="O(n)",
        space="O(n)",
        notes="n = total elements produced by the recorded operations (the forward cost); "
        "backward costs a constant multiple (about 2-3x) of the forward pass and keeps "
        "every recorded intermediate until it is released",
    ),
    references=(*_REFS, "Paszke et al., Automatic differentiation in PyTorch (2017)"),
)
class Tensor:
    r"""Float64 array that records operations for reverse-mode differentiation.

    Parameters
    ----------
    data : array_like
        Values (copied, stored read-only as float64).
    requires_grad : bool, default=False
        Track operations on this tensor so that ``backward`` fills ``.grad``.

    Attributes
    ----------
    data : ndarray
        The values (read-only).
    grad : ndarray or None
        Accumulated gradient (leaf tensors with ``requires_grad`` only).
    requires_grad : bool
        Whether gradients flow to or through this tensor.

    Notes
    -----
    Supported operations: ``+ - * / @``, ``**`` with a real exponent, unary ``-``,
    :meth:`sum`, :meth:`mean`, :meth:`exp`, :meth:`log`, :meth:`tanh`,
    :meth:`sigmoid`, :meth:`relu`, :meth:`softplus`, :meth:`log_softmax`,
    :meth:`reshape`, :meth:`transpose` / ``.T``, indexing (basic slices, integer
    arrays, boolean masks), and the functions :func:`concatenate`, :func:`stack`,
    :func:`maximum`, :func:`where`. All binary operations broadcast like NumPy.

    Complexity
    ----------
    Each operation costs what the NumPy operation costs; ``backward`` is a constant
    multiple of the forward cost; memory is linear in the size of the graph.

    References
    ----------
    Griewank & Walther (2008), ch. 4; Paszke et al. (2017).

    Examples
    --------
    >>> import numpy as np
    >>> w = Tensor([1.0, -2.0], requires_grad=True)
    >>> x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> loss = ((x @ w).tanh() ** 2).mean()
    >>> loss.backward()
    >>> w.grad.round(6)
    array([-0.010362, -0.020361])
    """

    __slots__ = ("_parents", "_released", "_vjp", "data", "grad", "requires_grad")
    __array_ufunc__ = None  # NumPy defers to Tensor's reflected operators

    def __init__(self, data: ArrayLike, requires_grad: bool = False) -> None:
        if not isinstance(requires_grad, (bool, np.bool_)):
            raise TypeError("requires_grad must be a bool")
        self.data = data._data_copy() if isinstance(data, Tensor) else _owned(data)
        self.requires_grad = bool(requires_grad)
        self.grad: np.ndarray | None = None
        self._parents: tuple[Tensor, ...] = ()
        self._vjp: _VJP | None = None
        self._released = False

    def _data_copy(self) -> np.ndarray:
        return _readonly(self.data.copy())

    @classmethod
    def _result(cls, data: np.ndarray, parents: tuple[Tensor, ...], vjp: _VJP) -> Tensor:
        """Create an op output; record the graph only if a parent needs gradients."""
        out = cls.__new__(cls)
        out.data = _readonly(np.asarray(data, dtype=np.float64))
        out.grad = None
        out._released = False
        out.requires_grad = any(p.requires_grad for p in parents)
        out._parents = parents if out.requires_grad else ()
        out._vjp = vjp if out.requires_grad else None
        return out

    # -- basic properties -----------------------------------------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def is_leaf(self) -> bool:
        """True for tensors created by the user (not by an operation)."""
        return self._vjp is None and not self._released

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        body = np.array2string(self.data, separator=", ", prefix="Tensor(")
        return f"Tensor({body}{', requires_grad=True' if self.requires_grad else ''})"

    def numpy(self) -> np.ndarray:
        """A writable copy of the data."""
        return self.data.copy()

    def item(self) -> float:
        """The value of a one-element tensor as a Python float."""
        if self.size != 1:
            raise ValueError(f"item() needs a one-element tensor, got shape {self.shape}")
        return float(self.data.reshape(()))

    def __float__(self) -> float:
        return self.item()

    def detach(self) -> Tensor:
        """A new leaf tensor with the same values and no graph."""
        return Tensor(self)

    def zero_grad(self) -> None:
        """Reset the accumulated gradient to None."""
        self.grad = None

    # -- backward -------------------------------------------------------------------
    def backward(self, grad: ArrayLike | None = None, retain_graph: bool = False) -> None:
        r"""Accumulate d(this)/d(leaf) into the ``.grad`` of every reachable leaf.

        Parameters
        ----------
        grad : array_like, optional
            Upstream gradient :math:`\bar y` with this tensor's shape; required unless
            the tensor has exactly one element (then 1 is used).
        retain_graph : bool, default=False
            Keep the graph so ``backward`` can be called through it again.

        Raises
        ------
        RuntimeError
            If the tensor does not require gradients, or the graph was released by an
            earlier ``backward`` without ``retain_graph=True``.
        """
        if not self.requires_grad:
            raise RuntimeError("backward() called on a tensor that does not require grad")
        if grad is None:
            if self.size != 1:
                raise RuntimeError(
                    f"grad must be given for non-scalar outputs (shape {self.shape})"
                )
            seed = np.ones(self.shape)
        else:
            seed = as_real(grad, "grad")
            if seed.shape != self.shape:
                raise ValueError(f"grad has shape {seed.shape}, expected {self.shape}")

        order = self._topological_order()
        for node in order:  # validate before mutating anything
            if node._released:
                raise RuntimeError(
                    "the graph was released by an earlier backward(); "
                    "pass retain_graph=True to backpropagate through it twice"
                )
        pending: dict[int, np.ndarray] = {id(self): seed}
        for node in order:
            g = pending.pop(id(node), None)
            if g is None:
                continue
            if node._vjp is None:  # leaf
                node.grad = g.copy() if node.grad is None else node.grad + g
                continue
            for parent, pg in zip(node._parents, node._vjp(g), strict=True):
                if pg is None or not parent.requires_grad:
                    continue
                key = id(parent)
                pending[key] = pg if key not in pending else pending[key] + pg
        if not retain_graph:
            for node in order:
                if node._vjp is not None:
                    node._vjp = None
                    node._parents = ()
                    node._released = True

    def _topological_order(self) -> list[Tensor]:
        """Nodes reachable from self that require grad, consumers before producers."""
        post: list[Tensor] = []
        visited: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                post.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for parent in node._parents:
                if parent.requires_grad and id(parent) not in visited:
                    stack.append((parent, False))
        post.reverse()
        return post

    # -- binary arithmetic ------------------------------------------------------------
    @staticmethod
    def _wrap(other: Any) -> Tensor:
        return other if isinstance(other, Tensor) else Tensor(other)

    def __add__(self, other: Any) -> Tensor:
        o = self._wrap(other)
        a, b = self.shape, o.shape
        return Tensor._result(
            self.data + o.data, (self, o), lambda g: (_unbroadcast(g, a), _unbroadcast(g, b))
        )

    def __radd__(self, other: Any) -> Tensor:
        return self._wrap(other) + self

    def __sub__(self, other: Any) -> Tensor:
        o = self._wrap(other)
        a, b = self.shape, o.shape
        return Tensor._result(
            self.data - o.data, (self, o), lambda g: (_unbroadcast(g, a), _unbroadcast(-g, b))
        )

    def __rsub__(self, other: Any) -> Tensor:
        return self._wrap(other) - self

    def __mul__(self, other: Any) -> Tensor:
        o = self._wrap(other)
        x, y = self.data, o.data
        return Tensor._result(
            x * y,
            (self, o),
            lambda g: (_unbroadcast(g * y, x.shape), _unbroadcast(g * x, y.shape)),
        )

    def __rmul__(self, other: Any) -> Tensor:
        return self._wrap(other) * self

    def __truediv__(self, other: Any) -> Tensor:
        o = self._wrap(other)
        x, y = self.data, o.data
        q = x / y
        return Tensor._result(
            q,
            (self, o),
            lambda g: (_unbroadcast(g / y, x.shape), _unbroadcast(-g * q / y, y.shape)),
        )

    def __rtruediv__(self, other: Any) -> Tensor:
        return self._wrap(other) / self

    def __matmul__(self, other: Any) -> Tensor:
        o = self._wrap(other)
        a, b = self.data, o.data
        if a.ndim == 0 or b.ndim == 0:
            raise ValueError("matmul does not accept 0-d tensors; use * instead")
        out = a @ b  # raises ValueError on shape mismatch

        def vjp(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            A = a if a.ndim > 1 else a[np.newaxis, :]
            B = b if b.ndim > 1 else b[:, np.newaxis]
            if a.ndim == 1 and b.ndim == 1:  # inner product: g is 0-d
                G = np.reshape(g, (1, 1))
            else:
                G = np.expand_dims(g, -2) if a.ndim == 1 else g
                G = np.expand_dims(G, -1) if b.ndim == 1 else G
            gA = _unbroadcast(G @ np.swapaxes(B, -1, -2), A.shape).reshape(a.shape)
            gB = _unbroadcast(np.swapaxes(A, -1, -2) @ G, B.shape).reshape(b.shape)
            return gA, gB

        return Tensor._result(out, (self, o), vjp)

    def __rmatmul__(self, other: Any) -> Tensor:
        return self._wrap(other) @ self

    def __pow__(self, exponent: Any) -> Tensor:
        if isinstance(exponent, Tensor) or not _is_real_number(exponent):
            raise TypeError("Tensor ** exponent supports a real (non-tensor) exponent only")
        p = float(exponent)
        x = self.data
        with np.errstate(divide="ignore", invalid="ignore"):
            out = x**p
        if p == 0.0:
            return Tensor._result(out, (self,), lambda g: (np.zeros_like(x),))

        def vjp(g: np.ndarray) -> tuple[np.ndarray]:
            with np.errstate(divide="ignore", invalid="ignore"):
                return (g * p * x ** (p - 1.0),)

        return Tensor._result(out, (self,), vjp)

    def __neg__(self) -> Tensor:
        return Tensor._result(-self.data, (self,), lambda g: (-g,))

    def __pos__(self) -> Tensor:
        return self

    # Comparisons return plain boolean arrays (not differentiable), e.g. for where().
    def __lt__(self, other: Any) -> np.ndarray:
        return self.data < self._wrap(other).data

    def __le__(self, other: Any) -> np.ndarray:
        return self.data <= self._wrap(other).data

    def __gt__(self, other: Any) -> np.ndarray:
        return self.data > self._wrap(other).data

    def __ge__(self, other: Any) -> np.ndarray:
        return self.data >= self._wrap(other).data

    # -- reductions -------------------------------------------------------------------
    def sum(self, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> Tensor:
        """Sum over ``axis`` (all axes by default)."""
        axes = _normalize_axes(axis, self.ndim)
        shape = self.shape
        out = self.data.sum(axis=axes, keepdims=keepdims)

        def vjp(g: np.ndarray) -> tuple[np.ndarray]:
            if not keepdims:
                g = np.expand_dims(g, axes)
            return (np.broadcast_to(g, shape).copy(),)

        return Tensor._result(out, (self,), vjp)

    def mean(self, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> Tensor:
        """Mean over ``axis`` (all axes by default)."""
        axes = _normalize_axes(axis, self.ndim)
        count = int(np.prod([self.shape[a] for a in axes])) if axes else 1
        if count == 0:
            raise ValueError("mean of an empty tensor is undefined")
        return self.sum(axis=axes, keepdims=keepdims) * (1.0 / count)

    # -- elementwise functions -------------------------------------------------------
    def exp(self) -> Tensor:
        e = np.exp(self.data)
        return Tensor._result(e, (self,), lambda g: (g * e,))

    def log(self) -> Tensor:
        x = self.data
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.log(x)
        return Tensor._result(out, (self,), lambda g: (g / x,))

    def tanh(self) -> Tensor:
        t = np.tanh(self.data)
        return Tensor._result(t, (self,), lambda g: (g * (1.0 - t * t),))

    def sigmoid(self) -> Tensor:
        s = np.asarray(_sigmoid(self.data))
        return Tensor._result(s, (self,), lambda g: (g * s * (1.0 - s),))

    def relu(self) -> Tensor:
        """``max(x, 0)``; subgradient 0 at ``x = 0``."""
        mask = self.data > 0
        return Tensor._result(np.where(mask, self.data, 0.0), (self,), lambda g: (g * mask,))

    def softplus(self) -> Tensor:
        """``log(1 + e^x)`` (stable); derivative ``sigmoid(x)``."""
        x = self.data
        return Tensor._result(
            np.asarray(_softplus(x)), (self,), lambda g: (g * np.asarray(_sigmoid(x)),)
        )

    def log_softmax(self, axis: int = -1) -> Tensor:
        r"""``x - logsumexp(x, axis)``.

        VJP: :math:`\bar x = \bar y - \operatorname{softmax}(x)\sum_{\text{axis}}\bar y`.
        """
        (ax,) = _normalize_axes(axis, self.ndim)
        out = np.asarray(_log_softmax(self.data, axis=ax))
        soft = np.exp(out)
        return Tensor._result(out, (self,), lambda g: (g - soft * g.sum(axis=ax, keepdims=True),))

    # -- shape operations --------------------------------------------------------------
    def reshape(self, *shape: Any) -> Tensor:
        new = shape[0] if len(shape) == 1 and not isinstance(shape[0], int) else shape
        old = self.shape
        return Tensor._result(self.data.reshape(new), (self,), lambda g: (g.reshape(old),))

    def transpose(self, axes: Sequence[int] | None = None) -> Tensor:
        perm = tuple(range(self.ndim))[::-1] if axes is None else tuple(axes)
        if sorted(perm) != list(range(self.ndim)):
            raise ValueError(f"axes {axes!r} is not a permutation of the {self.ndim} axes")
        inverse = tuple(np.argsort(perm))
        return Tensor._result(self.data.transpose(perm), (self,), lambda g: (g.transpose(inverse),))

    @property
    def T(self) -> Tensor:
        return self.transpose()

    def __getitem__(self, index: Any) -> Tensor:
        if isinstance(index, Tensor) or (
            isinstance(index, tuple) and any(isinstance(i, Tensor) for i in index)
        ):
            raise TypeError("index with integers, slices or NumPy arrays, not Tensors")
        index = _freeze_index(index)
        out = self.data[index]
        x = self.data

        def vjp(g: np.ndarray) -> tuple[np.ndarray]:
            gx = np.zeros_like(x)
            np.add.at(gx, index, g)  # repeated indices accumulate
            return (gx,)

        return Tensor._result(np.array(out, dtype=np.float64), (self,), vjp)


def _freeze_index(index: Any) -> Any:
    """Copy array/list parts of an index so later caller mutations cannot change backward."""
    if isinstance(index, tuple):
        return tuple(_freeze_index(i) for i in index)
    if isinstance(index, (list, np.ndarray)):
        arr = np.array(index, copy=True)
        if arr.dtype != np.bool_ and not np.issubdtype(arr.dtype, np.integer):
            raise IndexError(f"array indices must be integers or booleans, got {arr.dtype}")
        return arr
    return index


# ---------------------------------------------------------------------------
# multi-input tensor functions
# ---------------------------------------------------------------------------
_OP_COMPLEXITY = Complexity(time="O(n)", space="O(n)", notes="n = number of output elements")


def _tensors(seq: Sequence[Any], name: str) -> list[Tensor]:
    if isinstance(seq, (Tensor, np.ndarray)) or not isinstance(seq, Sequence) or not seq:
        raise TypeError(f"{name} expects a non-empty list or tuple of tensors")
    return [t if isinstance(t, Tensor) else Tensor(t) for t in seq]


@register(domain="numerics", tasks=("automatic-differentiation",), complexity=_OP_COMPLEXITY)
def concatenate(tensors: Sequence[Tensor | ArrayLike], axis: int = 0) -> Tensor:
    """Join tensors along an existing axis (differentiable ``np.concatenate``).

    Complexity
    ----------
    Time and space O(n) for n output elements.

    Examples
    --------
    >>> a = Tensor([1.0, 2.0], requires_grad=True)
    >>> c = concatenate([a, Tensor([3.0])])
    >>> (c * Tensor([1.0, 2.0, 3.0])).sum().backward()
    >>> a.grad
    array([1., 2.])
    """
    ts = _tensors(tensors, "concatenate")
    out = np.concatenate([t.data for t in ts], axis=axis)
    ax = axis % out.ndim
    splits = np.cumsum([t.shape[ax] for t in ts])[:-1]
    return Tensor._result(out, tuple(ts), lambda g: tuple(np.split(g, splits, axis=ax)))


@register(domain="numerics", tasks=("automatic-differentiation",), complexity=_OP_COMPLEXITY)
def stack(tensors: Sequence[Tensor | ArrayLike], axis: int = 0) -> Tensor:
    """Join same-shape tensors along a new axis (differentiable ``np.stack``).

    Complexity
    ----------
    Time and space O(n) for n output elements.

    Examples
    --------
    >>> a = Tensor([1.0, 2.0], requires_grad=True)
    >>> s = stack([a, a * 3.0])
    >>> s.sum().backward()
    >>> a.grad
    array([4., 4.])
    """
    ts = _tensors(tensors, "stack")
    out = np.stack([t.data for t in ts], axis=axis)
    ax = axis % out.ndim
    return Tensor._result(
        out, tuple(ts), lambda g: tuple(np.take(g, i, axis=ax) for i in range(len(ts)))
    )


@register(domain="numerics", tasks=("automatic-differentiation",), complexity=_OP_COMPLEXITY)
def maximum(a: Tensor | ArrayLike, b: Tensor | ArrayLike) -> Tensor:
    """Elementwise maximum with broadcasting; ties split the gradient equally.

    Complexity
    ----------
    Time and space O(n) for n output elements.

    Examples
    --------
    >>> a = Tensor([1.0, 5.0, 2.0], requires_grad=True)
    >>> maximum(a, 2.0).sum().backward()
    >>> a.grad
    array([0. , 1. , 0.5])
    """
    ta, tb = Tensor._wrap(a), Tensor._wrap(b)
    x, y = ta.data, tb.data
    out = np.maximum(x, y)
    wa = np.where(x > y, 1.0, np.where(x == y, 0.5, 0.0))
    return Tensor._result(
        out,
        (ta, tb),
        lambda g: (_unbroadcast(g * wa, x.shape), _unbroadcast(g * (1.0 - wa), y.shape)),
    )


@register(domain="numerics", tasks=("automatic-differentiation",), complexity=_OP_COMPLEXITY)
def where(condition: ArrayLike, a: Tensor | ArrayLike, b: Tensor | ArrayLike) -> Tensor:
    """Elementwise ``a`` where ``condition`` else ``b`` (differentiable in ``a`` and ``b``).

    Complexity
    ----------
    Time and space O(n) for n output elements.

    Examples
    --------
    >>> x = Tensor([-1.0, 2.0], requires_grad=True)
    >>> where(x > 0, x * x, -x).sum().backward()
    >>> x.grad
    array([-1.,  4.])
    """
    if isinstance(condition, Tensor):
        raise TypeError("condition must be a boolean array, not a Tensor")
    cond = np.asarray(condition)
    if cond.dtype != np.bool_:
        raise TypeError(f"condition must be boolean, got dtype {cond.dtype}")
    ta, tb = Tensor._wrap(a), Tensor._wrap(b)
    x, y = ta.data, tb.data
    out = np.where(cond, x, y)
    return Tensor._result(
        out,
        (ta, tb),
        lambda g: (
            _unbroadcast(np.where(cond, g, 0.0), x.shape),
            _unbroadcast(np.where(cond, 0.0, g), y.shape),
        ),
    )


# ---------------------------------------------------------------------------
# functional API
# ---------------------------------------------------------------------------
def _argnums(argnums: int | Sequence[int], nargs: int) -> tuple[int, ...]:
    nums = (argnums,) if isinstance(argnums, (int, np.integer)) else tuple(argnums)
    if not nums:
        raise ValueError("argnums must not be empty")
    for k in nums:
        if not isinstance(k, (int, np.integer)) or not -nargs <= k < nargs:
            raise ValueError(f"argnums {argnums!r} out of range for {nargs} arguments")
    return tuple(int(k) % nargs for k in nums)


def _run(
    f: Callable[..., Tensor], argnums: int | Sequence[int], args: tuple[Any, ...]
) -> tuple[float, tuple[np.ndarray, ...]]:
    nums = _argnums(argnums, len(args))
    targs = list(args)
    leaves = {}
    for k in nums:
        leaves[k] = Tensor(as_real(args[k], f"argument {k}", finite=False), requires_grad=True)
        targs[k] = leaves[k]
    out = f(*targs)
    if not isinstance(out, Tensor) or out.size != 1:
        raise TypeError("f must return a one-element Tensor")
    value = out.item()
    if out.requires_grad:
        out.backward()
    grads = tuple(
        leaves[k].grad if leaves[k].grad is not None else np.zeros(leaves[k].shape) for k in nums
    )
    return value, grads


@register(
    domain="numerics",
    tasks=("automatic-differentiation",),
    complexity=Complexity(time="O(n)", space="O(n)", notes="one forward and one backward pass"),
    references=_REFS,
)
def grad(f: Callable[..., Tensor], argnums: int | Sequence[int] = 0) -> Callable[..., Any]:
    r"""Return a function computing :math:`\nabla f` with respect to argument(s) ``argnums``.

    ``f`` receives :class:`Tensor` objects for the differentiated arguments (other
    arguments are passed unchanged) and must return a one-element tensor.

    Parameters
    ----------
    f : callable
        Scalar-valued function built from tensor operations.
    argnums : int or sequence of int, default=0
        Which positional arguments to differentiate.

    Returns
    -------
    callable
        ``g(*args)`` returning an ndarray (int ``argnums``) or a tuple of ndarrays.

    Complexity
    ----------
    One forward and one backward pass: a constant multiple of the cost of ``f``.

    References
    ----------
    Baydin et al., *JMLR* 18 (2018), §3.2.

    Examples
    --------
    >>> import numpy as np
    >>> g = grad(lambda w, x: ((x @ w) ** 2).sum())
    >>> g(np.array([1.0, 1.0]), np.array([[1.0, 2.0]]))  # 2 (x.w) x
    array([ 6., 12.])
    """
    check_callable(f, "f")
    single = isinstance(argnums, (int, np.integer))

    def gradient(*args: Any) -> Any:
        _, grads = _run(f, argnums, args)
        return grads[0] if single else grads

    return gradient


@register(
    domain="numerics",
    tasks=("automatic-differentiation",),
    complexity=Complexity(time="O(n)", space="O(n)", notes="one forward and one backward pass"),
    references=_REFS,
)
def value_and_grad(
    f: Callable[..., Tensor], argnums: int | Sequence[int] = 0
) -> Callable[..., tuple[float, Any]]:
    r"""Like :func:`grad`, but the returned function gives ``(f(*args), gradient)``.

    Complexity
    ----------
    One forward and one backward pass.

    References
    ----------
    Baydin et al., *JMLR* 18 (2018), §3.2.

    Examples
    --------
    >>> vg = value_and_grad(lambda x: (x * x).sum())
    >>> vg([1.0, 2.0])
    (5.0, array([2., 4.]))
    """
    check_callable(f, "f")
    single = isinstance(argnums, (int, np.integer))

    def fn(*args: Any) -> tuple[float, Any]:
        value, grads = _run(f, argnums, args)
        return value, grads[0] if single else grads

    return fn


@register(
    domain="numerics",
    tasks=("automatic-differentiation", "verification"),
    complexity=Complexity(
        time="O(d n)", space="O(n)", notes="d = number of inputs; 2d evaluations of f"
    ),
    references=("Griewank & Walther, Evaluating Derivatives, 2nd ed. (2008), §1.4",),
)
def check_grad(f: Callable[[Tensor], Tensor], x: ArrayLike, eps: float = 1e-6) -> float:
    r"""Compare the reverse-mode gradient of ``f`` with central differences.

    .. math::
        \tilde g_i = \frac{f(x + \epsilon e_i) - f(x - \epsilon e_i)}{2\epsilon},\qquad
        \text{err} = \frac{\lVert g - \tilde g\rVert_\infty}
                         {\max(\lVert g\rVert_\infty, \lVert\tilde g\rVert_\infty, 10^{-12})}.

    Central differences have truncation error :math:`O(\epsilon^2)` and rounding
    error :math:`O(u/\epsilon)`, so with the default :math:`\epsilon = 10^{-6}` a
    correct gradient of a well-scaled function typically gives err below
    :math:`10^{-7}`.

    Parameters
    ----------
    f : callable ``f(Tensor) -> Tensor``
        Scalar-valued function of one tensor.
    x : array_like
        Point of evaluation (any shape).
    eps : float, default=1e-6
        Finite-difference step.

    Returns
    -------
    float
        The normwise relative error ``err`` above.

    Complexity
    ----------
    2d evaluations of ``f`` for d inputs, plus one backward pass.

    References
    ----------
    Griewank & Walther, *Evaluating Derivatives*, §1.4.

    Examples
    --------
    >>> check_grad(lambda t: (t.tanh() * t).sum(), [0.3, -1.2]) < 1e-8
    True
    """
    check_callable(f, "f")
    x = as_real(x, "x")
    if not (_is_real_number(eps) and np.isfinite(eps) and eps > 0):
        raise ValueError(f"eps must be a positive number, got {eps!r}")
    _, (analytic,) = _run(f, 0, (x,))
    numeric = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        numeric[i] = (f(Tensor(xp)).item() - f(Tensor(xm)).item()) / (2.0 * eps)
    scale = max(float(np.max(np.abs(analytic), initial=0.0)),
                float(np.max(np.abs(numeric), initial=0.0)), 1e-12)  # fmt: skip
    return float(np.max(np.abs(analytic - numeric), initial=0.0)) / scale
