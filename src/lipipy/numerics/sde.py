r"""Stochastic differential equations :math:`dX_t = a(t, X_t)\,dt + b(t, X_t)\,dW_t`.

Both schemes use a uniform grid :math:`t_n = t_0 + n\Delta t`,
:math:`\Delta t = (t_1 - t_0)/N`, and Brownian increments
:math:`\Delta W_n \sim \mathcal N(0, \Delta t)`, independent across steps, paths and
components. Noise is **scalar or diagonal**: component :math:`i` is driven by its own
Brownian motion :math:`W^i` with coefficient :math:`b_i(t, X)`.

Euler-Maruyama
--------------
.. math::  X_{n+1} = X_n + a(t_n, X_n)\,\Delta t + b(t_n, X_n)\,\Delta W_n .

Strong order 1/2 (:math:`\mathbb E|X_N - X(T)| = O(\Delta t^{1/2})`), weak order 1.

Milstein
--------
Adding the next term of the Itô-Taylor expansion,

.. math::
    X_{n+1} = X_n + a\,\Delta t + b\,\Delta W_n
              + \tfrac12\, b\, \frac{\partial b}{\partial x}\,(\Delta W_n^2 - \Delta t),

gives strong order 1. For diagonal noise the formula is applied per component with
:math:`\partial b_i/\partial x_i`; it is the exact Milstein scheme when each
:math:`b_i` depends only on :math:`x_i` (otherwise Lévy-area terms would be needed).

Both methods are vectorised over paths: each step is one array operation on all
:math:`P` paths. The tests measure, on geometric Brownian motion
:math:`dX = \mu X\,dt + \sigma X\,dW` with exact solution
:math:`X_T = X_0\exp((\mu - \sigma^2/2)T + \sigma W_T)` and the same Brownian paths at
every resolution, strong orders :math:`0.5 \pm 0.1` (Euler-Maruyama) and
:math:`1.0 \pm 0.1` (Milstein).

References
----------
P. E. Kloeden & E. Platen, *Numerical Solution of Stochastic Differential Equations*
(Springer, 1992), §9.1, §10.2-10.3. D. J. Higham, "An algorithmic introduction to
numerical simulation of stochastic differential equations", *SIAM Review* 43 (2001)
525-546.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from lipipy.core.complexity import Complexity
from lipipy.core.random import RandomState, check_random_state
from lipipy.core.registry import register
from lipipy.numerics._common import (
    as_finite_float,
    as_real,
    check_callable,
    check_positive_int,
    jsonable,
)

__all__ = ["SDEResult", "euler_maruyama", "milstein"]

_KP = "Kloeden & Platen, Numerical Solution of SDEs (1992)"
_HIGHAM = "D. J. Higham, SIAM Review 43 (2001) 525-546"
_NOTES = (
    "T = n_steps, P = n_paths, per-path dimension d; plus T evaluations of the "
    "coefficient functions on (P, d) arrays. Space O(P T d) for the returned paths; "
    "O(P d) with return_paths=False"
)

Coefficient = Callable[[float, np.ndarray], ArrayLike]


@dataclass(frozen=True)
class SDEResult:
    """Outcome of :func:`euler_maruyama` and :func:`milstein`.

    Attributes
    ----------
    t : ndarray, shape (n_steps + 1,)
        Time grid.
    paths : ndarray or None
        Shape ``(n_steps + 1, n_paths)`` for scalar ``x0`` or
        ``(n_steps + 1, n_paths, d)`` for ``x0`` of shape ``(d,)``; ``paths[0]`` is
        ``x0``. ``None`` when ``return_paths=False``.
    terminal : ndarray
        Values at ``t[-1]``, shape ``(n_paths,)`` or ``(n_paths, d)``.
    """

    t: np.ndarray
    paths: np.ndarray | None
    terminal: np.ndarray

    def __post_init__(self) -> None:
        if self.paths is not None and self.paths.shape[0] != self.t.shape[0]:
            raise ValueError("paths must have one row per time point")
        if self.paths is not None and self.paths.shape[1:] != self.terminal.shape:
            raise ValueError("terminal must have the shape of one row of paths")

    def to_dict(self) -> dict[str, Any]:
        return {k: jsonable(v) for k, v in asdict(self).items()}


def _coefficient(func: Coefficient, t: float, x: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(func(t, x), dtype=np.float64)
    try:
        return np.broadcast_to(out, x.shape)
    except ValueError:
        raise ValueError(
            f"{name} must return an array broadcastable to the state shape {x.shape}, "
            f"got {out.shape}"
        ) from None


def _simulate(
    drift: Coefficient,
    diffusion: Coefficient,
    diffusion_dx: Coefficient | None,
    x0: ArrayLike,
    t_span: tuple[float, float],
    n_steps: int,
    n_paths: int,
    random_state: RandomState,
    dW: ArrayLike | None,
    return_paths: bool,
) -> SDEResult:
    check_callable(drift, "drift")
    check_callable(diffusion, "diffusion")
    if diffusion_dx is not None:
        check_callable(diffusion_dx, "diffusion_dx")
    if not isinstance(t_span, (tuple, list, np.ndarray)) or len(t_span) != 2:
        raise ValueError("t_span must be a pair (t0, t1)")
    t0 = as_finite_float(t_span[0], "t_span[0]")
    t1 = as_finite_float(t_span[1], "t_span[1]")
    if not t1 > t0:
        raise ValueError(f"need t0 < t1, got {t0}, {t1}")
    n_steps = check_positive_int(n_steps, "n_steps")
    n_paths = check_positive_int(n_paths, "n_paths")
    if not isinstance(return_paths, (bool, np.bool_)):
        raise TypeError("return_paths must be a bool")
    start = as_real(x0, "x0")
    if start.ndim > 1 or start.size == 0:
        raise ValueError(f"x0 must be a scalar or a non-empty 1-D array, got shape {start.shape}")
    state_shape = (n_paths,) + start.shape  # (P,) or (P, d)
    dt = (t1 - t0) / n_steps
    t = t0 + dt * np.arange(n_steps + 1)
    t[-1] = t1
    if dW is not None:
        increments = as_real(dW, "dW")
        if increments.shape != (n_steps,) + state_shape:
            raise ValueError(
                f"dW must have shape {(n_steps,) + state_shape}, got {increments.shape}"
            )
        rng = None
    else:
        increments = None
        rng = check_random_state(random_state)
    sqrt_dt = np.sqrt(dt)

    x = np.broadcast_to(start, state_shape).copy()
    paths = np.empty((n_steps + 1,) + state_shape) if return_paths else None
    if paths is not None:
        paths[0] = x
    with np.errstate(over="ignore", invalid="ignore"):
        for n in range(n_steps):
            if increments is not None:
                dw = increments[n]
            else:
                dw = sqrt_dt * rng.standard_normal(state_shape)  # type: ignore[union-attr]
            tn = float(t[n])
            a = _coefficient(drift, tn, x, "drift")
            b = _coefficient(diffusion, tn, x, "diffusion")
            x_new = x + a * dt + b * dw
            if diffusion_dx is not None:
                bx = _coefficient(diffusion_dx, tn, x, "diffusion_dx")
                x_new += 0.5 * b * bx * (dw * dw - dt)
            if not np.all(np.isfinite(x_new)):
                raise FloatingPointError(
                    f"the solution became non-finite at t = {float(t[n + 1])!r}; "
                    "reduce the step or check the coefficients"
                )
            x = x_new
            if paths is not None:
                paths[n + 1] = x
    return SDEResult(t=t, paths=paths, terminal=x)


@register(
    domain="numerics",
    tasks=("sde", "stochastic-simulation", "monte-carlo"),
    complexity=Complexity(time="O(T P d)", space="O(T P d)", notes=_NOTES),
    references=(_KP + ", §9.1, §10.2", _HIGHAM),
)
def euler_maruyama(
    drift: Coefficient,
    diffusion: Coefficient,
    x0: ArrayLike,
    t_span: tuple[float, float],
    n_steps: int,
    n_paths: int = 1,
    random_state: RandomState = None,
    dW: ArrayLike | None = None,
    return_paths: bool = True,
) -> SDEResult:
    r"""Simulate :math:`dX = a\,dt + b\,dW` with the Euler-Maruyama scheme (strong order 1/2).

    :math:`X_{n+1} = X_n + a(t_n, X_n)\Delta t + b(t_n, X_n)\Delta W_n`, vectorised over
    paths.

    Parameters
    ----------
    drift : callable ``a(t, x)``
        Drift; ``x`` has shape ``(n_paths,)`` (scalar ``x0``) or ``(n_paths, d)``.
        The result must broadcast to the shape of ``x``.
    diffusion : callable ``b(t, x)``
        Diagonal diffusion coefficient, same conventions as ``drift``.
    x0 : float or array_like, shape (d,)
        Initial state shared by all paths.
    t_span : (t0, t1)
        Finite interval with ``t0 < t1``.
    n_steps : int
        Number of uniform steps (>= 1).
    n_paths : int, default=1
        Number of independent paths (>= 1).
    random_state : None, int, SeedSequence or Generator
        Seed for the Brownian increments (unused when ``dW`` is given).
    dW : array_like, optional
        Brownian increments of shape ``(n_steps, n_paths)`` or
        ``(n_steps, n_paths, d)``, each :math:`\mathcal N(0, \Delta t)`. Pass the same
        increments to compare schemes or step sizes on identical paths.
    return_paths : bool, default=True
        If False, only the terminal values are kept (memory O(P d)).

    Returns
    -------
    SDEResult
        ``t`` (shape ``(n_steps + 1,)``), ``paths`` (shape ``(n_steps + 1, n_paths)``
        or ``(n_steps + 1, n_paths, d)``, or None) and ``terminal``.

    Raises
    ------
    FloatingPointError
        If the numerical solution overflows to NaN/inf.

    Complexity
    ----------
    Time O(T P d) for T steps and P paths (plus T coefficient evaluations on
    ``(P, d)`` arrays). Space O(T P d) for the returned paths, O(P d) with
    ``return_paths=False`` (increments are drawn one step at a time).

    References
    ----------
    Kloeden & Platen (1992), §9.1, §10.2; D. J. Higham, *SIAM Review* 43 (2001).

    Examples
    --------
    >>> res = euler_maruyama(lambda t, x: 0.05 * x, lambda t, x: 0.2 * x, 1.0, (0.0, 1.0),
    ...                      n_steps=100, n_paths=20_000, random_state=0)
    >>> res.paths.shape, res.terminal.shape
    ((101, 20000), (20000,))
    >>> bool(abs(res.terminal.mean() - 1.0512710963760241) < 0.01)  # E X_1 = e^0.05
    True
    """
    return _simulate(
        drift, diffusion, None, x0, t_span, n_steps, n_paths, random_state, dW, return_paths
    )


@register(
    domain="numerics",
    tasks=("sde", "stochastic-simulation", "monte-carlo"),
    complexity=Complexity(time="O(T P d)", space="O(T P d)", notes=_NOTES),
    references=(_KP + ", §10.3", _HIGHAM),
)
def milstein(
    drift: Coefficient,
    diffusion: Coefficient,
    diffusion_dx: Coefficient,
    x0: ArrayLike,
    t_span: tuple[float, float],
    n_steps: int,
    n_paths: int = 1,
    random_state: RandomState = None,
    dW: ArrayLike | None = None,
    return_paths: bool = True,
) -> SDEResult:
    r"""Simulate :math:`dX = a\,dt + b\,dW` with the Milstein scheme (strong order 1).

    :math:`X_{n+1} = X_n + a\Delta t + b\Delta W_n + \tfrac12 b\,b_x(\Delta W_n^2 - \Delta t)`,
    applied per component for diagonal noise (exact Milstein when each :math:`b_i`
    depends only on :math:`x_i`).

    Parameters
    ----------
    drift, diffusion : callable
        As in :func:`euler_maruyama`.
    diffusion_dx : callable ``b_x(t, x)``
        :math:`\partial b_i / \partial x_i`, same shape conventions as ``diffusion``.
    x0, t_span, n_steps, n_paths, random_state, dW, return_paths
        As in :func:`euler_maruyama`.

    Returns
    -------
    SDEResult

    Raises
    ------
    FloatingPointError
        If the numerical solution overflows to NaN/inf.

    Complexity
    ----------
    Time O(T P d) (plus T evaluations of each coefficient function); space O(T P d)
    for the returned paths, O(P d) with ``return_paths=False``.

    References
    ----------
    Kloeden & Platen (1992), §10.3; D. J. Higham, *SIAM Review* 43 (2001).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> dW = rng.normal(0.0, np.sqrt(1 / 64), size=(64, 1000))
    >>> res = milstein(lambda t, x: 0.1 * x, lambda t, x: 0.3 * x, lambda t, x: 0.3,
    ...                1.0, (0.0, 1.0), n_steps=64, n_paths=1000, dW=dW, return_paths=False)
    >>> exact = np.exp((0.1 - 0.045) + 0.3 * dW.sum(axis=0))
    >>> bool(np.mean(np.abs(res.terminal - exact)) < 3e-3)
    True
    """
    if diffusion_dx is None:
        raise TypeError("milstein needs diffusion_dx, the derivative of the diffusion")
    return _simulate(
        drift, diffusion, diffusion_dx, x0, t_span, n_steps, n_paths, random_state, dW,
        return_paths,
    )  # fmt: skip
