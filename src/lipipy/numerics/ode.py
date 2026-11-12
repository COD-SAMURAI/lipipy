r"""Initial value problems :math:`y'(t) = f(t, y)`, :math:`y(t_0) = y_0`.

Classical Runge-Kutta (``"rk4"``, fixed step)
--------------------------------------------
.. math::
    k_1 &= f(t_n, y_n),\quad k_2 = f(t_n + \tfrac h2, y_n + \tfrac h2 k_1),\quad
    k_3 = f(t_n + \tfrac h2, y_n + \tfrac h2 k_2),\quad k_4 = f(t_n + h, y_n + h k_3),\\
    y_{n+1} &= y_n + \tfrac h6 (k_1 + 2k_2 + 2k_3 + k_4).

Global error :math:`O(h^4)`. The step is :math:`h' = (t_1 - t_0)/N` with
:math:`N = \lceil |t_1 - t_0| / h\rceil`, so the grid lands exactly on :math:`t_1`
and :math:`|h'| \le h`.

Dormand-Prince 5(4) (``"rk45"``, adaptive)
-----------------------------------------
Seven stages (the last one is :math:`f(t_{n+1}, y_{n+1})`, reused as the next first
stage: "first same as last") give a 5th-order solution :math:`y_{n+1}` and an embedded
4th-order one :math:`\hat y_{n+1}`. The local error estimate
:math:`e = y_{n+1} - \hat y_{n+1} = h\sum_i (b_i - \hat b_i) k_i` is measured in the
weighted RMS norm

.. math::  \lVert e \rVert = \sqrt{\frac1d \sum_j \Bigl(\frac{e_j}{\text{atol} +
           \text{rtol}\,\max(|y_{n,j}|, |y_{n+1,j}|)}\Bigr)^2}.

The step is accepted when :math:`\lVert e\rVert \le 1`. Either way the next step is
:math:`h \cdot \min(10, \max(0.2, 0.9\lVert e\rVert^{-1/5}))` (Hairer, Nørsett & Wanner,
I, §II.4), with growth disabled right after a rejection. The first step is chosen by
the HNW algorithm from :math:`f(t_0, y_0)` and one explicit Euler trial. These are the
same rules as ``scipy.integrate.solve_ivp(method="RK45")``, so both take the same steps.

Output at ``t_eval`` uses **dense output**: the free 4th-order continuous extension of
Dormand-Prince (Shampine 1986) :math:`y(t_n + \theta h) = y_n + h\sum_i k_i P_i(\theta)`,
so ``t_eval`` does not shorten any step.

The tolerances bound the *local* error per step. The global error is usually of the
same order but is not guaranteed; the tests check that it stays below
:math:`10\,(\text{atol} + \text{rtol}\max_t\lVert y(t)\rVert_\infty)` on five non-stiff
problems with known solutions for :math:`10^{-10} \le \text{rtol} \le 10^{-3}`.

Implicit (backward) Euler (``"implicit_euler"``, fixed step)
-----------------------------------------------------------
:math:`y_{n+1} = y_n + h f(t_{n+1}, y_{n+1})` is solved for :math:`y_{n+1}` by Newton's
method on :math:`G(z) = z - y_n - h f(t_{n+1}, z)`:

.. math::  (I - h J(t_{n+1}, z))\,\Delta = -G(z),\qquad z \leftarrow z + \Delta,

starting from :math:`z = y_n`, with the Jacobian :math:`J = \partial f/\partial y` from
``jac`` or forward differences (step :math:`\sqrt{\varepsilon}\max(1, |z_j|)`). Iteration
stops when :math:`\lVert\Delta\rVert \le 10^{-3}` in the same weighted norm as above,
or when :math:`\Delta` reaches roundoff level (RMS of :math:`\Delta_j/(\text{atol}+|z_j|)`
at most :math:`100\varepsilon`), and fails after 50 iterations.
The Newton iteration is undamped and the step is fixed, so it can fail at
very fast nonlinear transitions (e.g. the jumps of a stiff van der Pol oscillator);
the solver then stops with ``success=False``. Global error :math:`O(h)`; the
method is A-stable (:math:`|R(z)| = |1/(1-z)| < 1` for
:math:`\mathrm{Re}\,z < 0`), so stiff problems can use steps far beyond the explicit
stability limit.

``t_eval`` for the fixed-step methods is served by interpolation between grid
points: cubic Hermite for ``rk4`` (error :math:`O(h^4)`, matching the method) and
linear for ``implicit_euler``.

References
----------
J. R. Dormand & P. J. Prince, "A family of embedded Runge-Kutta formulae",
*J. Comput. Appl. Math.* 6 (1980) 19-26. L. F. Shampine, "Some practical Runge-Kutta
formulas", *Math. Comp.* 46 (1986) 135-150. E. Hairer, S. P. Nørsett & G. Wanner,
*Solving Ordinary Differential Equations I*, 2nd ed. (1993), §II.4. E. Hairer &
G. Wanner, *Solving Ordinary Differential Equations II*, 2nd ed. (1996), §IV.3.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from lipipy.core.complexity import Complexity
from lipipy.core.registry import register
from lipipy.numerics._common import as_finite_float, as_real, check_callable, jsonable
from lipipy.numerics.linalg import LinAlgError, lu, lu_solve

__all__ = ["ODEResult", "solve_ode"]

_METHODS = ("rk45", "rk4", "implicit_euler")
_EPS = float(np.finfo(np.float64).eps)
_NEWTON_TOL = 1e-3
_NEWTON_MAXITER = 50
_NEWTON_ROUNDOFF = 100 * _EPS  # tolerances below roundoff cannot be met; stop there

# Dormand-Prince 5(4) tableau.
_C = np.array([0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0])
_A = [
    np.array([]),
    np.array([1 / 5]),
    np.array([3 / 40, 9 / 40]),
    np.array([44 / 45, -56 / 15, 32 / 9]),
    np.array([19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729]),
    np.array([9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656]),
]
_B = np.array([35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84])
_B_HAT = np.array([5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40])
_E = np.append(_B, 0.0) - _B_HAT  # error weights for the 7 stages
# Dense output: y(t + theta h) = y + h * K^T @ (_P @ [theta, theta^2, theta^3, theta^4]).
_P = np.array(
    [
        [1, -8048581381 / 2820520608, 8663915743 / 2820520608, -12715105075 / 11282082432],
        [0, 0, 0, 0],
        [0, 131558114200 / 32700410799, -68118460800 / 10900136933, 87487479700 / 32700410799],
        [0, -1754552775 / 470086768, 14199869525 / 1410260304, -10690763975 / 1880347072],
        [0, 127303824393 / 49829197408, -318862633887 / 49829197408, 701980252875 / 199316789632],
        [0, -282668133 / 205662961, 2019193451 / 616988883, -1453857185 / 822651844],
        [0, 40617522 / 29380423, -110615467 / 29380423, 69997945 / 29380423],
    ]
)
_SAFETY, _MIN_FACTOR, _MAX_FACTOR = 0.9, 0.2, 10.0


@dataclass(frozen=True)
class ODEResult:
    """Outcome of :func:`solve_ode`.

    Attributes
    ----------
    t : ndarray, shape (m,)
        Output times (``t_eval`` if given, else every step).
    y : ndarray, shape (d, m)
        Solution at ``t`` (column ``i`` is :math:`y(t_i)`), like SciPy's ``solve_ivp``.
    nfev : int
        Evaluations of ``fun`` (including finite-difference Jacobians).
    njev : int
        Evaluations of the user ``jac`` (or finite-difference Jacobians).
    success : bool
        Whether the end of ``t_span`` was reached.
    message : str
        Why the solver stopped.
    """

    t: np.ndarray
    y: np.ndarray
    nfev: int
    njev: int
    success: bool
    message: str

    def __post_init__(self) -> None:
        if self.y.ndim != 2 or self.y.shape[1] != self.t.shape[0]:
            raise ValueError(f"y must have shape (d, {self.t.shape[0]}), got {self.y.shape}")
        if self.nfev < 0 or self.njev < 0:
            raise ValueError("nfev and njev must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {k: jsonable(v) for k, v in asdict(self).items()}


class _StopIntegrationError(Exception):
    """fun returned NaN/inf (or the state overflowed)."""


class _Problem:
    """Wrap fun/jac with validation and evaluation counters."""

    def __init__(self, fun: Callable[..., ArrayLike], jac: Any, d: int) -> None:
        self.fun, self.jac, self.d = fun, jac, d
        self.nfev = self.njev = 0

    def f(self, t: float, y: np.ndarray) -> np.ndarray:
        self.nfev += 1
        out = np.asarray(self.fun(t, y), dtype=np.float64)
        if out.shape != (self.d,):
            if out.size != self.d:
                raise ValueError(f"fun must return shape ({self.d},), got {out.shape}")
            out = out.reshape(self.d)
        if not np.all(np.isfinite(out)):
            raise _StopIntegrationError(f"fun returned a non-finite value at t = {t!r}")
        return out

    def jacobian(self, t: float, y: np.ndarray, fy: np.ndarray) -> np.ndarray:
        self.njev += 1
        if self.jac is None:  # forward differences, column by column
            J = np.empty((self.d, self.d))
            for j in range(self.d):
                step = math.sqrt(_EPS) * max(1.0, abs(y[j]))
                yp = y.copy()
                yp[j] += step
                J[:, j] = (self.f(t, yp) - fy) / (yp[j] - y[j])
            return J
        J = np.asarray(self.jac(t, y) if callable(self.jac) else self.jac, dtype=np.float64)
        if J.shape != (self.d, self.d):
            raise ValueError(f"jac must have shape ({self.d}, {self.d}), got {J.shape}")
        if not np.all(np.isfinite(J)):
            raise _StopIntegrationError(f"jac returned a non-finite value at t = {t!r}")
        return J


def _rms(x: np.ndarray) -> float:
    return float(np.linalg.norm(x) / math.sqrt(x.size))


def _check_t_eval(t_eval: ArrayLike | None, t0: float, t1: float) -> np.ndarray | None:
    if t_eval is None:
        return None
    te = as_real(t_eval, "t_eval")
    te = te.reshape(-1) if te.ndim == 0 else te
    if te.ndim != 1:
        raise ValueError(f"t_eval must be 1-D, got shape {te.shape}")
    lo, hi = min(t0, t1), max(t0, t1)
    if np.any(te < lo) or np.any(te > hi):
        raise ValueError("t_eval values must lie within t_span")
    d = np.diff(te) * (1.0 if t1 >= t0 else -1.0)
    if np.any(d < 0):
        raise ValueError("t_eval must be sorted in the direction of integration")
    return te


# ---------------------------------------------------------------------------
# fixed-step methods
# ---------------------------------------------------------------------------
def _hermite(t0: float, t1: float, y0, y1, f0, f1, t: np.ndarray) -> np.ndarray:
    """Cubic Hermite interpolation on [t0, t1]; returns shape (d, len(t))."""
    h = t1 - t0
    s = (t - t0) / h
    h00 = (1 + 2 * s) * (1 - s) ** 2
    h10 = s * (1 - s) ** 2
    h01 = s * s * (3 - 2 * s)
    h11 = s * s * (s - 1)
    return np.outer(y0, h00) + np.outer(f0, h * h10) + np.outer(y1, h01) + np.outer(f1, h * h11)


def _rk4_step(p: _Problem, t: float, y: np.ndarray, k1: np.ndarray, h: float) -> np.ndarray:
    k2 = p.f(t + h / 2, y + (h / 2) * k1)
    k3 = p.f(t + h / 2, y + (h / 2) * k2)
    k4 = p.f(t + h, y + h * k3)
    return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def _implicit_euler_step(
    p: _Problem, t_new: float, y: np.ndarray, h: float, rtol: float, atol: float
) -> np.ndarray:
    z = y.copy()
    eye = np.eye(p.d)
    for _ in range(_NEWTON_MAXITER):
        fz = p.f(t_new, z)
        G = z - y - h * fz
        J = p.jacobian(t_new, z, fz)
        try:
            delta = lu_solve(lu(eye - h * J), -G)
        except LinAlgError:
            raise _StopIntegrationError(
                f"singular Newton matrix I - h J at t = {t_new!r}"
            ) from None
        z = z + delta
        if not np.all(np.isfinite(z)):
            raise _StopIntegrationError(f"Newton iterate became non-finite at t = {t_new!r}")
        if (
            _rms(delta / (atol + rtol * np.abs(z))) <= _NEWTON_TOL
            or _rms(delta / (atol + np.abs(z))) <= _NEWTON_ROUNDOFF
        ):
            return z
    raise _StopIntegrationError(
        f"Newton iteration did not converge in {_NEWTON_MAXITER} iterations at t = {t_new!r}"
    )


def _fixed_step(
    p: _Problem,
    method: str,
    t0: float,
    t1: float,
    y0: np.ndarray,
    h: float,
    t_eval: np.ndarray | None,
    rtol: float,
    atol: float,
) -> tuple[np.ndarray, np.ndarray, bool, str]:
    span = t1 - t0
    ratio = abs(span) / h
    n_steps = max(1, math.ceil(ratio * (1 - 1e-12)))
    step = span / n_steps
    ts, ys = [t0], [y0]
    out_t: list[float] = []
    out_y: list[np.ndarray] = []
    pending = 0  # index into t_eval of the next point to emit
    if t_eval is not None:
        while pending < t_eval.size and t_eval[pending] == t0:
            out_t.append(t0)
            out_y.append(y0.copy())
            pending += 1
    t, y = t0, y0
    success, message = True, "reached the end of t_span"
    try:
        f = p.f(t, y) if method == "rk4" else None
        for i in range(1, n_steps + 1):
            t_new = t1 if i == n_steps else t0 + i * step
            h_i = t_new - t
            if method == "rk4":
                y_new = _rk4_step(p, t, y, f, h_i)
                if not np.all(np.isfinite(y_new)):
                    raise _StopIntegrationError(f"solution became non-finite at t = {t_new!r}")
                need_f = i < n_steps or (t_eval is not None and pending < t_eval.size)
                f_new = p.f(t_new, y_new) if need_f else None
            else:
                y_new = _implicit_euler_step(p, t_new, y, h_i, rtol, atol)
                f_new = None
            if t_eval is not None:
                stop = pending
                while stop < t_eval.size and (t_eval[stop] - t_new) * np.sign(span) <= 0:
                    stop += 1
                if stop > pending:
                    tt = t_eval[pending:stop]
                    if method == "rk4":
                        block = _hermite(t, t_new, y, y_new, f, f_new, tt)
                    else:
                        s = (tt - t) / h_i
                        block = np.outer(y, 1 - s) + np.outer(y_new, s)
                    out_t.extend(tt.tolist())
                    out_y.extend(block.T)
                    pending = stop
            else:
                ts.append(t_new)
                ys.append(y_new)
            t, y, f = t_new, y_new, f_new
    except _StopIntegrationError as exc:
        success, message = False, str(exc)
    if t_eval is None:
        return np.array(ts), np.array(ys).T.reshape(p.d, -1), success, message
    return np.array(out_t), np.array(out_y).T.reshape(p.d, -1), success, message


# ---------------------------------------------------------------------------
# Dormand-Prince
# ---------------------------------------------------------------------------
def _initial_step(
    p: _Problem, t0: float, y0: np.ndarray, f0: np.ndarray, direction: float, length: float,
    rtol: float, atol: float, max_step: float,
) -> float:  # fmt: skip
    """Hairer, Nørsett & Wanner I, §II.4 starting step (error estimator order 4)."""
    scale = atol + np.abs(y0) * rtol
    d0, d1 = _rms(y0 / scale), _rms(f0 / scale)
    h0 = 1e-6 if d0 < 1e-5 or d1 < 1e-5 else 0.01 * d0 / d1
    h0 = min(h0, length)
    f1 = p.f(t0 + h0 * direction, y0 + h0 * direction * f0)
    d2 = _rms((f1 - f0) / scale) / h0
    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1 / 5)
    return min(100 * h0, h1, length, max_step)


def _dopri_step(
    p: _Problem, t: float, y: np.ndarray, f: np.ndarray, h: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One Dormand-Prince step: (y_new, f_new, K with the 7 stages as rows)."""
    K = np.empty((7, y.size))
    K[0] = f
    for s in range(1, 6):
        K[s] = p.f(t + _C[s] * h, y + h * (_A[s] @ K[:s]))
    y_new = y + h * (_B @ K[:6])
    if not np.all(np.isfinite(y_new)):
        raise _StopIntegrationError(f"solution became non-finite at t = {t + h!r}")
    K[6] = p.f(t + h, y_new)
    return y_new, K[6], K


def _rk45(
    p: _Problem,
    t0: float,
    t1: float,
    y0: np.ndarray,
    h_first: float | None,
    t_eval: np.ndarray | None,
    rtol: float,
    atol: float,
    max_step: float,
) -> tuple[np.ndarray, np.ndarray, bool, str]:
    direction = 1.0 if t1 >= t0 else -1.0
    length = abs(t1 - t0)
    ts, ys = [t0], [y0]
    out_t: list[float] = []
    out_y: list[np.ndarray] = []
    pending = 0
    if t_eval is not None:
        while pending < t_eval.size and t_eval[pending] == t0:
            out_t.append(t0)
            out_y.append(y0.copy())
            pending += 1
    t, y = t0, y0
    success, message = True, "reached the end of t_span"
    try:
        f = p.f(t, y)
        if length == 0.0:
            h_abs = 0.0
        elif h_first is None:
            h_abs = _initial_step(p, t0, y0, f, direction, length, rtol, atol, max_step)
        else:
            h_abs = min(h_first, length, max_step)
        while direction * (t1 - t) > 0:
            min_step = 10 * abs(np.nextafter(t, direction * np.inf) - t)
            h_abs = min(max(h_abs, min_step), max_step)
            rejected = False
            while True:
                if h_abs < min_step:
                    raise _StopIntegrationError(
                        f"required step size is below the spacing of floats at t = {t!r}"
                    )
                t_new = t + direction * h_abs
                if direction * (t_new - t1) > 0:
                    t_new = t1
                h = t_new - t
                h_abs = abs(h)
                y_new, f_new, K = _dopri_step(p, t, y, f, h)
                scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
                err = _rms(h * (_E @ K) / scale)
                if err < 1.0:
                    factor = _MAX_FACTOR if err == 0 else min(_MAX_FACTOR, _SAFETY * err**-0.2)
                    if rejected:
                        factor = min(1.0, factor)
                    h_abs *= factor
                    break
                h_abs *= max(_MIN_FACTOR, _SAFETY * err**-0.2)
                rejected = True
            if t_eval is not None:
                stop = pending
                while stop < t_eval.size and direction * (t_eval[stop] - t_new) <= 0:
                    stop += 1
                if stop > pending:
                    theta = (t_eval[pending:stop] - t) / h
                    powers = np.cumprod(np.tile(theta, (4, 1)), axis=0)  # theta^1..theta^4
                    block = y[:, None] + h * (K.T @ _P) @ powers
                    out_t.extend(t_eval[pending:stop].tolist())
                    out_y.extend(block.T)
                    pending = stop
            else:
                ts.append(t_new)
                ys.append(y_new)
            t, y, f = t_new, y_new, f_new
    except _StopIntegrationError as exc:
        success, message = False, str(exc)
    if t_eval is None:
        return np.array(ts), np.array(ys).T.reshape(p.d, -1), success, message
    return np.array(out_t), np.array(out_y).T.reshape(p.d, -1), success, message


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
@register(
    domain="numerics",
    tasks=("ode", "differential-equations"),
    complexity=Complexity(
        time="O(T d + T cost(f))",
        space="O(T d)",
        notes=(
            "T = steps (rk45: accepted + rejected, 6 f-evaluations each; rk4: 4 each); "
            "implicit_euler: O(T k (d^3 + d cost(f))) for k Newton iterations with "
            "finite-difference Jacobians; space is the stored trajectory "
            "(O((m + 1) d) with m = len(t_eval))"
        ),
    ),
    references=(
        "Dormand & Prince, J. Comput. Appl. Math. 6 (1980) 19-26",
        "Shampine, Math. Comp. 46 (1986) 135-150 (dense output)",
        "Hairer, Nørsett & Wanner, Solving ODEs I, 2nd ed. (1993), §II.4",
        "Hairer & Wanner, Solving ODEs II, 2nd ed. (1996), §IV.3",
    ),
)
def solve_ode(
    fun: Callable[[float, np.ndarray], ArrayLike],
    t_span: tuple[float, float],
    y0: ArrayLike,
    method: Literal["rk45", "rk4", "implicit_euler"] = "rk45",
    t_eval: ArrayLike | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    h: float | None = None,
    max_step: float = math.inf,
    jac: Callable[[float, np.ndarray], ArrayLike] | ArrayLike | None = None,
) -> ODEResult:
    r"""Solve the initial value problem :math:`y' = f(t, y)`, :math:`y(t_0) = y_0`.

    Parameters
    ----------
    fun : callable ``fun(t, y) -> array of shape (d,)``
        Right-hand side; ``y`` is a float64 array of shape ``(d,)``.
    t_span : (t0, t1)
        Finite interval; ``t1 < t0`` integrates backwards.
    y0 : float or array_like, shape (d,)
        Finite initial state.
    method : {"rk45", "rk4", "implicit_euler"}, default="rk45"
        Adaptive Dormand-Prince 5(4), classical fixed-step RK4, or fixed-step
        backward Euler for stiff problems.
    t_eval : array_like, optional
        Output times inside ``t_span``, sorted in the direction of integration.
        ``rk45`` uses its 4th-order dense output, ``rk4`` cubic Hermite and
        ``implicit_euler`` linear interpolation. Default: every step.
    rtol, atol : float, default=1e-6, 1e-9
        Error tolerances (``rtol >= 0``, ``atol > 0``). ``rk45`` controls the local
        error with them; ``implicit_euler`` uses them to stop Newton; ``rk4`` ignores them.
    h : float, optional
        Step size (> 0). Required by the fixed-step methods; for ``rk45`` it is the
        first step (default: chosen automatically).
    max_step : float, default=inf
        Largest allowed step (> 0). The fixed-step methods require ``h <= max_step``.
    jac : callable ``jac(t, y) -> (d, d)`` or array_like, optional
        Jacobian :math:`\partial f/\partial y` for ``implicit_euler`` (a constant
        matrix may be passed directly). Default: forward differences. Ignored by the
        explicit methods.

    Returns
    -------
    ODEResult
        ``t``, ``y`` (shape ``(d, len(t))``), ``nfev``, ``njev``, ``success``,
        ``message``. On failure (non-finite values, step size underflow, Newton
        failure) the solution up to the last good step is returned with
        ``success=False``.

    Raises
    ------
    ValueError, TypeError
        For invalid arguments or a ``fun`` returning the wrong shape.

    Complexity
    ----------
    ``rk45``/``rk4``: time O(T d) plus 6 (resp. 4) evaluations of ``fun`` per step,
    for T accepted + rejected steps. ``implicit_euler``: O(T k (d^3 + d cost(f))) for
    k Newton iterations per step with a finite-difference Jacobian (O(T k (d^3 +
    cost(f) + cost(jac))) with ``jac``). Space: O(T d) for the stored trajectory, or
    O(m d) with ``t_eval`` of length m, plus O(d^2) working memory for Newton.

    References
    ----------
    Dormand & Prince (1980); Shampine (1986); Hairer, Nørsett & Wanner, *Solving
    ODEs I* (1993), §II.4; Hairer & Wanner, *Solving ODEs II* (1996), §IV.3.

    Examples
    --------
    >>> import numpy as np
    >>> res = solve_ode(lambda t, y: -y, (0.0, 1.0), [1.0], t_eval=[0.5, 1.0])
    >>> res.success, bool(np.allclose(res.y[0], np.exp([-0.5, -1.0]), rtol=1e-6))
    (True, True)
    >>> stiff = solve_ode(lambda t, y: -1000 * (y - np.cos(t)), (0, 1), [0.0],
    ...                   method="implicit_euler", h=0.1)
    >>> stiff.success, bool(abs(stiff.y[0, -1] - np.cos(1.0)) < 1e-3)
    (True, True)
    """
    check_callable(fun, "fun")
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    if not isinstance(t_span, (tuple, list, np.ndarray)) or len(t_span) != 2:
        raise ValueError("t_span must be a pair (t0, t1)")
    t0 = as_finite_float(t_span[0], "t_span[0]")
    t1 = as_finite_float(t_span[1], "t_span[1]")
    y = as_real(y0, "y0")
    if y.ndim > 1:
        raise ValueError(f"y0 must be a scalar or 1-D, got shape {y.shape}")
    y = y.reshape(-1).copy()
    if y.size == 0:
        raise ValueError("y0 must be non-empty")
    rtol = as_finite_float(rtol, "rtol")
    atol = as_finite_float(atol, "atol")
    if rtol < 0 or atol <= 0:
        raise ValueError(f"need rtol >= 0 and atol > 0, got rtol={rtol}, atol={atol}")
    if isinstance(max_step, bool) or not isinstance(max_step, (int, float, np.number)):
        raise TypeError("max_step must be a number")
    max_step = float(max_step)
    if not max_step > 0:
        raise ValueError(f"max_step must be positive, got {max_step}")
    if h is not None:
        h = as_finite_float(h, "h")
        if h <= 0:
            raise ValueError(f"h must be positive, got {h}")
    if method != "rk45":
        if h is None:
            raise ValueError(f"method={method!r} is fixed-step and needs h")
        if h > max_step:
            raise ValueError(f"h={h} exceeds max_step={max_step}")
    if jac is not None and not callable(jac):
        J = as_real(jac, "jac")
        if J.shape != (y.size, y.size):
            raise ValueError(f"jac must have shape ({y.size}, {y.size}), got {J.shape}")
        jac = J
    te = _check_t_eval(t_eval, t0, t1)
    problem = _Problem(fun, jac if method == "implicit_euler" else None, y.size)

    with np.errstate(over="ignore", invalid="ignore"):
        if t0 == t1:
            ts = np.array([t0]) if te is None else te.copy()
            ys = np.repeat(y[:, None], ts.size, axis=1)
            success, message = True, "empty interval"
        elif method == "rk45":
            ts, ys, success, message = _rk45(problem, t0, t1, y, h, te, rtol, atol, max_step)
        else:
            step = float(h)  # type: ignore[arg-type]  # checked above for fixed-step methods
            ts, ys, success, message = _fixed_step(problem, method, t0, t1, y, step, te, rtol, atol)
    return ODEResult(ts, ys, problem.nfev, problem.njev, success, message)
