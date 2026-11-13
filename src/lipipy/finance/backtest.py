r"""Vectorized, causal portfolio backtester.

Timeline and look-ahead
-----------------------
Prices have shape ``(T, A)``; time ``t = 0, ..., T-1``. At each **decision time**
:math:`t \le T-2` that is a rebalance date, the target weights :math:`w_t` are decided
from information up to and including :math:`t`, and are held over :math:`(t, t+1]`,
earning the asset returns :math:`R_{t+1} = P_{t+1}/P_t - 1`.

* With a **strategy function**, ``weights_fn(history)`` is called with
  ``history`` of shape ``(t + 1, A)`` holding ``prices[: t + 1]``, a read-only view
  of a **separate buffer** that is filled one row at a time: when the strategy is
  called at :math:`t`, rows after :math:`t` of that buffer are NaN. So no memory
  reachable from ``history`` (``history.base``, strided views past its end, the
  buffer after flipping its read-only flag) contains a price after :math:`t`;
  ``history[t + 1]`` raises ``IndexError`` and writing raises ``ValueError``. The
  accounting uses a private copy of the prices, so a strategy that tampers with its
  buffer changes only what it sees itself, never the results. (Python cannot stop a
  strategy from reading data it captured elsewhere, e.g. the caller's own price
  array through a closure; the guarantee covers what the backtester hands over.)
* With a **weights array** of shape ``(T, A)``, row :math:`t` is the target decided
  at :math:`t` (applied to :math:`R_{t+1}`; the last row is unused); shape ``(A,)``
  means constant target weights. A precomputed array is **trusted input**: lipipy
  cannot check how it was built, so the no-look-ahead guarantee applies to the
  callable API only.

Portfolio accounting
--------------------
Weights are fractions of current equity :math:`E_t`; :math:`1 - \mathbf 1^\top w` is
held in cash earning zero (so weights may not sum to one; shorting and leverage are
allowed). Between rebalance dates positions are held (buy and hold): they drift with
prices, positions value :math:`E_t w_i P_{s,i}/P_{t,i}` at :math:`s > t`.

At a rebalance date :math:`t` with drifted pre-trade weights :math:`w^-_t` (all cash,
:math:`w^- = 0`, at :math:`t = 0`):

.. math::
    \text{turnover}_t = \sum_i |w_{t,i} - w^-_{t,i}|,\qquad
    c_t = \text{turnover}_t\cdot\frac{\text{cost\_bps} + \text{slippage\_bps}}{10^4},

and the cost :math:`c_t E_t` is paid from cash at :math:`t`. The net return over
:math:`(t, t+1]` is therefore exactly the gross return minus the cost fraction:

.. math::  r_{t+1} = w_t^\top R_{t+1} - c_t,\qquad E_{t+1} = E_t(1 + r_{t+1}).

Equity is floored at zero: if it reaches zero (or would turn negative with
leverage), the strategy is ruined, stops trading, and later returns are zero.

Within a block between rebalances the equity path is computed with array operations
(:math:`E_s = E_t(\text{cash} + \sum_i w_i P_{s,i}/P_{t,i})`), so the Python loop
runs over rebalance dates only.

References
----------
M. López de Prado, *Advances in Financial Machine Learning* (2018), ch. 11-12 (on
backtest pitfalls, including look-ahead bias). E. P. Chan, *Quantitative Trading*,
2nd ed. (2021), ch. 3.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from lipipy.core.complexity import Complexity
from lipipy.core.registry import register
from lipipy.finance._common import (
    DictMixin,
    as_float_array,
    check_finite,
    check_int,
    check_positive_number,
    check_prices,
    check_real,
)
from lipipy.finance.returns import (
    annualized_return,
    annualized_volatility,
    max_drawdown,
    sharpe_ratio,
)

__all__ = ["BacktestResult", "backtest"]

WeightsFn = Callable[[np.ndarray], ArrayLike]


@dataclass(frozen=True)
class BacktestResult(DictMixin):
    """Outcome of :func:`backtest`.

    Attributes
    ----------
    equity : ndarray of shape (T,)
        Portfolio value at each price time; ``equity[0] = initial_capital``.
    returns : ndarray of shape (T - 1,)
        Net return over :math:`(t, t+1]`.
    weights : ndarray of shape (T - 1, A)
        Weights held over :math:`(t, t+1]` right after the decision at :math:`t`
        (the target on rebalance dates, the drifted weights otherwise).
    turnover : ndarray of shape (T - 1,)
        One-way turnover :math:`\\sum_i |w_{t,i} - w^-_{t,i}|` at each decision time
        (0 when not rebalancing).
    costs : ndarray of shape (T - 1,)
        Cost fraction :math:`c_t` of equity paid at :math:`t`.
    total_return : float
        ``equity[-1] / equity[0] - 1``.
    annualized_return : float
        Geometric annualised return of ``returns``.
    annualized_volatility : float
        NaN when fewer than 2 returns.
    sharpe_ratio : float
        With zero risk-free rate; NaN when fewer than 2 returns.
    max_drawdown : float
        Maximum drawdown of the equity curve (<= 0).
    ruined : bool
        Whether equity hit zero.
    """

    equity: np.ndarray
    returns: np.ndarray
    weights: np.ndarray
    turnover: np.ndarray
    costs: np.ndarray
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    ruined: bool


def _rebalance_dates(rebalance: Any, T: int) -> np.ndarray:
    if isinstance(rebalance, str):
        if rebalance != "every":
            raise ValueError(f"rebalance must be 'every' or a positive int, got {rebalance!r}")
        step = 1
    else:
        step = check_int(rebalance, "rebalance", 1)
    return np.arange(0, T - 1, step)


def _target(
    weights: Any, history: np.ndarray | None, t: int, A: int, static: np.ndarray | None
) -> np.ndarray:
    if static is not None:
        return static[t] if static.ndim == 2 else static
    w = as_float_array(weights(history), "weights_fn output")
    if w.shape != (A,):
        raise ValueError(f"weights_fn must return shape ({A},), got {w.shape} at t={t}")
    check_finite(w, f"weights_fn output at t={t}")
    return w


@register(
    domain="finance",
    tasks=("backtesting",),
    complexity=Complexity(
        time="O(T A)",
        space="O(T A)",
        notes="plus the cost of the strategy calls; Python loop over rebalance dates only",
    ),
    references=(
        "M. Lopez de Prado, Advances in Financial Machine Learning (2018), ch. 11-12",
        "E. P. Chan, Quantitative Trading, 2nd ed. (2021), ch. 3",
    ),
)
def backtest(
    prices: ArrayLike,
    weights: WeightsFn | ArrayLike,
    rebalance: str | int = "every",
    cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    initial_capital: float = 1.0,
    periods_per_year: float = 252,
) -> BacktestResult:
    r"""Backtest a portfolio strategy without look-ahead bias.

    See the module docstring for the timeline, the accounting and the cost model.

    Parameters
    ----------
    prices : array_like of shape (T, A) or (T, )
        Strictly positive prices, oldest first, ``T >= 2``.
    weights : callable or array_like
        ``weights_fn(history) -> (A,)`` called at each rebalance date with a
        read-only ``(t + 1, A)`` view of the prices up to ``t`` whose underlying
        buffer holds no later prices (module docstring); or target weights of shape
        ``(T, A)`` (row ``t`` decided at ``t``) or ``(A,)`` (constant). A weights
        array is trusted input: the no-look-ahead guarantee covers the callable.
    rebalance : "every" or int, default="every"
        Rebalance at every step, or every ``k`` steps (``t = 0, k, 2k, ...``).
    cost_bps : float, default=0.0
        Proportional transaction cost per unit of turnover, in basis points.
    slippage_bps : float, default=0.0
        Slippage per unit of turnover, in basis points (added to ``cost_bps``).
    initial_capital : float, default=1.0
        Starting equity (> 0).
    periods_per_year : float, default=252
        For the annualised metrics.

    Returns
    -------
    BacktestResult
        Equity curve, returns, weights, turnover, costs and summary metrics.

    Complexity
    ----------
    Time O(T A) plus the strategy calls; space O(T A).

    References
    ----------
    López de Prado (2018), ch. 11-12; Chan (2021), ch. 3.

    Examples
    --------
    >>> prices = [[100.0, 50.0], [110.0, 50.0], [99.0, 55.0]]
    >>> res = backtest(prices, [0.5, 0.5])
    >>> res.returns.round(12)  # 0.5 * 10% ; 0.5 * (-10%) + 0.5 * 10%
    array([0.05, 0.  ])
    >>> res = backtest(prices, [0.5, 0.5], cost_bps=10.0)
    >>> float(res.turnover[0]), round(float(res.costs[0]), 12)  # buy from cash: 10 bp
    (1.0, 0.001)
    >>> round(float(res.returns[0]), 12)  # 5% gross minus the 0.1% cost
    0.049
    """
    P = check_prices(prices)
    if P.ndim == 1:
        P = P[:, None]
    P = P.copy()  # private: the strategy never gets a view of this array
    T, A = P.shape
    dates = _rebalance_dates(rebalance, T)
    cost = (check_real(cost_bps, "cost_bps") + check_real(slippage_bps, "slippage_bps")) / 1e4
    if cost < 0.0:
        raise ValueError("cost_bps + slippage_bps must be non-negative")
    capital = check_positive_number(initial_capital, "initial_capital")
    check_positive_number(periods_per_year, "periods_per_year")

    static: np.ndarray | None = None
    if not callable(weights):
        static = as_float_array(weights, "weights")
        if static.shape not in ((A,), (T, A)):
            raise ValueError(f"weights must have shape ({A},) or ({T}, {A}), got {static.shape}")
        check_finite(static, "weights")

    equity = np.empty(T)
    equity[0] = capital
    held = np.zeros((T - 1, A))
    turnover = np.zeros(T - 1)
    costs = np.zeros(T - 1)
    w_pre = np.zeros(A)  # start in cash
    # What the strategy can see: a buffer revealed one row at a time (NaN = future).
    seen = np.full((T, A), np.nan) if static is None else None
    revealed = 0
    ruined = False
    bounds = np.append(dates, T - 1)
    for start, end in zip(bounds[:-1], bounds[1:], strict=True):
        t = int(start)
        history = None
        if seen is not None:
            seen.flags.writeable = True
            seen[revealed : t + 1] = P[revealed : t + 1]
            seen.flags.writeable = False
            revealed = t + 1
            history = seen[: t + 1]
        w = _target(weights, history, t, A, static)
        turnover[t] = float(np.abs(w - w_pre).sum())
        costs[t] = turnover[t] * cost
        # Equity over (t, end], relative to E_t: cash (after costs) plus drifting positions.
        growth = P[t + 1 : end + 1] / P[t]  # (end - t, A)
        rel = (1.0 - w.sum() - costs[t]) + growth @ w
        held[t] = w
        dead = np.flatnonzero(rel <= 0.0)
        if dead.size:  # ruin: equity floored at zero, no more trading
            stop = t + 1 + int(dead[0])
            equity[t + 1 : stop] = equity[t] * rel[: stop - t - 1]
            equity[stop:] = 0.0
            held[t + 1 : stop] = w * growth[: stop - t - 1] / rel[: stop - t - 1, None]
            ruined = True
            break
        equity[t + 1 : end + 1] = equity[t] * rel
        held[t + 1 : end] = w * growth[:-1] / rel[:-1, None]  # drifted weights
        w_pre = w * growth[-1] / rel[-1]

    prev = equity[:-1]
    rets = np.zeros(T - 1)
    alive = prev > 0.0
    rets[alive] = equity[1:][alive] / prev[alive] - 1.0
    enough = T - 1 >= 2
    return BacktestResult(
        equity=equity,
        returns=rets,
        weights=held,
        turnover=turnover,
        costs=costs,
        total_return=float(equity[-1] / equity[0] - 1.0),
        annualized_return=float(annualized_return(rets, periods_per_year)),
        annualized_volatility=float(annualized_volatility(rets, periods_per_year))
        if enough
        else float("nan"),
        sharpe_ratio=float(sharpe_ratio(rets, 0.0, periods_per_year)) if enough else float("nan"),
        max_drawdown=max_drawdown(rets).max_drawdown,
        ruined=ruined,
    )
