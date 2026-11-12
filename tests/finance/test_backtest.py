"""Tests for lipipy.finance.backtest: causality, accounting identities, hand-computed paths."""

import numpy as np
import pytest

from lipipy.finance import backtest, max_drawdown, sharpe_ratio, simple_returns

pytestmark = pytest.mark.filterwarnings("error")

SEEDS = range(8)


def _prices(seed, T=120, A=3):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, size=(T, A)), axis=0))


# ---------------------------------------------------------------------------
# look-ahead is impossible by construction
# ---------------------------------------------------------------------------
def test_strategy_sees_only_the_past():
    P = _prices(0, T=30)
    seen = []

    def spy(history):
        seen.append(history.shape[0])
        np.testing.assert_array_equal(history, P[: history.shape[0]])
        return np.full(3, 1 / 3)

    backtest(P, spy)
    assert seen == list(range(1, 30))  # called at t = 0..T-2 with prices[: t + 1]


def test_cheating_strategy_cannot_reach_future_prices():
    P = _prices(1, T=20)

    def peek_tomorrow(history):
        t = history.shape[0] - 1
        tomorrow = history[t + 1]  # the future is not in the view
        return (tomorrow > history[t]).astype(float)

    with pytest.raises(IndexError):
        backtest(P, peek_tomorrow)


def test_strategy_cannot_modify_prices():
    P = _prices(2, T=10)

    def vandal(history):
        history[-1] = 1e9
        return np.ones(3) / 3

    with pytest.raises(ValueError, match="read-only"):
        backtest(P, vandal)


def _random_walks(seed=0, T=500, A=2):
    r = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(r.normal(0, 0.01, (T, A)), axis=0))


def test_no_future_prices_reachable_through_history_base():
    """Review regression: history.base used to be the full price matrix."""
    prices = _random_walks()
    leaks = []

    def cheater(history):
        t = len(history) - 1
        full = history.base
        if full is None or t + 1 >= len(full):
            return np.array([0.5, 0.5])
        leaks.append(np.isfinite(full[t + 1 :]).any())
        return np.eye(2)[int(np.argmax(full[t + 1] / full[t] - 1))]

    res = backtest(prices, cheater)
    assert leaks and not any(leaks)  # rows after t are all NaN
    # argmax over NaN picks asset 0: the cheater is just "hold asset 0"
    np.testing.assert_allclose(res.equity, backtest(prices, [1.0, 0.0]).equity, rtol=1e-12)
    assert res.total_return == pytest.approx(prices[-1, 0] / prices[0, 0] - 1, rel=1e-10)


def test_no_future_prices_through_strides_or_unlocking():
    prices = _random_walks(1, T=60)

    def strided(history):
        t, A = history.shape[0] - 1, history.shape[1]
        if t + 1 < len(prices):
            beyond = np.lib.stride_tricks.as_strided(history, shape=(t + 2, A))
            assert np.all(np.isnan(beyond[t + 1]))  # the next row exists only as NaN
        return np.full(A, 0.5)

    backtest(prices, strided)

    def unlocker(history):
        buf = history.base
        buf.flags.writeable = True  # the buffer can be unlocked ...
        assert np.all(np.isnan(buf[len(history) :]))  # ... but holds no future prices
        buf[: len(history)] = 1.0  # tampering only affects what the strategy sees
        return np.full(2, 0.5)

    tampered = backtest(prices, unlocker)
    honest = backtest(prices, [0.5, 0.5])
    np.testing.assert_array_equal(tampered.equity, honest.equity)


def test_weights_decided_at_t_earn_returns_from_t_to_t_plus_1():
    # a perfect-foresight weights array would be look-ahead; the engine applies row t
    # to (t, t+1], so an oracle must be shifted by the caller to profit.
    P = np.array([[100.0], [110.0], [99.0], [108.9]])
    R = simple_returns(P)[:, 0]  # +10%, -10%, +10%
    oracle = np.array([[1.0], [0.0], [1.0], [0.0]])  # long before each up move
    res = backtest(P, oracle)
    np.testing.assert_allclose(res.returns, [0.1, 0.0, 0.1], atol=1e-15)
    assert res.returns @ np.ones(3) == pytest.approx(R[0] + R[2])


# ---------------------------------------------------------------------------
# accounting identities
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_buy_and_hold_equals_asset_return(seed):
    P = _prices(seed)
    res = backtest(P[:, 1], [1.0], initial_capital=250.0)
    np.testing.assert_allclose(res.equity, 250.0 * P[:, 1] / P[0, 1], rtol=1e-12)
    np.testing.assert_allclose(res.returns, simple_returns(P[:, 1]), rtol=1e-11, atol=1e-15)
    assert res.turnover[0] == 1.0 and np.all(res.turnover[1:] == 0)
    # multi-asset buy and hold: rebalance only at t = 0
    w = np.array([0.5, 0.3, 0.2])
    hold = backtest(P, w, rebalance=len(P))
    np.testing.assert_allclose(hold.equity, (P / P[0]) @ w, rtol=1e-12)
    drift = w * (P[1:] / P[0]) / ((P[1:] / P[0]) @ w)[:, None]
    np.testing.assert_allclose(hold.weights[1:], drift[:-1], rtol=1e-12)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("rebalance", ["every", 5])
def test_costs_reduce_returns_by_exactly_turnover_times_bps(seed, rebalance):
    P = _prices(seed)
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(3), size=len(P))
    free = backtest(P, W, rebalance=rebalance)
    paid = backtest(P, W, rebalance=rebalance, cost_bps=7.0, slippage_bps=3.0)
    np.testing.assert_allclose(paid.costs, paid.turnover * 10e-4, rtol=1e-15, atol=0)
    # net return = gross return of the held weights - cost fraction, every period
    gross = np.sum(paid.weights * simple_returns(P), axis=1)
    np.testing.assert_allclose(paid.returns, gross - paid.costs, rtol=0, atol=1e-14)
    assert paid.equity[-1] < free.equity[-1]
    np.testing.assert_allclose(
        free.returns, np.sum(free.weights * simple_returns(P), axis=1), rtol=0, atol=1e-14
    )


def test_constant_weight_rebalancing_hand_computed():
    P = np.array([[100.0, 100.0], [120.0, 90.0], [108.0, 108.0], [108.0, 81.0]])
    # asset returns: (+20%, -10%), (-10%, +20%), (0%, -25%); 60/40 rebalanced every period
    res = backtest(P, [0.6, 0.4], cost_bps=10.0)
    r_gross = np.array([0.6 * 0.2 + 0.4 * -0.1, 0.6 * -0.1 + 0.4 * 0.2, 0.4 * -0.25])
    # turnover: 1 to buy in; then drift back to 60/40
    e1 = 1.0 + r_gross[0] - 0.001
    drift1 = np.array([0.6 * 1.2, 0.4 * 0.9]) / e1
    t1 = np.abs(np.array([0.6, 0.4]) - drift1).sum()
    e2 = e1 * (1.0 + r_gross[1] - t1 * 0.001)
    drift2 = np.array([0.6 * 0.9, 0.4 * 1.2]) * e1 / e2
    t2 = np.abs(np.array([0.6, 0.4]) - drift2).sum()
    e3 = e2 * (1.0 + r_gross[2] - t2 * 0.001)
    np.testing.assert_allclose(res.turnover, [1.0, t1, t2], rtol=1e-13)
    np.testing.assert_allclose(res.equity, [1.0, e1, e2, e3], rtol=1e-13)
    zero_cost = backtest(P, [0.6, 0.4])
    np.testing.assert_allclose(zero_cost.equity, np.cumprod(np.r_[1.0, 1 + r_gross]), rtol=1e-14)
    np.testing.assert_allclose(zero_cost.returns, r_gross, atol=1e-15)


def test_cash_and_leverage():
    P = np.array([[100.0], [110.0], [121.0]])
    half = backtest(P, [0.5])  # 50% cash earning zero, rebalanced every period
    np.testing.assert_allclose(half.returns, [0.05, 0.05], rtol=1e-13)
    held = backtest(P, [0.5], rebalance=10)  # buy and hold: the stock weight drifts up
    np.testing.assert_allclose(held.returns, [0.05, 0.1 * 0.55 / 1.05], rtol=1e-13)
    lev = backtest(P, [2.0], rebalance="every")
    np.testing.assert_allclose(lev.returns, [0.2, 0.2], rtol=1e-14)
    short = backtest(P, [-1.0])
    np.testing.assert_allclose(short.returns, [-0.1, -0.1], rtol=1e-14)


def test_ruin_floors_equity_at_zero():
    P = np.array([[100.0], [140.0], [70.0], [140.0], [150.0]])
    res = backtest(P, [2.0])  # -50% with 2x leverage wipes the account out
    assert res.ruined
    np.testing.assert_allclose(res.equity, [1.0, 1.8, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(res.returns, [0.8, -1.0, 0.0, 0.0])
    assert res.max_drawdown == -1.0 and res.total_return == -1.0
    assert res.annualized_return == -1.0
    deep = backtest(P, [4.0], rebalance=10)  # -3 + 4 * 0.7 < 0: negative mid-block
    assert deep.ruined and deep.equity[2] == 0.0 and np.all(deep.weights[2:] == 0.0)


@pytest.mark.parametrize("seed", SEEDS)
def test_metrics_and_periodic_rebalance(seed):
    P = _prices(seed)
    calls = []

    def equal_weight(history):
        calls.append(history.shape[0] - 1)
        return np.full(3, 1 / 3)

    res = backtest(P, equal_weight, rebalance=10, cost_bps=5.0, periods_per_year=52)
    assert calls == list(range(0, len(P) - 1, 10))
    assert np.all(res.turnover[[t for t in range(len(P) - 1) if t % 10]] == 0.0)
    assert res.sharpe_ratio == pytest.approx(sharpe_ratio(res.returns, 0.0, 52), rel=1e-14)
    assert res.max_drawdown == max_drawdown(res.returns).max_drawdown
    assert res.total_return == pytest.approx(res.equity[-1] - 1.0, rel=1e-14)
    np.testing.assert_allclose(res.weights.sum(axis=1)[[0, 10, 20]], 1.0, rtol=1e-14)
    d = res.to_dict()
    assert isinstance(d["equity"], list) and isinstance(d["ruined"], bool)


def test_single_period_and_validation():
    res = backtest([[100.0, 50.0], [101.0, 49.0]], [0.5, 0.5])
    assert res.returns.shape == (1,) and np.isnan(res.sharpe_ratio)
    assert np.isnan(res.annualized_volatility)
    P = _prices(0, T=10)
    with pytest.raises(ValueError, match="shape"):
        backtest(P, [0.5, 0.5])
    with pytest.raises(ValueError, match="shape"):
        backtest(P, lambda h: np.ones(2))
    with pytest.raises(ValueError, match="NaN"):
        backtest(P, lambda h: np.array([np.nan, 0.5, 0.5]))
    with pytest.raises(ValueError, match="NaN"):
        backtest(np.where(np.arange(10)[:, None] == 3, np.nan, P), [1 / 3] * 3)
    with pytest.raises(ValueError, match="positive"):
        backtest(-P, [1 / 3] * 3)
    with pytest.raises(ValueError, match="rebalance"):
        backtest(P, [1 / 3] * 3, rebalance="weekly")
    with pytest.raises(ValueError, match="rebalance"):
        backtest(P, [1 / 3] * 3, rebalance=0)
    with pytest.raises(ValueError, match="non-negative"):
        backtest(P, [1 / 3] * 3, cost_bps=-5.0)
    with pytest.raises(ValueError, match="initial_capital"):
        backtest(P, [1 / 3] * 3, initial_capital=0.0)
    with pytest.raises(ValueError, match="at least 2"):
        backtest(P[:1], [1 / 3] * 3)
