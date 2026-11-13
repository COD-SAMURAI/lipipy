"""Layer 2: quantitative finance and trading research.

For research and education only; nothing here is investment advice.

Design rules: metrics are pure functions on arrays of returns or prices; models
declare their assumptions; the backtest engine gives a strategy data only up to the
current time step, so look-ahead bias is impossible by construction; transaction
costs and slippage are never silently zero.

Catalog. ``[x]`` = implemented, ``[ ]`` = planned.

Returns and performance
-----------------------
[x] Simple and log returns, cumulative returns, annualization
[x] Volatility (historical, EWMA covariance), Sharpe, Sortino, Calmar, information ratio
[x] Drawdown, maximum drawdown, drawdown duration
[x] Alpha / beta (CAPM regression), tracking error

Risk
----
[x] Value at risk and expected shortfall (historical, Gaussian, Cornish-Fisher)
[ ] Monte Carlo value at risk
[x] Covariance estimation: sample, EWMA, Ledoit-Wolf shrinkage
[ ] Factor models (Fama-French style regression), PCA risk factors
[ ] Stress testing and scenario analysis

Portfolio construction
----------------------
[x] Mean-variance (Markowitz) optimization, efficient frontier
[x] Minimum variance (also long-only), maximum Sharpe (tangency), risk parity
[ ] Hierarchical risk parity
[x] Black-Litterman
[ ] Kelly criterion, volatility targeting, rebalancing rules

Derivatives pricing
-------------------
[x] Black-Scholes-Merton prices and Greeks, implied volatility (Brent)
[x] Binomial (Cox-Ross-Rubinstein) trees, European and American options
[ ] Trinomial trees
[x] Monte Carlo pricing with antithetic and control variates (European, chunked)
[ ] Path-dependent options
[ ] Heston stochastic volatility, Merton jump diffusion
[ ] Finite-difference PDE pricing
[ ] Bond math: yield, duration, convexity, yield-curve bootstrapping

Time-series models for markets
------------------------------
[ ] ARIMA / GARCH for returns and volatility
[ ] Kalman filters for pairs trading / dynamic hedge ratios
[ ] Cointegration tests (Engle-Granger, Johansen)
[ ] Regime switching (hidden Markov models)

Trading and execution
---------------------
[x] Vectorized backtester (research speed, O(T A))
[ ] Event-driven backtester (path-dependent logic, order types)
[ ] Transaction cost and slippage models
[ ] Walk-forward validation, purged and embargoed cross-validation
[ ] Strategy templates: momentum, mean reversion, pairs trading, trend following
[ ] Execution: TWAP, VWAP, Almgren-Chriss optimal execution
[ ] Market microstructure: order book simulation, bid-ask spread estimators
"""

from lipipy.finance.backtest import BacktestResult, backtest
from lipipy.finance.options import (
    Greeks,
    MonteCarloResult,
    binomial_price,
    bs_greeks,
    bs_price,
    implied_volatility,
    monte_carlo_price,
)
from lipipy.finance.portfolio import (
    BlackLittermanResult,
    EfficientFrontier,
    black_litterman,
    efficient_frontier,
    max_sharpe_weights,
    min_variance_weights,
    risk_contributions,
    risk_parity_weights,
)
from lipipy.finance.returns import (
    CAPMResult,
    DrawdownResult,
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    capm_alpha_beta,
    cumulative_returns,
    drawdown_series,
    information_ratio,
    log_returns,
    max_drawdown,
    sharpe_ratio,
    simple_returns,
    sortino_ratio,
    tracking_error,
)
from lipipy.finance.risk import (
    LedoitWolfResult,
    ewma_covariance,
    expected_shortfall,
    ledoit_wolf,
    sample_covariance,
    value_at_risk,
)

__all__ = [
    "BacktestResult",
    "BlackLittermanResult",
    "CAPMResult",
    "DrawdownResult",
    "EfficientFrontier",
    "Greeks",
    "LedoitWolfResult",
    "MonteCarloResult",
    "annualized_return",
    "annualized_volatility",
    "backtest",
    "binomial_price",
    "black_litterman",
    "bs_greeks",
    "bs_price",
    "calmar_ratio",
    "capm_alpha_beta",
    "cumulative_returns",
    "drawdown_series",
    "efficient_frontier",
    "ewma_covariance",
    "expected_shortfall",
    "implied_volatility",
    "information_ratio",
    "ledoit_wolf",
    "log_returns",
    "max_drawdown",
    "max_sharpe_weights",
    "min_variance_weights",
    "monte_carlo_price",
    "risk_contributions",
    "risk_parity_weights",
    "sample_covariance",
    "sharpe_ratio",
    "simple_returns",
    "sortino_ratio",
    "tracking_error",
    "value_at_risk",
]
