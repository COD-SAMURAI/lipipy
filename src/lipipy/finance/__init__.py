"""Layer 2: quantitative finance and trading research.

For research and education only; nothing here is investment advice.

Design rules: metrics are pure functions on arrays of returns or prices; models
declare their assumptions; the backtest engine gives a strategy data only up to the
current time step, so look-ahead bias is impossible by construction; transaction
costs and slippage are never silently zero.

Catalog. ``[x]`` = implemented, ``[ ]`` = planned.

Returns and performance
-----------------------
[ ] Simple and log returns, cumulative returns, annualization
[ ] Volatility (historical, EWMA), Sharpe, Sortino, Calmar, information ratio
[ ] Drawdown, maximum drawdown, drawdown duration
[ ] Alpha / beta (CAPM regression), tracking error

Risk
----
[ ] Value at risk (historical, parametric, Monte Carlo), conditional VaR / expected shortfall
[ ] Covariance estimation: sample, EWMA, Ledoit-Wolf shrinkage
[ ] Factor models (Fama-French style regression), PCA risk factors
[ ] Stress testing and scenario analysis

Portfolio construction
----------------------
[ ] Mean-variance (Markowitz) optimization, efficient frontier
[ ] Minimum variance, maximum Sharpe (tangency), risk parity, hierarchical risk parity
[ ] Black-Litterman
[ ] Kelly criterion, volatility targeting, rebalancing rules

Derivatives pricing
-------------------
[ ] Black-Scholes-Merton prices and Greeks, implied volatility (Newton, Brent)
[ ] Binomial and trinomial trees (European and American options)
[ ] Monte Carlo pricing with antithetic and control variates; path-dependent options
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
[ ] Vectorized backtester (research speed, O(T A))
[ ] Event-driven backtester (path-dependent logic, order types)
[ ] Transaction cost and slippage models
[ ] Walk-forward validation, purged and embargoed cross-validation
[ ] Strategy templates: momentum, mean reversion, pairs trading, trend following
[ ] Execution: TWAP, VWAP, Almgren-Chriss optimal execution
[ ] Market microstructure: order book simulation, bid-ask spread estimators
"""
