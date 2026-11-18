r"""Time-series tools for markets: autocorrelation, AR models, unit roots, cointegration.

Conventions: series are ``(T,)`` arrays, oldest first; NaN raises ``ValueError``.

Autocorrelation
---------------
Sample ACF with the biased (divide-by-:math:`T`) autocovariance, which keeps the
autocovariance sequence positive semi-definite:
:math:`\hat\rho_k = \sum_{t=k}^{T-1}(x_t-\bar x)(x_{t-k}-\bar x) / \sum_t (x_t-\bar x)^2`.
The PACF is obtained from the ACF by the **Durbin-Levinson** recursion
(:math:`\phi_{kk}` is the partial autocorrelation at lag :math:`k`):

.. math::
    \phi_{kk} = \frac{\rho_k - \sum_{j<k}\phi_{k-1,j}\rho_{k-j}}
                     {1 - \sum_{j<k}\phi_{k-1,j}\rho_j},\qquad
    \phi_{kj} = \phi_{k-1,j} - \phi_{kk}\phi_{k-1,k-j}.

**Ljung-Box**: :math:`Q = T(T+2)\sum_{k=1}^h \hat\rho_k^2/(T-k)`, compared with
:math:`\chi^2_{h - \text{model\_df}}`; the survival function is the regularised upper
incomplete gamma function :math:`Q(\nu/2, Q/2)` (series / continued fraction,
Numerical Recipes §6.2).

AR(p) fitting
-------------
* **Yule-Walker**: solve the Toeplitz system :math:`\Gamma\phi = \gamma` of biased
  autocovariances (Durbin-Levinson), :math:`\sigma^2 = \gamma_0 - \phi^\top\gamma`,
  intercept :math:`\bar x(1 - \sum\phi)`.
* **OLS**: regress :math:`x_t` on :math:`(1, x_{t-1}, \dots, x_{t-p})` for
  :math:`t = p..T-1` (Householder QR); :math:`\sigma^2 = \text{SSR}/(T-p)`.

Augmented Dickey-Fuller
-----------------------
:math:`\Delta x_t = d_t + \gamma x_{t-1} + \sum_{i=1}^{L}\delta_i\Delta x_{t-i} + e_t`
with deterministic terms :math:`d_t` = none (``"n"``), constant (``"c"``) or constant
and linear trend (``"ct"``); the statistic is the OLS t-ratio of :math:`\gamma`
(:math:`H_0`: unit root). With ``autolag="aic"`` the lag order :math:`L \le` ``maxlag``
minimises :math:`n\ln(\text{SSR}/n) + 2k` over regressions on the common sample, and
the chosen model is re-estimated on all available observations; the default
``maxlag`` is Schwert's :math:`\lceil 12 (T/100)^{1/4}\rceil` (capped at
:math:`T/2 - k_d - 1`). This follows ``statsmodels.tsa.stattools.adfuller``.

* Critical values: MacKinnon (2010) response surfaces
  :math:`c(n) = \beta_\infty + \beta_1/n + \beta_2/n^2 + \beta_3/n^3` (n = regression
  observations).
* p-values: MacKinnon (1994) approximation :math:`p = \Phi(\sum_j a_j\tau^j)`, with
  separate small- and large-:math:`\tau` polynomials (0 / 1 outside the fitted range).

**Engle-Granger**: regress :math:`y` on :math:`(x, d_t)` by OLS and apply the ADF test
(``regression="n"``) to the residuals; critical values and p-values come from the
MacKinnon tables for :math:`N = 1 + k` integrated series (:math:`k` regressors,
:math:`k \le 2` supported) and ``n = T - 1``, as in ``statsmodels.tsa.stattools.coint``.

References
----------
G. E. P. Box, G. M. Jenkins, G. C. Reinsel & G. M. Ljung, *Time Series Analysis*,
5th ed. (2015), ch. 3. J. Durbin, *Rev. Int. Statist. Inst.* 28 (1960) 233-244. G. M.
Ljung & G. E. P. Box, *Biometrika* 65 (1978) 297-303. D. A. Dickey & W. A. Fuller,
*J. Amer. Statist. Assoc.* 74 (1979) 427-431. S. E. Said & D. A. Dickey, *Biometrika*
71 (1984) 599-607. J. G. MacKinnon, *J. Business & Economic Statistics* 12 (1994)
167-176; "Critical values for cointegration tests", Queen's Economics Dept. WP 1227
(2010). R. F. Engle & C. W. J. Granger, *Econometrica* 55 (1987) 251-276.
W. H. Press et al., *Numerical Recipes*, 3rd ed. (2007), §6.2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from lipipy.core.complexity import Complexity
from lipipy.core.registry import register
from lipipy.finance._common import DictMixin, check_int, check_series, is_negligible
from lipipy.numerics.linalg import qr, solve_triangular
from lipipy.numerics.special import norm_cdf

__all__ = [
    "ADFResult",
    "ARResult",
    "CointegrationResult",
    "LjungBoxResult",
    "acf",
    "adf",
    "ar_fit",
    "engle_granger",
    "ljung_box",
    "pacf",
]

_BJ = "Box, Jenkins, Reinsel & Ljung, Time Series Analysis, 5th ed. (2015)"
_MACKINNON = (
    "J. G. MacKinnon, J. Business & Economic Statistics 12 (1994) 167-176",
    "J. G. MacKinnon, Critical values for cointegration tests, Queen's WP 1227 (2010)",
)

# MacKinnon (2010), Table 2: rows N = 1..3, each 1% / 5% / 10% with
# (beta_inf, beta_1, beta_2, beta_3). "n" is only tabulated for N = 1.
_CRIT: dict[str, list[list[tuple[float, float, float, float]]]] = {
    "n": [
        [(-2.56574, -2.2358, -3.627, 0.0), (-1.941, -0.2686, -3.365, 31.223),
         (-1.61682, 0.2656, -2.714, 25.364)],
    ],
    "c": [
        [(-3.43035, -6.5393, -16.786, -79.433), (-2.86154, -2.8903, -4.234, -40.04),
         (-2.56677, -1.5384, -2.809, 0.0)],
        [(-3.89644, -10.9519, -33.527, 0.0), (-3.33613, -6.1101, -6.823, 0.0),
         (-3.04445, -4.2412, -2.72, 0.0)],
        [(-4.29374, -14.4354, -33.195, 47.433), (-3.74066, -8.5632, -10.852, 27.982),
         (-3.45218, -6.2143, -3.718, 0.0)],
    ],
    "ct": [
        [(-3.95877, -9.0531, -28.428, -134.155), (-3.41049, -4.3904, -9.036, -45.374),
         (-3.12705, -2.5856, -3.925, -22.38)],
        [(-4.32762, -15.4387, -35.679, 0.0), (-3.78057, -9.5106, -12.074, 0.0),
         (-3.49631, -7.0815, -7.538, 21.892)],
        [(-4.66305, -18.7688, -49.793, 104.244), (-4.1189, -11.8922, -19.031, 77.332),
         (-3.83511, -9.0723, -8.504, 35.403)],
    ],
}  # fmt: skip

# MacKinnon (1994), Table 3 / p-value polynomials, N = 1..3 (coefficients of tau^0..).
_TAU_MAX = {"n": (math.inf, 1.51, 0.86), "c": (2.74, 0.92, 0.55), "ct": (0.7, 0.63, 0.71)}
_TAU_MIN = {"n": (-19.04, -19.62, -21.21), "c": (-18.83, -18.86, -23.48),
            "ct": (-16.18, -21.15, -25.37)}  # fmt: skip
_TAU_STAR = {"n": (-1.04, -1.53, -2.68), "c": (-1.61, -2.62, -3.13), "ct": (-2.89, -3.19, -3.5)}
_TAU_SMALL = {
    "n": ((0.6344, 1.2378, 0.032496), (1.9129, 1.3857, 0.035322), (2.7648, 1.4502, 0.034186)),
    "c": ((2.1659, 1.4412, 0.038269), (2.92, 1.5012, 0.039796), (3.4699, 1.4856, 0.03164)),
    "ct": ((3.2512, 1.6047, 0.049588), (3.6646, 1.5419, 0.036448), (4.0983, 1.5173, 0.029898)),
}
_TAU_LARGE = {
    "n": ((0.4797, 0.93557, -0.06999, 0.033066), (1.5578, 0.8558, -0.2083, -0.033549),
          (2.2268, 0.68093, -0.32362, -0.054448)),
    "c": ((1.7339, 0.93202, -0.12745, -0.010368), (2.1945, 0.64695, -0.29198, -0.042377),
          (2.5893, 0.45168, -0.36529, -0.050074)),
    "ct": ((2.5261, 0.61654, -0.37956, -0.060285), (2.85, 0.5272, -0.36622, -0.051695),
           (3.221, 0.5255, -0.32685, -0.041501)),
}  # fmt: skip
_REGRESSIONS = ("n", "c", "ct")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _series(x: ArrayLike, name: str = "x", min_rows: int = 3) -> np.ndarray:
    a = check_series(x, name, min_rows=min_rows)
    if a.ndim != 1:
        raise ValueError(f"{name} must have shape (T,), got {a.shape}")
    return a


def _check_nonconstant(x: np.ndarray, name: str = "x") -> None:
    if is_negligible(np.std(x), np.max(np.abs(x))):
        raise ValueError(f"{name} is constant; autocorrelations are undefined")


def _autocov(x: np.ndarray, nlags: int) -> np.ndarray:
    xc = x - x.mean()
    T = x.size
    return np.array([xc[k:] @ xc[: T - k] for k in range(nlags + 1)]) / T


def _durbin_levinson(rho: np.ndarray, nlags: int) -> tuple[np.ndarray, np.ndarray]:
    """PACF phi_kk for k = 1..nlags and the order-nlags AR coefficients from rho_0..rho_n."""
    pac = np.empty(nlags)
    phi = np.zeros(0)
    for k in range(1, nlags + 1):
        num = rho[k] - phi @ rho[k - 1 : 0 : -1] if k > 1 else rho[1]
        den = 1.0 - phi @ rho[1:k] if k > 1 else 1.0
        kk = num / den
        phi = np.append(phi - kk * phi[::-1], kk)
        pac[k - 1] = kk
    return pac, phi


def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Coefficients, SSR and diag((X^T X)^{-1}) by Householder QR."""
    Q, R = qr(X)
    beta = solve_triangular(R, Q.T @ y, lower=False)
    resid = y - X @ beta
    Rinv = solve_triangular(R, np.eye(R.shape[0]), lower=False)
    return beta, float(resid @ resid), np.sum(Rinv**2, axis=1)


def _gammq(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a, x) (Numerical Recipes 3e, §6.2)."""
    if x <= 0.0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1.0:  # series for P(a, x)
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(10_000):
            ap += 1.0
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-16:
                break
        return max(0.0, 1.0 - s * math.exp(-x + a * math.log(x) - gln))
    tiny = 1e-300  # modified Lentz continued fraction for Q(a, x)
    b = x + 1.0 - a
    c, d = 1.0 / tiny, 1.0 / b
    h = d
    for i in range(1, 10_000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def _chi2_sf(x: float, df: int) -> float:
    return _gammq(0.5 * df, 0.5 * x)


def _mackinnon_crit(regression: str, N: int, nobs: float) -> dict[str, float]:
    out = {}
    for label, (b0, b1, b2, b3) in zip(("1%", "5%", "10%"), _CRIT[regression][N - 1], strict=True):
        out[label] = b0 + b1 / nobs + b2 / nobs**2 + b3 / nobs**3
    return out


def _mackinnon_p(stat: float, regression: str, N: int) -> float:
    if stat > _TAU_MAX[regression][N - 1]:
        return 1.0
    if stat < _TAU_MIN[regression][N - 1]:
        return 0.0
    coefs = (_TAU_SMALL if stat <= _TAU_STAR[regression][N - 1] else _TAU_LARGE)[regression]
    poly = sum(c * stat**j for j, c in enumerate(coefs[N - 1]))
    return float(norm_cdf(poly))


# ---------------------------------------------------------------------------
# ACF / PACF / Ljung-Box
# ---------------------------------------------------------------------------
@register(
    domain="finance",
    tasks=("time-series",),
    complexity=Complexity(time="O(T K)", space="O(T + K)", notes="K = nlags"),
    references=(_BJ,),
)
def acf(x: ArrayLike, nlags: int = 20) -> np.ndarray:
    r"""Sample autocorrelation function :math:`\hat\rho_0 = 1, \dots, \hat\rho_K`.

    Biased (divide-by-:math:`T`) autocovariances, as ``statsmodels`` ``acf`` with
    ``adjusted=False``.

    Parameters
    ----------
    x : array_like of shape (T, )
        Series, not constant.
    nlags : int, default=20
        Largest lag :math:`K` (< T).

    Returns
    -------
    ndarray of shape (K + 1, )

    Complexity
    ----------
    Time O(T K), extra space O(T + K).

    References
    ----------
    Box, Jenkins, Reinsel & Ljung, *Time Series Analysis*, §2.1.

    Examples
    --------
    >>> acf([1.0, 2.0, 3.0, 4.0], nlags=2)
    array([ 1.  ,  0.25, -0.3 ])
    """
    a = _series(x)
    nlags = check_int(nlags, "nlags", 0)
    if nlags >= a.size:
        raise ValueError(f"nlags must be < T = {a.size}, got {nlags}")
    _check_nonconstant(a)
    g = _autocov(a, nlags)
    return g / g[0]


@register(
    domain="finance",
    tasks=("time-series",),
    complexity=Complexity(time="O(T K + K^2)", space="O(T + K)", notes="K = nlags"),
    references=(_BJ, "J. Durbin, Rev. Int. Statist. Inst. 28 (1960) 233-244"),
)
def pacf(x: ArrayLike, nlags: int = 20) -> np.ndarray:
    r"""Partial autocorrelation function by the Durbin-Levinson recursion.

    Returns :math:`(1, \phi_{11}, \dots, \phi_{KK})` from the biased ACF, as
    ``statsmodels`` ``pacf(method="ldb")``.

    Parameters
    ----------
    x : array_like of shape (T, )
        Series, not constant.
    nlags : int, default=20
        Largest lag :math:`K` (< T).

    Returns
    -------
    ndarray of shape (K + 1, )

    Complexity
    ----------
    Time O(T K + K^2), extra space O(T + K).

    References
    ----------
    J. Durbin (1960); Box et al., *Time Series Analysis*, §3.2.

    Examples
    --------
    >>> pacf([1.0, 2.0, 3.0, 4.0], nlags=2).round(6)
    array([ 1.      ,  0.25    , -0.386667])
    """
    rho = acf(x, nlags)
    if nlags == 0:
        return rho
    pac, _ = _durbin_levinson(rho, nlags)
    return np.concatenate([[1.0], pac])


@dataclass(frozen=True)
class LjungBoxResult(DictMixin):
    """Ljung-Box portmanteau test: ``statistic`` Q, ``pvalue`` and degrees of freedom."""

    statistic: float
    pvalue: float
    df: int


@register(
    domain="finance",
    tasks=("time-series", "hypothesis-test"),
    complexity=Complexity(time="O(T h)", space="O(T + h)"),
    references=("G. M. Ljung & G. E. P. Box, Biometrika 65 (1978) 297-303",),
)
def ljung_box(x: ArrayLike, lags: int = 10, model_df: int = 0) -> LjungBoxResult:
    r"""Ljung-Box test of no autocorrelation up to lag ``h``.

    :math:`Q = T(T+2)\sum_{k=1}^h \hat\rho_k^2/(T-k) \sim \chi^2_{h - \text{model\_df}}`
    under :math:`H_0`.

    Parameters
    ----------
    x : array_like of shape (T, )
        Series (e.g. model residuals).
    lags : int, default=10
        Number of lags :math:`h` (1 <= h < T).
    model_df : int, default=0
        Degrees of freedom used by a fitted model (e.g. p + q for ARMA residuals).

    Returns
    -------
    LjungBoxResult

    Complexity
    ----------
    Time O(T h), extra space O(T + h).

    References
    ----------
    Ljung & Box (1978).

    Examples
    --------
    >>> res = ljung_box([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0], lags=1)
    >>> round(res.statistic, 6), res.df
    (8.75, 1)
    """
    a = _series(x)
    h = check_int(lags, "lags", 1)
    model_df = check_int(model_df, "model_df", 0)
    if h >= a.size:
        raise ValueError(f"lags must be < T = {a.size}, got {h}")
    if model_df >= h:
        raise ValueError(f"model_df ({model_df}) must be smaller than lags ({h})")
    rho = acf(a, h)[1:]
    T = a.size
    q = float(T * (T + 2) * np.sum(rho**2 / (T - np.arange(1, h + 1))))
    df = h - model_df
    return LjungBoxResult(statistic=q, pvalue=_chi2_sf(q, df), df=df)


# ---------------------------------------------------------------------------
# AR(p)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ARResult(DictMixin):
    """Fitted AR(p) :math:`x_t = c + \\sum_i \\phi_i x_{t-i} + e_t`, noise variance ``sigma2``."""

    coefficients: np.ndarray
    intercept: float
    sigma2: float
    method: str


@register(
    domain="finance",
    tasks=("time-series", "regression"),
    complexity=Complexity(
        time="O(T p^2)", space="O(T p)", notes="yule_walker: O(T p + p^2) time, O(T) space"
    ),
    references=(_BJ, "J. Durbin, Rev. Int. Statist. Inst. 28 (1960) 233-244"),
)
def ar_fit(x: ArrayLike, p: int = 1, method: str = "yule_walker") -> ARResult:
    r"""Fit an AR(p) model by Yule-Walker or ordinary least squares.

    See the module docstring. Yule-Walker matches ``statsmodels``
    ``yule_walker(method="mle")``; OLS matches ``AutoReg(x, p, trend="c")``.

    Parameters
    ----------
    x : array_like of shape (T, )
        Series, ``T > 2 p``.
    p : int, default=1
        Order (>= 1).
    method : {"yule_walker", "ols"}, default="yule_walker"
        Estimator.

    Returns
    -------
    ARResult
        ``coefficients`` :math:`(\phi_1..\phi_p)`, ``intercept``, ``sigma2``, ``method``.

    Complexity
    ----------
    OLS: time O(T p^2), space O(T p). Yule-Walker: time O(T p + p^2), space O(T).

    References
    ----------
    Box et al., *Time Series Analysis*, §3.2 and §7.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = np.zeros(20_000)
    >>> for t in range(1, x.size):
    ...     x[t] = 0.6 * x[t - 1] + rng.standard_normal()
    >>> round(float(ar_fit(x, 1).coefficients[0]), 2)
    0.6
    """
    a = _series(x)
    p = check_int(p, "p", 1)
    if method not in ("yule_walker", "ols"):
        raise ValueError(f"method must be 'yule_walker' or 'ols', got {method!r}")
    if a.size <= 2 * p:
        raise ValueError(f"need T > 2 p observations, got T = {a.size}, p = {p}")
    _check_nonconstant(a)
    if method == "yule_walker":
        g = _autocov(a, p)
        _, phi = _durbin_levinson(g / g[0], p)
        sigma2 = float(g[0] - phi @ g[1:])
        return ARResult(phi, float(a.mean() * (1.0 - phi.sum())), sigma2, method)
    T = a.size
    X = np.column_stack([np.ones(T - p)] + [a[p - i : T - i] for i in range(1, p + 1)])
    beta, ssr, _ = _ols(a[p:], X)
    return ARResult(beta[1:], float(beta[0]), ssr / (T - p), method)


# ---------------------------------------------------------------------------
# ADF and Engle-Granger
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ADFResult(DictMixin):
    """Augmented Dickey-Fuller test.

    Attributes
    ----------
    statistic : float
        t-ratio of :math:`\\gamma`.
    pvalue : float
        MacKinnon (1994) approximate p-value.
    used_lag : int
        Number of lagged differences :math:`L`.
    nobs : int
        Observations in the final regression.
    critical_values : dict[str, float]
        MacKinnon (2010) 1%, 5% and 10% values for ``nobs``.
    regression : str
        Deterministic terms (``"n"``, ``"c"``, ``"ct"``).
    """

    statistic: float
    pvalue: float
    used_lag: int
    nobs: int
    critical_values: dict[str, float]
    regression: str


def _trend_cols(n: int, regression: str, start: int) -> list[np.ndarray]:
    cols = []
    if regression in ("c", "ct"):
        cols.append(np.ones(n))
    if regression == "ct":
        cols.append(np.arange(start + 1, start + n + 1, dtype=float))
    return cols


def _adf_design(x: np.ndarray, lag: int, maxlag: int, regression: str) -> tuple[np.ndarray, ...]:
    """Regressand and design on the sample that has ``maxlag`` lags available."""
    dx = np.diff(x)
    n = dx.size - maxlag
    y = dx[maxlag:]
    cols = [x[maxlag : maxlag + n]]  # level x_{t-1}
    cols += [dx[maxlag - i : maxlag - i + n] for i in range(1, lag + 1)]
    cols += _trend_cols(n, regression, maxlag)
    return y, np.column_stack(cols)


def _adf(x: np.ndarray, maxlag: int | None, regression: str, autolag: str | None) -> ADFResult:
    if regression not in _REGRESSIONS:
        raise ValueError(f"regression must be one of {_REGRESSIONS}, got {regression!r}")
    if autolag not in (None, "aic"):
        raise ValueError(f"autolag must be None or 'aic', got {autolag!r}")
    T = x.size
    ntrend = {"n": 0, "c": 1, "ct": 2}[regression]
    cap = T // 2 - ntrend - 1
    if maxlag is None:
        maxlag = min(int(math.ceil(12.0 * (T / 100.0) ** 0.25)), cap)
        if maxlag < 0:
            raise ValueError("series too short for this regression")
    else:
        maxlag = check_int(maxlag, "maxlag", 0)
        if maxlag > cap:
            raise ValueError(f"maxlag must be <= T/2 - 1 - (deterministic terms) = {cap}")
    lag = maxlag
    if autolag == "aic":
        best = math.inf
        for L in range(maxlag + 1):
            y, X = _adf_design(x, L, maxlag, regression)
            _, ssr, _ = _ols(y, X)
            n = y.size
            ic = n * math.log(ssr / n) + 2 * X.shape[1] if ssr > 0 else -math.inf
            if ic < best:
                best, lag = ic, L
    y, X = _adf_design(x, lag, lag, regression)
    beta, ssr, diag = _ols(y, X)
    n, k = X.shape
    if ssr <= 0.0:
        raise ValueError("the ADF regression fits exactly; the statistic is undefined")
    se = math.sqrt(ssr / (n - k) * diag[0])
    stat = float(beta[0] / se)
    return ADFResult(
        statistic=stat,
        pvalue=_mackinnon_p(stat, regression, 1),
        used_lag=lag,
        nobs=n,
        critical_values=_mackinnon_crit(regression, 1, n),
        regression=regression,
    )


@register(
    domain="finance",
    tasks=("time-series", "hypothesis-test", "unit-root"),
    complexity=Complexity(
        time="O(T L^3)", space="O(T L)", notes="L = maxlag (one QR per candidate lag)"
    ),
    references=(
        "D. A. Dickey & W. A. Fuller, J. Amer. Statist. Assoc. 74 (1979) 427-431",
        "S. E. Said & D. A. Dickey, Biometrika 71 (1984) 599-607",
        *_MACKINNON,
    ),
)
def adf(
    x: ArrayLike,
    maxlag: int | None = None,
    regression: str = "c",
    autolag: str | None = "aic",
) -> ADFResult:
    r"""Augmented Dickey-Fuller unit-root test.

    See the module docstring for the regression, the lag selection and the
    MacKinnon critical values and p-values (approximations from response surfaces;
    they match ``statsmodels`` ``adfuller``).

    Parameters
    ----------
    x : array_like of shape (T, )
        Series, oldest first.
    maxlag : int, optional
        Largest lag order; default Schwert's rule.
    regression : {"c", "ct", "n"}, default="c"
        Deterministic terms: constant, constant and trend, or none.
    autolag : {"aic", None}, default="aic"
        Choose the lag by AIC, or use ``maxlag`` itself.

    Returns
    -------
    ADFResult

    Complexity
    ----------
    Time O(T L^3) with autolag (one regression per candidate lag), space O(T L).

    References
    ----------
    Dickey & Fuller (1979); Said & Dickey (1984); MacKinnon (1994, 2010).

    Examples
    --------
    >>> import numpy as np
    >>> walk = np.cumsum(np.random.default_rng(0).standard_normal(500))
    >>> adf(walk).pvalue > 0.05
    True
    """
    a = _series(x, min_rows=6)
    _check_nonconstant(a)
    return _adf(a, maxlag, regression, autolag)


@dataclass(frozen=True)
class CointegrationResult(DictMixin):
    """Engle-Granger cointegration test.

    Attributes
    ----------
    statistic : float
        ADF statistic of the cointegrating-regression residuals.
    pvalue : float
        MacKinnon (1994) p-value for N = 1 + k series.
    critical_values : dict[str, float]
        MacKinnon (2010) values for N = 1 + k, n = T - 1.
    coefficients : ndarray
        Cointegrating regression coefficients on ``x`` (hedge ratios).
    intercept : float
        Constant of the cointegrating regression (0 if ``trend="n"``).
    residuals : ndarray of shape (T,)
        The spread :math:`y - \\hat y`.
    used_lag : int
        Lag order of the residual ADF regression.
    """

    statistic: float
    pvalue: float
    critical_values: dict[str, float]
    coefficients: np.ndarray
    intercept: float
    residuals: np.ndarray
    used_lag: int


@register(
    domain="finance",
    tasks=("time-series", "hypothesis-test", "cointegration", "pairs-trading"),
    complexity=Complexity(time="O(T (k^2 + L^3))", space="O(T (k + L))", notes="L = maxlag"),
    references=("R. F. Engle & C. W. J. Granger, Econometrica 55 (1987) 251-276", *_MACKINNON),
)
def engle_granger(
    y: ArrayLike,
    x: ArrayLike,
    trend: str = "c",
    maxlag: int | None = None,
    autolag: str | None = "aic",
) -> CointegrationResult:
    r"""Engle-Granger two-step cointegration test of ``y`` on ``x``.

    :math:`H_0`: no cointegration (the residuals have a unit root). See the module
    docstring; results match ``statsmodels.tsa.stattools.coint``.

    Parameters
    ----------
    y : array_like of shape (T, )
        Dependent series.
    x : array_like of shape (T, ) or (T, k)
        Regressor series, k <= 2.
    trend : {"c", "ct"}, default="c"
        Deterministic terms in the cointegrating regression.
    maxlag, autolag
        As in :func:`adf`, for the residual test.

    Returns
    -------
    CointegrationResult

    Raises
    ------
    ValueError
        If ``x`` explains ``y`` exactly (zero residuals).

    Complexity
    ----------
    Time O(T (k^2 + L^3)), extra space O(T (k + L)).

    References
    ----------
    Engle & Granger (1987); MacKinnon (1994, 2010).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(3)
    >>> x = np.cumsum(rng.standard_normal(1000))
    >>> y = 2.0 * x + rng.standard_normal(1000)
    >>> res = engle_granger(y, x)
    >>> res.pvalue < 0.01, round(float(res.coefficients[0]), 2)
    (True, 2.0)
    """
    yy = _series(y, "y", min_rows=10)
    xx = check_series(x, "x", min_rows=10)
    if xx.ndim == 1:
        xx = xx[:, None]
    if xx.shape[0] != yy.size:
        raise ValueError(f"x has {xx.shape[0]} rows but y has {yy.size}")
    k = xx.shape[1]
    if k > 2:
        raise ValueError("engle_granger supports at most 2 regressors (MacKinnon N <= 3)")
    if trend not in ("c", "ct"):
        raise ValueError(f"trend must be 'c' or 'ct', got {trend!r}")
    T = yy.size
    X = np.column_stack([xx, *_trend_cols(T, trend, 0)])
    beta, ssr, _ = _ols(yy, X)
    resid = yy - X @ beta
    if is_negligible(math.sqrt(ssr / T), float(np.max(np.abs(yy)))):
        raise ValueError("x explains y exactly; the residual test is undefined")
    res = _adf(resid, maxlag, "n", autolag)
    return CointegrationResult(
        statistic=res.statistic,
        pvalue=_mackinnon_p(res.statistic, trend, k + 1),
        critical_values=_mackinnon_crit(trend, k + 1, T - 1),
        coefficients=beta[:k],
        intercept=float(beta[k]),
        residuals=resid,
        used_lag=res.used_lag,
    )
