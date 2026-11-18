r"""GARCH-family conditional volatility models fitted by maximum likelihood.

Model
-----
Returns :math:`r_t` (``(T,)``, oldest first) are :math:`r_t = \mu + \varepsilon_t`,
:math:`\varepsilon_t = \sigma_t z_t`, :math:`z_t \sim N(0, 1)` i.i.d., with
``mean="zero"`` (:math:`\mu = 0`) or ``mean="constant"`` (:math:`\mu` estimated).

* **GARCH(p, q)** (Bollerslev 1986), following the ``arch`` package's naming —
  ``p`` ARCH lags of :math:`\varepsilon^2`, ``q`` GARCH lags of :math:`\sigma^2`:

  .. math::  \sigma_t^2 = \omega + \sum_{i=1}^p \alpha_i\varepsilon_{t-i}^2
             + \sum_{j=1}^q \beta_j\sigma_{t-j}^2 .

* **GJR-GARCH(1, 1)** (Glosten, Jagannathan & Runkle 1993):
  :math:`\sigma_t^2 = \omega + (\alpha + \gamma\,\mathbf 1\{\varepsilon_{t-1}<0\})
  \varepsilon_{t-1}^2 + \beta\sigma_{t-1}^2`.

**Pre-sample values** (as in ``arch``): every :math:`\varepsilon^2_{t}` and
:math:`\sigma^2_t` with :math:`t < 0` equals the backcast
:math:`\bar\sigma^2 = \sum_{i<\tau} w_i e_i^2`, :math:`w_i\propto 0.94^i`,
:math:`\tau = \min(75, T)`, computed once from :math:`e = r` (zero mean) or
:math:`e = r - \bar r` (constant mean), so it does not move with :math:`\mu`; the
asymmetric pre-sample term
:math:`\mathbf 1\{\varepsilon<0\}\varepsilon^2` equals :math:`\bar\sigma^2/2`.

**Log-likelihood** :math:`\ell = -\tfrac12\sum_t\bigl(\ln 2\pi + \ln\sigma_t^2 +
\varepsilon_t^2/\sigma_t^2\bigr)`.

Constraints by reparametrisation
--------------------------------
The optimiser (:func:`lipipy.numerics.minimize`, BFGS with central-difference
gradients) works on unconstrained :math:`\theta`. :math:`\omega = e^{\theta_0}`, the
persistence :math:`s = \operatorname{sigmoid}(\theta_1) \in (0, 1)` and the shares
:math:`\pi = \operatorname{softmax}(\theta_2, \dots, 0)` give
:math:`(\alpha_1,\dots,\alpha_p,\beta_1,\dots,\beta_q) = s\,\pi` for GARCH and
:math:`(\alpha, \gamma/2, \beta) = s\,\pi` for GJR. So :math:`\omega > 0`, all
coefficients are positive and covariance stationarity
:math:`\sum\alpha + \sum\beta < 1` (GJR: :math:`\alpha + \gamma/2 + \beta < 1`) holds
for every :math:`\theta`. Estimates on the boundary (a coefficient that should be 0)
are approached but never reached.

Standard errors are the square roots of the diagonal of the inverse of the
numerical (central-difference) Hessian of :math:`-\ell` in the natural parameters
:math:`(\mu, \omega, \alpha, \gamma, \beta)` at the optimum (non-robust, "inverse
information" errors); NaN when that Hessian is not positive definite.

Forecasts
---------
With :math:`\mathbb E_T[\varepsilon_{T+k}^2] = \sigma_{T+k}^2` for :math:`k\ge1` (and
:math:`\mathbb E_T[\mathbf 1\{\varepsilon<0\}\varepsilon^2] = \sigma^2/2`), the
variance forecasts follow the recursion with future shocks replaced by their
expectations; they converge to :math:`\omega/(1 - \sum\alpha - \sum\beta)`.

References
----------
T. Bollerslev, "Generalized autoregressive conditional heteroskedasticity",
*J. Econometrics* 31 (1986) 307-327. L. R. Glosten, R. Jagannathan & D. E. Runkle,
*J. Finance* 48 (1993) 1779-1801. K. Sheppard, *arch* documentation (backcasting).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from lipipy.core.complexity import Complexity
from lipipy.core.errors import ConvergenceWarning
from lipipy.core.registry import register
from lipipy.finance._common import DictMixin, check_int, check_series, is_negligible
from lipipy.numerics.optimize import minimize

__all__ = ["GarchResult", "garch", "gjr_garch"]

_LOG2PI = math.log(2.0 * math.pi)
_REFS = (
    "T. Bollerslev, J. Econometrics 31 (1986) 307-327",
    "K. Sheppard, arch: ARCH models in Python (backcasting convention)",
)
_GJR_REF = "L. R. Glosten, R. Jagannathan & D. E. Runkle, J. Finance 48 (1993) 1779-1801"


# ---------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GarchResult(DictMixin):
    """A fitted GARCH or GJR-GARCH model.

    Attributes
    ----------
    model : str
        ``"GARCH(p,q)"`` or ``"GJR-GARCH(1,1)"``.
    params : dict[str, float]
        Estimates: ``mu`` (0 for ``mean="zero"``), ``omega``, ``alpha[i]``,
        ``gamma[1]`` (GJR only), ``beta[j]``.
    std_errors : dict[str, float]
        Inverse-Hessian standard errors (``mu`` only when estimated).
    loglik : float
        Maximised log-likelihood.
    conditional_volatility : ndarray of shape (T,)
        :math:`\\sigma_t` at the estimates.
    residuals : ndarray of shape (T,)
        :math:`\\varepsilon_t = r_t - \\mu`.
    converged : bool
        Whether the optimiser met its gradient tolerance.
    """

    model: str
    params: dict[str, float]
    std_errors: dict[str, float]
    loglik: float
    conditional_volatility: np.ndarray
    residuals: np.ndarray
    converged: bool
    _p: int = field(repr=False, default=1)
    _q: int = field(repr=False, default=1)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.pop("_p")
        d.pop("_q")
        return d

    @property
    def persistence(self) -> float:
        """:math:`\\sum\\alpha + \\sum\\beta` (+ :math:`\\gamma/2` for GJR)."""
        pr = self.params
        return float(
            sum(v for k, v in pr.items() if k.startswith(("alpha", "beta")))
            + pr.get("gamma[1]", 0.0) / 2.0
        )

    def forecast(self, h: int) -> np.ndarray:
        """Conditional variance forecasts :math:`\\sigma^2_{T+1}, \\dots, \\sigma^2_{T+h}`.

        Examples
        --------
        >>> import numpy as np
        >>> r = np.random.default_rng(0).normal(0, 0.01, 500)
        >>> res = garch(r)
        >>> res.forecast(3).shape
        (3,)
        """
        h = check_int(h, "h", 1)
        p, q = self._p, self._q
        pr = self.params
        omega = pr["omega"]
        alpha = [pr[f"alpha[{i + 1}]"] for i in range(p)]
        beta = [pr[f"beta[{j + 1}]"] for j in range(q)]
        gamma = pr.get("gamma[1]", 0.0)
        eps2 = list((self.residuals**2)[::-1][:p])  # most recent first
        neg = list(((self.residuals < 0) * self.residuals**2)[::-1][:p])
        sig2 = list((self.conditional_volatility**2)[::-1][:q])
        out = np.empty(h)
        for k in range(h):
            s = omega
            s += sum(a * e for a, e in zip(alpha, eps2, strict=False))
            s += gamma * (neg[0] if neg else 0.0)
            s += sum(b * v for b, v in zip(beta, sig2, strict=False))
            out[k] = s
            eps2 = [s, *eps2][:p]  # E[eps^2] = sigma^2 in the future
            neg = [s / 2.0, *neg][:p]
            sig2 = [s, *sig2][:q]
        return out


# ---------------------------------------------------------------------------
# likelihood
# ---------------------------------------------------------------------------
def _backcast(eps: np.ndarray) -> float:
    tau = min(75, eps.shape[0])
    w = 0.94 ** np.arange(tau)
    return float(np.sum(eps[:tau] ** 2 * (w / w.sum())))


def _variance_path(
    eps: np.ndarray,
    omega: float,
    alpha: list[float],
    beta: list[float],
    gamma: float,
    backcast: float,
) -> np.ndarray:
    """sigma_t^2 for t = 0..T-1 (plain Python loop over floats: fast and exact)."""
    p, q = len(alpha), len(beta)
    e2 = (eps * eps).tolist()
    neg = [x * x if x < 0 else 0.0 for x in eps.tolist()]
    s2: list[float] = []
    for t in range(len(e2)):
        v = omega
        for i in range(p):
            v += alpha[i] * (e2[t - 1 - i] if t - 1 - i >= 0 else backcast)
        if gamma:
            v += gamma * (neg[t - 1] if t >= 1 else 0.5 * backcast)
        for j in range(q):
            v += beta[j] * (s2[t - 1 - j] if t - 1 - j >= 0 else backcast)
        s2.append(v)
    return np.asarray(s2)


def _loglik(eps: np.ndarray, s2: np.ndarray) -> float:
    if not np.all(s2 > 0.0):
        return -np.inf
    return float(-0.5 * np.sum(_LOG2PI + np.log(s2) + eps * eps / s2))


class _Model:
    """Maps unconstrained theta <-> natural parameters for one specification."""

    def __init__(self, r: np.ndarray, p: int, q: int, gjr: bool, const_mean: bool) -> None:
        self.r, self.p, self.q, self.gjr, self.const = r, p, q, gjr, const_mean
        self.k = p + q + (1 if gjr else 0)  # number of positive dynamic coefficients

    def names(self) -> list[str]:
        n = ["mu"] if self.const else []
        n += ["omega"] + [f"alpha[{i + 1}]" for i in range(self.p)]
        n += ["gamma[1]"] if self.gjr else []
        return n + [f"beta[{j + 1}]" for j in range(self.q)]

    def natural(self, theta: np.ndarray) -> np.ndarray:
        """theta -> (mu?, omega, alpha..., gamma?, beta...)."""
        i = 0
        out = []
        if self.const:
            out.append(theta[0])
            i = 1
        omega = math.exp(min(theta[i], 700.0))
        s = 1.0 / (1.0 + math.exp(-max(min(theta[i + 1], 700.0), -700.0)))
        logits = np.append(theta[i + 2 :], 0.0)
        z = np.exp(logits - logits.max())
        shares = s * z / z.sum()
        coefs = list(shares)
        if self.gjr:  # shares are (alpha, gamma / 2, beta)
            coefs[1] *= 2.0
        return np.array(out + [omega] + coefs)

    def split(self, nat: np.ndarray) -> tuple[float, float, list[float], float, list[float]]:
        i = 1 if self.const else 0
        mu = float(nat[0]) if self.const else 0.0
        omega = float(nat[i])
        rest = [float(x) for x in nat[i + 1 :]]
        alpha = rest[: self.p]
        gamma = rest[self.p] if self.gjr else 0.0
        beta = rest[self.p + (1 if self.gjr else 0) :]
        return mu, omega, alpha, gamma, beta

    def negll_natural(self, nat: np.ndarray, backcast: float) -> float:
        mu, omega, alpha, gamma, beta = self.split(nat)
        eps = self.r - mu
        s2 = _variance_path(eps, omega, alpha, beta, gamma, backcast)
        return -_loglik(eps, s2)

    def theta0(self, backcast: float) -> np.ndarray:
        var = float(np.var(self.r))
        out = [float(np.mean(self.r))] if self.const else []
        persistence = 0.95
        out += [math.log(max(var * (1.0 - persistence), 1e-300)), math.log(persistence / 0.05)]
        # shares: alpha ~ 0.05/0.95 of persistence in total, beta the rest
        k = self.k
        weights = np.full(k, 1.0)
        a_idx = list(range(self.p)) + ([self.p] if self.gjr else [])
        weights[a_idx] = 0.05 / len(a_idx)
        weights[[j for j in range(k) if j not in a_idx]] = 0.9 / max(self.q, 1)
        logits = np.log(weights / weights[-1])[:-1]
        return np.array(out + list(logits))


def _num_grad(f: Any, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    g = np.empty_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = h * max(1.0, abs(x[i]))
        g[i] = (f(x + e) - f(x - e)) / (2.0 * e[i])
    return g


def _num_hessian(f: Any, x: np.ndarray) -> np.ndarray:
    n = x.size
    H = np.empty((n, n))
    step = 1e-4 * np.maximum(np.abs(x), 1e-4)
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = step[i]
            ej[j] = step[j]
            val = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (
                4.0 * step[i] * step[j]
            )
            H[i, j] = H[j, i] = val
    return H


def _fit(returns: ArrayLike, p: int, q: int, gjr: bool, mean: str, dist: str) -> GarchResult:
    r = check_series(returns, min_rows=10)
    if r.ndim != 1:
        raise ValueError(f"returns must have shape (T,), got {r.shape}")
    p = check_int(p, "p", 1)
    q = check_int(q, "q", 0)
    if mean not in ("zero", "constant"):
        raise ValueError(f"mean must be 'zero' or 'constant', got {mean!r}")
    if dist != "normal":
        raise ValueError(f"only dist='normal' is supported, got {dist!r}")
    if is_negligible(np.std(r), np.max(np.abs(r))):
        raise ValueError("returns are constant; the conditional variance is not identifiable")

    model = _Model(r, p, q, gjr, mean == "constant")
    scale = float(np.std(r))
    r_scaled = r / scale  # fit on unit-variance data for a well-conditioned problem
    ms = _Model(r_scaled, p, q, gjr, mean == "constant")
    bc = _backcast(r_scaled - (r_scaled.mean() if ms.const else 0.0))

    n_obs = float(r.shape[0])

    def obj(theta: np.ndarray) -> float:  # mean negative log-likelihood, O(1) in size
        v = ms.negll_natural(ms.natural(theta), bc) / n_obs
        return v if np.isfinite(v) else 1e300

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        res = minimize(obj, ms.theta0(bc), lambda th: _num_grad(obj, th), method="bfgs", tol=1e-7)

    nat_s = ms.natural(res.x)
    # back to the original scale: mu * scale, omega * scale^2, coefficients unchanged
    nat = nat_s.copy()
    i = 1 if model.const else 0
    if model.const:
        nat[0] *= scale
    nat[i] *= scale * scale
    backcast = _backcast(r - r.mean() if model.const else r)
    mu, omega, alpha, gamma, beta = model.split(nat)
    eps = r - mu
    s2 = _variance_path(eps, omega, alpha, beta, gamma, backcast)
    ll = _loglik(eps, s2)

    # standard errors from the Hessian in natural parameters (scaled problem, then rescaled)
    H = _num_hessian(lambda x: ms.negll_natural(x, bc), nat_s)
    scales = np.ones(nat.size)
    if model.const:
        scales[0] = scale
    scales[i] = scale * scale
    with np.errstate(invalid="ignore"):
        try:
            L = np.linalg.cholesky(H)
            cov = np.linalg.inv(L).T @ np.linalg.inv(L)
            se = np.sqrt(np.diag(cov)) * scales
        except np.linalg.LinAlgError:
            se = np.full(nat.size, np.nan)
    names = model.names()
    params = {"mu": 0.0} | {n: float(v) for n, v in zip(names, nat, strict=True)}
    if not res.success:
        warnings.warn(f"GARCH optimisation did not converge: {res.message}", ConvergenceWarning, 3)
    label = "GJR-GARCH(1,1)" if gjr else f"GARCH({p},{q})"
    return GarchResult(
        model=label,
        params=params,
        std_errors={n: float(v) for n, v in zip(names, se, strict=True)},
        loglik=ll,
        conditional_volatility=np.sqrt(s2),
        residuals=eps,
        converged=bool(res.success),
        _p=p,
        _q=q,
    )


@register(
    domain="finance",
    tasks=("volatility-modeling", "time-series"),
    complexity=Complexity(
        time="O(T (p + q) k^2 I)",
        space="O(T + k^2)",
        notes="I BFGS iterations, k = p + q + 2 parameters (numerical gradients)",
    ),
    references=_REFS,
)
def garch(
    returns: ArrayLike, p: int = 1, q: int = 1, mean: str = "zero", dist: str = "normal"
) -> GarchResult:
    r"""Fit a GARCH(p, q) model with normal innovations by maximum likelihood.

    See the module docstring for the model, the backcast, the reparametrisation
    (stationarity is enforced) and the standard errors. The data are rescaled to
    unit variance for the optimisation and the estimates mapped back.

    Parameters
    ----------
    returns : array_like of shape (T, )
        Returns, oldest first, ``T >= 10``.
    p : int, default=1
        Number of ARCH lags (:math:`\varepsilon^2`), >= 1.
    q : int, default=1
        Number of GARCH lags (:math:`\sigma^2`), >= 0.
    mean : {"zero", "constant"}, default="zero"
        Mean model.
    dist : {"normal"}, default="normal"
        Innovation distribution.

    Returns
    -------
    GarchResult
        Parameters, standard errors, log-likelihood, conditional volatilities,
        residuals and a ``forecast(h)`` method.

    Complexity
    ----------
    Each likelihood evaluation is O(T (p + q)); the fit takes O(k) evaluations per
    gradient and I iterations, plus O(k^2) evaluations for the Hessian.

    References
    ----------
    Bollerslev (1986); ``arch`` documentation for the backcast.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> r, s2 = np.empty(3000), 1e-4
    >>> for t in range(3000):
    ...     r[t] = np.sqrt(s2) * rng.standard_normal()
    ...     s2 = 1e-6 + 0.08 * r[t] ** 2 + 0.9 * s2
    >>> res = garch(r)
    >>> res.model, res.persistence < 1
    ('GARCH(1,1)', True)
    """
    return _fit(returns, p, q, False, mean, dist)


@register(
    domain="finance",
    tasks=("volatility-modeling", "time-series"),
    complexity=Complexity(
        time="O(T k^2 I)", space="O(T + k^2)", notes="I BFGS iterations, k <= 5 parameters"
    ),
    references=(_GJR_REF, *_REFS),
)
def gjr_garch(returns: ArrayLike, mean: str = "zero", dist: str = "normal") -> GarchResult:
    r"""Fit a GJR-GARCH(1, 1) (asymmetric "leverage") model by maximum likelihood.

    :math:`\sigma_t^2 = \omega + (\alpha + \gamma\mathbf 1\{\varepsilon_{t-1}<0\})
    \varepsilon_{t-1}^2 + \beta\sigma_{t-1}^2` with :math:`\alpha, \gamma, \beta > 0`
    and :math:`\alpha + \gamma/2 + \beta < 1` (module docstring).

    Parameters
    ----------
    returns : array_like of shape (T, )
        Returns, oldest first, ``T >= 10``.
    mean : {"zero", "constant"}, default="zero"
        Mean model.
    dist : {"normal"}, default="normal"
        Innovation distribution.

    Returns
    -------
    GarchResult
        As :func:`garch`, with ``params["gamma[1]"]``.

    Complexity
    ----------
    O(T) per likelihood evaluation; see :func:`garch`.

    References
    ----------
    Glosten, Jagannathan & Runkle (1993).

    Examples
    --------
    >>> import numpy as np
    >>> r = np.random.default_rng(2).standard_t(6, 1500) * 0.01
    >>> sorted(gjr_garch(r).params)
    ['alpha[1]', 'beta[1]', 'gamma[1]', 'mu', 'omega']
    """
    return _fit(returns, 1, 1, True, mean, dist)
