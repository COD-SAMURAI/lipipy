r"""Linear Gaussian state-space models: Kalman filter, RTS smoother, dynamic hedge ratios.

Model
-----
State :math:`x_t \in \mathbb R^n`, observation :math:`z_t \in \mathbb R^m`:

.. math::
    x_t = F x_{t-1} + w_t,\ w_t\sim N(0, Q),\qquad
    z_t = H_t x_t + v_t,\ v_t\sim N(0, R),\qquad x_0 \sim N(m_0, P_0).

``initial_mean`` / ``initial_cov`` are the prior of the **first** state :math:`x_0`
(before :math:`z_0` is seen): the filter updates with :math:`z_0` first, then for
:math:`t \ge 1` predicts and updates.

Filter
------
Predict :math:`\hat x_{t|t-1} = F\hat x_{t-1|t-1}`,
:math:`P_{t|t-1} = FP_{t-1|t-1}F^\top + Q`. Update with the innovation
:math:`v_t = z_t - H_t\hat x_{t|t-1}`, :math:`S_t = H_tP_{t|t-1}H_t^\top + R`,
:math:`K_t = P_{t|t-1}H_t^\top S_t^{-1}` (Cholesky solve),
:math:`\hat x_{t|t} = \hat x_{t|t-1} + K_tv_t`, and the **Joseph form**
:math:`P_{t|t} = (I - K_tH_t)P_{t|t-1}(I-K_tH_t)^\top + K_tRK_t^\top`, which stays
symmetric positive semi-definite under rounding. The log-likelihood is
:math:`\sum_t \log N(v_t; 0, S_t)` (prediction-error decomposition). Observations
that are entirely NaN are missing: the update is skipped (the step is a pure
prediction) and they add nothing to the likelihood.

**Causality.** The filter is a loop over time that reads observation :math:`t` only
at step :math:`t` and calls no user code, so filtered and predicted quantities at
:math:`t` are functions of :math:`z_0, \dots, z_t` only (tested by perturbing the
future). The smoother is *not* causal: it uses all observations by design.

Smoother (Rauch-Tung-Striebel)
------------------------------
Backwards from :math:`t = T-1`: :math:`C_t = P_{t|t}F^\top P_{t+1|t}^{-1}`,
:math:`\hat x_{t|T} = \hat x_{t|t} + C_t(\hat x_{t+1|T} - \hat x_{t+1|t})`,
:math:`P_{t|T} = P_{t|t} + C_t(P_{t+1|T} - P_{t+1|t})C_t^\top`.

Dynamic hedge ratio
-------------------
Pairs trading regression :math:`y_t = \beta_t x_t + \alpha_t + v_t` with
:math:`(\beta_t, \alpha_t)` a random walk, :math:`Q = \frac{\delta}{1-\delta} I`,
:math:`R` = ``obs_var``, :math:`H_t = (x_t, 1)` (Chan, *Algorithmic Trading*, §3).
The tradeable spread at :math:`t` is the innovation
:math:`y_t - H_t\hat x_{t|t-1}`, built from the estimate **before** :math:`y_t` is
seen, with standard deviation :math:`\sqrt{S_t}`.

References
----------
R. E. Kalman, *J. Basic Engineering* 82 (1960) 35-45. H. E. Rauch, F. Tung &
C. T. Striebel, *AIAA J.* 3 (1965) 1445-1450. B. D. O. Anderson & J. B. Moore,
*Optimal Filtering* (1979), ch. 3-4 and 7. J. Durbin & S. J. Koopman, *Time Series
Analysis by State Space Methods*, 2nd ed. (2012), ch. 4. E. P. Chan, *Algorithmic
Trading* (2013), ch. 3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from lipipy.core.complexity import Complexity
from lipipy.core.registry import register
from lipipy.finance._common import DictMixin, as_float_array, check_finite, check_series
from lipipy.numerics.linalg import LinAlgError, cholesky, cholesky_solve

__all__ = [
    "FilterResult",
    "HedgeRatioResult",
    "KalmanFilter",
    "SmootherResult",
    "dynamic_hedge_ratio",
]

_REFS = (
    "R. E. Kalman, J. Basic Engineering 82 (1960) 35-45",
    "B. D. O. Anderson & J. B. Moore, Optimal Filtering (1979)",
)
_LOG2PI = math.log(2.0 * math.pi)


def _matrix(a: ArrayLike, shape: tuple[int, ...], name: str) -> np.ndarray:
    m = as_float_array(a, name)
    if m.ndim == 0 and shape == (1, 1):
        m = m.reshape(1, 1)
    if m.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {m.shape}")
    check_finite(m, name)
    return m


def _check_cov(m: np.ndarray, name: str, strict: bool) -> np.ndarray:
    scale = float(np.max(np.abs(m))) if m.size else 0.0
    if np.any(np.abs(m - m.T) > 1e-10 * max(scale, 1e-300)):
        raise ValueError(f"{name} must be symmetric")
    m = (m + m.T) / 2.0
    eig = np.linalg.eigvalsh(m)
    if eig[0] < -1e-12 * max(scale, 1e-300) or (strict and eig[0] <= 0.0):
        kind = "positive definite" if strict else "positive semi-definite"
        raise ValueError(f"{name} must be {kind}")
    return m


@dataclass(frozen=True)
class FilterResult(DictMixin):
    """Output of :meth:`KalmanFilter.filter` (``T`` steps, state size ``n``).

    Attributes
    ----------
    filtered_means, filtered_covs : ndarray of shape (T, n), (T, n, n)
        :math:`\\hat x_{t|t}`, :math:`P_{t|t}`.
    predicted_means, predicted_covs : ndarray of shape (T, n), (T, n, n)
        :math:`\\hat x_{t|t-1}`, :math:`P_{t|t-1}` (for t = 0: the prior).
    innovations : ndarray of shape (T, m)
        :math:`v_t` (NaN where missing).
    innovation_covs : ndarray of shape (T, m, m)
        :math:`S_t` (NaN where missing).
    loglik : float
        Log-likelihood of the observed data.
    """

    filtered_means: np.ndarray
    filtered_covs: np.ndarray
    predicted_means: np.ndarray
    predicted_covs: np.ndarray
    innovations: np.ndarray
    innovation_covs: np.ndarray
    loglik: float


@dataclass(frozen=True)
class SmootherResult(DictMixin):
    """RTS smoother output: ``smoothed_means`` (T, n) and ``smoothed_covs`` (T, n, n)."""

    smoothed_means: np.ndarray
    smoothed_covs: np.ndarray


@register(
    domain="finance",
    tasks=("state-space", "filtering", "time-series"),
    complexity=Complexity(
        time="O(T (n^3 + m^3 + n^2 m))", space="O(T n^2 + T m^2)", notes="T steps"
    ),
    references=(*_REFS, "H. E. Rauch, F. Tung & C. T. Striebel, AIAA J. 3 (1965) 1445-1450"),
)
class KalmanFilter:
    r"""Linear Gaussian Kalman filter with a Joseph-form update and an RTS smoother.

    See the module docstring for the model, the timeline and the formulas. The
    object holds the current state estimate (``mean``, ``cov``, ``t``); ``predict``
    and ``update`` advance it, and validate everything before changing it (a
    rejected call leaves the state unchanged). ``filter`` and ``smooth`` run on a
    whole series from the initial prior without touching that state.

    Parameters
    ----------
    transition : array_like of shape (n, n)
        :math:`F`.
    observation : array_like of shape (m, n)
        :math:`H` (time-invariant default; ``update``/``filter`` accept per-step ones).
    transition_cov : array_like of shape (n, n)
        :math:`Q`, symmetric positive semi-definite.
    observation_cov : array_like of shape (m, m)
        :math:`R`, symmetric positive definite.
    initial_mean : array_like of shape (n, )
        :math:`m_0`.
    initial_cov : array_like of shape (n, n)
        :math:`P_0`, symmetric positive semi-definite.

    Complexity
    ----------
    Per step O(n^3 + m^3 + n^2 m) time; ``filter`` stores O(T n^2) results.

    References
    ----------
    Kalman (1960); Anderson & Moore (1979); Rauch, Tung & Striebel (1965).

    Examples
    --------
    >>> kf = KalmanFilter([[1.0]], [[1.0]], [[0.1]], [[1.0]], [0.0], [[1.0]])
    >>> res = kf.filter([1.0, 1.2, 0.9])
    >>> res.filtered_means.shape, round(float(res.filtered_means[0, 0]), 6)
    ((3, 1), 0.5)
    """

    def __init__(
        self,
        transition: ArrayLike,
        observation: ArrayLike,
        transition_cov: ArrayLike,
        observation_cov: ArrayLike,
        initial_mean: ArrayLike,
        initial_cov: ArrayLike,
    ) -> None:
        F = as_float_array(transition, "transition")
        if F.ndim != 2 or F.shape[0] != F.shape[1]:
            raise ValueError(f"transition must be square (n, n), got {F.shape}")
        n = F.shape[0]
        H = as_float_array(observation, "observation")
        if H.ndim != 2 or H.shape[1] != n:
            raise ValueError(f"observation must have shape (m, {n}), got {H.shape}")
        m = H.shape[0]
        check_finite(F, "transition")
        check_finite(H, "observation")
        self.n, self.m = n, m
        self.F, self.H = F, H
        self.Q = _check_cov(
            _matrix(transition_cov, (n, n), "transition_cov"), "transition_cov", False
        )
        self.R = _check_cov(
            _matrix(observation_cov, (m, m), "observation_cov"), "observation_cov", True
        )
        self.initial_mean = _matrix(initial_mean, (n,), "initial_mean")
        self.initial_cov = _check_cov(
            _matrix(initial_cov, (n, n), "initial_cov"), "initial_cov", False
        )
        self.reset()

    # -- online interface ----------------------------------------------------
    def reset(self) -> KalmanFilter:
        """Set the current estimate back to the prior of :math:`x_0` (``t = 0``)."""
        self.mean = self.initial_mean.copy()
        self.cov = self.initial_cov.copy()
        self.t = 0
        return self

    def predict(self) -> KalmanFilter:
        """Predict: :math:`\\hat x \\leftarrow F\\hat x`, :math:`P \\leftarrow FPF^\\top+Q`."""
        mean, cov = self._predict(self.mean, self.cov)
        self.mean, self.cov = mean, cov
        self.t += 1
        return self

    def update(self, z: ArrayLike, observation: ArrayLike | None = None) -> KalmanFilter:
        """Condition the current (predicted) estimate on observation ``z``.

        ``z`` all-NaN is treated as missing (no change). Invalid input (wrong shape,
        infinity, partly-NaN ``z``, a non-positive-definite innovation covariance)
        raises before anything changes.
        """
        zz, H = self._check_obs(z, observation)
        if zz is None:
            return self
        mean, cov, _, _ = self._update(self.mean, self.cov, zz, H)
        self.mean, self.cov = mean, cov
        return self

    # -- whole-series interface ----------------------------------------------
    def filter(
        self, observations: ArrayLike, observation_matrices: ArrayLike | None = None
    ) -> FilterResult:
        """Run the filter over ``observations`` (shape (T,) for m = 1, or (T, m)).

        ``observation_matrices`` of shape (T, m, n) gives a per-step :math:`H_t`.
        Does not change the online state.
        """
        Z, Hs = self._check_series(observations, observation_matrices)
        T, n, m = Z.shape[0], self.n, self.m
        fm, fc = np.empty((T, n)), np.empty((T, n, n))
        pm, pc = np.empty((T, n)), np.empty((T, n, n))
        inn, S_all = np.full((T, m), np.nan), np.full((T, m, m), np.nan)
        mean, cov = self.initial_mean.copy(), self.initial_cov.copy()
        ll = 0.0
        for t in range(T):
            if t > 0:
                mean, cov = self._predict(mean, cov)
            pm[t], pc[t] = mean, cov
            H = self.H if Hs is None else Hs[t]
            if not np.all(np.isnan(Z[t])):
                mean, cov, v, S = self._update(mean, cov, Z[t], H)
                inn[t], S_all[t] = v, S
                L = cholesky(S)
                a = cholesky_solve(L, v)
                ll += -0.5 * (m * _LOG2PI + 2.0 * float(np.sum(np.log(np.diag(L)))) + float(v @ a))
            fm[t], fc[t] = mean, cov
        return FilterResult(fm, fc, pm, pc, inn, S_all, ll)

    def smooth(
        self, observations: ArrayLike, observation_matrices: ArrayLike | None = None
    ) -> SmootherResult:
        """Rauch-Tung-Striebel smoother (uses all observations; not causal)."""
        res = self.filter(observations, observation_matrices)
        T = res.filtered_means.shape[0]
        sm, sc = res.filtered_means.copy(), res.filtered_covs.copy()
        for t in range(T - 2, -1, -1):
            Pp = res.predicted_covs[t + 1]
            # C = P_{t|t} F^T P_{t+1|t}^{-1}, via a symmetric solve (pseudo-inverse if singular)
            C = np.linalg.lstsq(Pp, self.F @ res.filtered_covs[t], rcond=None)[0].T
            sm[t] = res.filtered_means[t] + C @ (sm[t + 1] - res.predicted_means[t + 1])
            sc[t] = res.filtered_covs[t] + C @ (sc[t + 1] - Pp) @ C.T
            sc[t] = (sc[t] + sc[t].T) / 2.0
        return SmootherResult(sm, sc)

    # -- internals -------------------------------------------------------------
    def _predict(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cov = self.F @ cov @ self.F.T + self.Q
        return self.F @ mean, (cov + cov.T) / 2.0

    def _update(
        self, mean: np.ndarray, cov: np.ndarray, z: np.ndarray, H: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        v = z - H @ mean
        S = H @ cov @ H.T + self.R
        S = (S + S.T) / 2.0
        try:
            L = cholesky(S)
        except LinAlgError:
            raise ValueError("innovation covariance is not positive definite") from None
        K = cholesky_solve(L, H @ cov).T  # P H^T S^{-1}
        A = np.eye(self.n) - K @ H
        new_cov = A @ cov @ A.T + K @ self.R @ K.T
        return mean + K @ v, (new_cov + new_cov.T) / 2.0, v, S

    def _check_H(self, observation: ArrayLike | None) -> np.ndarray:
        if observation is None:
            return self.H
        return _matrix(observation, (self.m, self.n), "observation")

    def _check_obs(
        self, z: ArrayLike, observation: ArrayLike | None
    ) -> tuple[np.ndarray | None, np.ndarray]:
        H = self._check_H(observation)
        zz = as_float_array(z, "z").reshape(-1)
        if zz.shape != (self.m,):
            raise ValueError(f"z must have {self.m} element(s), got shape {np.shape(z)}")
        if np.all(np.isnan(zz)):
            return None, H
        check_finite(zz, "z (partly missing observations are not supported)")
        return zz, H

    def _check_series(
        self, observations: ArrayLike, observation_matrices: ArrayLike | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        Z = as_float_array(observations, "observations")
        if Z.ndim == 1 and self.m == 1:
            Z = Z[:, None]
        if Z.ndim != 2 or Z.shape[1] != self.m or Z.shape[0] == 0:
            raise ValueError(f"observations must have shape (T, {self.m}), got {Z.shape}")
        rows_missing = np.all(np.isnan(Z), axis=1)
        if not np.all(np.isfinite(Z[~rows_missing])):
            raise ValueError("observations contain infinity or partly-NaN rows")
        Hs = None
        if observation_matrices is not None:
            Hs = as_float_array(observation_matrices, "observation_matrices")
            if Hs.shape != (Z.shape[0], self.m, self.n):
                raise ValueError(
                    f"observation_matrices must have shape {(Z.shape[0], self.m, self.n)}, "
                    f"got {Hs.shape}"
                )
            check_finite(Hs, "observation_matrices")
        return Z, Hs


@dataclass(frozen=True)
class HedgeRatioResult(DictMixin):
    """Output of :func:`dynamic_hedge_ratio` (all arrays of shape (T,)).

    Attributes
    ----------
    hedge_ratio, intercept : ndarray
        Filtered :math:`\\beta_{t|t}`, :math:`\\alpha_{t|t}` (use at t for trading at t+1).
    spread : ndarray
        Innovation :math:`y_t - \\beta_{t|t-1}x_t - \\alpha_{t|t-1}` (uses only data before t
        for the coefficients).
    spread_std : ndarray
        :math:`\\sqrt{S_t}`, the predicted standard deviation of ``spread``.
    """

    hedge_ratio: np.ndarray
    intercept: np.ndarray
    spread: np.ndarray
    spread_std: np.ndarray


@register(
    domain="finance",
    tasks=("pairs-trading", "filtering", "state-space"),
    complexity=Complexity(time="O(T)", space="O(T)"),
    references=("E. P. Chan, Algorithmic Trading (2013), ch. 3", *_REFS),
)
def dynamic_hedge_ratio(
    y: ArrayLike, x: ArrayLike, delta: float = 1e-4, obs_var: float = 1e-3
) -> HedgeRatioResult:
    r"""Time-varying hedge ratio of ``y`` on ``x`` by Kalman filtering (pairs trading).

    See the module docstring. The prior is :math:`(\beta_0, \alpha_0) = (0, 0)` with
    covariance :math:`10^6 I` (diffuse), so the first estimates follow the data.

    Parameters
    ----------
    y, x : array_like of shape (T, )
        Prices (or log prices) of the two legs, oldest first.
    delta : float, default=1e-4
        Random-walk intensity in (0, 1); larger adapts faster.
    obs_var : float, default=1e-3
        Observation noise variance :math:`R` (> 0).

    Returns
    -------
    HedgeRatioResult

    Complexity
    ----------
    Time O(T), extra space O(T).

    References
    ----------
    E. P. Chan, *Algorithmic Trading* (2013), ch. 3.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(10.0, 20.0, 400)
    >>> res = dynamic_hedge_ratio(1.5 * x + 2.0, x, obs_var=1e-6)
    >>> round(float(res.hedge_ratio[-1]), 3), round(float(res.intercept[-1]), 2)
    (1.5, 2.0)
    """
    yy = check_series(y, "y", min_rows=1)
    xx = check_series(x, "x", min_rows=1)
    if yy.ndim != 1 or xx.shape != yy.shape:
        raise ValueError(f"y and x must both have shape (T,), got {yy.shape} and {xx.shape}")
    d = float(delta)
    if not 0.0 < d < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta!r}")
    r = float(obs_var)
    if not (np.isfinite(r) and r > 0.0):
        raise ValueError(f"obs_var must be positive, got {obs_var!r}")
    kf = KalmanFilter(
        np.eye(2), [[1.0, 1.0]], d / (1.0 - d) * np.eye(2), [[r]], np.zeros(2), 1e6 * np.eye(2)
    )
    Hs = np.stack([xx, np.ones_like(xx)], axis=1)[:, None, :]
    res = kf.filter(yy, Hs)
    return HedgeRatioResult(
        hedge_ratio=res.filtered_means[:, 0],
        intercept=res.filtered_means[:, 1],
        spread=res.innovations[:, 0],
        spread_std=np.sqrt(res.innovation_covs[:, 0, 0]),
    )
