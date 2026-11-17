"""Tests for lipipy.finance.kalman: exact conditioning, Riccati limit, causality, atomicity."""

import math

import numpy as np
import pytest

from lipipy.finance import KalmanFilter, dynamic_hedge_ratio

pytestmark = pytest.mark.filterwarnings("error")


def _model(seed, n=2, m=2):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(n, n)) * 0.5
    H = rng.normal(size=(m, n))
    A = rng.normal(size=(n, n))
    B = rng.normal(size=(m, m))
    Q = A @ A.T * 0.1 + 0.01 * np.eye(n)
    R = B @ B.T * 0.2 + 0.05 * np.eye(m)
    m0 = rng.normal(size=n)
    C = rng.normal(size=(n, n))
    P0 = C @ C.T + 0.1 * np.eye(n)
    return F, H, Q, R, m0, P0


def _simulate(F, H, Q, R, m0, P0, T, seed):
    rng = np.random.default_rng(seed + 99)
    x = rng.multivariate_normal(m0, P0)
    zs = []
    for t in range(T):
        if t > 0:
            x = F @ x + rng.multivariate_normal(np.zeros(len(m0)), Q)
        zs.append(H @ x + rng.multivariate_normal(np.zeros(H.shape[0]), R))
    return np.array(zs)


def _joint(F, H, Q, R, m0, P0, T):
    """Mean and covariance of (x_0..x_{T-1}, z_0..z_{T-1}) by stacking the linear map."""
    n, m = F.shape[0], H.shape[0]
    # x = M_x @ (x0, w_1..w_{T-1}),  z = Hbig x + v
    dim_e = n * T
    Mx = np.zeros((n * T, dim_e))
    for t in range(T):
        for s in range(t + 1):
            Mx[t * n : (t + 1) * n, s * n : (s + 1) * n] = np.linalg.matrix_power(F, t - s)
    cov_e = np.zeros((dim_e, dim_e))
    cov_e[:n, :n] = P0
    for s in range(1, T):
        cov_e[s * n : (s + 1) * n, s * n : (s + 1) * n] = Q
    mean_e = np.zeros(dim_e)
    mean_e[:n] = m0
    mx = Mx @ mean_e
    Cxx = Mx @ cov_e @ Mx.T
    Hb = np.kron(np.eye(T), H)
    mz = Hb @ mx
    Czz = Hb @ Cxx @ Hb.T + np.kron(np.eye(T), R)
    Cxz = Cxx @ Hb.T
    return mx, mz, Cxx, Czz, Cxz, n, m


@pytest.mark.parametrize("seed", range(6))
def test_filter_and_smoother_equal_exact_gaussian_conditioning(seed):
    F, H, Q, R, m0, P0 = _model(seed)
    T = 6
    Z = _simulate(F, H, Q, R, m0, P0, T, seed)
    kf = KalmanFilter(F, H, Q, R, m0, P0)
    res = kf.filter(Z)
    sm = kf.smooth(Z)
    mx, mz, Cxx, Czz, Cxz, n, m = _joint(F, H, Q, R, m0, P0, T)
    z = Z.reshape(-1)
    for t in range(T):
        k = (t + 1) * m  # condition on z_0..z_t only
        G = np.linalg.solve(Czz[:k, :k], np.eye(k))
        sl = slice(t * n, (t + 1) * n)
        mean = mx[sl] + Cxz[sl, :k] @ G @ (z[:k] - mz[:k])
        cov = Cxx[sl, sl] - Cxz[sl, :k] @ G @ Cxz[sl, :k].T
        np.testing.assert_allclose(res.filtered_means[t], mean, rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(res.filtered_covs[t], cov, rtol=1e-9, atol=1e-11)
        Gall = np.linalg.solve(Czz, np.eye(T * m))
        smean = mx[sl] + Cxz[sl] @ Gall @ (z - mz)
        scov = Cxx[sl, sl] - Cxz[sl] @ Gall @ Cxz[sl].T
        np.testing.assert_allclose(sm.smoothed_means[t], smean, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(sm.smoothed_covs[t], scov, rtol=1e-8, atol=1e-10)
    # log-likelihood = log N(z; mz, Czz) of the whole observed vector
    sign, logdet = np.linalg.slogdet(Czz)
    d = z - mz
    ll = -0.5 * (T * m * math.log(2 * math.pi) + logdet + d @ np.linalg.solve(Czz, d))
    assert res.loglik == pytest.approx(ll, rel=1e-10)


@pytest.mark.parametrize("q,r", [(0.01, 1.0), (1.0, 1.0), (5.0, 0.1), (1e-4, 1.0)])
def test_random_walk_steady_state_gain_equals_riccati_closed_form(q, r):
    """Local level model: P = (q + sqrt(q^2 + 4 q r)) / 2, K = P / (P + r)."""
    Z = np.random.default_rng(0).normal(size=4000)
    res = KalmanFilter([[1.0]], [[1.0]], [[q]], [[r]], [0.0], [[1.0]]).filter(Z)
    P = (q + math.sqrt(q * q + 4 * q * r)) / 2
    K = P / (P + r)
    pred = res.predicted_covs[-1, 0, 0]
    gain = pred / (pred + r)
    assert pred == pytest.approx(P, rel=1e-10)
    assert gain == pytest.approx(K, rel=1e-10)


def test_filter_is_causal_perturbing_the_future():
    F, H, Q, R, m0, P0 = _model(3)
    Z = _simulate(F, H, Q, R, m0, P0, 40, 3)
    kf = KalmanFilter(F, H, Q, R, m0, P0)
    base = kf.filter(Z)
    for t in (0, 10, 25, 38):
        Zp = Z.copy()
        Zp[t + 1 :] += np.random.default_rng(t).normal(scale=1e3, size=Zp[t + 1 :].shape)
        alt = kf.filter(Zp)
        np.testing.assert_array_equal(alt.filtered_means[: t + 1], base.filtered_means[: t + 1])
        np.testing.assert_array_equal(alt.filtered_covs[: t + 1], base.filtered_covs[: t + 1])
        np.testing.assert_array_equal(alt.predicted_means[: t + 2], base.predicted_means[: t + 2])
        assert not np.array_equal(alt.filtered_means[t + 1 :], base.filtered_means[t + 1 :])


def test_online_interface_matches_batch_filter():
    F, H, Q, R, m0, P0 = _model(4)
    Z = _simulate(F, H, Q, R, m0, P0, 15, 4)
    kf = KalmanFilter(F, H, Q, R, m0, P0)
    res = kf.filter(Z)
    for t in range(15):
        if t > 0:
            kf.predict()
        kf.update(Z[t])
        np.testing.assert_allclose(kf.mean, res.filtered_means[t], rtol=1e-14, atol=1e-15)
        np.testing.assert_allclose(kf.cov, res.filtered_covs[t], rtol=1e-14, atol=1e-15)
    assert kf.t == 14
    kf.reset()
    np.testing.assert_array_equal(kf.mean, m0)


def test_rejected_update_leaves_state_unchanged():
    F, H, Q, R, m0, P0 = _model(5)
    kf = KalmanFilter(F, H, Q, R, m0, P0)
    kf.update([0.3, -0.2]).predict()
    mean, cov, t = kf.mean.copy(), kf.cov.copy(), kf.t
    for bad in ([1.0], [np.inf, 0.0], [np.nan, 1.0], "zz"):
        with pytest.raises((ValueError, TypeError)):
            kf.update(bad)
    with pytest.raises(ValueError, match="observation"):
        kf.update([0.0, 0.0], observation=np.ones((3, 2)))
    np.testing.assert_array_equal(kf.mean, mean)
    np.testing.assert_array_equal(kf.cov, cov)
    assert kf.t == t
    kf.update([np.nan, np.nan])  # missing: no change
    np.testing.assert_array_equal(kf.mean, mean)


def test_missing_observations_are_pure_predictions():
    F, H, Q, R, m0, P0 = _model(6)
    Z = _simulate(F, H, Q, R, m0, P0, 10, 6)
    Z[4] = np.nan
    res = KalmanFilter(F, H, Q, R, m0, P0).filter(Z)
    np.testing.assert_array_equal(res.filtered_means[4], res.predicted_means[4])
    assert np.all(np.isnan(res.innovations[4]))
    full = KalmanFilter(F, H, Q, R, m0, P0).filter(np.delete(Z, 4, axis=0)[:4])
    np.testing.assert_allclose(res.filtered_means[:4], full.filtered_means, rtol=1e-14)


def test_joseph_form_keeps_covariances_symmetric_psd():
    # nearly noiseless observations of a 3-state system: the textbook (I - KH)P loses PSD
    rng = np.random.default_rng(7)
    F = np.eye(3) + 0.01 * rng.normal(size=(3, 3))
    H = rng.normal(size=(3, 3))
    kf = KalmanFilter(F, H, 1e-8 * np.eye(3), 1e-12 * np.eye(3), np.zeros(3), 1e6 * np.eye(3))
    res = kf.filter(rng.normal(size=(300, 3)))
    for P in res.filtered_covs:
        np.testing.assert_array_equal(P, P.T)
        assert np.linalg.eigvalsh(P)[0] > -1e-12 * np.abs(P).max()


def test_constructor_validation():
    with pytest.raises(ValueError, match="square"):
        KalmanFilter(np.ones((2, 3)), np.ones((1, 2)), np.eye(2), [[1.0]], [0, 0], np.eye(2))
    with pytest.raises(ValueError, match="positive definite"):
        KalmanFilter(np.eye(1), [[1.0]], [[1.0]], [[0.0]], [0.0], [[1.0]])
    with pytest.raises(ValueError, match="symmetric"):
        KalmanFilter(np.eye(2), np.eye(2), [[1.0, 0.5], [0.0, 1.0]], np.eye(2), [0, 0], np.eye(2))
    with pytest.raises(ValueError, match="semi-definite"):
        KalmanFilter(np.eye(1), [[1.0]], [[-1.0]], [[1.0]], [0.0], [[1.0]])
    kf = KalmanFilter(np.eye(1), [[1.0]], [[1.0]], [[1.0]], [0.0], [[1.0]])
    with pytest.raises(ValueError, match="infinity"):
        kf.filter([1.0, np.inf])
    with pytest.raises(ValueError, match="observation_matrices"):
        kf.filter([1.0, 2.0], np.ones((3, 1, 1)))
    assert set(kf.filter([1.0]).to_dict()) >= {"filtered_means", "loglik"}


# ---------------------------------------------------------------------------
# dynamic hedge ratio
# ---------------------------------------------------------------------------
def test_dynamic_hedge_ratio_tracks_a_changing_beta():
    rng = np.random.default_rng(8)
    T = 3000
    x = 50 + np.cumsum(rng.normal(size=T))
    beta = np.where(np.arange(T) < T // 2, 1.0, 1.6)
    y = beta * x + 5.0 + rng.normal(scale=0.5, size=T)
    res = dynamic_hedge_ratio(y, x, delta=1e-4, obs_var=0.25)
    assert abs(res.hedge_ratio[T // 2 - 1] - 1.0) < 0.05
    assert abs(res.hedge_ratio[-1] - 1.6) < 0.05
    # spread is the one-step-ahead innovation and is standardised by its predicted sd
    z = res.spread[100:] / res.spread_std[100:]
    assert abs(np.std(z) - 1.0) < 0.15


def test_dynamic_hedge_ratio_is_causal():
    rng = np.random.default_rng(9)
    x = 20 + np.cumsum(rng.normal(size=500))
    y = 1.3 * x + rng.normal(size=500)
    base = dynamic_hedge_ratio(y, x)
    for t in (0, 100, 498):
        yp, xp = y.copy(), x.copy()
        yp[t + 1 :] *= 3.0
        xp[t + 1 :] += 50.0
        alt = dynamic_hedge_ratio(yp, xp)
        for field in ("hedge_ratio", "intercept", "spread", "spread_std"):
            np.testing.assert_array_equal(
                getattr(alt, field)[: t + 1], getattr(base, field)[: t + 1]
            )
    with pytest.raises(ValueError, match="delta"):
        dynamic_hedge_ratio(y, x, delta=1.0)
    with pytest.raises(ValueError, match="obs_var"):
        dynamic_hedge_ratio(y, x, obs_var=0.0)
    with pytest.raises(ValueError, match="shape"):
        dynamic_hedge_ratio(y, x[:-1])
    with pytest.raises(ValueError, match="NaN"):
        dynamic_hedge_ratio(np.r_[y[:-1], np.nan], x)
