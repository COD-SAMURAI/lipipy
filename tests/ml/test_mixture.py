"""GaussianMixture: EM parity with sklearn from identical starts, monotone EM, BIC/AIC."""

import numpy as np
import pytest

from lipipy.core import ConvergenceWarning, NotFittedError
from lipipy.ml import GaussianMixture, adjusted_rand_score

COV_TYPES = ["full", "tied", "diag", "spherical"]


def make_data(seed, n_per=80, d=3, K=3, spread=4.0):
    rng = np.random.default_rng(seed)
    parts = [
        rng.normal(size=(n_per, d)) @ rng.normal(scale=0.6, size=(d, d))
        + rng.normal(scale=spread, size=d)
        for _ in range(K)
    ]
    return np.concatenate(parts), np.repeat(np.arange(K), n_per)


def explicit_start(X, seed, K, cov_type):
    rng = np.random.default_rng(seed + 1000)
    d = X.shape[1]
    weights = rng.dirichlet(np.ones(K) * 5)
    means = X[rng.choice(len(X), K, replace=False)]
    A = rng.normal(size=(K, d, d))
    full = np.einsum("kij,klj->kil", A, A) + d * np.eye(d)
    precisions = {
        "full": full,
        "tied": full[0],
        "diag": rng.uniform(0.2, 2.0, size=(K, d)),
        "spherical": rng.uniform(0.2, 2.0, size=K),
    }[cov_type]
    return {"weights_init": weights, "means_init": means, "precisions_init": precisions}


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("cov_type", COV_TYPES)
def test_parity_with_sklearn_from_explicit_start(seed, cov_type):
    sk_mixture = pytest.importorskip("sklearn.mixture")
    X, _ = make_data(seed)
    kw = dict(
        n_components=3,
        covariance_type=cov_type,
        tol=1e-10,
        max_iter=1000,
        **explicit_start(X, seed, 3, cov_type),
    )
    ours = GaussianMixture(**kw).fit(X)
    ref = sk_mixture.GaussianMixture(**kw).fit(X)
    for attr in ("weights_", "means_", "covariances_", "precisions_", "precisions_cholesky_"):
        np.testing.assert_allclose(
            getattr(ours, attr), getattr(ref, attr), rtol=1e-8, atol=1e-10, err_msg=attr
        )
    assert ours.lower_bound_ == pytest.approx(ref.lower_bound_, rel=1e-10)
    assert ours.n_iter_ == ref.n_iter_
    np.testing.assert_allclose(ours.lower_bounds_, ref.lower_bounds_, rtol=1e-9)
    X_new = make_data(seed + 50)[0]
    np.testing.assert_allclose(ours.score_samples(X_new), ref.score_samples(X_new), rtol=1e-9)
    np.testing.assert_allclose(ours.predict_proba(X_new), ref.predict_proba(X_new), atol=1e-9)
    assert ours.bic(X) == pytest.approx(ref.bic(X), rel=1e-10)
    assert ours.aic(X) == pytest.approx(ref.aic(X), rel=1e-10)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("init_params", ["kmeans", "random"])
def test_log_likelihood_never_decreases(seed, cov_type, init_params):
    """Exact EM (reg_covar=0) is monotone: every iteration's lower bound is >= the last."""
    X, _ = make_data(seed, n_per=60)
    gm = GaussianMixture(
        n_components=3,
        covariance_type=cov_type,
        reg_covar=0.0,
        tol=1e-8,
        max_iter=5000,
        init_params=init_params,
        random_state=seed,
    ).fit(X)
    bounds = gm.lower_bounds_
    assert len(bounds) == gm.n_iter_ >= 2
    slack = 1e-12 * np.max(np.abs(bounds))  # rounding of a mean of n log-densities
    assert np.all(np.diff(bounds) >= -slack), np.diff(bounds).min()


def test_bic_aic_formulas():
    X, _ = make_data(0)
    n, d, K = X.shape[0], X.shape[1], 3
    counts = {"full": K * d * (d + 1) // 2, "tied": d * (d + 1) // 2, "diag": K * d, "spherical": K}
    for cov_type, cov_params in counts.items():
        gm = GaussianMixture(n_components=K, covariance_type=cov_type, random_state=0).fit(X)
        p = cov_params + K * d + K - 1
        L = gm.score_samples(X).sum()
        assert gm.bic(X) == pytest.approx(-2 * L + p * np.log(n), rel=1e-12)
        assert gm.aic(X) == pytest.approx(-2 * L + 2 * p, rel=1e-12)


def test_score_samples_is_the_mixture_density():
    X, _ = make_data(1, d=2)
    gm = GaussianMixture(n_components=3, random_state=0).fit(X)
    x = X[:5]
    dens = np.zeros(5)
    for w, mu, S in zip(gm.weights_, gm.means_, gm.covariances_, strict=True):
        diff = x - mu
        q = np.einsum("ij,jk,ik->i", diff, np.linalg.inv(S), diff)
        dens += w * np.exp(-0.5 * q) / np.sqrt(np.linalg.det(2 * np.pi * S))
    np.testing.assert_allclose(gm.score_samples(x), np.log(dens), rtol=1e-12)
    np.testing.assert_allclose(gm.predict_proba(x).sum(axis=1), 1.0)
    assert gm.score(x) == pytest.approx(np.mean(np.log(dens)))


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("seed", range(5))
def test_recovers_well_separated_components(seed):
    X, y = make_data(seed, spread=15.0)
    gm = GaussianMixture(n_components=3, n_init=3, random_state=seed).fit(X)
    assert adjusted_rand_score(y, gm.predict(X)) == 1.0
    np.testing.assert_array_equal(gm.fit_predict(X), gm.predict(X))


@pytest.mark.filterwarnings("error")
def test_far_points_do_not_underflow():
    X, _ = make_data(2)
    gm = GaussianMixture(n_components=3, random_state=0).fit(X)
    far = np.full((2, 3), 1e4)
    proba = gm.predict_proba(far)
    assert np.all(np.isfinite(proba)) and np.allclose(proba.sum(axis=1), 1.0)
    assert np.all(np.isfinite(gm.score_samples(far)))


def test_seeded_runs_and_sampling_are_reproducible():
    X, _ = make_data(3)
    a = GaussianMixture(n_components=3, init_params="random", random_state=5).fit(X)
    b = GaussianMixture(n_components=3, init_params="random", random_state=5).fit(X)
    np.testing.assert_array_equal(a.means_, b.means_)
    S1, l1 = a.sample(500)
    S2, l2 = a.sample(500)
    np.testing.assert_array_equal(S1, S2)
    assert S1.shape == (500, 3) and set(np.unique(l1)) <= {0, 1, 2}
    np.testing.assert_array_equal(l1, l2)


@pytest.mark.parametrize("cov_type", COV_TYPES)
def test_sample_moments(cov_type):
    X, _ = make_data(4, d=2)
    gm = GaussianMixture(n_components=2, covariance_type=cov_type, random_state=0).fit(X)
    S, labels = gm.sample(200_000)
    for k in range(2):
        Sk = S[labels == k]
        assert len(Sk) / len(S) == pytest.approx(gm.weights_[k], abs=0.01)
        np.testing.assert_allclose(Sk.mean(axis=0), gm.means_[k], atol=0.05)
        if cov_type == "full":
            cov = gm.covariances_[k]
        elif cov_type == "tied":
            cov = gm.covariances_
        elif cov_type == "diag":
            cov = np.diag(gm.covariances_[k])
        else:
            cov = gm.covariances_[k] * np.eye(2)
        np.testing.assert_allclose(np.cov(Sk.T), cov, atol=0.05 * np.abs(cov).max())


def test_convergence_warning():
    X, _ = make_data(0)
    with pytest.warns(ConvergenceWarning, match="did not converge"):
        gm = GaussianMixture(n_components=3, max_iter=1, tol=1e-12, random_state=0).fit(X)
    assert not gm.converged_ and gm.n_iter_ == 1


def test_singular_covariance_raises_and_fit_is_atomic():
    X = np.vstack([np.zeros((5, 2)), np.ones((5, 2))])  # collapsed components
    gm = GaussianMixture(n_components=2, reg_covar=0.0, random_state=0)
    with pytest.raises(ValueError, match="ill-defined covariance"):
        gm.fit(X)
    assert not hasattr(gm, "means_")
    good = GaussianMixture(n_components=2, random_state=0).fit(X)  # reg_covar > 0 fixes it
    means = good.means_.copy()
    with pytest.raises(ValueError):
        good.set_params(reg_covar=0.0).fit(X)
    np.testing.assert_array_equal(good.means_, means)


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"n_components": 0}, "n_components"),
        ({"n_components": 1000}, "n_components"),
        ({"covariance_type": "banded"}, "covariance_type"),
        ({"init_params": "k-means++"}, "init_params"),
        ({"tol": -1.0}, "tol"),
        ({"reg_covar": -1.0}, "reg_covar"),
        ({"max_iter": 0}, "max_iter"),
        ({"n_init": 0}, "n_init"),
        ({"weights_init": [0.5, 0.5, 0.5]}, "weights_init"),
        ({"weights_init": [0.5, 0.5]}, "weights_init"),
        ({"means_init": np.zeros((2, 3))}, "means_init"),
        ({"precisions_init": -np.ones(3), "covariance_type": "spherical"}, "precisions_init"),
        ({"precisions_init": np.stack([-np.eye(3)] * 3)}, "precisions_init"),
        ({"precisions_init": np.ones((3, 3, 3))}, "precisions_init"),
    ],
)
def test_bad_params(params, match):
    X, _ = make_data(0)
    kw = {"n_components": 3, **params}
    with pytest.raises(ValueError, match=match):
        GaussianMixture(**kw).fit(X)


def test_unfitted_and_feature_mismatch():
    X, _ = make_data(0)
    with pytest.raises(NotFittedError):
        GaussianMixture().predict(X)
    with pytest.raises(NotFittedError):
        GaussianMixture().sample()
    gm = GaussianMixture(n_components=2, random_state=0).fit(X)
    with pytest.raises(ValueError, match="features"):
        gm.score_samples(X[:, :2])
    with pytest.raises(ValueError, match="n_samples"):
        gm.sample(0)
