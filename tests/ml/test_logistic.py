"""LogisticRegression: parity with scikit-learn, optimality, stability, error paths."""

import tracemalloc

import numpy as np
import pytest

from lipipy.bench import estimate_complexity
from lipipy.core import ConvergenceWarning, NotFittedError
from lipipy.ml import LogisticRegression


def make_data(rng, n=200, d=5, k=3, scale=1.0, shift=0.0):
    X = rng.normal(size=(n, d)) * scale + shift
    W = rng.normal(size=(k, d)) / np.maximum(scale, 1e-12)
    y = np.argmax((X - shift) @ W.T + rng.gumbel(size=(n, k)), axis=1)
    return X, y


def sk_logistic(C, fit_intercept):
    """L2-penalised lbfgs reference (L2 is the default penalty in every sklearn version;
    passing ``penalty="l2"`` explicitly is deprecated since 1.8)."""
    lm = pytest.importorskip("sklearn.linear_model")
    return lm.LogisticRegression(
        C=C, fit_intercept=fit_intercept, solver="lbfgs", tol=1e-10, max_iter=10000
    )


def scaled_gradient(model, X, y):
    """Gradient of mean log-loss + ||W||^2 / (2 C n), recomputed independently."""
    n = X.shape[0]
    Y = (y[:, None] == model.classes_[None, :]).astype(float)
    W, b = model.coef_, model.intercept_
    Z = X @ W.T + b
    if W.shape[0] == 1:
        R = (1.0 / (1.0 + np.exp(-Z[:, 0])) - Y[:, 1])[:, None]
    else:
        Z = Z - Z.max(axis=1, keepdims=True)
        P = np.exp(Z) / np.exp(Z).sum(axis=1, keepdims=True)
        R = P - Y
    gW = R.T @ X / n + W / (model.C * n)
    gb = R.mean(axis=0) if model.fit_intercept else np.zeros(0)
    return np.concatenate([gW.ravel(), gb])


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("k", [2, 3, 5])
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("C", [0.01, 1.0, 100.0])
def test_parity_with_sklearn(k, fit_intercept, C):
    for seed in range(8):
        rng = np.random.default_rng(seed)
        X, y = make_data(rng, k=k)
        X_test = rng.normal(size=(100, X.shape[1])) * 1.5
        ours = LogisticRegression(C=C, fit_intercept=fit_intercept).fit(X, y)
        ref = sk_logistic(C, fit_intercept).fit(X, y)
        assert ours.coef_.shape == ref.coef_.shape
        scale = np.max(np.abs(ref.coef_))
        np.testing.assert_allclose(ours.coef_, ref.coef_, rtol=0, atol=1e-5 * scale)
        np.testing.assert_allclose(ours.intercept_, ref.intercept_, rtol=0, atol=1e-5 * (1 + scale))
        p_ours, p_ref = ours.predict_proba(X_test), ref.predict_proba(X_test)
        np.testing.assert_allclose(p_ours, p_ref, rtol=0, atol=1e-6)
        clear = np.sort(p_ref, axis=1)[:, -1] - np.sort(p_ref, axis=1)[:, -2] > 1e-5
        np.testing.assert_array_equal(ours.predict(X_test)[clear], ref.predict(X_test)[clear])
        np.testing.assert_allclose(ours.predict_log_proba(X_test), np.log(p_ref), rtol=0, atol=1e-5)
        np.testing.assert_allclose(
            ours.decision_function(X_test), ref.decision_function(X_test), atol=1e-5 * (1 + scale)
        )


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("k", [2, 4])
def test_converged_gradient_below_documented_bound(k):
    """Docstring claim: the scaled gradient at the solution is below 1e-7 (default tol)."""
    for seed in range(15):
        rng = np.random.default_rng(seed)
        for scale, shift in [(1.0, 0.0), (3.0, 2.0), (0.2, -1.0)]:
            X, y = make_data(rng, n=150, d=4, k=k, scale=scale, shift=shift)
            for C in (0.01, 1.0, 100.0):
                for fit_intercept in (True, False):
                    m = LogisticRegression(C=C, fit_intercept=fit_intercept).fit(X, y)
                    assert np.max(np.abs(scaled_gradient(m, X, y))) < 1e-7


def test_parity_ill_conditioned_features():
    """Shifted, scaled features: sklearn stops on its ftol rule, so compare loosely."""
    for seed in range(5):
        rng = np.random.default_rng(seed)
        X, y = make_data(rng, k=3, scale=3.0, shift=2.0)
        ours = LogisticRegression(C=1.0).fit(X, y)
        ref = sk_logistic(1.0, True).fit(X, y)
        np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X), atol=1e-5)
        # ours is at least as optimal as the reference
        assert np.max(np.abs(scaled_gradient(ours, X, y))) <= max(
            1e-7, np.max(np.abs(scaled_gradient(ref, X, y)))
        )


def test_multinomial_two_classes_equals_binomial_with_double_C(rng):
    """Softmax with 2 classes and penalty on both rows == binomial with C' = 2C.

    At the optimum w_1 = -w_0 = beta/2, so ||W||^2/2 = ||beta||^2/4.
    """
    X, y = make_data(rng, k=2)
    multi = LogisticRegression(C=0.5, multi_class="multinomial").fit(X, y)
    binom = LogisticRegression(C=1.0).fit(X, y)
    assert multi.coef_.shape == (2, X.shape[1]) and binom.coef_.shape == (1, X.shape[1])
    np.testing.assert_allclose(multi.coef_[1] - multi.coef_[0], binom.coef_[0], atol=1e-6)
    np.testing.assert_allclose(multi.predict_proba(X), binom.predict_proba(X), atol=1e-7)


def test_multinomial_intercepts_sum_to_zero(rng):
    X, y = make_data(rng, k=4)
    m = LogisticRegression().fit(X + 5.0, y)
    assert abs(m.intercept_.sum()) < 1e-12
    assert abs(m.coef_.sum(axis=0)).max() < 1e-6  # penalty forces sum_c w_c = 0 too


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("k", [2, 3])
def test_separable_data_stays_finite(k):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 2))
    y = np.argmax(np.column_stack([X[:, 0], -X[:, 0], X[:, 1] * 0.0 - 5])[:, :k], axis=1)
    for C in (1e-3, 1.0, 1e4):
        m = LogisticRegression(C=C, max_iter=5000).fit(X, y)
        assert np.all(np.isfinite(m.coef_)) and np.all(np.isfinite(m.intercept_))
        assert m.score(X, y) == 1.0 or C < 1
        far = X * 1e8  # logits of order 1e8+: probabilities saturate, logs stay finite
        lp = m.predict_log_proba(far)
        assert np.all(np.isfinite(lp)) and np.all(lp <= 0)
        np.testing.assert_allclose(m.predict_proba(far).sum(axis=1), 1.0)


@pytest.mark.filterwarnings("error")
@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("magnitude", [1e-3, 1e3, 1e6])
def test_badly_scaled_features_converge(magnitude):
    """Scaling X by s is the same problem as C -> C s^2 with W -> W / s (exact identity).

    ``tol`` bounds the gradient absolutely, and the gradient w.r.t. W scales with the
    features, so tiny features need a proportionally smaller ``tol`` (documented).
    """
    rng = np.random.default_rng(3)
    X, y = make_data(rng, n=150, d=3, k=3)
    tol = 1e-8 * min(1.0, magnitude)
    m = LogisticRegression(C=10.0 / magnitude**2, tol=tol).fit(X * magnitude, y)
    ref = LogisticRegression(C=10.0).fit(X, y)
    np.testing.assert_allclose(m.coef_ * magnitude, ref.coef_, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(m.predict_proba(X * magnitude), ref.predict_proba(X), atol=1e-6)


def test_constant_and_duplicate_features(rng):
    X, y = make_data(rng, k=3)
    X = np.column_stack([X, np.full(len(X), 4.0), X[:, 0]])
    m = LogisticRegression().fit(X, y)
    # a duplicated column gets identical weights (strictly convex in W)
    np.testing.assert_allclose(m.coef_[:, 0], m.coef_[:, -1], atol=1e-6)
    assert np.max(np.abs(scaled_gradient(m, X, y))) < 1e-7


def test_n_less_than_d(rng):
    X, y = make_data(rng, n=10, d=40, k=3)
    m = LogisticRegression(C=0.1).fit(X, y)
    assert np.all(np.isfinite(m.coef_))
    assert np.max(np.abs(scaled_gradient(m, X, y))) < 1e-7


def test_string_labels_score_and_attributes(rng):
    X, y = make_data(rng, k=3)
    labels = np.array(["cat", "dog", "emu"])[y]
    m = LogisticRegression().fit(X, labels)
    assert m.classes_.tolist() == ["cat", "dog", "emu"]
    assert set(m.predict(X)) <= {"cat", "dog", "emu"}
    assert m.score(X, labels) > 0.6
    assert isinstance(m.n_iter_, int) and m.n_iter_ > 0
    assert m.n_features_in_ == X.shape[1]


def test_convergence_warning():
    rng = np.random.default_rng(0)
    X, y = make_data(rng, k=3)
    with pytest.warns(ConvergenceWarning, match="did not converge"):
        m = LogisticRegression(max_iter=2).fit(X, y)
    assert m.n_iter_ == 2


def test_failed_fit_leaves_previous_state(rng):
    X, y = make_data(rng, k=3)
    m = LogisticRegression().fit(X, y)
    coef, classes = m.coef_, m.classes_
    with pytest.raises(ValueError, match="at least 2 classes"):
        m.fit(X, np.zeros(len(X)))
    assert m.coef_ is coef and m.classes_ is classes
    fresh = LogisticRegression()
    with pytest.raises(ValueError):
        fresh.fit(X, np.zeros(len(X)))
    assert not hasattr(fresh, "coef_") and not hasattr(fresh, "classes_")


def test_bad_inputs(rng):
    X, y = make_data(rng)
    for C in (0, -1.0, np.inf, np.nan, True, "1"):
        with pytest.raises(ValueError, match="C must"):
            LogisticRegression(C=C).fit(X, y)
    for tol in (0, -1e-3, np.nan):
        with pytest.raises(ValueError, match="tol"):
            LogisticRegression(tol=tol).fit(X, y)
    for it in (0, 1.5, True):
        with pytest.raises(ValueError, match="max_iter"):
            LogisticRegression(max_iter=it).fit(X, y)
    with pytest.raises(ValueError, match="multi_class"):
        LogisticRegression(multi_class="ovr").fit(X, y)
    with pytest.raises(TypeError, match="fit_intercept"):
        LogisticRegression(fit_intercept="yes").fit(X, y)
    with pytest.raises(ValueError, match="samples"):
        LogisticRegression().fit(X, y[:-1])
    with pytest.raises(ValueError, match="NaN"):
        LogisticRegression().fit(np.full_like(X, np.nan), y)
    with pytest.raises(NotFittedError):
        LogisticRegression().predict(X)
    m = LogisticRegression().fit(X, y)
    with pytest.raises(ValueError, match="features"):
        m.predict_proba(X[:, :2])


def _fit_peak_bytes(n, k=3, d=10):
    rng = np.random.default_rng(0)
    X, y = make_data(rng, n=n, d=d, k=k)
    tracemalloc.start()
    LogisticRegression(max_iter=30).fit(X, y)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak


@pytest.mark.filterwarnings("ignore::lipipy.core.ConvergenceWarning")
def test_fit_memory_is_linear_in_n_and_bounded_by_n_k():
    est = estimate_complexity(
        lambda X, y: LogisticRegression(max_iter=30).fit(X, y),
        lambda n, rng: make_data(rng, n=n, d=10, k=3),
        sizes=(4000, 8000, 16000, 32000),
        repeats=1,
    )
    assert est.space_exponent < 1.1
    # Nothing larger than O(n k): the peak stays within a few (n, k) float arrays
    # (Z, log-softmax, R, the gather index), never an (n, d) copy of X (d = 10 > k).
    n, k = 32000, 3
    assert _fit_peak_bytes(n, k=k, d=40) < 12 * n * k * 8


@pytest.mark.complexity
@pytest.mark.filterwarnings("ignore::lipipy.core.ConvergenceWarning")
def test_fit_time_is_linear_in_n():
    est = estimate_complexity(
        lambda X, y: LogisticRegression(max_iter=20, tol=1e-30).fit(X, y),
        lambda n, rng: make_data(rng, n=n, d=50, k=5),
        sizes=(20000, 40000, 80000, 160000),
        repeats=3,
    )
    assert est.time_exponent < 1.25
