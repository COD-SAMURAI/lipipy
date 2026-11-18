"""SVC: parity with sklearn.svm.SVC (LIBSVM), KKT conditions, degenerate cases.

LIBSVM keeps kernel values in float32, so its decision values carry relative
errors of about 1e-7 of the kernel scale; agreement is checked to 1e-5 relative to
the largest decision value, and predictions must be identical.
"""

import warnings

import numpy as np
import pytest

from lipipy.bench import estimate_complexity
from lipipy.core import ConvergenceWarning, NotFittedError
from lipipy.ml import SVC

PSD_KERNELS = ["linear", "poly", "rbf"]


def make_data(seed, n=80, d=3, k=2, shift=0.9):
    rng = np.random.default_rng(seed)
    y = rng.integers(k, size=n)
    X = rng.normal(size=(n, d)) + shift * y[:, None] * rng.uniform(0.5, 1.5, size=d)
    X_test = rng.normal(scale=1.5, size=(40, d)) + shift
    return X, y, X_test


def kernel_kwargs(kernel):
    if kernel == "poly":
        return {"degree": 3, "coef0": 0.5}
    if kernel == "sigmoid":
        return {"gamma": 0.1, "coef0": 0.0}
    return {}


def assert_decisions_close(ours, ref):
    scale = max(1.0, float(np.max(np.abs(ref))))
    np.testing.assert_allclose(ours, ref, rtol=0, atol=1e-5 * scale)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("kernel", PSD_KERNELS)
@pytest.mark.parametrize("k", [2, 3, 4])
def test_parity_with_sklearn(seed, kernel, k):
    sk_svm = pytest.importorskip("sklearn.svm")
    X, y, X_test = make_data(seed, k=k)
    kw = dict(C=1.5, kernel=kernel, tol=1e-10, **kernel_kwargs(kernel))
    ours = SVC(**kw).fit(X, y)
    ref = sk_svm.SVC(**kw).fit(X, y)
    np.testing.assert_array_equal(ours.predict(X_test), ref.predict(X_test))
    np.testing.assert_array_equal(ours.predict(X), ref.predict(X))
    assert_decisions_close(ours.decision_function(X_test), ref.decision_function(X_test))
    assert_decisions_close(ours.intercept_, ref.intercept_)
    assert ours.dual_coef_.shape[0] == ref.dual_coef_.shape[0] == k - 1
    assert ours.n_support_.sum() == len(ours.support_) == len(ours.support_vectors_)


@pytest.mark.parametrize("seed", range(8))
def test_decision_function_shape_ovo_parity(seed):
    sk_svm = pytest.importorskip("sklearn.svm")
    X, y, X_test = make_data(seed, k=4)
    kw = dict(kernel="rbf", tol=1e-10, decision_function_shape="ovo")
    ours = SVC(**kw).fit(X, y).decision_function(X_test)
    ref = sk_svm.SVC(**kw).fit(X, y).decision_function(X_test)
    assert ours.shape == (len(X_test), 6)
    assert_decisions_close(ours, ref)


@pytest.mark.parametrize("seed", range(20))
def test_sigmoid_kernel_binary_parity(seed):
    """The sigmoid kernel is not PSD, so the dual is non-convex and SMO finds a KKT
    point; for binary problems we and LIBSVM follow the same path to the same point."""
    sk_svm = pytest.importorskip("sklearn.svm")
    X, y, X_test = make_data(seed, n=60)
    kw = dict(kernel="sigmoid", tol=1e-10, **kernel_kwargs("sigmoid"))
    ours, ref = SVC(**kw).fit(X, y), sk_svm.SVC(**kw).fit(X, y)
    assert_decisions_close(ours.decision_function(X_test), ref.decision_function(X_test))
    np.testing.assert_array_equal(ours.predict(X_test), ref.predict(X_test))


def binary_alpha(model, n):
    """Multipliers alpha_i per training sample of a binary model (0 for non-SVs)."""
    alpha = np.zeros(n)
    alpha[model.support_] = np.abs(model.dual_coef_[0])
    return alpha


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("kernel", PSD_KERNELS + ["sigmoid"])
@pytest.mark.parametrize("C", [0.1, 1.0, 100.0])
def test_kkt_conditions_hold(seed, kernel, C):
    X, y, _ = make_data(seed)
    tol = 1e-6
    model = SVC(C=C, kernel=kernel, tol=tol, **kernel_kwargs(kernel)).fit(X, y)
    s = np.where(y == model.classes_[1], 1.0, -1.0)
    alpha = binary_alpha(model, len(y))
    margin = s * model.decision_function(X)
    assert np.all(alpha >= 0) and np.all(alpha <= C)
    assert abs(float(np.sum(s * alpha))) <= 1e-9 * max(1.0, C * len(y))
    eps = 2 * tol
    assert np.all(margin[alpha == 0] >= 1 - eps)
    assert np.all(margin[alpha == C] <= 1 + eps)
    free = (alpha > 0) & (alpha < C)
    np.testing.assert_allclose(margin[free], 1.0, atol=eps)


@pytest.mark.parametrize("seed", range(15))
@pytest.mark.parametrize("kernel", PSD_KERNELS)
def test_support_sets_equal_when_non_degenerate(seed, kernel):
    """With strict complementarity (no multiplier at a bound with margin exactly 1)
    the support set is unique, so it must equal LIBSVM's."""
    sk_svm = pytest.importorskip("sklearn.svm")
    X, y, _ = make_data(seed)
    kw = dict(C=1.0, kernel=kernel, tol=1e-10, **kernel_kwargs(kernel))
    ours, ref = SVC(**kw).fit(X, y), sk_svm.SVC(**kw).fit(X, y)
    s = np.where(y == ours.classes_[1], 1.0, -1.0)
    alpha = binary_alpha(ours, len(y))
    margin = s * ours.decision_function(X)
    delta = 1e-4
    strict = np.where(
        alpha == 0,
        margin > 1 + delta,
        np.where(alpha >= 1.0, margin < 1 - delta, (alpha > delta) & (alpha < 1 - delta)),
    )
    if not strict.all():
        pytest.skip("degenerate solution: the support set is not unique")
    np.testing.assert_array_equal(ours.support_, ref.support_)
    np.testing.assert_allclose(ours.dual_coef_, ref.dual_coef_, atol=1e-5)


def test_non_degenerate_case_is_exercised():
    """Guard for the test above: most seeds are non-degenerate."""
    count = 0
    for seed in range(15):
        X, y, _ = make_data(seed)
        m = SVC(kernel="rbf", tol=1e-10).fit(X, y)
        s = np.where(y == m.classes_[1], 1.0, -1.0)
        alpha = binary_alpha(m, len(y))
        margin = s * m.decision_function(X)
        ok = np.where(
            alpha == 0,
            margin > 1 + 1e-4,
            np.where(alpha >= 1.0, margin < 1 - 1e-4, (alpha > 1e-4) & (alpha < 1 - 1e-4)),
        )
        count += bool(ok.all())
    assert count >= 5


@pytest.mark.filterwarnings("error")
def test_separable_data_with_huge_C_is_hard_margin():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(size=(30, 2)) - 4, rng.normal(size=(30, 2)) + 4])
    y = np.repeat([0, 1], 30)
    m = SVC(kernel="linear", C=1e6, tol=1e-8).fit(X, y)
    margin = np.where(y == 1, 1.0, -1.0) * m.decision_function(X)
    assert margin.min() == pytest.approx(1.0, abs=1e-6)  # support vectors on the margin
    assert np.all(binary_alpha(m, 60) < 1e6)  # no margin violations
    w = m.dual_coef_[0] @ m.support_vectors_
    hard = np.where(y == 1, 1, -1) * (X @ w + m.intercept_[0])
    assert hard.min() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.filterwarnings("error")
def test_duplicate_points_give_same_decision_values():
    sk_svm = pytest.importorskip("sklearn.svm")
    X, y, X_test = make_data(3, n=40)
    X2, y2 = np.vstack([X, X[:10]]), np.concatenate([y, y[:10]])
    ours = SVC(kernel="rbf", tol=1e-10).fit(X2, y2)
    ref = sk_svm.SVC(kernel="rbf", tol=1e-10).fit(X2, y2)
    # multipliers of duplicates are not unique; the decision function is
    assert_decisions_close(ours.decision_function(X_test), ref.decision_function(X_test))


@pytest.mark.filterwarnings("error")
def test_tiny_cache_gives_same_model():
    X, y, X_test = make_data(1, n=120, k=3)
    big = SVC(tol=1e-8).fit(X, y)
    small = SVC(tol=1e-8, cache_size=1e-6).fit(X, y)  # 2 columns
    np.testing.assert_array_equal(big.support_, small.support_)
    np.testing.assert_array_equal(big.dual_coef_, small.dual_coef_)


def test_string_labels_and_binary_sign_convention():
    X, y, X_test = make_data(2)
    names = np.array(["neg", "pos"])[y]
    m = SVC(kernel="linear").fit(X, names)
    d = m.decision_function(X_test)
    np.testing.assert_array_equal(m.predict(X_test), np.where(d > 0, "pos", "neg"))
    assert m.score(X, names) > 0.7


def test_gamma_options():
    X, y, _ = make_data(0)
    assert SVC(gamma="auto").fit(X, y)._gamma == pytest.approx(1 / 3)
    assert SVC().fit(X, y)._gamma == pytest.approx(1 / (3 * X.var()))
    const = np.ones((6, 2))
    const[3:] = 1.0  # zero variance: gamma = 1
    m = SVC().fit(np.vstack([const, const + 1]), [0] * 6 + [1] * 6)
    assert m._gamma == pytest.approx(1 / (2 * np.vstack([const, const + 1]).var()))
    with pytest.raises(ValueError, match="gamma"):
        SVC(gamma=-1.0).fit(X, y)


def test_max_iter_warns():
    X, y, _ = make_data(0)
    with pytest.warns(ConvergenceWarning, match="max_iter"):
        SVC(max_iter=2).fit(X, y)


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"C": 0.0}, "C"),
        ({"kernel": "cosine"}, "kernel"),
        ({"degree": -1}, "degree"),
        ({"tol": 0.0}, "tol"),
        ({"max_iter": 0}, "max_iter"),
        ({"cache_size": 0}, "cache_size"),
        ({"decision_function_shape": "ova"}, "decision_function_shape"),
    ],
)
def test_bad_params(params, match):
    X, y, _ = make_data(0)
    with pytest.raises(ValueError, match=match):
        SVC(**params).fit(X, y)


def test_failed_fit_leaves_estimator_unchanged():
    X, y, _ = make_data(0)
    m = SVC().fit(X, y)
    before = m.dual_coef_.copy()
    with pytest.raises(ValueError):
        m.set_params(C=-1.0).fit(X[:10], y[:10])
    np.testing.assert_array_equal(m.dual_coef_, before)
    with pytest.raises(ValueError, match="2 classes"):
        SVC().fit(X, np.zeros(len(X)))
    fresh = SVC()
    with pytest.raises(ValueError):
        fresh.fit(X, np.zeros(len(X)))
    assert not hasattr(fresh, "support_")


def test_unfitted_and_feature_mismatch():
    X, y, _ = make_data(0)
    with pytest.raises(NotFittedError):
        SVC().predict(X)
    m = SVC().fit(X, y)
    with pytest.raises(ValueError, match="features"):
        m.decision_function(X[:, :2])


def test_no_warnings_on_default_fit():
    X, y, _ = make_data(5, k=3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SVC().fit(X, y).predict(X)


def test_kernel_cache_bounds_memory():
    """With a tiny cache only two kernel columns are kept: memory O(n d), not O(n^2)."""

    def make(n, rng):
        y = rng.integers(2, size=n)
        return rng.normal(size=(n, 2)) + 2.0 * y[:, None], y

    def fit(X, y):
        return SVC(kernel="rbf", cache_size=1e-6, tol=1e-2, max_iter=200).fit(X, y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        est = estimate_complexity(fit, make, sizes=(1000, 2000, 4000, 8000), repeats=1)
    assert est.space_exponent < 1.3, est.summary()
