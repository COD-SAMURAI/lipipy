"""lipipy.numerics.stats: streaming moments, covariance, correlation."""

import numpy as np
import pytest

from lipipy.bench import estimate_complexity
from lipipy.numerics import RunningMoments, correlation, covariance


def feed(data, batch_sizes):
    rm = RunningMoments()
    start = 0
    for size in batch_sizes:
        rm.update(data[start : start + size])
        start += size
    assert start == len(data)
    return rm


@pytest.mark.parametrize("batches", [[1000], [1] * 50 + [950], [333, 1, 666], [7] * 142 + [6]])
def test_matches_numpy_2d(rng, batches):
    X = rng.normal(size=(1000, 4)) * [1.0, 10.0, 0.1, 3.0] + [5.0, -2.0, 0.0, 1e3]
    rm = feed(X, batches)
    assert rm.count == 1000
    np.testing.assert_allclose(rm.mean, X.mean(axis=0), rtol=1e-12)
    for ddof in (0, 1):
        np.testing.assert_allclose(rm.variance(ddof), X.var(axis=0, ddof=ddof), rtol=1e-12)
        np.testing.assert_allclose(rm.std(ddof), X.std(axis=0, ddof=ddof), rtol=1e-12)


def test_scalar_stream_one_at_a_time(rng):
    x = rng.exponential(size=500)
    rm = RunningMoments()
    for value in x:
        rm.update(value)  # pure Welford updates
    assert rm.mean.shape == ()
    assert rm.mean == pytest.approx(x.mean(), rel=1e-12)
    assert rm.variance(ddof=1) == pytest.approx(x.var(ddof=1), rel=1e-12)


def test_huge_offset_where_naive_formula_fails(rng):
    x = 1e9 + rng.normal(size=100_000)
    rm = RunningMoments()
    for chunk in np.array_split(x, 37):
        rm.update(chunk)
    reference = np.var(x - 1e9)  # shift is exact for doubles near 1e9
    assert rm.variance() == pytest.approx(reference, rel=1e-12)
    assert rm.variance() == pytest.approx(np.var(x), rel=1e-12)
    assert rm.mean == pytest.approx(np.mean(x), rel=1e-15)
    naive = np.mean(x**2) - np.mean(x) ** 2
    assert abs(naive - reference) / reference > 1e-3  # the textbook formula is wrong


def test_merge_equals_concatenation(rng):
    X = rng.normal(size=(900, 3)) + 1e6
    shards = [X[:100], X[100:101], X[101:900]]
    parts = []
    for shard in shards:
        rm = RunningMoments()
        rm.update(shard)
        parts.append(rm)
    total = RunningMoments().merge(parts[0]).merge(parts[1]).merge(parts[2])
    total.merge(RunningMoments())  # merging an empty summary is a no-op
    assert total.count == 900
    np.testing.assert_allclose(total.mean, X.mean(axis=0), rtol=1e-14)
    np.testing.assert_allclose(total.variance(1), X.var(axis=0, ddof=1), rtol=1e-11)


def test_to_dict_and_repr():
    rm = RunningMoments()
    assert rm.to_dict() == {"count": 0, "mean": None, "variance": None}
    rm.update([[1.0, 2.0], [3.0, 4.0]])
    assert rm.to_dict() == {"count": 2, "mean": [2.0, 3.0], "variance": [1.0, 1.0]}
    assert "count=2" in repr(rm)


def test_rejected_multi_chunk_batch_leaves_state_unchanged(rng):
    rm = RunningMoments()
    rm.update(np.array([1.0, 2.0, 4.0]))
    before = (rm.count, rm.mean, rm.variance())
    bad = rng.normal(size=5000)  # spans two 4096-row chunks; NaN is in the second
    bad[4500] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        rm.update(bad)
    assert (rm.count, rm.mean, rm.variance()) == before
    X = np.ones((5000, 2))
    X[4999, 1] = np.inf
    rm2 = RunningMoments()
    rm2.update([[1.0, 2.0], [3.0, 5.0]])
    with pytest.raises(ValueError, match="NaN"):
        rm2.update(X)
    assert rm2.count == 2
    np.testing.assert_array_equal(rm2.mean, [2.0, 3.5])


def test_rejected_first_batch_leaves_object_pristine():
    rm = RunningMoments()
    with pytest.raises(ValueError, match="NaN"):
        rm.update([[1.0, 2.0], [np.nan, 3.0]])
    assert rm.count == 0
    assert rm.to_dict() == {"count": 0, "mean": None, "variance": None}
    rm.update([1.0, 2.0, 3.0])  # a different feature shape is still accepted
    assert rm.mean.shape == ()
    assert float(rm.mean) == 2.0


def test_running_moments_errors():
    rm = RunningMoments()
    with pytest.raises(ValueError, match="no observations"):
        _ = rm.mean
    with pytest.raises(ValueError, match="no observations"):
        rm.variance()
    rm.update([[1.0, 2.0]])
    with pytest.raises(ValueError, match="ddof"):
        rm.variance(ddof=1)
    with pytest.raises(ValueError, match="ddof"):
        rm.variance(ddof=-1)
    with pytest.raises(ValueError, match="feature shape"):
        rm.update([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="NaN"):
        rm.update([[np.nan, 1.0]])
    with pytest.raises(ValueError, match="dimensional"):
        rm.update(np.ones((2, 2, 2)))
    with pytest.raises(TypeError):
        rm.merge([1.0])
    other = RunningMoments()
    other.update([1.0])
    with pytest.raises(ValueError, match="merge"):
        rm.merge(other)


def test_running_moments_update_memory_constant_in_n():
    """Declared extra space O(d): independent of the batch length n."""
    d = 8

    def run(X):
        RunningMoments().update(X)

    est = estimate_complexity(
        run,
        lambda n, rng: rng.normal(size=(n, d)),
        sizes=(20_000, 40_000, 80_000, 160_000),
        repeats=1,
    )
    assert est.space_exponent == pytest.approx(0.0, abs=0.15)
    assert max(est.peak_bytes) < 3 * 4096 * d * 8  # a few chunk-sized temporaries


@pytest.mark.complexity
def test_running_moments_update_time_is_linear_in_n():
    est = estimate_complexity(
        lambda X: RunningMoments().update(X),
        lambda n, rng: rng.normal(size=(n, 8)),
        sizes=(50_000, 100_000, 200_000, 400_000),
    )
    assert est.time_exponent < 1.35


# -- covariance / correlation ---------------------------------------------------
@pytest.mark.parametrize("ddof", [0, 1, 2])
def test_covariance_matches_numpy(rng, ddof):
    X = rng.normal(size=(50, 5)) @ rng.normal(size=(5, 5)) + 100.0
    np.testing.assert_allclose(
        covariance(X, ddof=ddof), np.cov(X, rowvar=False, ddof=ddof), rtol=1e-12, atol=1e-12
    )


def test_correlation_matches_numpy(rng):
    X = rng.normal(size=(80, 6)) @ rng.normal(size=(6, 6))
    R = correlation(X)
    np.testing.assert_allclose(R, np.corrcoef(X, rowvar=False), rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(R, R.T)
    assert np.all(np.abs(R) <= 1.0)


def test_correlation_constant_column_is_nan():
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    R = correlation(X)
    assert R[0, 0] == pytest.approx(1.0)
    assert np.all(np.isnan(R[1]))


def test_covariance_errors():
    with pytest.raises(ValueError, match="2D"):
        covariance([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="ddof"):
        covariance([[1.0, 2.0]], ddof=1)
    with pytest.raises(ValueError, match="ddof"):
        covariance([[1.0], [2.0]], ddof=-1)
    with pytest.raises(ValueError, match="NaN"):
        correlation([[1.0, np.nan], [2.0, 3.0]])
