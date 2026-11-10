"""Tests for lipipy.bench.estimate_complexity.

Memory exponents are deterministic and run on every PR. Timing exponents depend on
the machine, so those tests carry the ``complexity`` marker (``make complexity``).
"""

import numpy as np
import pytest

from lipipy.bench import estimate_complexity, fit_exponent
from lipipy.ml import LinearRegression


def test_fit_exponent_exact_power_laws():
    n = [10, 100, 1000]
    assert fit_exponent(n, [3 * x for x in n]) == pytest.approx(1.0)
    assert fit_exponent(n, [x**3 for x in n]) == pytest.approx(3.0)
    assert fit_exponent(n, [0, 0, 0], floor=1.0) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        fit_exponent([10], [1])


def test_input_validation():
    with pytest.raises(ValueError, match="increasing"):
        estimate_complexity(len, lambda n, rng: [0] * n, sizes=(10, 5))
    with pytest.raises(ValueError, match="repeats"):
        estimate_complexity(len, lambda n, rng: [0] * n, sizes=(5, 10), repeats=0)


def test_space_exponents():
    vec = lambda n, rng: rng.random(n)  # noqa: E731
    vec32 = lambda n, rng: rng.random(n, dtype=np.float32)  # noqa: E731
    # sizes large enough that the ~130 KB fixed measurement overhead is negligible
    quadratic = estimate_complexity(lambda x: np.outer(x, x), vec32, sizes=(400, 800, 1600, 3200))
    linear = estimate_complexity(lambda x: x * 2.0, vec, sizes=(10**4, 4 * 10**4, 16 * 10**4))
    constant = estimate_complexity(lambda x: x.sum(), vec, sizes=(10**4, 4 * 10**4, 16 * 10**4))
    assert quadratic.space_exponent == pytest.approx(2.0, abs=0.15)
    assert linear.space_exponent == pytest.approx(1.0, abs=0.15)
    assert constant.space_exponent == pytest.approx(0.0, abs=0.15)
    assert quadratic.summary().startswith("time ≈ O(n^")
    assert quadratic.to_dict()["sizes"] == (400, 800, 1600, 3200)


def test_linear_regression_memory_is_linear_in_n():
    """Declared space O(n d): doubling n should double extra memory."""
    est = estimate_complexity(
        lambda X, y: LinearRegression().fit(X, y),
        lambda n, rng: (rng.normal(size=(n, 8)), rng.normal(size=n)),
        sizes=(4_000, 8_000, 16_000, 32_000),
        repeats=1,
    )
    assert est.space_exponent == pytest.approx(1.0, abs=0.2)


@pytest.mark.complexity
def test_time_exponents():
    vec = lambda n, rng: rng.random(n)  # noqa: E731
    sizes = (2 * 10**5, 4 * 10**5, 8 * 10**5, 16 * 10**5)
    sort = estimate_complexity(np.sort, vec, sizes=sizes)
    assert sort.time_exponent == pytest.approx(1.05, abs=0.35)
    # Direct convolution is compute-bound O(n^2) with O(n) memory, so its timing is
    # stable. (np.outer at these sizes is memory-bandwidth bound and under-reads, ~1.6.)
    quad = estimate_complexity(lambda x: np.convolve(x, x), vec, sizes=(2000, 4000, 8000, 16000))
    assert quad.time_exponent == pytest.approx(2.0, abs=0.25)


@pytest.mark.complexity
def test_linear_regression_time_is_linear_in_n():
    est = estimate_complexity(
        lambda X, y: LinearRegression().fit(X, y),
        lambda n, rng: (rng.normal(size=(n, 16)), rng.normal(size=n)),
        sizes=(20_000, 40_000, 80_000, 160_000),
    )
    assert est.time_exponent < 1.35
