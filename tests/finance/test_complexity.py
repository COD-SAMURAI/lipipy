"""Empirical checks of the declared finance complexities (lipipy.bench.estimate_complexity).

Memory exponents are deterministic and run on every PR; timing exponents carry the
``complexity`` marker (``make complexity``) and use compute-bound sizes.
"""

import numpy as np
import pytest

from lipipy.bench import estimate_complexity
from lipipy.finance import backtest, binomial_price, ledoit_wolf, monte_carlo_price


def _mc(n):
    return monte_carlo_price(
        100.0, 100.0, 1.0, 0.05, 0.2, n_paths=int(n), random_state=0, chunk_size=8192
    )


def test_monte_carlo_memory_is_independent_of_paths():
    """Declared space O(chunk_size): 8x more paths, same peak memory."""
    est = estimate_complexity(
        _mc, lambda n, rng: n, sizes=(100_000, 200_000, 400_000, 800_000), repeats=1
    )
    assert est.space_exponent == pytest.approx(0.0, abs=0.1)
    assert max(est.peak_bytes) < 2_000_000  # a few chunk-sized arrays, not P floats


def test_binomial_memory_is_linear_in_steps():
    """Declared space O(N): the node grid, the value array and one scratch array.

    The peak includes a fixed measurement overhead, so linearity is checked on the
    increments: each doubling of N adds twice the memory of the previous one
    (a quadratic cost would add four times as much).
    """
    est = estimate_complexity(
        lambda n: binomial_price(100.0, 100.0, 1.0, 0.05, 0.2, steps=n, american=True),
        lambda n, rng: n,
        sizes=(2_000, 4_000, 8_000),
        repeats=1,
    )
    p = est.peak_bytes
    assert (p[2] - p[1]) / (p[1] - p[0]) == pytest.approx(2.0, abs=0.2)
    assert p[2] < 5 * 8 * 8_000  # grid (2N + 1), values (N + 1) and scratch (N) floats


def test_backtest_memory_is_linear_in_time():
    est = estimate_complexity(
        lambda P: backtest(P, np.full(8, 0.125), rebalance=20),
        lambda n, rng: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(n, 8)), axis=0)),
        sizes=(5_000, 10_000, 20_000, 40_000),
        repeats=1,
    )
    assert est.space_exponent == pytest.approx(1.0, abs=0.15)


def test_ledoit_wolf_memory_is_linear_in_time():
    """Declared space O(T A + A^2): with A fixed, linear in T."""
    est = estimate_complexity(
        ledoit_wolf,
        lambda n, rng: rng.normal(size=(n, 10)),
        sizes=(5_000, 10_000, 20_000, 40_000),
        repeats=1,
    )
    assert est.space_exponent == pytest.approx(1.0, abs=0.15)


@pytest.mark.complexity
def test_binomial_time_is_quadratic_in_steps():
    est = estimate_complexity(
        lambda n: binomial_price(100.0, 100.0, 1.0, 0.05, 0.2, steps=n),
        lambda n, rng: n,
        sizes=(8_000, 16_000, 32_000),
        repeats=3,
    )
    # per-step NumPy call overhead pulls the fit slightly below 2 at moderate N
    assert est.time_exponent == pytest.approx(2.0, abs=0.4)


@pytest.mark.complexity
def test_monte_carlo_time_is_linear_in_paths():
    est = estimate_complexity(_mc, lambda n, rng: n, sizes=(400_000, 800_000, 1_600_000, 3_200_000))
    assert est.time_exponent == pytest.approx(1.0, abs=0.2)


@pytest.mark.complexity
def test_backtest_time_is_linear_in_time_steps():
    est = estimate_complexity(
        lambda P: backtest(P, np.full(16, 1 / 16)),
        lambda n, rng: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(n, 16)), axis=0)),
        sizes=(2_500, 5_000, 10_000, 20_000),
        repeats=3,
    )
    assert est.time_exponent == pytest.approx(1.0, abs=0.25)
