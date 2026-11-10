r"""Empirical complexity estimation.

For input sizes :math:`n_1 < n_2 < \dots`, measure run time :math:`t(n)` and peak
extra memory :math:`m(n)`, then fit :math:`\log t = a + b\log n` by least squares.
The slope :math:`b` estimates the exponent in :math:`O(n^b)`; for example
:math:`b \approx 1` is linear, :math:`b\approx 2` quadratic. Log factors show up as a
slightly raised exponent (``n log n`` fits to about 1.1).

Timing takes the minimum over repeats (the least noisy statistic for a
deterministic function). Memory is measured in a separate run with
:mod:`tracemalloc`, which tracks Python and NumPy allocations made during the call.
Use sizes large enough that each call takes at least ~1 ms and allocates well
above ~150 KB; smaller costs are dominated by fixed measurement overhead and bias
the fitted exponent downwards.

Cost of the measurement itself: time :math:`O(R\sum_i t(n_i))` for ``R`` repeats.
"""

from __future__ import annotations

import gc
import time
import tracemalloc
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from lipipy.core.random import RandomState, check_random_state


@dataclass(frozen=True)
class ComplexityEstimate:
    """Result of :func:`estimate_complexity`."""

    sizes: tuple[int, ...]
    seconds: tuple[float, ...]
    peak_bytes: tuple[int, ...]
    time_exponent: float
    space_exponent: float

    def summary(self) -> str:
        return f"time ≈ O(n^{self.time_exponent:.2f}), space ≈ O(n^{self.space_exponent:.2f})"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_exponent(sizes: Sequence[float], values: Sequence[float], *, floor: float = 1e-12) -> float:
    """Slope of the least-squares line through ``(log n, log value)``.

    Values below ``floor`` are raised to it, so a constant ≈0 cost (e.g. no extra
    memory, with ``floor=1`` byte) gives an exponent near 0 instead of failing.

    Examples
    --------
    >>> round(fit_exponent([10, 100, 1000], [100, 10_000, 1_000_000]), 6)
    2.0
    """
    n = np.asarray(sizes, dtype=float)
    v = np.maximum(np.asarray(values, dtype=float), floor)
    if n.size < 2 or np.any(n <= 0):
        raise ValueError("need at least two positive sizes")
    slope, _ = np.polyfit(np.log(n), np.log(v), 1)
    return float(slope)


def _call(func: Callable[..., Any], inputs: Any) -> Any:
    return func(*inputs) if isinstance(inputs, tuple) else func(inputs)


def estimate_complexity(
    func: Callable[..., Any],
    make_input: Callable[[int, np.random.Generator], Any],
    sizes: Sequence[int] = (1_000, 2_000, 4_000, 8_000, 16_000),
    *,
    repeats: int = 5,
    random_state: RandomState = 0,
) -> ComplexityEstimate:
    """Measure how ``func``'s time and peak memory grow with input size.

    Parameters
    ----------
    func : callable
        The function under test.
    make_input : callable ``(n, rng) -> input``
        Builds the input of size ``n``. Return a tuple to pass several positional
        arguments. Input creation is excluded from both measurements.
    sizes : sequence of int
        Increasing input sizes (at least two; five or more recommended).
    repeats : int
        Timing repeats per size; the minimum is kept.
    random_state : int, Generator or None
        Seed for ``make_input``.

    Examples
    --------
    >>> import numpy as np
    >>> est = estimate_complexity(
    ...     lambda x: np.outer(x, x), lambda n, rng: rng.random(n), sizes=(200, 400, 800, 1600)
    ... )
    >>> round(est.space_exponent)
    2
    """
    sizes = tuple(int(s) for s in sizes)
    if len(sizes) < 2 or list(sizes) != sorted(set(sizes)):
        raise ValueError("sizes must contain at least two strictly increasing values")
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    rng = check_random_state(random_state)

    seconds, peaks = [], []
    for n in sizes:
        inputs = make_input(n, rng)
        _call(func, inputs)  # warm-up (caches, lazy imports)

        best = float("inf")
        for _ in range(repeats):
            gc.collect()
            start = time.perf_counter()
            _call(func, inputs)
            best = min(best, time.perf_counter() - start)
        seconds.append(best)

        gc.collect()
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            base, _ = tracemalloc.get_traced_memory()
            _call(func, inputs)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        peaks.append(max(int(peak - base), 0))

    return ComplexityEstimate(
        sizes=sizes,
        seconds=tuple(seconds),
        peak_bytes=tuple(peaks),
        time_exponent=fit_exponent(sizes, seconds),
        space_exponent=fit_exponent(sizes, peaks, floor=1.0),
    )
