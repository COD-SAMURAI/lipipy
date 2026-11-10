"""Layer 1: measurement tools.

Measure how an algorithm's run time and memory grow with input size, to check the
complexity it declares in the registry (see ARCHITECTURE.md §6.2).

Planned
-------
[x] Empirical time/space exponent estimation (``estimate_complexity``)
[ ] ``pytest -m complexity`` helpers that compare against registered ``Complexity``
[ ] Benchmark runner over registry entries with JSON/Markdown result tables
[ ] Backend comparison (NumPy vs CuPy vs PyTorch vs JAX) for the same algorithm
"""

from lipipy.bench.complexity import ComplexityEstimate, estimate_complexity, fit_exponent

__all__ = ["ComplexityEstimate", "estimate_complexity", "fit_exponent"]
