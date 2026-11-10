"""Reproducible randomness.

lipipy never uses global random state. Every stochastic API accepts
``random_state`` and normalises it with :func:`check_random_state`.
"""

from __future__ import annotations

import numbers

import numpy as np

RandomState = None | int | np.random.SeedSequence | np.random.Generator


def check_random_state(seed: RandomState = None) -> np.random.Generator:
    """Turn ``seed`` into a :class:`numpy.random.Generator`.

    ``None`` gives fresh OS entropy, an int or ``SeedSequence`` gives a seeded
    generator, and a ``Generator`` is returned as is (so state is shared).

    Examples
    --------
    >>> a = check_random_state(0).normal(size=3)
    >>> b = check_random_state(0).normal(size=3)
    >>> bool((a == b).all())
    True
    """
    if seed is None or isinstance(seed, np.random.SeedSequence):
        return np.random.default_rng(seed)
    if isinstance(seed, np.random.Generator):
        return seed
    if isinstance(seed, numbers.Integral) and not isinstance(seed, bool):
        return np.random.default_rng(int(seed))
    raise TypeError(
        f"random_state must be None, an int, a SeedSequence or a Generator; got {type(seed)!r}"
    )


def spawn(seed: RandomState, n: int) -> list[np.random.Generator]:
    """Create ``n`` statistically independent generators for parallel work.

    Results are reproducible and do not depend on how the work is scheduled.

    Examples
    --------
    >>> [g.integers(100) for g in spawn(42, 3)] == [g.integers(100) for g in spawn(42, 3)]
    True
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    children = check_random_state(seed).bit_generator.seed_seq.spawn(n)
    return [np.random.default_rng(child) for child in children]
