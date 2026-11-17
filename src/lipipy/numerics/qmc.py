r"""Quasi-Monte Carlo point sets: Halton and Sobol sequences, L2-star discrepancy.

Low-discrepancy sequences fill :math:`[0, 1)^d` more evenly than independent
uniform points, so quasi-Monte Carlo integration errors decay like
:math:`O((\log n)^d / n)` instead of :math:`O(n^{-1/2})` (Koksma-Hlawka inequality).

Halton
------
Point :math:`i` (starting at :math:`i = 0`) has coordinate :math:`j` equal to the
radical inverse of :math:`i` in the :math:`j`-th prime base :math:`b_j`: if
:math:`i = \sum_k a_k b^k`, then :math:`\phi_b(i) = \sum_k a_k b^{-k-1}` (the digits
mirrored about the radix point). With ``scramble=True`` every digit position
:math:`k` of every dimension gets an independent random permutation
:math:`\pi_{j,k}` of :math:`\{0, \dots, b_j - 1\}`, and
:math:`x_{ij} = \sum_{k<K} \pi_{j,k}(a_k) b_j^{-k-1}` over the
:math:`K = \lfloor 52/\log_2 b_j\rfloor` digits resolvable in double precision
(random digit permutation scrambling; it breaks the strong correlations between
high dimensions of the plain sequence).

Sobol
-----
Coordinate :math:`j` uses direction numbers :math:`v_{j,1}, v_{j,2},\dots` (32-bit
integers) generated from a primitive polynomial of degree :math:`s_j` with
coefficients :math:`a_j` and initial values :math:`m_{j,1..s_j}`:

.. math::
    v_{j,k} = v_{j,k-s} \oplus (v_{j,k-s} \gg s) \oplus
              \bigoplus_{l=1}^{s-1} c_l\, v_{j,k-l},

where :math:`c_l` are the bits of :math:`a_j`. We use the Joe-Kuo
``new-joe-kuo-6.21201`` parameters for dimensions 2-21 (dimension 1 is the van der
Corput sequence in base 2). Points are generated in Gray-code order,
:math:`x_i = \bigoplus_{k:\ g_k(i)=1} v_k \cdot 2^{-32}` with
:math:`g(i) = i \oplus (i \gg 1)`, which equals the classical recursive
construction :math:`x_{i+1} = x_i \oplus v_{c(i)}`.

**First-point convention:** both sequences start at index 0, so the first point is
the origin :math:`(0, \dots, 0)` (unscrambled), exactly as
``scipy.stats.qmc.Sobol(scramble=False)`` / ``Halton(scramble=False)``. The Sobol
net properties (every elementary box of volume :math:`2^{-m}` holds the right number
of points) hold for the first :math:`n = 2^m` points; drop or keep the origin
consistently when comparing with other software.

L2-star discrepancy
-------------------
Warnock's closed form for :math:`D_2^*(P) = \bigl(\int_{[0,1]^d}
(\tfrac{\#\{x_i \in [0, u)\}}{n} - \prod_k u_k)^2 du\bigr)^{1/2}`:

.. math::
    (D_2^*)^2 = 3^{-d} - \frac{2^{1-d}}{n}\sum_i \prod_k (1 - x_{ik}^2)
              + \frac{1}{n^2}\sum_{i,j}\prod_k (1 - \max(x_{ik}, x_{jk})).

For i.i.d. uniform points :math:`\mathbb E[(D_2^*)^2] = (2^{-d} - 3^{-d})/n`.

**Stated comparison (tested):** for :math:`d \in \{2, 3, 5\}` and
:math:`n \in \{256, 1024\}`, the Sobol points, the unscrambled Halton points and
scrambled Halton points (10 seeds) all have a smaller :math:`D_2^*` than the *best*
of 50 independent uniform point sets of the same size. This is a low-dimensional
statement: :math:`D_2^*` puts most weight near the origin, and at these :math:`n`
the unscrambled sequences (which contain the origin and have weak high-dimensional
projections) can be *worse* than random points for :math:`d \gtrsim 10`.

References
----------
J. H. Halton, "On the efficiency of certain quasi-random sequences of points in
evaluating multi-dimensional integrals", *Numer. Math.* 2 (1960) 84-90.
I. M. Sobol', "On the distribution of points in a cube and the approximate
evaluation of integrals", *USSR Comput. Math. Math. Phys.* 7 (1967) 86-112.
S. Joe & F. Y. Kuo, "Constructing Sobol sequences with better two-dimensional
projections", *SIAM J. Sci. Comput.* 30 (2008) 2635-2654.
T. T. Warnock, "Computational investigations of low-discrepancy point sets" (1972).
H. Niederreiter, *Random Number Generation and Quasi-Monte Carlo Methods*, SIAM (1992).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from lipipy.core.complexity import Complexity
from lipipy.core.random import RandomState, check_random_state
from lipipy.core.registry import register
from lipipy.numerics._common import as_real, check_positive_int

__all__ = ["SOBOL_MAX_DIM", "halton", "l2_star_discrepancy", "sobol"]

# Joe-Kuo new-joe-kuo-6.21201, dimensions 2..21: (degree s, coefficients a, m_1..m_s).
_JOE_KUO = (
    (1, 0, (1,)),
    (2, 1, (1, 3)),
    (3, 1, (1, 3, 1)),
    (3, 2, (1, 1, 1)),
    (4, 1, (1, 1, 3, 3)),
    (4, 4, (1, 3, 5, 13)),
    (5, 2, (1, 1, 5, 5, 17)),
    (5, 4, (1, 1, 5, 5, 5)),
    (5, 7, (1, 1, 7, 11, 19)),
    (5, 11, (1, 1, 5, 1, 1)),
    (5, 13, (1, 1, 1, 3, 11)),
    (5, 14, (1, 3, 5, 5, 31)),
    (6, 1, (1, 3, 3, 9, 7, 49)),
    (6, 13, (1, 1, 1, 15, 21, 21)),
    (6, 16, (1, 3, 1, 13, 27, 49)),
    (6, 19, (1, 1, 1, 15, 7, 5)),
    (6, 22, (1, 3, 1, 15, 13, 25)),
    (6, 25, (1, 1, 5, 5, 19, 61)),
    (7, 1, (1, 3, 7, 11, 23, 15, 103)),
    (7, 4, (1, 3, 7, 13, 13, 15, 69)),
)
_BITS = 32
#: Number of Sobol dimensions supported (Joe-Kuo parameters included).
SOBOL_MAX_DIM = len(_JOE_KUO) + 1


def _direction_numbers() -> np.ndarray:
    """(SOBOL_MAX_DIM, 32) uint64 array; row j holds v_{j,1..32} scaled to 32 bits."""
    V = np.zeros((SOBOL_MAX_DIM, _BITS), dtype=np.uint64)
    V[0] = [1 << (_BITS - 1 - k) for k in range(_BITS)]
    for j, (s, a, m) in enumerate(_JOE_KUO, start=1):
        v = [0] * (_BITS + 1)  # 1-based
        for k in range(1, s + 1):
            v[k] = m[k - 1] << (_BITS - k)
        for k in range(s + 1, _BITS + 1):
            val = v[k - s] ^ (v[k - s] >> s)
            for ell in range(1, s):
                if (a >> (s - 1 - ell)) & 1:
                    val ^= v[k - ell]
            v[k] = val
        V[j] = v[1:]
    V.flags.writeable = False
    return V


_V = _direction_numbers()


def _first_primes(d: int) -> list[int]:
    primes: list[int] = []
    candidate = 2
    while len(primes) < d:
        if all(candidate % p for p in primes if p * p <= candidate):
            primes.append(candidate)
        candidate += 1
    return primes


@register(
    domain="numerics",
    tasks=("quasi-monte-carlo", "sampling"),
    complexity=Complexity(
        time="O(n d log n)", space="O(n d)", notes="log_b n digits per coordinate"
    ),
    references=(
        "Halton, Numer. Math. 2 (1960) 84-90",
        "Niederreiter, Random Number Generation and Quasi-Monte Carlo Methods (1992)",
    ),
)
def halton(n: int, d: int, random_state: RandomState = None, scramble: bool = False) -> np.ndarray:
    r"""First ``n`` points of the ``d``-dimensional Halton sequence in :math:`[0, 1)^d`.

    Coordinate :math:`j` is the radical inverse of the index :math:`i = 0, \dots, n-1`
    in the :math:`j`-th prime base (see the module docstring for the scrambled
    variant). Unscrambled, the first point is the origin, as in
    ``scipy.stats.qmc.Halton(scramble=False)``.

    Parameters
    ----------
    n : int
        Number of points (>= 1).
    d : int
        Dimension (>= 1).
    random_state : None, int, SeedSequence or Generator
        Randomness for ``scramble=True`` (ignored otherwise).
    scramble : bool, default=False
        Apply random digit permutations.

    Returns
    -------
    ndarray of shape (n, d)

    Complexity
    ----------
    Time O(n d log n), memory O(n d).

    References
    ----------
    Halton (1960); Niederreiter (1992), §3.1.

    Examples
    --------
    >>> halton(4, 2)
    array([[0.        , 0.        ],
           [0.5       , 0.33333333],
           [0.25      , 0.66666667],
           [0.75      , 0.11111111]])
    """
    n = check_positive_int(n, "n")
    d = check_positive_int(d, "d")
    if not isinstance(scramble, (bool, np.bool_)):
        raise TypeError("scramble must be a bool")
    rng = check_random_state(random_state) if scramble else None
    index = np.arange(n, dtype=np.int64)
    out = np.empty((n, d))
    for j, base in enumerate(_first_primes(d)):
        if rng is None:
            n_digits = max(1, math.ceil(math.log(n, base)) + 1) if n > 1 else 1
        else:
            n_digits = int(52 / math.log2(base))
        x = np.zeros(n)
        rest = index.copy()
        scale = 1.0 / base
        for _ in range(n_digits):
            digit = rest % base
            if rng is not None:
                digit = rng.permutation(base)[digit]
            x += digit * scale
            rest //= base
            scale /= base
        out[:, j] = x
    return np.minimum(out, np.nextafter(1.0, 0.0))


@register(
    domain="numerics",
    tasks=("quasi-monte-carlo", "sampling"),
    complexity=Complexity(time="O(n d log n)", space="O(n d)", notes="d <= 21"),
    references=(
        "Sobol', USSR Comput. Math. Math. Phys. 7 (1967) 86-112",
        "Joe & Kuo, SIAM J. Sci. Comput. 30 (2008) 2635-2654",
    ),
)
def sobol(n: int, d: int) -> np.ndarray:
    r"""First ``n`` points of the unscrambled ``d``-dimensional Sobol sequence.

    Joe-Kuo direction numbers, Gray-code order; the first point is the origin
    (module docstring). Balance properties hold for ``n`` a power of two.

    Parameters
    ----------
    n : int
        Number of points, ``1 <= n <= 2**32``.
    d : int
        Dimension, ``1 <= d <= 21``.

    Returns
    -------
    ndarray of shape (n, d)
        Dyadic rationals :math:`k / 2^{32}` in :math:`[0, 1)`.

    Complexity
    ----------
    Time O(n d log n), memory O(n d).

    References
    ----------
    Sobol' (1967); Joe & Kuo (2008).

    Examples
    --------
    >>> sobol(4, 3)
    array([[0.  , 0.  , 0.  ],
           [0.5 , 0.5 , 0.5 ],
           [0.75, 0.25, 0.25],
           [0.25, 0.75, 0.75]])
    """
    n = check_positive_int(n, "n")
    d = check_positive_int(d, "d")
    if d > SOBOL_MAX_DIM:
        raise ValueError(f"sobol supports d <= {SOBOL_MAX_DIM}, got {d}")
    if n > 2**_BITS:
        raise ValueError(f"sobol supports n <= 2**{_BITS}, got {n}")
    k = np.arange(n, dtype=np.uint64)
    gray = k ^ (k >> np.uint64(1))
    X = np.zeros((n, d), dtype=np.uint64)
    for bit in range(max(1, int(n - 1).bit_length())):
        mask = ((gray >> np.uint64(bit)) & np.uint64(1)).astype(bool)
        X[mask] ^= _V[:d, bit]
    return X.astype(np.float64) * 2.0**-_BITS


@register(
    domain="numerics",
    tasks=("quasi-monte-carlo", "statistics"),
    complexity=Complexity(time="O(n^2 d)", space="O(n d)", notes="rows processed in chunks"),
    references=("Warnock, Computational investigations of low-discrepancy point sets (1972)",),
)
def l2_star_discrepancy(x: ArrayLike) -> float:
    r"""L2-star discrepancy :math:`D_2^*` of points in :math:`[0, 1]^d` (Warnock's formula).

    Matches ``scipy.stats.qmc.discrepancy(x, method="L2-star")``.

    Parameters
    ----------
    x : array_like of shape (n, d)
        Points in the unit cube.

    Returns
    -------
    float
        :math:`D_2^*`; smaller means more uniform.

    Complexity
    ----------
    Time O(n^2 d), extra memory O(n d) (pairwise terms are summed in row chunks).

    References
    ----------
    Warnock (1972); Niederreiter (1992), §2.

    Examples
    --------
    >>> round(l2_star_discrepancy([[0.1, 0.2], [0.6, 0.7], [0.3, 0.9]]), 12)
    0.147365758121
    """
    x = as_real(x, "x", ndim=2)
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("x must contain at least one point and one dimension")
    if np.any((x < 0) | (x > 1)):
        raise ValueError("points must lie in [0, 1]^d")
    n, d = x.shape
    one_minus = 1.0 - x
    pair_sum = 0.0
    chunk = max(1, 2**20 // max(n * d, 1))
    for start in range(0, n, chunk):
        block = x[start : start + chunk]
        prod = np.prod(1.0 - np.maximum(block[:, None, :], x[None, :, :]), axis=2)
        pair_sum += float(prod.sum())
    d2 = (
        3.0**-d
        - 2.0 ** (1 - d) / n * float(np.prod(one_minus * (1.0 + x), axis=1).sum())
        + pair_sum / n**2
    )
    return float(np.sqrt(max(d2, 0.0)))
