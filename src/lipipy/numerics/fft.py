r"""Fast Fourier transforms and FFT-based convolution.

Discrete Fourier transform
--------------------------
For :math:`x \in \mathbb{C}^n` (NumPy's sign and scaling conventions):

.. math::
    X_k = \sum_{j=0}^{n-1} x_j\, \omega_n^{jk}, \qquad
    x_j = \frac1n \sum_{k=0}^{n-1} X_k\, \omega_n^{-jk}, \qquad
    \omega_n = e^{-2\pi i/n}.

Evaluating the sum directly costs :math:`O(n^2)`; every length here costs
:math:`O(n \log n)`.

Radix-2 Cooley-Tukey (power-of-two :math:`n`)
---------------------------------------------
Split into even and odd samples, :math:`E_k = \mathrm{DFT}_{n/2}(x_{2j})`,
:math:`O_k = \mathrm{DFT}_{n/2}(x_{2j+1})`; then for :math:`k < n/2`

.. math::  X_k = E_k + \omega_n^k O_k,\qquad X_{k+n/2} = E_k - \omega_n^k O_k .

The iterative form permutes the input into bit-reversed order once and then runs
:math:`\log_2 n` butterfly stages of block size :math:`2, 4, \dots, n`; each stage is
one vectorised :math:`O(n)` pass. Twiddles :math:`\omega_n^k` are computed once
directly from :math:`\cos/\sin` (not by repeated multiplication), so their error
is a few ulps.

Bluestein's chirp-z algorithm (any :math:`n`)
---------------------------------------------
With :math:`jk = \tfrac12\left(j^2 + k^2 - (k-j)^2\right)` and the chirp
:math:`c_j = e^{-i\pi j^2/n}`,

.. math::  X_k = c_k \sum_{j=0}^{n-1} (x_j c_j)\, \overline{c_{k-j}},

a convolution, evaluated exactly (circularly, without wrap-around) with radix-2
FFTs of length :math:`M = 2^{\lceil\log_2(2n-1)\rceil} < 4n`. The exponent is
reduced as :math:`j^2 \bmod 2n` in integer arithmetic so the chirp stays accurate for
large :math:`j`.

Real transforms
---------------
For real :math:`x` and even :math:`n`, pack :math:`z_j = x_{2j} + i x_{2j+1}` and take
one complex FFT of half length, :math:`Z = \mathrm{DFT}_{n/2}(z)`; then

.. math::
    X_k = \tfrac12\bigl(Z_k + \overline{Z_{n/2-k}}\bigr)
        - \tfrac{i}{2}\,\omega_n^k \bigl(Z_k - \overline{Z_{n/2-k}}\bigr),
    \qquad k = 0, \dots, n/2,

(indices mod :math:`n/2`). Odd :math:`n` falls back to the full complex transform.
``irfft`` rebuilds the Hermitian spectrum :math:`X_{n-k} = \overline{X_k}` and takes
the real part of the inverse; like NumPy it ignores the imaginary parts of
:math:`X_0` and (for even :math:`n`) :math:`X_{n/2}`.

Convolution
-----------
:math:`(a * b)_m = \sum_j a_j b_{m-j}` has length :math:`n_a + n_b - 1`; zero-padding
both inputs to a power of two :math:`M \ge n_a + n_b - 1` makes the circular
convolution :math:`\mathrm{IFFT}(\mathrm{FFT}(a)\cdot\mathrm{FFT}(b))` equal to the
linear one.

Accuracy
--------
Cooley-Tukey has relative :math:`\ell_2` error :math:`O(\varepsilon \log n)` (Higham,
*Accuracy and Stability of Numerical Algorithms*, 2nd ed., §24.1). The tests assert,
over all lengths 1..130, all powers of two up to :math:`2^{16}`, primes up to 10007
and random complex data,

.. math::
    \frac{\lVert \hat X - X \rVert_2}{\lVert X \rVert_2}
    \le 2\times10^{-15}\,(1 + \log_2 n)

against ``numpy.fft`` for ``fft``, ``ifft``, ``rfft`` and ``irfft`` (Bluestein
lengths included), and the same bound for the round trip ``ifft(fft(x))``
against ``x``.

References
----------
J. W. Cooley & J. W. Tukey, "An algorithm for the machine calculation of complex
Fourier series", *Math. Comp.* 19 (1965) 297-301. L. I. Bluestein, "A linear filtering
approach to the computation of discrete Fourier transform", *IEEE Trans. Audio
Electroacoust.* 18 (1970) 451-455. W. H. Press et al., *Numerical Recipes*, 3rd ed.
(2007), §12.2-12.3 (real FFT packing). N. J. Higham, *Accuracy and Stability of
Numerical Algorithms*, 2nd ed. (2002), ch. 24.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from lipipy.core.complexity import Complexity
from lipipy.core.registry import register
from lipipy.numerics._common import as_finite_float, check_positive_int

__all__ = ["fft", "fftconvolve", "fftfreq", "ifft", "irfft", "rfft", "rfftfreq"]

_CT = "Cooley & Tukey, Math. Comp. 19 (1965) 297-301"
_BLUESTEIN = "Bluestein, IEEE Trans. Audio Electroacoust. 18 (1970) 451-455"
_NR = "Press et al., Numerical Recipes 3rd ed. (2007), §12.2-12.3"
_NLOGN = Complexity(
    time="O(n log n)",
    space="O(n)",
    notes="radix-2 for powers of two; Bluestein (three radix-2 FFTs of length < 4n) otherwise",
)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def _as_signal(x: ArrayLike, name: str, *, real: bool) -> np.ndarray:
    if isinstance(x, (str, bytes)):
        raise TypeError(f"{name} must be numeric, got {type(x).__name__}")
    arr = np.asarray(x)
    if arr.dtype == object or not (
        np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_)
    ):
        raise TypeError(f"{name} must be a numeric array, got dtype {arr.dtype}")
    if real and np.iscomplexobj(arr):
        raise TypeError(f"{name} must be real-valued, got a complex array")
    arr = arr.astype(np.float64 if real else np.complex128)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinity")
    return arr


def _fit_length(x: np.ndarray, n: int | None, name: str = "x") -> np.ndarray:
    """Crop or zero-pad ``x`` to length ``n`` (NumPy's ``n`` semantics)."""
    if n is None:
        if x.size == 0:
            raise ValueError(f"{name} must be non-empty (or pass n >= 1)")
        return x
    n = check_positive_int(n, "n")
    if n <= x.size:
        return x[:n]
    out = np.zeros(n, dtype=x.dtype)
    out[: x.size] = x
    return out


# ---------------------------------------------------------------------------
# kernels (no validation)
# ---------------------------------------------------------------------------
def _is_pow2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def _bit_reverse_indices(n: int) -> np.ndarray:
    bits = n.bit_length() - 1
    idx = np.arange(n)
    rev = np.zeros(n, dtype=np.intp)
    for b in range(bits):
        rev |= ((idx >> b) & 1) << (bits - 1 - b)
    return rev


def _fft_pow2(x: np.ndarray) -> np.ndarray:
    """Iterative radix-2 decimation-in-time FFT of a complex power-of-two-length array."""
    n = x.size
    if n == 1:
        return x.copy()
    angle = (2.0 * np.pi / n) * np.arange(n // 2)
    twiddle = np.cos(angle) - 1j * np.sin(angle)  # omega_n^k, k < n/2
    a = x[_bit_reverse_indices(n)]
    half = 1
    while half < n:
        blocks = a.reshape(-1, 2 * half)
        even = blocks[:, :half]
        odd = blocks[:, half:] * twiddle[:: n // (2 * half)]  # omega_{2 half}^j
        a = np.concatenate((even + odd, even - odd), axis=1).ravel()
        half *= 2
    return a


def _ifft_pow2(x: np.ndarray) -> np.ndarray:
    return np.conj(_fft_pow2(np.conj(x))) / x.size


def _fft_bluestein(x: np.ndarray) -> np.ndarray:
    n = x.size
    k = np.arange(n, dtype=np.int64)
    angle = (np.pi / n) * ((k * k) % (2 * n))  # j^2 mod 2n keeps the angle small and exact
    chirp = np.cos(angle) - 1j * np.sin(angle)  # c_j = exp(-i pi j^2 / n)
    m = 1 << (2 * n - 2).bit_length()  # power of two >= 2n - 1
    a = np.zeros(m, dtype=np.complex128)
    a[:n] = x * chirp
    b = np.zeros(m, dtype=np.complex128)
    b[:n] = np.conj(chirp)
    b[m - n + 1 :] = np.conj(chirp[1:])[::-1]  # conj(c_{-j}) at index m - j
    conv = _ifft_pow2(_fft_pow2(a) * _fft_pow2(b))
    return chirp * conv[:n]


def _fft_any(x: np.ndarray) -> np.ndarray:
    return _fft_pow2(x) if _is_pow2(x.size) else _fft_bluestein(x)


def _ifft_any(x: np.ndarray) -> np.ndarray:
    return np.conj(_fft_any(np.conj(x))) / x.size


def _rfft_real(x: np.ndarray) -> np.ndarray:
    n = x.size
    if n % 2:
        return _fft_any(x.astype(np.complex128))[: n // 2 + 1]
    m = n // 2
    Z = _fft_any(x[0::2] + 1j * x[1::2])
    k = np.arange(m + 1)
    Zk = Z[k % m]
    Zc = np.conj(Z[(-k) % m])
    angle = (2.0 * np.pi / n) * k
    w = np.cos(angle) - 1j * np.sin(angle)
    return 0.5 * (Zk + Zc) - 0.5j * w * (Zk - Zc)


def _irfft_real(X: np.ndarray, n: int) -> np.ndarray:
    full = np.zeros(n, dtype=np.complex128)
    half = n // 2 + 1
    m = min(half, X.size)
    full[:m] = X[:m]
    full[0] = full[0].real
    if n % 2 == 0:
        full[n // 2] = full[n // 2].real
    full[half:] = np.conj(full[1 : (n - 1) // 2 + 1])[::-1]
    return _ifft_any(full).real


# ---------------------------------------------------------------------------
# public transforms
# ---------------------------------------------------------------------------
@register(
    domain="numerics",
    tasks=("fft", "signal-processing"),
    complexity=_NLOGN,
    references=(_CT, _BLUESTEIN),
)
def fft(x: ArrayLike, n: int | None = None) -> np.ndarray:
    r"""Discrete Fourier transform :math:`X_k = \sum_j x_j e^{-2\pi i jk/n}` of a 1-D array.

    Radix-2 Cooley-Tukey when ``n`` is a power of two, Bluestein's chirp-z algorithm
    otherwise, so every length costs :math:`O(n \log n)`.

    Parameters
    ----------
    x : array_like, shape (m,)
        Real or complex input (finite).
    n : int, optional
        Transform length: ``x`` is cropped or zero-padded to ``n`` (default ``m``).

    Returns
    -------
    ndarray of complex128, shape (n,)

    Complexity
    ----------
    Time O(n log n). Extra space O(n) (Bluestein uses buffers of length < 4n).

    References
    ----------
    Cooley & Tukey (1965); Bluestein (1970).

    Examples
    --------
    >>> import numpy as np
    >>> np.round(fft([1.0, 2.0, 3.0, 4.0]), 12)
    array([10.+0.j, -2.+2.j, -2.+0.j, -2.-2.j])
    >>> np.allclose(fft(np.arange(7.0)), np.fft.fft(np.arange(7.0)))  # Bluestein
    True
    """
    return _fft_any(_fit_length(_as_signal(x, "x", real=False), n))


@register(
    domain="numerics",
    tasks=("fft", "signal-processing"),
    complexity=_NLOGN,
    references=(_CT, _BLUESTEIN),
)
def ifft(x: ArrayLike, n: int | None = None) -> np.ndarray:
    r"""Inverse DFT :math:`x_j = \frac1n\sum_k X_k e^{2\pi i jk/n}`.

    Computed as :math:`\overline{\mathrm{FFT}(\bar X)}/n`, so it inherits the cost and
    accuracy of :func:`fft`.

    Parameters
    ----------
    x : array_like, shape (m,)
        Spectrum (finite).
    n : int, optional
        Length; the input is cropped or zero-padded to ``n``.

    Returns
    -------
    ndarray of complex128, shape (n,)

    Complexity
    ----------
    Time O(n log n), extra space O(n).

    References
    ----------
    Cooley & Tukey (1965); Bluestein (1970).

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1.0, -2.0, 0.5, 3.0, 4.0])
    >>> np.allclose(ifft(fft(x)), x)
    True
    """
    return _ifft_any(_fit_length(_as_signal(x, "x", real=False), n))


@register(
    domain="numerics",
    tasks=("fft", "signal-processing"),
    complexity=_NLOGN,
    references=(_NR, _CT, _BLUESTEIN),
)
def rfft(x: ArrayLike, n: int | None = None) -> np.ndarray:
    r"""DFT of a real 1-D array (non-negative frequencies :math:`X_0..X_{n//2}` only).

    Even ``n`` packs the signal into a complex array of half the length (one FFT of
    size ``n/2``); odd ``n`` uses the full complex transform.

    Parameters
    ----------
    x : array_like, shape (m,)
        Real input (finite).
    n : int, optional
        Transform length (crop or zero-pad).

    Returns
    -------
    ndarray of complex128, shape (n // 2 + 1,)

    Complexity
    ----------
    Time O(n log n), extra space O(n).

    References
    ----------
    Press et al., *Numerical Recipes* 3rd ed. (2007), §12.3.

    Examples
    --------
    >>> import numpy as np
    >>> X = rfft([1.0, 2.0, 3.0, 4.0])
    >>> X.shape, np.allclose(X, [10.0, -2.0 + 2.0j, -2.0])
    ((3,), True)
    """
    return _rfft_real(_fit_length(_as_signal(x, "x", real=True), n))


@register(
    domain="numerics",
    tasks=("fft", "signal-processing"),
    complexity=_NLOGN,
    references=(_NR, _CT, _BLUESTEIN),
)
def irfft(x: ArrayLike, n: int | None = None) -> np.ndarray:
    r"""Inverse of :func:`rfft`: the real signal of length ``n`` with half-spectrum ``x``.

    The full spectrum is rebuilt from Hermitian symmetry :math:`X_{n-k} = \overline{X_k}`;
    the imaginary parts of :math:`X_0` and, for even ``n``, :math:`X_{n/2}` are ignored
    (as in NumPy). Extra input entries are dropped, missing ones are zero.

    Parameters
    ----------
    x : array_like, shape (m,)
        Half spectrum (finite, complex or real).
    n : int, optional
        Output length; default ``2 (m - 1)``.

    Returns
    -------
    ndarray of float64, shape (n,)

    Complexity
    ----------
    Time O(n log n), extra space O(n).

    References
    ----------
    Press et al., *Numerical Recipes* 3rd ed. (2007), §12.3.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0.5, -1.0, 2.0, 3.0, 0.0])
    >>> np.allclose(irfft(rfft(x), n=5), x)
    True
    """
    X = _as_signal(x, "x", real=False)
    if X.size == 0:
        raise ValueError("x must be non-empty")
    if n is None:
        n = 2 * (X.size - 1)
        if n < 1:
            raise ValueError("irfft of a length-1 spectrum needs an explicit n >= 1")
    n = check_positive_int(n, "n")
    return _irfft_real(X, n)


@register(
    domain="numerics",
    tasks=("fft", "signal-processing"),
    complexity=Complexity(time="O(n)", space="O(n)", notes="n = window length"),
)
def fftfreq(n: int, d: float = 1.0) -> np.ndarray:
    r"""Sample frequencies of :func:`fft`: ``[0, 1, ..., ⌈n/2⌉-1, -⌊n/2⌋, ..., -1] / (d n)``.

    Parameters
    ----------
    n : int
        Window length (>= 1).
    d : float, default=1.0
        Sample spacing (> 0).

    Returns
    -------
    ndarray of float64, shape (n,)

    Complexity
    ----------
    Time O(n), space O(n).

    Examples
    --------
    >>> fftfreq(5, d=0.1)
    array([ 0.,  2.,  4., -4., -2.])
    """
    n = check_positive_int(n, "n")
    d = as_finite_float(d, "d")
    if d <= 0:
        raise ValueError(f"d must be positive, got {d}")
    k = np.arange(n)
    k[(n + 1) // 2 :] -= n
    return k / (n * d)


@register(
    domain="numerics",
    tasks=("fft", "signal-processing"),
    complexity=Complexity(time="O(n)", space="O(n)", notes="n = window length"),
)
def rfftfreq(n: int, d: float = 1.0) -> np.ndarray:
    r"""Sample frequencies of :func:`rfft`: ``[0, 1, ..., ⌊n/2⌋] / (d n)``.

    Parameters
    ----------
    n : int
        Window length (>= 1).
    d : float, default=1.0
        Sample spacing (> 0).

    Returns
    -------
    ndarray of float64, shape (n // 2 + 1,)

    Complexity
    ----------
    Time O(n), space O(n).

    Examples
    --------
    >>> rfftfreq(8, d=0.5)
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    """
    n = check_positive_int(n, "n")
    d = as_finite_float(d, "d")
    if d <= 0:
        raise ValueError(f"d must be positive, got {d}")
    return np.arange(n // 2 + 1) / (n * d)


@register(
    domain="numerics",
    tasks=("convolution", "signal-processing"),
    complexity=Complexity(
        time="O(n log n)",
        space="O(n)",
        notes="n = len(a) + len(b); FFTs of the next power of two >= len(a) + len(b) - 1",
    ),
    references=(_CT, _NR),
)
def fftconvolve(
    a: ArrayLike, b: ArrayLike, mode: Literal["full", "same", "valid"] = "full"
) -> np.ndarray:
    r"""Linear convolution :math:`(a*b)_m = \sum_j a_j b_{m-j}` of 1-D arrays via zero-padded FFTs.

    Modes follow ``scipy.signal.fftconvolve``: ``"full"`` (length
    :math:`n_a+n_b-1`), ``"same"`` (length :math:`n_a`, centred on the full result)
    and ``"valid"`` (length :math:`\max(n_a,n_b)-\min(n_a,n_b)+1`, only the points
    where the shorter input overlaps the longer one completely). Real inputs give
    a real output (via :func:`rfft`); if either is complex the output is complex.
    An empty input gives an empty output.

    Parameters
    ----------
    a, b : array_like, 1-D
        Finite inputs.
    mode : {"full", "same", "valid"}, default="full"

    Returns
    -------
    ndarray of float64 or complex128

    Complexity
    ----------
    Time O(n log n), extra space O(n) for :math:`n = n_a + n_b`.

    References
    ----------
    Press et al., *Numerical Recipes* 3rd ed. (2007), §13.1.

    Examples
    --------
    >>> import numpy as np
    >>> np.round(fftconvolve([1.0, 2.0, 3.0], [0.0, 1.0, 0.5]), 12)
    array([0. , 1. , 2.5, 4. , 1.5])
    >>> np.round(fftconvolve([1.0, 2.0, 3.0], [0.0, 1.0, 0.5], mode="valid"), 12)
    array([2.5])
    """
    if mode not in ("full", "same", "valid"):
        raise ValueError(f"mode must be 'full', 'same' or 'valid', got {mode!r}")
    real = not (np.iscomplexobj(a) or np.iscomplexobj(b))
    A = _as_signal(a, "a", real=real)
    B = _as_signal(b, "b", real=real)
    dtype = np.float64 if real else np.complex128
    if A.size == 0 or B.size == 0:
        return np.zeros(0, dtype=dtype)
    full_len = A.size + B.size - 1
    m = 1 << (full_len - 1).bit_length()  # power of two >= full_len
    if real:
        out = _irfft_real(_rfft_real(_fit_length(A, m)) * _rfft_real(_fit_length(B, m)), m)
    else:
        out = _ifft_pow2(_fft_pow2(_fit_length(A, m)) * _fft_pow2(_fit_length(B, m)))
    if mode == "full":
        return out[:full_len].copy()
    if mode == "same":
        start = (full_len - A.size) // 2
        return out[start : start + A.size].copy()
    start = min(A.size, B.size) - 1
    return out[start : start + abs(A.size - B.size) + 1].copy()
