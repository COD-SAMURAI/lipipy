"""lipipy.numerics.fft vs numpy.fft and scipy.signal.fftconvolve."""

import numpy as np
import pytest

from lipipy.bench import estimate_complexity
from lipipy.numerics import fft, fftconvolve, fftfreq, ifft, irfft, rfft, rfftfreq

EPS_BOUND = 2e-15  # documented: relative l2 error <= 2e-15 * (1 + log2 n)

PRIMES = [2, 3, 5, 7, 11, 13, 101, 127, 257, 509, 1021, 2053, 4099, 8191, 10007]
LENGTHS = sorted(set(range(1, 131)) | {2**k for k in range(8, 17)} | set(PRIMES) | {1000, 4095})


def rel(a, b):
    nb = np.linalg.norm(b)
    return np.linalg.norm(a - b) / nb if nb else np.linalg.norm(a)


def bound(n):
    return EPS_BOUND * (1 + np.log2(n))


def cdata(rng, n):
    return rng.normal(size=n) + 1j * rng.normal(size=n)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("n", LENGTHS)
def test_fft_ifft_match_numpy_and_round_trip(n):
    rng = np.random.default_rng(n)
    for _ in range(3):
        x = cdata(rng, n)
        X = fft(x)
        assert X.dtype == np.complex128 and X.shape == (n,)
        assert rel(X, np.fft.fft(x)) <= bound(n)
        assert rel(ifft(x), np.fft.ifft(x)) <= bound(n)
        assert rel(ifft(X), x) <= bound(n)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("n", LENGTHS)
def test_rfft_irfft_match_numpy(n):
    rng = np.random.default_rng(10_000 + n)
    for _ in range(3):
        x = rng.normal(size=n)
        R = rfft(x)
        assert R.shape == (n // 2 + 1,)
        assert rel(R, np.fft.rfft(x)) <= bound(n)
        if n >= 2:
            back = irfft(R, n)
            assert back.dtype == np.float64
            assert rel(back, x) <= bound(n)
            spec = cdata(rng, n // 2 + 1)  # arbitrary, non-Hermitian at DC / Nyquist
            assert rel(irfft(spec, n), np.fft.irfft(spec, n)) <= bound(n)


def test_real_input_fft_matches_and_small_cases_exact():
    np.testing.assert_array_equal(fft([3.0]), [3.0 + 0j])
    np.testing.assert_allclose(fft([1.0, 2.0]), [3.0, -1.0], atol=0)
    x = np.arange(12.0)
    np.testing.assert_allclose(fft(x), np.fft.fft(x), rtol=0, atol=1e-13)
    # integer and boolean inputs are cast to float
    np.testing.assert_allclose(fft([1, 0, 0, 0]), np.ones(4))
    np.testing.assert_allclose(fft(np.array([True, False, False])), np.ones(3))


def test_impulse_and_constant_are_exact_dual_pairs():
    for n in (1, 6, 16, 17):
        delta = np.zeros(n)
        delta[0] = 1.0
        np.testing.assert_allclose(fft(delta), np.ones(n), atol=1e-14)
        np.testing.assert_allclose(fft(np.ones(n)), n * delta, atol=1e-12)


@pytest.mark.parametrize("n_in, n", [(5, 8), (8, 5), (7, 7), (3, 11), (16, 1)])
def test_n_pads_or_crops_like_numpy(n_in, n):
    x = cdata(np.random.default_rng(3), n_in)
    np.testing.assert_allclose(fft(x, n), np.fft.fft(x, n), atol=1e-13)
    np.testing.assert_allclose(ifft(x, n), np.fft.ifft(x, n), atol=1e-13)
    np.testing.assert_allclose(rfft(x.real, n), np.fft.rfft(x.real, n), atol=1e-13)
    if n >= 2:
        np.testing.assert_allclose(irfft(x, n), np.fft.irfft(x, n), atol=1e-13)
    np.testing.assert_allclose(irfft(x), np.fft.irfft(x), atol=1e-13)


@pytest.mark.filterwarnings("error")
def test_extreme_magnitudes_scale_exactly():
    x = cdata(np.random.default_rng(4), 37)
    ref = fft(x)
    for scale in (1e-300, 1e-200, 1e200, 1e300):
        np.testing.assert_allclose(fft(x * scale) / scale, ref, rtol=1e-12, atol=1e-12)
    sub = np.full(8, 5e-324)
    np.testing.assert_allclose(fft(sub)[0].real, 8 * 5e-324)


def test_linearity_and_parseval():
    rng = np.random.default_rng(5)
    for n in (64, 97):
        x, y = cdata(rng, n), cdata(rng, n)
        np.testing.assert_allclose(fft(2 * x - 3j * y), 2 * fft(x) - 3j * fft(y), atol=1e-11)
        assert np.sum(np.abs(fft(x)) ** 2) / n == pytest.approx(np.sum(np.abs(x) ** 2), rel=1e-13)


@pytest.mark.parametrize("n", [1, 2, 7, 8, 9, 100])
@pytest.mark.parametrize("d", [1.0, 0.25, 3.0])
def test_frequencies_match_numpy(n, d):
    np.testing.assert_allclose(fftfreq(n, d), np.fft.fftfreq(n, d), rtol=1e-15, atol=0)
    np.testing.assert_allclose(rfftfreq(n, d), np.fft.rfftfreq(n, d), rtol=1e-15, atol=0)


@pytest.mark.parametrize("mode", ["full", "same", "valid"])
def test_fftconvolve_matches_scipy(mode):
    signal = pytest.importorskip("scipy.signal")
    rng = np.random.default_rng(6)
    sizes = [1, 2, 3, 7, 16, 31, 100, 257, 1000]
    for na in sizes:
        for nb in sizes:
            a, b = rng.normal(size=na), rng.normal(size=nb)
            ours = fftconvolve(a, b, mode)
            ref = signal.fftconvolve(a, b, mode)
            assert ours.dtype == np.float64 and ours.shape == ref.shape
            scale = np.linalg.norm(a) * np.linalg.norm(b)
            n = na + nb
            assert np.linalg.norm(ours - ref) <= 4 * bound(n) * scale
            direct = np.convolve(a, b, mode) if mode != "same" or na >= nb else None
            if direct is not None:
                assert np.linalg.norm(ours - direct) <= 4 * bound(n) * scale


def test_fftconvolve_complex_and_empty():
    signal = pytest.importorskip("scipy.signal")
    rng = np.random.default_rng(7)
    a, b = cdata(rng, 13), rng.normal(size=6)
    for mode in ("full", "same", "valid"):
        ours = fftconvolve(a, b, mode)
        assert ours.dtype == np.complex128
        np.testing.assert_allclose(ours, signal.fftconvolve(a, b, mode), atol=1e-12)
    assert fftconvolve([], [1.0]).shape == (0,)
    assert fftconvolve([1.0], [], mode="same").shape == (0,)


def test_fftconvolve_exact_on_integers():
    a = np.arange(1, 50)
    b = np.array([1, -2, 3, 0, 5])
    np.testing.assert_allclose(fftconvolve(a, b), np.convolve(a, b), atol=1e-9)


@pytest.mark.parametrize(
    "bad, exc",
    [
        ([], ValueError),
        ([[1.0, 2.0]], ValueError),
        ([1.0, np.nan], ValueError),
        ([1.0, np.inf], ValueError),
        ("abc", TypeError),
        (np.array(["a"], dtype=object), TypeError),
    ],
)
def test_fft_rejects_bad_input(bad, exc):
    with pytest.raises(exc):
        fft(bad)
    with pytest.raises(exc):
        ifft(bad)


def test_bad_parameters():
    with pytest.raises(ValueError):
        fft([1.0], n=0)
    with pytest.raises(TypeError):
        fft([1.0], n=2.5)
    with pytest.raises(TypeError):
        rfft([1j, 2.0])
    with pytest.raises(ValueError):
        irfft([1.0])  # default n = 0
    with pytest.raises(ValueError):
        irfft([])
    with pytest.raises(ValueError):
        fftfreq(0)
    with pytest.raises(ValueError):
        fftfreq(4, d=0.0)
    with pytest.raises(ValueError):
        rfftfreq(4, d=-1.0)
    with pytest.raises(ValueError):
        fftconvolve([1.0], [1.0], mode="circular")
    with pytest.raises(ValueError):
        fftconvolve([[1.0]], [1.0])


def test_inputs_not_mutated():
    x = np.arange(10.0)
    keep = x.copy()
    fft(x)
    rfft(x)
    fftconvolve(x, x)
    np.testing.assert_array_equal(x, keep)


def test_registered():
    import lipipy

    for name in ("fft", "ifft", "rfft", "irfft", "fftfreq", "rfftfreq", "fftconvolve"):
        info = lipipy.registry.get(f"numerics.{name}")
        assert info.domain == "numerics"
    assert lipipy.registry.get("numerics.fft").complexity.time == "O(n log n)"


@pytest.mark.parametrize("pow2", [True, False])
def test_fft_extra_memory_is_linear(pow2):
    def make(n, rng):
        m = n if pow2 else n + 1  # n + 1 is odd -> Bluestein
        return cdata(rng, m)

    est = estimate_complexity(fft, make, sizes=(2**14, 2**15, 2**16, 2**17), repeats=1)
    assert est.space_exponent == pytest.approx(1.0, abs=0.15)


@pytest.mark.complexity
@pytest.mark.parametrize("pow2", [True, False])
def test_fft_time_is_n_log_n(pow2):
    # A quadratic DFT would give an exponent near 2; n log n fits to about 1.1.
    def make(n, rng):
        return cdata(rng, n if pow2 else n + 1)

    est = estimate_complexity(fft, make, sizes=(2**14, 2**15, 2**16, 2**17, 2**18), repeats=3)
    assert est.time_exponent < 1.35
