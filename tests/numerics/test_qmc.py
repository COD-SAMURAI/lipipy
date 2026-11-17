"""lipipy.numerics.qmc: Halton / Sobol vs scipy.stats.qmc, discrepancy comparisons."""

import numpy as np
import pytest

from lipipy.bench import estimate_complexity
from lipipy.numerics.qmc import SOBOL_MAX_DIM, halton, l2_star_discrepancy, sobol


@pytest.mark.filterwarnings("ignore:The balance properties:UserWarning")  # SciPy, n != 2^m
@pytest.mark.parametrize("n", [1, 2, 7, 64, 100, 1000, 4096])
@pytest.mark.parametrize("d", [1, 2, 5, 13, 21])
def test_sobol_equals_scipy_unscrambled(n, d):
    qmc = pytest.importorskip("scipy.stats.qmc")
    ref = qmc.Sobol(d, scramble=False).random(n)
    np.testing.assert_array_equal(sobol(n, d), ref)  # bit-identical dyadic rationals


@pytest.mark.parametrize("n", [1, 2, 9, 100, 2000])
@pytest.mark.parametrize("d", [1, 2, 6, 20])
def test_halton_equals_scipy_unscrambled(n, d):
    qmc = pytest.importorskip("scipy.stats.qmc")
    ref = qmc.Halton(d, scramble=False).random(n)
    np.testing.assert_allclose(halton(n, d), ref, rtol=0, atol=4e-16)


def test_first_point_convention_and_range():
    for pts in (sobol(1024, 21), halton(1024, 21)):
        np.testing.assert_array_equal(pts[0], 0.0)
        assert np.all((pts >= 0) & (pts < 1))
    x = halton(500, 7, random_state=0, scramble=True)
    assert np.all((x >= 0) & (x < 1))


@pytest.mark.parametrize("m", range(1, 11))
@pytest.mark.parametrize("d", [1, 2, 8, 21])
def test_sobol_balance_for_powers_of_two(m, d):
    """Each 1-D projection of the first 2^m points is one point per interval of width 2^-m."""
    x = sobol(2**m, d)
    for j in range(d):
        np.testing.assert_array_equal(np.sort(np.floor(x[:, j] * 2**m)), np.arange(2**m))


def test_sobol_two_dimensional_net_property():
    """First 2 dimensions form a (0, m, 2)-net: every 2^-a x 2^-(m-a) box holds one point."""
    m = 8
    x = sobol(2**m, 2)
    for a in range(m + 1):
        cells = np.floor(x[:, 0] * 2**a) * 2 ** (m - a) + np.floor(x[:, 1] * 2 ** (m - a))
        assert np.unique(cells).size == 2**m


def test_halton_radical_inverse_by_hand():
    x = halton(10, 3)
    assert x[5, 0] == 0.625  # 5 = 101_2 -> 0.101_2
    assert x[5, 1] == pytest.approx(2 / 3 + 1 / 9)  # 5 = 12_3 -> 0.21_3
    assert x[7, 2] == pytest.approx(2 / 5 + 1 / 25)  # 7 = 12_5 -> 0.21_5


def test_scrambled_halton_is_deterministic_and_stratified():
    a = halton(243, 3, random_state=42, scramble=True)
    b = halton(243, 3, random_state=np.random.default_rng(42), scramble=True)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, halton(243, 3, random_state=43, scramble=True))
    assert not np.array_equal(a, halton(243, 3))
    # Digit permutations keep the stratification of the leading digits:
    # 243 = 3^5 points occupy each of the 3^5 base-3 intervals exactly once.
    np.testing.assert_array_equal(np.sort(np.floor(a[:, 1] * 243)), np.arange(243))
    c = halton(256, 1, random_state=7, scramble=True)  # base 2: 2^8 points
    np.testing.assert_array_equal(np.sort(np.floor(c[:, 0] * 256)), np.arange(256))


def test_l2_star_discrepancy_matches_scipy(rng):
    qmc = pytest.importorskip("scipy.stats.qmc")
    for n, d in [(1, 1), (5, 3), (200, 2), (300, 7), (64, 21)]:
        x = rng.random((n, d))
        assert l2_star_discrepancy(x) == pytest.approx(
            qmc.discrepancy(x, method="L2-star"), rel=1e-10
        )


def test_l2_star_random_points_match_expectation():
    """E[(D2*)^2] = (2^-d - 3^-d) / n for i.i.d. uniform points."""
    for d in (1, 3, 6):
        n = 200
        sq = [l2_star_discrepancy(np.random.default_rng(s).random((n, d))) ** 2 for s in range(200)]
        assert np.mean(sq) == pytest.approx((2.0**-d - 3.0**-d) / n, rel=0.1)


@pytest.mark.parametrize("d", [2, 3, 5])
@pytest.mark.parametrize("n", [256, 1024])
def test_qmc_beats_best_of_50_random_point_sets(d, n):
    """The comparison stated in the module docstring."""
    best_random = min(
        l2_star_discrepancy(np.random.default_rng(seed).random((n, d))) for seed in range(50)
    )
    assert l2_star_discrepancy(sobol(n, d)) < best_random
    assert l2_star_discrepancy(halton(n, d)) < best_random
    for seed in range(10):
        assert l2_star_discrepancy(halton(n, d, random_state=seed, scramble=True)) < best_random


def test_input_validation():
    with pytest.raises(ValueError, match="d <= 21"):
        sobol(4, SOBOL_MAX_DIM + 1)
    with pytest.raises(ValueError, match="n must be"):
        sobol(0, 2)
    with pytest.raises(TypeError, match="integer"):
        halton(4.0, 2)
    with pytest.raises(TypeError, match="scramble"):
        halton(4, 2, scramble=1)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        l2_star_discrepancy([[1.5]])
    with pytest.raises(ValueError, match="2-D"):
        l2_star_discrepancy([0.5])


def test_sobol_memory_linear_in_n():
    est = estimate_complexity(
        lambda n: sobol(n, 4), lambda n, rng: n, sizes=(20_000, 40_000, 80_000, 160_000), repeats=1
    )
    assert est.space_exponent == pytest.approx(1.0, abs=0.15)


def test_discrepancy_memory_not_quadratic():
    """Declared extra memory O(n d): pairwise terms are summed in chunks."""
    est = estimate_complexity(
        l2_star_discrepancy, lambda n, rng: rng.random((n, 2)), sizes=(1_000, 2_000, 4_000),
        repeats=1,
    )  # fmt: skip
    assert est.space_exponent < 1.3
