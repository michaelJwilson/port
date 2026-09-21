"""What the quantile inversion costs, and what removing it saves (#174).

The Measurement rule puts the speedup claim at a stress size, and this one is
worth stating at both: the ratio is the same, and the **absolute** number is
what makes the change worth making. The filter is 93.6 per cent of the whole
preprocessing chain at 2,500 spots and 400 bins.

Benchmarked on pooled counts and totals directly rather than through the
filter. The fit around them is 0.3 s and does not move, so timing the whole
function would measure the same difference against a constant offset and take
a beta-binomial fit per round to do it.
"""

import numpy as np
import pytest
import scipy.stats
from pytest_benchmark.fixture import BenchmarkFixture

pytestmark = pytest.mark.preprocessing

ALPHA = 15.0
BETA = 15.0
"""`p = 0.5` and `tau = 30`, which is what the filter proceeds with.

`cnaster` overwrites the fitted success probability with 0.5 and floors the
concentration at `min_betabinom_tau = 30`, so these are not a choice this
module made -- they are what the quantile is evaluated at on every run.
"""

SHIPPED_CONFIDENCE = (0.01, 0.99)

GATE = (40, 30_000)
"""Bins and the pooled trial count per bin, at the dev instance's scale.

40 bins, and a pooled total of order 30,000 because the normal candidates are
520 spots at 15 to 60 reads a bin.
"""

STRESS = (400, 30_000)
"""Ten times the genome, where the filter takes 29.4 s of a 31.6 s chain."""


def _instance(n_bins: int, scale: int, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Pooled B-allele counts and totals, balanced, as the filter sees them."""
    rng = np.random.default_rng(seed)
    totals = rng.integers(scale // 2, scale, n_bins).astype(float)
    counts = rng.binomial(totals.astype(int), 0.5).astype(float)

    return counts, totals


def cnaster_indicator(
    counts: np.ndarray,
    totals: np.ndarray,
    alpha: float,
    beta: float,
    confidence_interval: tuple[float, float],
) -> np.ndarray:
    """`normal_spot.py:988-1000`, transcribed.

    Transcribed rather than imported because it is four inline statements in a
    120-line function, so there is no way to call it on its own -- which is
    also why the cost had never been attributed to it.
    """
    lower = counts < scipy.stats.betabinom.ppf(
        confidence_interval[0], totals, alpha, beta
    )
    upper = counts > scipy.stats.betabinom.ppf(
        confidence_interval[1], totals, alpha, beta
    )

    return np.asarray(lower | upper)


@pytest.mark.benchmark
def test_the_quantile_inversion_at_the_gate_size(benchmark: BenchmarkFixture) -> None:
    """`cnaster`'s two `ppf` calls over 40 bins: 1.44 s."""
    counts, totals = _instance(*GATE)

    benchmark(
        lambda: cnaster_indicator(counts, totals, ALPHA, BETA, SHIPPED_CONFIDENCE)
    )


@pytest.mark.benchmark
def test_the_distribution_function_at_the_gate_size(
    benchmark: BenchmarkFixture,
) -> None:
    """The same mask from two `cdf` calls: 95 ms, so **15x**."""
    from port.patch.normal_spot import removal_indicator

    counts, totals = _instance(*GATE)

    benchmark(
        lambda: removal_indicator(counts, totals, ALPHA, BETA, SHIPPED_CONFIDENCE)
    )


@pytest.mark.benchmark
@pytest.mark.release
def test_the_quantile_inversion_at_the_stress_size(
    benchmark: BenchmarkFixture,
) -> None:
    """400 bins: 13.95 s, and the filter around it 29.4 s."""
    counts, totals = _instance(*STRESS)

    benchmark(
        lambda: cnaster_indicator(counts, totals, ALPHA, BETA, SHIPPED_CONFIDENCE)
    )


@pytest.mark.benchmark
@pytest.mark.release
def test_the_distribution_function_at_the_stress_size(
    benchmark: BenchmarkFixture,
) -> None:
    """919 ms against 13.95 s: **15.2x**, and the whole filter 14.3x."""
    from port.patch.normal_spot import removal_indicator

    counts, totals = _instance(*STRESS)

    benchmark(
        lambda: removal_indicator(counts, totals, ALPHA, BETA, SHIPPED_CONFIDENCE)
    )
