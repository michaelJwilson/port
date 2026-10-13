"""Benchmark: `cnaster`'s beta-binomial quantile mask against `removal_indicator`'s cdf mask (#174).

Gate 40 bins, stress 400 bins, on pooled counts directly.
"""

from collections.abc import Callable

import numpy as np
import pytest
import scipy.stats
from port.patch.normal_spot import removal_indicator
from pytest_benchmark.fixture import BenchmarkFixture

from tests.fixtures import tiers

pytestmark = pytest.mark.preprocessing

ALPHA = 15.0
BETA = 15.0
"""`p = 0.5` and `tau = 30`: the values `cnaster`'s filter evaluates at on every run."""

SHIPPED_CONFIDENCE = (0.01, 0.99)

GATE = (40, 30_000)
"""Bins and pooled trials per bin at the dev instance's scale."""

STRESS = (400, 30_000)
"""Ten times the genome."""


def _instance(n_bins: int, scale: int, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Return pooled balanced B-allele counts and totals."""
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
    """Return `normal_spot.py:988-1000`'s mask, transcribed: inline in `cnaster`."""
    lower = counts < scipy.stats.betabinom.ppf(
        confidence_interval[0], totals, alpha, beta
    )
    upper = counts > scipy.stats.betabinom.ppf(
        confidence_interval[1], totals, alpha, beta
    )

    return np.asarray(lower | upper)


@pytest.mark.benchmark
@pytest.mark.parametrize("size", tiers(GATE, STRESS))
@pytest.mark.parametrize(
    "arm", [cnaster_indicator, removal_indicator], ids=["quantile", "distribution"]
)
def test_the_indicator(
    benchmark: BenchmarkFixture,
    arm: Callable[..., np.ndarray],
    size: tuple[int, int],
) -> None:
    """Time `cnaster`'s two `ppf` calls against two `cdf` calls for the same mask."""
    counts, totals = _instance(*size)

    benchmark(arm, counts, totals, ALPHA, BETA, SHIPPED_CONFIDENCE)
