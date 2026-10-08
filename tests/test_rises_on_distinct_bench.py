"""`port.patch.emission.scaled_rising`'s cost on distinct counts, at a dev_tree run's
median and largest call (#702, T- #776).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from port.patch.emission import scaled_rising as rises_on_distinct
from pytest_benchmark.fixture import BenchmarkFixture
from sal.emissions.rising import scaled_rising_array as log_rising

GATE = (4, 655, 220, 685)
STRESS = (25, 2_829, 594, 2_797)
"""(states, bins, distinct counts, largest count)."""

FORMS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "per-bin": log_rising,
    "distinct": rises_on_distinct,
}


def _arms(
    states: int, bins: int, distinct: int, top: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(702)
    pool = rng.choice(np.arange(top + 1), distinct, replace=False).astype(float)
    return rng.uniform(1.0, 400.0, (states, 1)), rng.choice(pool, bins)


@pytest.mark.benchmark
@pytest.mark.parametrize("form", list(FORMS))
def test_rises_gate(benchmark: BenchmarkFixture, form: str) -> None:
    """The median call: 4 states x 655 bins."""
    x, m = _arms(*GATE)
    benchmark(FORMS[form], x, m)


@pytest.mark.benchmark
@pytest.mark.release
@pytest.mark.parametrize("form", list(FORMS))
def test_rises_stress(benchmark: BenchmarkFixture, form: str) -> None:
    """The largest call: 25 states x 2,829 bins."""
    x, m = _arms(*STRESS)
    benchmark(FORMS[form], x, m)
