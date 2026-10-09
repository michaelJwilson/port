"""Benchmark: `port.patch.io`'s loader and range filter against `cnaster`'s (#167).

Loader at the dev instance (no speedup claimed); range filter at gate and stress sizes.
"""

from collections.abc import Callable
from typing import Any

import pytest
from cnaster.io import load_input_data as cnaster_loader
from port.patch.io import _range_mask
from port.patch.io import load_input_data as patched_loader
from pytest_benchmark.fixture import BenchmarkFixture

from tests.adapters import range_filter_loop
from tests.fixtures import synthetic_ranges, tiers

pytestmark = pytest.mark.preprocessing

GATE_RANGES = (10_000, 200)
"""SNPs and ranges at the per-PR size."""

STRESS_RANGES = (100_000, 2_000)
"""A fifth of a Visium slide's SNP count, where the speedup is read."""


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "loader", [cnaster_loader, patched_loader], ids=["cnaster", "patched"]
)
def test_the_loader(
    benchmark: BenchmarkFixture,
    loader: Callable[[Any], Any],
    gate_config: Any,
) -> None:
    """Time `cnaster`'s loader against the patch on the dev instance."""
    benchmark(loader, gate_config)


@pytest.mark.benchmark
@pytest.mark.parametrize("size", tiers(GATE_RANGES, STRESS_RANGES))
@pytest.mark.parametrize(
    "arm", [range_filter_loop, _range_mask], ids=["loop", "vectorized"]
)
def test_the_range_filter(
    benchmark: BenchmarkFixture,
    arm: Callable[[Any, Any], Any],
    size: tuple[int, int],
) -> None:
    """Time `cnaster`'s range loop against the vectorized mask."""
    snp_ids, ranges = synthetic_ranges(*size)

    benchmark(arm, snp_ids, ranges)
