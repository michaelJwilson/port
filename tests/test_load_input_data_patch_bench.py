"""What `port.patch.input_data` costs against what it replaces.

#167. Two measurements, and they say different things.

**The loader as a whole is not a speedup and is not offered as one.** It is
1.15 times `cnaster`'s on the dev instance (60.5 against 52.4 ms median) and
1.05 at 2,500 spots, with peak allocation the same to within a per cent,
because neither number is set in the body the patch rewrites: the peak lives in `get_spaceranger_counts`, which builds three
full copies of the count matrix to cast it, and in the dense allele matrices
the return contract requires. It lands as a **simplification**, whose evidence
is the equivalence in `tests/test_load_input_data_patch.py`.

**The range filter is.** `cnaster` walks the SNPs in Python and rebuilds
`ranges.Chr.to_numpy()` inside the inner loop, so it is quadratic in a way that
does not show at a fixture's scale and dominates at a genome's.
"""

from typing import Any

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from tests.test_load_input_data_patch import (
    gate_config,  # noqa: F401  -- used by name, and it needs the one below
    planted_instance,  # noqa: F401  -- `gate_config` resolves it in this module
    range_filter_loop,
    synthetic_ranges,
)

pytestmark = pytest.mark.preprocessing

GATE_RANGES = (10_000, 200)
"""SNPs and ranges at the per-pull-request size, where the loop costs 261 ms."""

STRESS_RANGES = (100_000, 2_000)
"""A fifth of a Visium slide's SNP count, where the loop costs 2.88 s.

The Measurement rule puts the speedup claim here and not at the gate size. A
real slide carries 500,000 SNPs, where the loop is 15 s by the same slope.
"""


@pytest.mark.benchmark
def test_cnasters_loader(benchmark: BenchmarkFixture, gate_config: Any) -> None:  # noqa: F811
    """The baseline: 60.5 ms median on the dev instance."""
    from cnaster.io import load_input_data

    benchmark(lambda: load_input_data(gate_config))


@pytest.mark.benchmark
def test_the_patched_loader(benchmark: BenchmarkFixture, gate_config: Any) -> None:  # noqa: F811
    """52.4 ms, so 1.15x. Reported; no speedup is claimed from it."""
    from port.patch.input_data import load_input_data

    benchmark(lambda: load_input_data(gate_config))


@pytest.mark.benchmark
def test_the_range_filter_loop_at_the_gate_size(benchmark: BenchmarkFixture) -> None:
    """`cnaster`'s loop over 10,000 SNPs and 200 ranges."""
    snp_ids, ranges = synthetic_ranges(*GATE_RANGES)

    benchmark(lambda: range_filter_loop(snp_ids, ranges))


@pytest.mark.benchmark
def test_the_vectorized_range_filter_at_the_gate_size(
    benchmark: BenchmarkFixture,
) -> None:
    """The same, vectorized at 7.4 ms: 35x at this size."""
    from port.patch.input_data import _range_mask

    snp_ids, ranges = synthetic_ranges(*GATE_RANGES)

    benchmark(lambda: _range_mask(snp_ids, ranges))


@pytest.mark.benchmark
@pytest.mark.release
def test_the_range_filter_loop_at_the_stress_size(benchmark: BenchmarkFixture) -> None:
    """100,000 SNPs and 2,000 ranges, which is where the claim is made."""
    snp_ids, ranges = synthetic_ranges(*STRESS_RANGES)

    benchmark(lambda: range_filter_loop(snp_ids, ranges))


@pytest.mark.benchmark
@pytest.mark.release
def test_the_vectorized_range_filter_at_the_stress_size(
    benchmark: BenchmarkFixture,
) -> None:
    """2.88 s against 73 ms: **39x**, and exact agreement."""
    from port.patch.input_data import _range_mask

    snp_ids, ranges = synthetic_ranges(*STRESS_RANGES)

    benchmark(lambda: _range_mask(snp_ids, ranges))
