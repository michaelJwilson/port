"""What `port.patch.io` costs against what it replaces.

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

from collections.abc import Callable
from typing import Any

import pytest
from cnaster.io import load_input_data as cnaster_loader
from port.patch.io import _range_mask
from port.patch.io import load_input_data as patched_loader
from pytest_benchmark.fixture import BenchmarkFixture

from tests.fixtures import tiers
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
@pytest.mark.parametrize(
    "loader", [cnaster_loader, patched_loader], ids=["cnaster", "patched"]
)
def test_the_loader(
    benchmark: BenchmarkFixture,
    loader: Callable[[Any], Any],
    gate_config: Any,  # noqa: F811
) -> None:
    """60.5 ms median on the dev instance, against 52.4 ms patched.

    1.15x. Reported; no speedup is claimed from it.
    """
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
    """`cnaster`'s loop against the vectorized mask, with exact agreement.

    7.4 ms vectorized at the gate size, 35x. At the stress size, where the
    claim is made, 2.88 s against 73 ms: **39x**.
    """
    snp_ids, ranges = synthetic_ranges(*size)

    benchmark(arm, snp_ids, ranges)
