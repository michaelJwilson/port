"""What the boundary recomputes per iteration (issue #59 item 4).

Recorded, not argued. `(single_base_nb_mean > 0).sum(axis=0)` and
`(single_total_bb_RD > 0).sum(axis=0)` are two full passes over
`(n_obs, n_spots)`, run once per outer iteration for quantities the outer
loop cannot change:

| `n_obs` | `n_spots` | both counts, per iteration |
| ---: | ---: | ---: |
| 240 | 160 | 0.041 ms |
| 3,000 | 5,000 | 25.2 ms |
| 10,000 | 2,500 | 48.5 ms |

25.2 ms against a boundary costing about 16 s is under two tenths of a per
cent, so this lands as a simplification under `CLAUDE.md`'s split and the
numbers are here to stop anyone claiming otherwise.

The hoisted weight is benchmarked beside them -- 0.018 ms at 160 spots,
0.229 ms at 5,000 -- because it is the cost the hoist *adds* at its single
call site. Recorded so the trade is visible rather than assumed: the patch
does not move work out of the loop into something more expensive outside it.

`n_obs = 10,000`, `n_spots = 2,500` is `cnaster`'s own working size. The
counts are bench-local arrays rather than the fixture's, because what they
cost depends on the shape and on the fraction of zeros, not on the
generative model -- and the fixture at 3,000 by 5,000 would carry an
emission this benchmark never reads.
"""

import numpy as np
import pytest
from port.patch.hmrf.invariants import BoundaryInvariants, boundary_invariants
from pytest_benchmark.fixture import BenchmarkFixture
from scipy.sparse import csr_matrix

from tests.fixtures import tiers

GATE = (240, 160)
STRESS = (3_000, 5_000)
WORKING = (10_000, 2_500)

NB_DROPOUT = 0.1
BB_DROPOUT = 0.4
"""Different rates, so the two counts differ and the weight is not one."""

NEIGHBOURS = 6
"""A smoothing neighbourhood's degree, near enough."""


def _counts_inputs(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(8_101)

    base = rng.random(shape)
    base[base < NB_DROPOUT] = 0.0

    total = rng.random(shape)
    total[total < BB_DROPOUT] = 0.0

    return base, total


def _cnaster_counts(
    base: np.ndarray, total: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """`cnaster.hmrf:262-263`, verbatim -- what the patch removes from the loop."""
    return (base > 0).sum(axis=0), (total > 0).sum(axis=0)


def _smooth(n_spots: int) -> csr_matrix:
    rng = np.random.default_rng(17)
    rows = np.repeat(np.arange(n_spots), NEIGHBOURS)
    cols = rng.integers(0, n_spots, NEIGHBOURS * n_spots)
    return csr_matrix(
        (np.ones(NEIGHBOURS * n_spots), (rows, cols)), shape=(n_spots, n_spots)
    )


def _weight_inputs(n_spots: int) -> tuple[BoundaryInvariants, csr_matrix]:
    base, total = _counts_inputs((64, n_spots))
    return boundary_invariants(base, total), _smooth(n_spots)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "shape",
    [
        *tiers(GATE, STRESS),
        pytest.param(WORKING, id="working", marks=pytest.mark.release),
    ],
)
def test_cnaster_counts(benchmark: BenchmarkFixture, shape: tuple[int, int]) -> None:
    """Both passes, per outer iteration, for a constant.

    0.041 ms at the gate shape, 25.2 ms at 3,000 by 5,000, and 48.5 ms at
    `cnaster`'s own working size, 10,000 by 2,500.
    """
    base, total = _counts_inputs(shape)
    benchmark(_cnaster_counts, base, total)


@pytest.mark.benchmark
@pytest.mark.parametrize("shape", tiers(GATE, STRESS))
def test_hoisted_weight(benchmark: BenchmarkFixture, shape: tuple[int, int]) -> None:
    """What the hoist costs once: 0.018 ms at the gate, 0.229 ms at 5,000 spots."""
    invariants, smooth = _weight_inputs(shape[1])
    benchmark(invariants.relative_channel_weight, smooth.indptr, smooth.indices)
