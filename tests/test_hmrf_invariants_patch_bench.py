"""The per-iteration count passes `hmrf` recomputes, and the hoisted weight's one-off
cost (#59 item 4).

Shapes up to cnaster's working size, 10,000 by 2,500; lands as a simplification.
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
    """`cnaster.hmrf:262-263`, verbatim: what the patch removes from the loop."""
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
    """Both passes, per outer iteration, at the gate, 3,000 x 5,000 and 10,000 x 2,500."""
    base, total = _counts_inputs(shape)
    benchmark(_cnaster_counts, base, total)


@pytest.mark.benchmark
@pytest.mark.parametrize("shape", tiers(GATE, STRESS))
def test_hoisted_weight(benchmark: BenchmarkFixture, shape: tuple[int, int]) -> None:
    """What the hoist costs once."""
    invariants, smooth = _weight_inputs(shape[1])
    benchmark(invariants.relative_channel_weight, smooth.indptr, smooth.indices)
