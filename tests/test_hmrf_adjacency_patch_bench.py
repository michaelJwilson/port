"""Cost of the adjacency round trip against `adjacency_coo`, 1,200 to 20,000 spots (#59 item 3)."""

from collections.abc import Callable

import numpy as np
import pytest
from port.patch.hmrf.adjacency import adjacency_coo
from pytest_benchmark.fixture import BenchmarkFixture
from scipy.sparse import csr_matrix

from tests.fixtures import tiers

GATE_SPOTS = 1_200
STRESS_SPOTS = 20_000
NEIGHBOURS = 6
"""A slice's degree, near enough: `multislice_adjacency` measures 4 to 7."""


def _graph(n_spots: int) -> csr_matrix:
    rng = np.random.default_rng(11)
    rows = np.repeat(np.arange(n_spots), NEIGHBOURS)
    cols = rng.integers(0, n_spots, NEIGHBOURS * n_spots)
    return csr_matrix(
        (np.ones(NEIGHBOURS * n_spots), (rows, cols)), shape=(n_spots, n_spots)
    )


def _cnaster_round_trip(matrix: csr_matrix) -> tuple[np.ndarray, ...]:
    from cnaster.hmrf_utils import cast_csr
    from cnaster.icm import unpack_adjacency

    spots, neighbors, weights = unpack_adjacency(cast_csr(matrix))
    return spots, neighbors, weights


@pytest.mark.benchmark
@pytest.mark.parametrize("n_spots", tiers(GATE_SPOTS, STRESS_SPOTS))
@pytest.mark.parametrize(
    "arm", [_cnaster_round_trip, adjacency_coo], ids=["cnaster", "patched"]
)
def test_adjacency(
    benchmark: BenchmarkFixture, arm: Callable[[csr_matrix], object], n_spots: int
) -> None:
    """Two Python passes over the non-zeros against three `numpy` expressions."""
    benchmark(arm, _graph(n_spots))
