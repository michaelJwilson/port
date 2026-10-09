"""Cost of the adjacency round trip against `adjacency_coo`, 1,200 to 20,000 spots (#59 item 3)."""

from collections.abc import Callable

import numpy as np
import pytest
from port.patch.hmrf.adjacency import adjacency_coo
from pytest_benchmark.fixture import BenchmarkFixture
from scipy.sparse import csr_matrix

from tests.adapters import cnaster_adjacency_triple
from tests.builders import regular_graph
from tests.fixtures import tiers

GATE_SPOTS = 1_200
STRESS_SPOTS = 20_000
NEIGHBOURS = 6
"""A slice's degree, near enough: `multislice_adjacency` measures 4 to 7."""


def _graph(n_spots: int) -> csr_matrix:
    return regular_graph(np.random.default_rng(11), n_spots, NEIGHBOURS)


_cnaster_round_trip = cnaster_adjacency_triple


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
