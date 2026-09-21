"""What the adjacency round trip costs (issue #59 item 3).

Recorded rather than argued: the ratio is large and the saving is not.

| spots | non-zeros | `cast_csr` + `unpack_adjacency` | `adjacency_coo` | ratio |
| ---: | ---: | ---: | ---: | ---: |
| 1,200 | 7,192 | 2.8 ms | 0.014 ms | 202 |
| 5,000 | 29,987 | 11.4 ms | 0.040 ms | 282 |
| 20,000 | 119,994 | 52.7 ms | 0.951 ms | 55 |

11 ms per outer iteration against a boundary costing about 16 s is under a
tenth of a per cent, so this lands as a simplification and the numbers are
here to stop anyone claiming otherwise. `CLAUDE.md` separates the two cases
and this is the first.
"""

import numpy as np
import pytest
from port.patch.hmrf.adjacency import adjacency_coo
from pytest_benchmark.fixture import BenchmarkFixture
from scipy.sparse import csr_matrix

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
def test_cnaster_adjacency_round_trip_gate(benchmark: BenchmarkFixture) -> None:
    """Two Python passes over the non-zeros, at gate size."""
    benchmark(_cnaster_round_trip, _graph(GATE_SPOTS))


@pytest.mark.benchmark
def test_patched_adjacency_gate(benchmark: BenchmarkFixture) -> None:
    """Three `numpy` expressions, at gate size."""
    benchmark(adjacency_coo, _graph(GATE_SPOTS))


@pytest.mark.benchmark
@pytest.mark.release
def test_cnaster_adjacency_round_trip_stress(benchmark: BenchmarkFixture) -> None:
    """The same at 20,000 spots, where the Python loop is 52.7 ms."""
    benchmark(_cnaster_round_trip, _graph(STRESS_SPOTS))


@pytest.mark.benchmark
@pytest.mark.release
def test_patched_adjacency_stress(benchmark: BenchmarkFixture) -> None:
    """And the patch, at 0.951 ms."""
    benchmark(adjacency_coo, _graph(STRESS_SPOTS))
