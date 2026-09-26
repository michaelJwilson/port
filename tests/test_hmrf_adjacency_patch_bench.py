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
    """Two Python passes over the non-zeros, against three `numpy` expressions.

    At 20,000 spots the Python loop is 52.7 ms and the patch 0.951 ms.
    """
    benchmark(arm, _graph(n_spots))
