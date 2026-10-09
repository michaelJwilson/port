"""`adjacency_coo` against `cnaster`'s `unpack_adjacency(cast_csr(m))`, bitwise (#59 item 3)."""

import numpy as np
import pytest
from port.patch.hmrf.adjacency import adjacency_coo
from scipy.sparse import csr_matrix

from tests.adapters import cnaster_adjacency_triple
from tests.builders import random_graph

_cnaster_triple = cnaster_adjacency_triple


def _lattice(n_side: int, seed: int) -> csr_matrix:
    """Return a seeded sparse graph with uneven degree."""
    return random_graph(np.random.default_rng(seed), n_side * n_side, (1, 7))


@pytest.mark.patch
def test_the_triple_survives_an_empty_row() -> None:
    """An isolated spot contributes no entry, as in `cnaster`'s."""
    matrix = csr_matrix(([1.0, 2.0], ([0, 2], [2, 0])), shape=(3, 3))
    assert np.diff(matrix.indptr)[1] == 0

    expected = _cnaster_triple(matrix)
    actual = adjacency_coo(matrix)

    for reference, patched in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(reference, patched)

    assert 1 not in actual[0].tolist()


@pytest.mark.analytic
def test_the_triple_is_the_graph_it_came_from() -> None:
    """The triple rebuilds the matrix it came from."""
    matrix = _lattice(8, seed=3)
    spots, neighbors, weights = adjacency_coo(matrix)

    rebuilt = csr_matrix((weights, (spots, neighbors)), shape=matrix.shape)

    assert (rebuilt != matrix).nnz == 0
