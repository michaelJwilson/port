"""`adjacency_coo` against `cnaster`'s `unpack_adjacency(cast_csr(m))`, bitwise (#59 item 3)."""

import numpy as np
import pytest
from port.patch.hmrf.adjacency import adjacency_coo
from scipy.sparse import csr_matrix


def _cnaster_triple(matrix: csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from cnaster.hmrf_utils import cast_csr
    from cnaster.icm import unpack_adjacency

    spots, neighbors, weights = unpack_adjacency(cast_csr(matrix))
    return spots, neighbors, weights


def _lattice(n_side: int, seed: int) -> csr_matrix:
    """Return a seeded sparse graph with uneven degree."""
    rng = np.random.default_rng(seed)
    n_nodes = n_side * n_side

    rows, cols, data = [], [], []
    for node in range(n_nodes):
        for neighbor in rng.choice(n_nodes, size=rng.integers(1, 7), replace=False):
            rows.append(node)
            cols.append(int(neighbor))
            data.append(float(rng.uniform(0.5, 2.0)))

    return csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))


@pytest.mark.patch
@pytest.mark.parametrize("n_side", [3, 8, 20])
def test_the_triple_is_bitwise_cnasters(n_side: int) -> None:
    """All three arrays and their dtypes equal `cnaster`'s (dtype decides `@njit` specialization)."""
    matrix = _lattice(n_side, seed=11)

    expected = _cnaster_triple(matrix)
    actual = adjacency_coo(matrix)

    for reference, patched in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(reference, patched)
        assert reference.dtype == patched.dtype


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


@pytest.mark.smoke
def test_the_triple_is_the_graph_it_came_from() -> None:
    """The triple rebuilds the matrix it came from."""
    matrix = _lattice(8, seed=3)
    spots, neighbors, weights = adjacency_coo(matrix)

    rebuilt = csr_matrix((weights, (spots, neighbors)), shape=matrix.shape)

    assert (rebuilt != matrix).nnz == 0
