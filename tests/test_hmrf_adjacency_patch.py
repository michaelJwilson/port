"""The COO adjacency patch, pinned bitwise against `cnaster`'s round trip.

Issue #59 item 3. `port.patch.hmrf_adjacency.adjacency_coo` replaces
`unpack_adjacency(cast_csr(m))` -- two pure-Python passes over the non-zeros
-- with three `numpy` expressions on the CSR arrays already in hand.

Landed as a **simplification**: `CLAUDE.md` says a patch that makes the code
plainer is worth its evidence of equivalence alone, and the 55-282x it
measures is under a tenth of a per cent of the boundary. The benchmark
records it; these tests decide it.
"""

import numpy as np
import pytest
from port.patch.hmrf_adjacency import adjacency_coo
from scipy.sparse import csr_matrix


def _cnaster_triple(matrix: csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from cnaster.hmrf_utils import cast_csr
    from cnaster.icm import unpack_adjacency

    spots, neighbors, weights = unpack_adjacency(cast_csr(matrix))
    return spots, neighbors, weights


def _lattice(n_side: int, seed: int) -> csr_matrix:
    """A seeded sparse graph with an uneven degree, which is what a slice is."""
    rng = np.random.default_rng(seed)
    n_nodes = n_side * n_side

    rows, cols, data = [], [], []
    for node in range(n_nodes):
        for neighbor in rng.choice(n_nodes, size=rng.integers(1, 7), replace=False):
            rows.append(node)
            cols.append(int(neighbor))
            data.append(float(rng.uniform(0.5, 2.0)))

    return csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))


@pytest.mark.cnaster
@pytest.mark.parametrize("n_side", [3, 8, 20])
def test_the_triple_is_bitwise_cnasters(n_side: int) -> None:
    """All three arrays identical, and the dtypes with them.

    The dtypes matter: `icm.calc_assignment_cost` is `@njit`, so a `float32`
    weight or an `int32` index would specialize the kernel differently. A
    comparison on values alone would pass while changing what `cnaster`
    compiles.
    """
    matrix = _lattice(n_side, seed=11)

    expected = _cnaster_triple(matrix)
    actual = adjacency_coo(matrix)

    for reference, patched in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(reference, patched)
        assert reference.dtype == patched.dtype


@pytest.mark.cnaster
def test_the_triple_survives_an_empty_row() -> None:
    """A spot with no neighbours contributes nothing, rather than a zero.

    The degenerate case `np.repeat` gets right by construction and a hand
    loop gets wrong by forgetting: a filtered slice can leave an isolated
    spot, and an entry for it would give `calc_assignment_cost` a
    self-neighbour at weight zero.
    """
    matrix = csr_matrix(([1.0, 2.0], ([0, 2], [2, 0])), shape=(3, 3))
    assert np.diff(matrix.indptr)[1] == 0

    expected = _cnaster_triple(matrix)
    actual = adjacency_coo(matrix)

    for reference, patched in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(reference, patched)

    assert 1 not in actual[0].tolist()


@pytest.mark.analytic
def test_the_triple_is_the_graph_it_came_from() -> None:
    """Round trip: the triple rebuilds the matrix.

    An invariant rather than a comparison, so it holds whichever
    implementation is right -- and it is what says the patch is a
    *representation* change and not a computation.
    """
    matrix = _lattice(8, seed=3)
    spots, neighbors, weights = adjacency_coo(matrix)

    rebuilt = csr_matrix((weights, (spots, neighbors)), shape=matrix.shape)

    assert (rebuilt != matrix).nnz == 0
