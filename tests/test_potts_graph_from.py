"""`potts_graph_from` through sal's `from_csr`, one-way edges at half (#410 step 3, sal #1113).

Referees: the upper-triangle loop it replaces, bitwise on symmetric graphs;
and, on `cnaster`'s directed kNN graph, the coupling `cnaster`'s ICM sums --
each spot's own row -- which a `PottsGraph` must carry at half, as it
carries every symmetric graph.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp


def _upper_triangle(
    graph: Any, beta: float
) -> tuple[tuple[Any, ...], tuple[float, ...]]:
    """The loop `from_csr` replaced: each edge once, from the upper triangle."""
    edges, coupling = [], []
    for site in range(graph.indptr.size - 1):
        for slot in range(int(graph.indptr[site]), int(graph.indptr[site + 1])):
            neighbour = int(graph.indices[slot])
            if neighbour > site:
                edges.append((site, neighbour))
                coupling.append(float(beta) * float(graph.weights[slot]))
    return tuple(edges), tuple(coupling)


def _square(side: int) -> np.ndarray:
    return np.stack(np.unravel_index(np.arange(side * side), (side, side)), axis=1)


@pytest.mark.patch
@pytest.mark.parametrize("side", [6, 40])
def test_a_symmetric_graph_converts_bitwise_as_the_loop_did(side: int) -> None:
    """The lattice construction's reinforced graph: same edges, same couplings."""
    from port.extensions.adjacency import lattice_adjacency
    from port.patch.icm.alpha_expansion import potts_graph_from
    from port.patch.icm.interface import CsrGraph

    graph = CsrGraph.from_matrix(lattice_adjacency(_square(side), "moore"))
    ours = potts_graph_from(graph, 0.6)
    edges, coupling = _upper_triangle(graph, 0.6)

    assert ours.edges == edges
    assert ours.coupling == coupling


@pytest.mark.analytic
@pytest.mark.cnaster
def test_a_one_way_edge_carries_half_the_coupling_cnasters_row_sum_does() -> None:
    """Zero field, any labelling: `-E = beta / 2 * sum_i sum_{j in row i} [s_i = s_j]`.

    On `cnaster`'s own kNN graph (40 x 40, eight per row, one-way only at
    the boundary), for five random labellings.
    """
    from cnaster.spatial import construct_lattice_adjacency
    from port.patch.icm.alpha_expansion import potts_graph_from
    from port.patch.icm.interface import CsrGraph
    from sal.sim.potts import energy

    _, directed = construct_lattice_adjacency(
        _square(40).astype(float), unit_xsquared=1, unit_ysquared=1
    )
    beta = 0.6
    potts = potts_graph_from(CsrGraph.from_matrix(directed), beta)
    coo = directed.tocoo()
    rng = np.random.default_rng(0)

    for _ in range(5):
        labels = rng.integers(0, 4, 1600)
        same = labels[coo.row] == labels[coo.col]
        expected = 0.5 * beta * float(np.sum(coo.data[same]))

        assert -energy(potts, np.zeros((1600, 4)), labels) == pytest.approx(
            expected, rel=1e-12
        )


@pytest.mark.patch
def test_a_mostly_one_way_graph_is_refused() -> None:
    """Eight random targets per spot: almost nothing reciprocated, under 0.6.

    `patch`, as `test_alpha_expansion`'s negative-coupling refusal is: the
    conversion's contract, where the loop it replaced accepted anything.
    """
    from port.extensions.adjacency import AdjacencyError
    from port.patch.icm.alpha_expansion import potts_graph_from
    from port.patch.icm.interface import CsrGraph

    rng = np.random.default_rng(0)
    n = 144
    targets = np.array(
        [rng.choice(np.delete(np.arange(n), i), 8, replace=False) for i in range(n)]
    )
    one_way = sp.csr_matrix(
        (np.ones(n * 8), (np.repeat(np.arange(n), 8), targets.ravel())), shape=(n, n)
    )

    with pytest.raises(AdjacencyError, match="reciprocated"):
        potts_graph_from(CsrGraph.from_matrix(one_way), 0.6)
