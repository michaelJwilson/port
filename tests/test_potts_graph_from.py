"""`potts_graph_from` via sal's `from_csr`, one-way edges at half (#410 step 3, sal #1113).

Referees: the replaced upper-triangle loop, bitwise; `cnaster`'s ICM row-sum coupling.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp
from cnaster.spatial import construct_lattice_adjacency
from port.extensions.adjacency import AdjacencyError, lattice_adjacency
from port.patch.icm.alpha_expansion import potts_graph_from
from port.patch.icm.interface import CsrGraph
from sal.sim.graph import PottsGraph
from sal.sim.potts import energy

from tests.adapters import square_coords


def _upper_triangle(
    graph: Any, beta: float
) -> tuple[tuple[Any, ...], tuple[float, ...]]:
    """Return edges and couplings by the replaced loop: each edge once, upper triangle."""
    edges, coupling = [], []
    for site in range(graph.indptr.size - 1):
        for slot in range(int(graph.indptr[site]), int(graph.indptr[site + 1])):
            neighbour = int(graph.indices[slot])
            if neighbour > site:
                edges.append((site, neighbour))
                coupling.append(float(beta) * float(graph.weights[slot]))
    return tuple(edges), tuple(coupling)


@pytest.mark.patch
@pytest.mark.parametrize("side", [6, 40])
def test_a_symmetric_graph_converts_bitwise_as_the_loop_did(side: int) -> None:
    """The lattice construction's reinforced graph: same edges, same couplings."""

    graph = CsrGraph.from_matrix(lattice_adjacency(square_coords(side, side), "moore"))
    ours = potts_graph_from(graph, 0.6)
    edges, coupling = _upper_triangle(graph, 0.6)

    assert ours.edges == edges
    assert ours.coupling == coupling


@pytest.mark.analytic
@pytest.mark.cnaster
def test_a_one_way_edge_carries_half_the_coupling_cnasters_row_sum_does() -> None:
    """Zero-field energy equals half `cnaster`'s row-sum coupling on its kNN graph, to 1e-12."""

    _, directed = construct_lattice_adjacency(
        square_coords(40, 40).astype(float), unit_xsquared=1, unit_ysquared=1
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
    """A graph with under 0.6 of edges reciprocated is refused."""

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


def _symmetrized_from_csr(graph: Any, beta: float) -> Any:
    """Return the replaced construction (T- #632): `(A + A^T) * beta / 2` via `from_csr`."""

    n = int(graph.indptr.size - 1)
    matrix = sp.csr_matrix((graph.weights, graph.indices, graph.indptr), shape=(n, n))
    matrix.eliminate_zeros()
    symmetric = ((matrix + matrix.T) * (0.5 * float(beta))).tocsr()
    symmetric.sort_indices()
    return PottsGraph.from_csr(symmetric.indptr, symmetric.indices, symmetric.data)


@pytest.mark.patch
@pytest.mark.parametrize("beta", [0.6, 1.3, 7.0])
def test_from_directed_csr_builds_the_symmetrized_graph_bitwise(beta: float) -> None:
    """On `cnaster`'s directed kNN graph, edges and couplings equal `_symmetrized_from_csr`'s."""

    _, directed = construct_lattice_adjacency(
        square_coords(40, 40).astype(float), unit_xsquared=1, unit_ysquared=1
    )
    directed = directed.tocsr()
    directed.data = np.random.default_rng(1).uniform(0.1, 3.0, directed.nnz)
    graph = CsrGraph.from_matrix(directed)
    ours, oracle = potts_graph_from(graph, beta), _symmetrized_from_csr(graph, beta)

    assert ours.edges == oracle.edges
    assert ours.coupling == oracle.coupling
