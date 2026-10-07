"""`port.extensions.recolour` and `port.studies.recoloured_clones` (#751).

`recolour` is pinned against a flood fill written here, independent of
SciPy's components. The study's sampler is pinned on a property of the model
it targets: at J = 0 with no field, neighbouring spots share a colour at 1/q.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest
import scipy.sparse as sp


def _hex(rows: int, columns: int) -> sp.csr_matrix:
    """An offset-row hex lattice's adjacency, symmetric, weight 1."""
    index = np.arange(rows * columns).reshape(rows, columns)
    pairs = []
    for r in range(rows):
        for c in range(columns):
            here = index[r, c]
            if c + 1 < columns:
                pairs.append((here, index[r, c + 1]))
            if r + 1 < rows:
                for dc in (0, -1) if r % 2 == 0 else (0, 1):
                    if 0 <= c + dc < columns:
                        pairs.append((here, index[r + 1, c + dc]))
    i, j = np.array(pairs).T
    n = rows * columns
    a = sp.coo_matrix((np.ones(i.size), (i, j)), shape=(n, n))
    return (a + a.T).tocsr()


def _flood(labels: np.ndarray, adjacency: sp.csr_matrix) -> np.ndarray:
    """Connected same-label patches by breadth-first search."""
    patch = np.full(labels.size, -1)
    count = 0
    for root in range(labels.size):
        if patch[root] >= 0:
            continue
        patch[root] = count
        queue = deque([root])
        while queue:
            node = queue.popleft()
            for other in adjacency.indices[
                adjacency.indptr[node] : adjacency.indptr[node + 1]
            ]:
                if patch[other] < 0 and labels[other] == labels[node]:
                    patch[other] = count
                    queue.append(other)
        count += 1
    return patch


@pytest.mark.oracle
def test_recolour_is_the_flood_fill_up_to_naming() -> None:
    """On a 4 x 6 hex lattice and 200 random 3-labellings, `recolour`'s patches are the flood fill's, as a partition."""
    from port.extensions.recolour import recolour

    adjacency = _hex(4, 6)
    rng = np.random.default_rng(751)
    for _ in range(200):
        labels = rng.integers(0, 3, adjacency.shape[0])
        ours, theirs = recolour(labels, adjacency), _flood(labels, adjacency)
        # NB one partition: each patch of one maps to one patch of the other
        pairs = set(zip(ours.tolist(), theirs.tolist(), strict=True))
        assert len(pairs) == len(set(ours.tolist())) == len(set(theirs.tolist()))


@pytest.mark.oracle
def test_clones_of_sums_umis_and_reads_purity() -> None:
    """Two disjoint patches of one label become two clones; UMIs sum over each, purity at `PURE`."""
    from port.studies.recoloured_clones import clones_of

    adjacency = _hex(1, 5)
    labels = np.array([0, 0, 1, 0, 0])
    planted = np.array([0, 0, 0, 1, 1])
    umis = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    found = sorted(clones_of(labels, adjacency, umis, planted), key=lambda k: k["umis"])
    assert [(k["spots"], k["umis"], k["pure"]) for k in found] == [
        (2, 3.0, True), (1, 4.0, True), (2, 24.0, True)
    ]  # fmt: skip


@pytest.mark.analytic
def test_at_zero_coupling_neighbours_share_a_colour_at_one_over_q() -> None:
    """Heat-bath Swendsen-Wang at J = 0, no field, q = 4 on a 20 x 20 hex lattice: same-colour neighbour pairs at 1/4."""
    from port.patch.icm.alpha_expansion import potts_graph_from
    from port.patch.icm.interface import CsrGraph
    from sal.sample.potts_mcmc import PottsMove, sample_potts

    adjacency = _hex(20, 20)
    adjacency.sort_indices()
    graph = potts_graph_from(
        CsrGraph(adjacency.indptr, adjacency.indices, adjacency.data), 0.0
    )
    field = np.zeros((adjacency.shape[0], 4))
    chain = sample_potts(graph, field, PottsMove("swendsen-wang-heat-bath"),
                         np.random.default_rng(1), n_sweeps=50, burn_in=5)  # fmt: skip
    upper = sp.triu(adjacency, k=1).tocoo()
    same = np.array([np.mean(s[upper.row] == s[upper.col]) for s in chain.states])
    pairs = upper.row.size * len(same)
    # NB 4 binomial standard errors over every recorded pair: sweeps at J = 0 are independent
    assert abs(same.mean() - 0.25) < 4 * np.sqrt(0.25 * 0.75 / pairs)
