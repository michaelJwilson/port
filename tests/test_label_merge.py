"""#556, T- #829: the label merge `potts_stream` polishes with, the hex graph and the BAF overdispersion estimator.

The merge is sal's (`merge_labels`, inside `Polish.ICM_MERGE`); port's
closed-form copy (`port.studies.color_merge`) is retired by T- #829. These pin
sal's against a brute-force energy difference, against `cnaster`'s own merge
where they correspond, and where `cnaster`'s differs.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from port.studies.field_strength import overdispersion
from port.studies.potts_stream import hex_graph


def _hex(n_side: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A hex patch's points and CSR neighbours, from `potts_stream.hex_graph`."""
    rows, cols = np.divmod(np.arange(n_side * n_side), n_side)
    points = np.column_stack([cols + 0.5 * (rows % 2), rows * np.sqrt(3) / 2])
    return (points, *hex_graph(points))


def _graph(beta: float):  # type: ignore[no-untyped-def]
    """`_hex()` as the run couples it: `potts_graph_from`, beta (A_ij + A_ji) / 2 per pair."""
    from port.patch.icm.alpha_expansion import potts_graph_from
    from port.patch.icm.interface import CsrGraph

    _, indptr, indices, weights = _hex()
    return potts_graph_from(CsrGraph(indptr, indices, weights), beta)


def _deltas(graph, field: np.ndarray, labels: np.ndarray) -> np.ndarray:  # type: ignore[no-untyped-def]
    """`(q, q)` energy change of relabelling all of u to v, by `sal.sim.potts.energy`; `inf` where u = v or either is empty."""
    from sal.sim.potts import energy

    q = field.shape[1]
    before = energy(graph, field, labels)
    alive = np.bincount(labels, minlength=q) > 0
    delta = np.full((q, q), np.inf)
    for u in np.flatnonzero(alive):
        for v in np.flatnonzero(alive):
            if u != v:
                delta[u, v] = (
                    energy(graph, field, np.where(labels == u, v, labels)) - before
                )
    return delta


def _greedy(graph, field: np.ndarray, labels: np.ndarray) -> np.ndarray:  # type: ignore[no-untyped-def]
    """cnaster's `merge_assignment` rule by brute force: merge the most negative pair, first in row-major order, until none is."""
    labels = labels.copy()
    while True:
        delta = _deltas(graph, field, labels)
        u, v = np.unravel_index(int(np.argmin(delta)), delta.shape)
        if not delta[u, v] < 0:
            return labels
        labels[labels == u] = v


@pytest.mark.oracle
@pytest.mark.parametrize("seed", [2, 4, 5])
def test_sals_merge_is_the_greedy_merge_by_brute_force(seed: int) -> None:
    """`merge_labels` on random labels: the brute-force greedy labelling, bitwise, and its energy to 1e-12 relative."""
    from sal.search.icm import merge_labels
    from sal.sim.potts import energy

    graph = _graph(0.7)
    rng = np.random.default_rng(seed)
    field = rng.normal(size=(graph.n_nodes, 6)) * 0.2
    labels = rng.integers(0, 6, graph.n_nodes)
    sal = np.asarray(merge_labels(graph, field, labels).labelling, dtype=np.int64)
    ours = _greedy(graph, field, labels)

    assert np.unique(ours).size < np.unique(labels).size
    np.testing.assert_array_equal(sal, ours)
    assert energy(graph, field, sal) == pytest.approx(
        energy(graph, field, ours), rel=1e-12
    )


def _cnaster_gain(
    field: np.ndarray, labels: np.ndarray, beta: float
) -> tuple[float, tuple[int, int]]:
    """`cnaster.icm.merge_assignment`'s best cost gain and pair on `_hex()`."""
    from cnaster.icm import merge_assignment

    _, indptr, indices, weights = _hex()
    coo = sp.csr_matrix((weights, indices, indptr)).tocoo()
    cost, best, pair = merge_assignment(
        field, coo.row, coo.col, coo.data, labels.copy(), beta
    )
    return float(best - cost), (int(pair[0]), int(pair[1]))


@pytest.mark.oracle
def test_the_merges_field_term_is_cnasters() -> None:
    """At beta = 1e-9, against `cnaster.icm.merge_assignment`: its gain is the brute-force drop, its pair the argmin.

    cnaster scores only pairs that share a boundary, so beta is not 0; random
    labels put every pair on one, and the boundary term is then below 1e-6.
    """
    rng = np.random.default_rng(3)
    graph = _graph(1e-9)
    field = rng.normal(size=(graph.n_nodes, 4)) * 0.3
    labels = rng.integers(0, 4, graph.n_nodes)
    delta = _deltas(graph, field, labels)
    ours = np.unravel_index(int(np.argmin(delta)), delta.shape)
    gain, pair = _cnaster_gain(field, labels, 1e-9)

    assert pair == (int(ours[0]), int(ours[1]))
    assert gain == pytest.approx(-delta[ours], abs=1e-6)


@pytest.mark.bug
def test_cnaster_scores_a_merges_boundary_at_half_the_energy_it_removes() -> None:
    """`cnaster.icm.merge_assignment` adds `boundary_gain[u, v]`, one direction of the u-v boundary, to its gain.

    Its own cost (`calc_assignment_cost`) counts beta w / 2 per directed edge,
    beta w per undirected one, so merging u into v removes beta (B[u, v] +
    B[v, u]) / 2 of energy: twice what it scores. It merges less than its own
    cost would, and can pick another pair. Fails when cnaster adds both
    directions.
    """
    rng = np.random.default_rng(3)
    field = rng.normal(size=(_graph(1.0).n_nodes, 4)) * 0.3
    labels = rng.integers(0, 4, int(field.shape[0]))
    gain, (u, v) = _cnaster_gain(field, labels, 1.0)
    unary = -_deltas(_graph(0.0), field, labels)[u, v]
    spatial = -_deltas(_graph(1.0), field, labels)[u, v] - unary

    assert spatial > 1.0
    assert gain == pytest.approx(unary + spatial / 2, rel=1e-9)


@pytest.mark.analytic
def test_sals_merge_never_raises_the_energy_and_stops_where_no_pair_lowers_it() -> None:
    """Monotone descent: the merged energy is no higher, and at the end every pair's change is >= 0."""
    from sal.search.icm import merge_labels
    from sal.sim.potts import energy

    graph = _graph(1.0)
    rng = np.random.default_rng(4)
    field = rng.normal(size=(graph.n_nodes, 6)) * 0.2
    labels = rng.integers(0, 6, graph.n_nodes)
    merged = np.asarray(merge_labels(graph, field, labels).labelling, dtype=np.int64)

    assert np.unique(merged).size < np.unique(labels).size
    assert energy(graph, field, merged) < energy(graph, field, labels)
    assert (_deltas(graph, field, merged) >= -1e-12).all()


@pytest.mark.analytic
def test_the_hex_graph_gives_interior_points_six_neighbours() -> None:
    """A hex patch: 6 neighbours inside, fewer on the edge, symmetric, weight 1."""
    points, indptr, indices, weights = _hex()
    degree = np.diff(indptr)
    adjacency = sp.csr_matrix((weights, indices, indptr))

    assert degree.max() == 6
    assert (adjacency != adjacency.T).nnz == 0
    assert set(np.unique(weights)) == {1.0}


@pytest.mark.analytic
def test_the_moment_overdispersion_recovers_the_rho_it_was_drawn_at() -> None:
    """Beta-binomial draws at rho 0 and 0.05 on 20,000 entries of 2-4 reads: recovered within 0.01."""
    rng = np.random.default_rng(6)
    n = rng.integers(2, 5, size=20_000).astype(float)
    p = rng.uniform(0.2, 0.8, size=n.size)
    for rho in (0.0, 0.05):
        drawn = p if rho == 0 else rng.beta(p * (1 / rho - 1), (1 - p) * (1 / rho - 1))
        b = rng.binomial(n.astype(np.int64), drawn).astype(float)
        assert overdispersion(b, n, p) == pytest.approx(rho, abs=0.01)
