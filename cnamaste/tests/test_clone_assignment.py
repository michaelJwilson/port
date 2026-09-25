"""The spatial graph, the spot-clone field and the label solver (#392 stage 1),
each against brute force or against the labels the instance was planted with."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.stats
from sim.truth import Truth, planted
from sklearn.metrics import adjusted_rand_score

from cnamaste.hmm_nophasing import hmm_nophasing
from cnamaste.hmrf import fused_spot_clone_field, pipeline_clone_assignment
from cnamaste.icm_interface import CsrGraph, icm_sweep
from cnamaste.spatial import (
    construct_multislice_lattice_adjacency,
    initialize_rectangular_clones,
)

TOLERANCE = 1e-8
"""Absolute, in nats, on a sum of up to 120 per-bin log-pmfs."""


@pytest.fixture(scope="module")
def truth() -> Truth:
    return planted(n_clones=3, n_states=4, lattice=(30, 12), n_obs=60, n_segments=3)


def _coords(truth: Truth) -> np.ndarray:
    rows, columns = np.unravel_index(np.arange(truth.n_spots), truth.lattice)
    return np.stack([rows, columns], axis=1).astype(np.float64)


@pytest.fixture(scope="module")
def adjacency(truth: Truth) -> Any:
    """Unit spacing on both axes, as the whole-run configuration sets it."""
    adjacency_mat, _ = construct_multislice_lattice_adjacency(
        np.zeros(truth.n_spots, dtype=np.int64),
        ["S1"],
        _coords(truth),
        None,
        maxspots_pooling=1,
        unit_xsquared=1,
        unit_ysquared=1,
    )
    return sp.csr_matrix(adjacency_mat)


@pytest.fixture
def installed() -> Any:
    """The one key the assignment reads from the global configuration."""
    from cnamaste.config import YAMLConfig, get_global_config, set_global_config

    previous = get_global_config()
    set_global_config(YAMLConfig({"hmrf": {"fixed_assignment": False}}))
    yield
    set_global_config(previous)


def _field(truth: Truth, weight: np.ndarray) -> np.ndarray:
    field: np.ndarray = fused_spot_clone_field(
        truth.counts_nb.astype(np.float64),
        truth.base_nb_mean,
        truth.unphased_bb().astype(np.float64),
        truth.total_bb_RD.astype(np.float64),
        truth.log_mu,
        truth.alphas,
        truth.p_binom,
        truth.taus,
        np.ascontiguousarray(truth.states.T),
        weight,
        np.empty((truth.n_spots, truth.n_clones)),
    )
    return field


@pytest.mark.oracle
def test_each_spot_is_joined_to_its_eight_nearest_spots(
    truth: Truth, adjacency: Any
) -> None:
    """Eight out-edges per spot, none farther than the spot's eighth-nearest.

    `construct_lattice_adjacency` queries a k-d tree for `coordination_num = 8`
    neighbours: the Moore neighbourhood in the interior, and at an edge or
    corner whichever of the tied farther spots the tree returns first -- so
    the referee bounds the distance rather than naming the set.
    """
    coords = _coords(truth)
    distance = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    np.fill_diagonal(distance, np.inf)
    eighth = np.sort(distance, axis=1)[:, 7]

    np.testing.assert_array_equal(np.diff(adjacency.indptr), 8)
    for spot in range(truth.n_spots):
        neighbours = adjacency.indices[
            adjacency.indptr[spot] : adjacency.indptr[spot + 1]
        ]
        assert np.all(distance[spot, neighbours] <= eighth[spot] + 1e-12), spot

    interior = (
        (coords[:, 0] > 0)
        & (coords[:, 0] < truth.lattice[0] - 1)
        & (coords[:, 1] > 0)
        & (coords[:, 1] < truth.lattice[1] - 1)
    )
    moore = (distance <= np.sqrt(2) + 1e-12)[interior]
    np.testing.assert_array_equal(adjacency.toarray()[interior] != 0, moore)


@pytest.mark.analytic
@pytest.mark.xfail(
    strict=True,
    reason="a k-nearest-neighbour graph is directed: at an edge, spot j can be "
    "among i's eight nearest without i among j's. The paper's Potts prior is on "
    "an undirected graph, and the ICM re-queues only a changed spot's "
    "out-neighbours, so a directed graph can leave a spot off its optimum",
)
def test_the_spot_graph_is_undirected(adjacency: Any) -> None:
    """Realized: 192 of 2,880 entries have no reverse edge, each within one spot of the boundary."""
    assert (adjacency != adjacency.T).nnz == 0


@pytest.mark.oracle
def test_the_field_is_the_summed_log_likelihood_under_each_clones_path(
    truth: Truth,
) -> None:
    """`w_s * sum_g log NB + sum_g log BB`, per spot and candidate clone."""
    weight = np.random.default_rng(0).uniform(0.5, 2.0, truth.n_spots)
    field = _field(truth, weight)

    b = truth.unphased_bb()
    for clone in range(truth.n_clones):
        path = truth.states[clone]
        mean = truth.base_nb_mean * np.exp(truth.log_mu[path])[:, None]
        r = 1.0 / truth.alphas[path][:, None]
        nb = scipy.stats.nbinom.logpmf(truth.counts_nb, r, r / (r + mean)).sum(axis=0)
        p = truth.p_binom[path][:, None]
        tau = truth.taus[path][:, None]
        bb = scipy.stats.betabinom.logpmf(
            b, truth.total_bb_RD, p * tau, (1 - p) * tau
        ).sum(axis=0)
        np.testing.assert_allclose(
            field[:, clone], weight * nb + bb, rtol=0, atol=TOLERANCE
        )


@pytest.mark.end2end
def test_the_field_alone_assigns_every_spot_its_planted_clone(truth: Truth) -> None:
    """Under the planted parameters and paths, no spatial prior is needed."""
    field = _field(truth, np.ones(truth.n_spots))
    np.testing.assert_array_equal(field.argmax(axis=1), truth.labels)


@pytest.mark.analytic
def test_the_solver_stops_at_a_local_optimum(adjacency: Any) -> None:
    """No single spot can raise `field[s, c] + beta * sum_j w_sj [c == z_j]` alone.

    On the symmetrized graph: on the directed one a spot is not re-queued
    when a spot it lists changes, and the property does not hold.
    """
    adjacency = ((adjacency + adjacency.T) > 0).astype(np.float64).tocsr()
    rng = np.random.default_rng(1)
    n_spots, n_clones, beta = adjacency.shape[0], 3, 0.7
    field = rng.normal(0.0, 1.0, (n_spots, n_clones))
    assignment = rng.integers(0, n_clones, n_spots)

    np.random.seed(2)  # noqa: NPY002 -- the solver draws from the legacy global
    icm_sweep(
        field, CsrGraph.from_matrix(adjacency), assignment, beta, min_clone_spots=0
    )

    agreement = np.stack(
        [
            adjacency @ (assignment == clone).astype(np.float64)
            for clone in range(n_clones)
        ],
        axis=1,
    )
    score = field + beta * agreement
    chosen = score[np.arange(n_spots), assignment]
    assert np.all(chosen >= score.max(axis=1) - 1e-12)


@pytest.mark.end2end
@pytest.mark.usefixtures("installed")
def test_the_assignment_recovers_the_planted_clones_from_rectangles(
    truth: Truth, adjacency: Any
) -> None:
    """From the pipeline's own rectangular start, under the planted model.

    The returned likelihood is the field at the chosen labels plus the
    coupling times the number of agreeing edges, which is asserted too.
    """
    _, start = initialize_rectangular_clones(_coords(truth), truth.n_clones)
    res = {
        "new_log_mu": truth.log_mu[:, None],
        "new_alphas": truth.alphas[:, None],
        "new_p_binom": truth.p_binom[:, None],
        "new_taus": truth.taus[:, None],
    }
    single_X = np.stack([truth.counts_nb, truth.unphased_bb()], axis=1).astype(
        np.float64
    )

    np.random.seed(0)  # noqa: NPY002 -- the solver draws from the legacy global
    labels, field, log_likelihood = pipeline_clone_assignment(
        single_X,
        truth.base_nb_mean,
        truth.total_bb_RD.astype(np.float64),
        res,
        np.ascontiguousarray(truth.states.T),
        adjacency,
        np.asarray(start, dtype=np.int64),
        np.zeros(truth.n_spots, dtype=np.int64),
        1.0,
        hmmclass=hmm_nophasing,
    )

    assert adjusted_rand_score(truth.labels, labels) == 1.0

    rows, columns = sp.triu(adjacency, k=1).nonzero()
    expected = field[np.arange(truth.n_spots), labels].sum() + np.sum(
        labels[rows] == labels[columns]
    )
    assert abs(log_likelihood - expected) < 1e-6 * abs(expected)


@pytest.mark.analytic
@pytest.mark.parametrize("n_clones", [2, 3, 4, 5])
def test_rectangular_clones_are_unions_of_a_grid_of_blocks(
    truth: Truth, n_clones: int
) -> None:
    """A partition into clones of at least a fifth of an equal share, each a
    union of the cells of a `p x p` grid, `p = ceil(sqrt(n_clones))`, and the
    same partition for the same seed.

    On the lattice that is: the label image changes between at most `p - 1`
    pairs of adjacent rows and at most `p - 1` pairs of adjacent columns.
    """
    coords = _coords(truth)
    groups, labels = initialize_rectangular_clones(coords, n_clones, random_state=3)
    _, again = initialize_rectangular_clones(coords, n_clones, random_state=3)

    np.testing.assert_array_equal(labels, again)
    assert len(groups) == n_clones
    np.testing.assert_array_equal(
        np.sort(np.concatenate(groups)), np.arange(truth.n_spots)
    )
    assert min(len(g) for g in groups) > 0.2 * truth.n_spots / n_clones

    p = int(np.ceil(np.sqrt(n_clones)))
    image = labels.reshape(truth.lattice)
    row_breaks = np.any(image[1:] != image[:-1], axis=1).sum()
    column_breaks = np.any(image[:, 1:] != image[:, :-1], axis=0).sum()
    assert row_breaks <= p - 1
    assert column_breaks <= p - 1
