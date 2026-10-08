"""Potts labelling fixtures, and `cnaster`'s objective against upstream's `energy` (#40).

`cnaster`'s assignment cost is the exact negation of upstream's energy; the planted
labelling is not the optimum, checked by exhaustive enumeration.
"""

import numpy as np
import pytest
from sal.enumeration import MAX_ENUMERABLE_CONFIGURATIONS

from tests.adapters import (
    cnaster_assignment_cost,
    cnaster_icm_labelling,
    cnaster_potts_adjacency,
    upstream_potts_energy,
)
from tests.fixtures import PottsLabels, enumerate_minimum_energy, potts_labels

SIGN_TOLERANCE = 1e-12
"""Absolute tolerance on `cost + energy == 0`: float64 reassociation (measured 7.1e-15)."""


@pytest.fixture
def lattice() -> PottsLabels:
    """The default draw: a 6x6 open lattice, three clones, 60 edges."""
    return potts_labels()


@pytest.fixture
def enumerable() -> PottsLabels:
    """A 10-site, three-clone lattice with weak signal, small enough to enumerate."""
    return potts_labels(shape=(5, 2), n_clones=3, signal=0.6, noise=1.0)


@pytest.mark.smoke
def test_the_edge_set_survives_the_csr_conversion(lattice: PottsLabels) -> None:
    """The CSR adjacency holds each undirected edge twice, symmetric, no self-loops (#12)."""
    matrix = cnaster_potts_adjacency(lattice)

    assert (matrix != matrix.T).nnz == 0, "the adjacency is not symmetric"
    assert matrix.diagonal().sum() == 0.0, "the conversion introduced a self-loop"
    assert matrix.nnz == 2 * len(lattice.graph.edges)

    expected: dict[tuple[int, int], float] = {}
    for (left, right), coupling in zip(
        lattice.graph.edges, lattice.graph.coupling, strict=True
    ):
        key = (min(left, right), max(left, right))
        expected[key] = expected.get(key, 0.0) + coupling

    coo = matrix.tocoo()
    seen: dict[tuple[int, int], float] = {}
    for row, col, value in zip(coo.row, coo.col, coo.data, strict=True):
        if row < col:
            seen[(int(row), int(col))] = seen.get((int(row), int(col)), 0.0) + value

    assert seen == pytest.approx(expected)


@pytest.mark.smoke
@pytest.mark.parametrize("coupling", [0.0, 0.5, 2.0])
def test_cnaster_maximises_what_upstream_minimises(coupling: float) -> None:
    """`calc_assignment_cost` is the negation of upstream `energy`, to `SIGN_TOLERANCE`."""
    fixture = potts_labels(
        shape=(4, 4), n_clones=3, coupling=coupling, spatial_weight=1.5
    )

    rng = np.random.default_rng(fixture.seed)
    labellings = [
        fixture.labels,
        np.zeros(fixture.n_nodes, dtype=np.int64),
        np.arange(fixture.n_nodes, dtype=np.int64) % fixture.n_clones,
        rng.integers(0, fixture.n_clones, size=fixture.n_nodes),
    ]

    for labelling in labellings:
        cost = cnaster_assignment_cost(fixture, labelling)
        energy = upstream_potts_energy(fixture, labelling)
        assert cost + energy == pytest.approx(0.0, abs=SIGN_TOLERANCE)


@pytest.mark.analytic
def test_at_zero_coupling_the_problem_separates(lattice: PottsLabels) -> None:
    """At zero coupling the field's argmax is the optimum, node by node."""
    uncoupled = potts_labels(
        shape=lattice.shape, n_clones=lattice.n_clones, coupling=0.0
    )
    argmax = np.argmax(uncoupled.field, axis=1)

    best = cnaster_assignment_cost(uncoupled, argmax)
    assert best == pytest.approx(float(np.max(uncoupled.field, axis=1).sum()))

    rng = np.random.default_rng(uncoupled.seed)
    for _ in range(8):
        other = rng.integers(0, uncoupled.n_clones, size=uncoupled.n_nodes)
        assert cnaster_assignment_cost(uncoupled, other) <= best


@pytest.mark.end2end
def test_the_planted_labelling_is_not_the_optimum(enumerable: PottsLabels) -> None:
    """Exhaustive search beats the planted labelling's energy."""
    optimum, best = enumerate_minimum_energy(enumerable)
    planted = upstream_potts_energy(enumerable, enumerable.labels)

    assert best < planted, (
        f"the planted labelling is already optimal at energy {planted:.6f}; "
        "the fixture's signal is too strong for the comparison to have content"
    )
    assert not np.array_equal(optimum, enumerable.labels)


@pytest.mark.smoke
def test_enumeration_refuses_what_it_cannot_search() -> None:
    """Enumeration past `MAX_ENUMERABLE_CONFIGURATIONS` raises."""
    too_large = potts_labels(shape=(6, 6), n_clones=3)
    assert too_large.n_clones**too_large.n_nodes > MAX_ENUMERABLE_CONFIGURATIONS

    with pytest.raises(ValueError, match="refusing to enumerate"):
        enumerate_minimum_energy(too_large)


@pytest.mark.oracle
def test_icm_reports_the_cost_of_the_labelling_it_returns(lattice: PottsLabels) -> None:
    """ICM's reported cost equals the recomputed objective change, to `SIGN_TOLERANCE`."""
    start = np.zeros(lattice.n_nodes, dtype=np.int64)

    before = cnaster_assignment_cost(lattice, start)
    labelling, reported, _ = cnaster_icm_labelling(lattice, start)
    after = cnaster_assignment_cost(lattice, labelling)

    assert reported == pytest.approx(after - before, abs=SIGN_TOLERANCE)


@pytest.mark.analytic
@pytest.mark.parametrize("signal", [0.5, 1.0, 2.0])
def test_icm_never_lowers_the_objective_it_maximises(signal: float) -> None:
    """ICM never lowers its objective, to `SIGN_TOLERANCE` (#36, #40)."""
    fixture = potts_labels(shape=(5, 5), n_clones=3, signal=signal)
    start = np.zeros(fixture.n_nodes, dtype=np.int64)

    before = cnaster_assignment_cost(fixture, start)
    labelling, _, _ = cnaster_icm_labelling(fixture, start)
    after = cnaster_assignment_cost(fixture, labelling)

    assert after >= before - SIGN_TOLERANCE, (
        f"the sweep lowered its own objective, {before:.9f} -> {after:.9f}"
    )


@pytest.mark.smoke
def test_icm_is_reproducible_only_because_the_adapter_seeds_it(
    enumerable: PottsLabels,
) -> None:
    """ICM reads the global RNG; the adapter's seed alone makes it reproducible (#8)."""
    start = np.zeros(enumerable.n_nodes, dtype=np.int64)

    first, first_cost, _ = cnaster_icm_labelling(enumerable, start, seed=0)
    again, again_cost, _ = cnaster_icm_labelling(enumerable, start, seed=0)

    np.testing.assert_array_equal(first, again)
    assert first_cost == again_cost

    # NB disturb the legacy global state the solver reads, to test the adapter's isolation
    np.random.seed(12345)  # noqa: NPY002
    unseeded = np.random.permutation(enumerable.n_nodes)  # noqa: NPY002
    third, _, _ = cnaster_icm_labelling(enumerable, start, seed=0)
    np.testing.assert_array_equal(
        third, first, err_msg="seeding in the adapter did not isolate the global state"
    )
    assert unseeded.shape == (enumerable.n_nodes,)
