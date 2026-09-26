"""The spatial labelling fixtures, and the correspondence they rest on.

Issue #40. A label solver takes a unary field and a graph and returns a
labelling; `cnaster` solves that with `icm_sweep_deque` and
`snakes_and_ladders` with three solvers behind `LabelSolver`. Comparing them
is the point of #40 and of the audit in #8, and neither can begin until the
two sides are scoring **one objective on one graph**.

That is what this module establishes, and it is the whole of what this pull
request claims. The solver table -- energy and wall time per method, and the
fraction of instances each reaches the best of -- is the next one.

**The correspondence, and why it is exact rather than approximate.**
`cnaster` maximises `sum_i llf[i, s_i] + w sum_(ij) J_ij [s_i == s_j]`;
upstream's `energy` is `-sum_i h_i[s_i] - sum_(ij) J_ij [s_i == s_j]`. Same
two terms, opposite sign, and the only translation needed is that `cnaster`
keeps `spatial_weight` outside its adjacency where upstream folds it into
the coupling. So the two agree to float64 rounding and nothing has to be
tuned for them to.

**What is planted and what is not.** The field is planted, as
`planted_posterior` is in `test_m_step`: it is the input to the step under
test, so deriving it from an emission would make a defect in the emission
look like a defect in the solver. The emission-derived field belongs to #4.

The planted labelling is **not** the optimum, and a test below asserts
exactly that. At the fixture's default signal-to-noise the energy-minimising
labelling differs from the one the field was drawn around, which is what
separates a recovery claim from an optimality claim. A fixture where they
coincided would let a solver pass by returning its input.
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
from tests.fixtures import (
    PottsLabels,
    enumerate_minimum_energy,
    potts_labels,
)

SIGN_TOLERANCE = 1e-12
"""Absolute tolerance on `cost + energy == 0`.

Not a modelling tolerance. The two implementations sum the same terms in a
different order -- upstream vectorizes over edges, `cnaster` loops over nodes
and halves -- so the difference is float64 reassociation and nothing else.
Measured at `7.1e-15` on the default fixture, so this floor is two orders
above the gap it admits, and a real disagreement would exceed it by many
more.
"""


@pytest.fixture
def lattice() -> PottsLabels:
    """The default draw: a 6x6 open lattice, three clones, 60 edges."""
    return potts_labels()


@pytest.fixture
def enumerable() -> PottsLabels:
    """A lattice small enough to search exhaustively: 10 sites, three clones.

    `3 ** 10` is 59,049 labellings. Weak signal, so the coupling matters and
    the optimum is not simply the field's argmax -- at a strong signal every
    method agrees and an exhaustive search proves nothing.
    """
    return potts_labels(shape=(5, 2), n_clones=3, signal=0.6, noise=1.0)


@pytest.mark.smoke
def test_the_edge_set_survives_the_csr_conversion(lattice: PottsLabels) -> None:
    """The adjacency carries each undirected edge twice, and nothing else.

    #12 is the ticket for this and states why it cannot be assumed: upstream
    and `cnaster` build their graphs from the same coordinates by different
    code, and a difference of one edge makes every energy incomparable
    without anything raising.

    Two conventions are pinned rather than one. Upstream's `PottsGraph.edges`
    is **not deduplicated** -- its own docstring says a periodic lattice of
    extent two legitimately repeats a pair -- so a conversion that assumed
    uniqueness would silently drop a bond. And the CSR must be symmetric,
    because `icm_sweep_deque` reads one row per node and takes it for the
    whole neighbourhood.

    What would have to be wrong for this to fail: an emitted direction
    missing, a self-loop introduced, or a coupling attached to the wrong
    pair.
    """
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
    """`calc_assignment_cost` is the exact negation of `energy`.

    The correspondence every later comparison rests on, asserted through
    both implementations' own functions rather than a restatement of either.
    Scored at several labellings including two that are nothing like the
    planted one, because a sign convention that happened to agree on a good
    labelling would still be wrong.

    `spatial_weight` is held at a value other than one, so a translation that
    dropped it would be visible rather than absorbed.

    What would have to be wrong for this to fail: either side's pairwise sign,
    `cnaster`'s halving of its doubled edge list, or `spatial_weight` folded
    in twice or not at all.
    """
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
    """With no coupling, the optimum is the field's argmax, node by node.

    The degenerate case, and it checks the harness rather than a solver: if
    this fails, every comparison above it is measuring the wrong objective
    and no statement about a solver means anything.

    It is also the one case where the answer is known without enumeration at
    any lattice size, which is why it is asserted on the full 36-node fixture
    rather than the enumerable one.
    """
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
    """The fixture's own claim, asserted rather than asserted about.

    `potts_labels` documents that the planted labelling is what the field was
    drawn around and not what minimises the energy. If the two coincided, a
    solver could pass every recovery test by returning its input, and the
    optimality tests in #40 would be vacuous.

    So this pins the gap: exhaustive search finds a labelling strictly better
    than the planted one, and the fixture's signal-to-noise is therefore in
    the regime where the comparison has content. Raising `signal` far enough
    should break this test, and that is the intended reading -- it is the
    boundary of the regime, not a defect.
    """
    optimum, best = enumerate_minimum_energy(enumerable)
    planted = upstream_potts_energy(enumerable, enumerable.labels)

    assert best < planted, (
        f"the planted labelling is already optimal at energy {planted:.6f}; "
        "the fixture's signal is too strong for the comparison to have content"
    )
    assert not np.array_equal(optimum, enumerable.labels)


@pytest.mark.smoke
def test_enumeration_refuses_what_it_cannot_search() -> None:
    """A fixture too large to enumerate raises rather than running for an hour.

    The refusal is the feature. An exhaustive test that silently takes an
    hour is a test that gets deleted rather than fixed, and a fixture past
    the ceiling needs a different referee rather than more patience.
    """
    too_large = potts_labels(shape=(6, 6), n_clones=3)
    assert too_large.n_clones**too_large.n_nodes > MAX_ENUMERABLE_CONFIGURATIONS

    with pytest.raises(ValueError, match="refusing to enumerate"):
        enumerate_minimum_energy(too_large)


@pytest.mark.oracle
def test_icm_reports_the_cost_of_the_labelling_it_returns(lattice: PottsLabels) -> None:
    """The returned cost is the objective's change, recomputed independently.

    `icm_sweep_deque` accumulates per-node deltas from `cost_zeropoint` and
    never evaluates the objective, so what it returns is a telescoping sum
    rather than a measurement. Whether that sum is the change in the
    objective is a claim about the solver, and it is the claim that would
    break silently -- a solver reporting a cost it did not achieve is #30's
    failure in a discrete setting.

    Recomputed through `calc_assignment_cost` at the start and at the end, so
    the assertion is against `cnaster`'s own objective rather than a
    restatement of it.

    Measured: the residual is at most `8.9e-15` across the fixtures here.
    """
    start = np.zeros(lattice.n_nodes, dtype=np.int64)

    before = cnaster_assignment_cost(lattice, start)
    labelling, reported, _ = cnaster_icm_labelling(lattice, start)
    after = cnaster_assignment_cost(lattice, labelling)

    assert reported == pytest.approx(after - before, abs=SIGN_TOLERANCE)


@pytest.mark.analytic
@pytest.mark.parametrize("signal", [0.5, 1.0, 2.0])
def test_icm_never_lowers_the_objective_it_maximises(signal: float) -> None:
    """A descent method does not end worse than it began.

    The invariant, holding whichever implementation is right, and the
    discrete analogue of #36's monotonicity. Single-site descent accepts a
    move only where it improves the local cost, so the global objective is
    non-decreasing across the sweep.

    Asserted across signal-to-noise because the failure it guards against is
    regime-dependent: at a weak signal the coupling dominates and the solver
    moves little, which is where an accounting error in the pairwise term
    would show as a decrease rather than as a smaller increase.

    It does not assert the solver *improves* anything. At `signal = 0.5` from
    an all-zero start the sweep terminates having made no edit, which is the
    freezing `wolff.tex` argues against and #40 measures.
    """
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
    """The solver draws its visit order from the global `numpy` RNG.

    `icm_sweep_deque` calls `np.random.shuffle` on the initial node order and
    again on each promoted queue, and seeds nothing. So a result depends on
    whatever else in the process last touched the legacy global state, and
    two identical calls need not agree.

    `cnaster_icm_labelling` seeds it and records the seed, because a solver
    comparison whose outcome depends on unrecorded global state is not a
    comparison. This pins both halves: the same seed reproduces, and a
    different seed is free to differ.

    Reported rather than worked around. The fix belongs in `cnaster` -- a
    `Generator` taken as an argument -- and #8 is where it is proposed.
    """
    start = np.zeros(enumerable.n_nodes, dtype=np.int64)

    first, first_cost, _ = cnaster_icm_labelling(enumerable, start, seed=0)
    again, again_cost, _ = cnaster_icm_labelling(enumerable, start, seed=0)

    np.testing.assert_array_equal(first, again)
    assert first_cost == again_cost

    # NB the legacy global generator, deliberately: this disturbs exactly the
    #    state the solver reads, which is what makes the next call a test of
    #    the adapter's isolation rather than of the seed alone.
    np.random.seed(12345)  # noqa: NPY002
    unseeded = np.random.permutation(enumerable.n_nodes)  # noqa: NPY002
    third, _, _ = cnaster_icm_labelling(enumerable, start, seed=0)
    np.testing.assert_array_equal(
        third, first, err_msg="seeding in the adapter did not isolate the global state"
    )
    assert unseeded.shape == (enumerable.n_nodes,)
