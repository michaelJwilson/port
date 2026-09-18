"""`cnaster.icm` refereed by `snakes_and_ladders.search` (#140).

**334 statements at zero: `icm.py` is the largest `cnaster` module no second
implementation decides a value for.** The code is exercised --
`tests/test_potts_labels.py` drives it against invariants -- but an invariant
is not an independent answer, and #139's floor counts only the tests where
upstream decides one.

Upstream gives three referees of increasing strength, and this module uses all
three rather than picking one:

*   `iterated_conditional_modes` is the **same move set** on the same
    objective, so a disagreement is about the code rather than the method.
*   `alpha_expansion` is a **stronger move set** on the same objective, so its
    energy bounds `cnaster`'s from below with no tolerance to argue about.
*   `enumerate_minimum_energy` is **exact** at enumerable sizes, and bounds
    both.

The chain is `energy(cnaster) >= energy(alpha_expansion) >= energy(exact)`.
Bounds rather than equality is the point: two local searches reach different
labellings for good reasons, and an equality test would fail on that without
saying anything about either. A bound that breaks is a defect.

Every comparison is scored by `upstream_potts_energy`, and
`test_potts_labels.py::test_cnaster_maximises_what_upstream_minimises` is what
licenses that: `cnaster`'s `calc_assignment_cost` is the exact negation of
upstream's `energy`, so scoring one labelling under both is comparing solvers
rather than sign conventions.
"""

from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.adapters import cnaster_icm_labelling, upstream_potts_energy
from tests.fixtures import PottsLabels, enumerate_minimum_energy, potts_labels

if TYPE_CHECKING:
    from snakes_and_ladders.search.alpha_expansion import ExpansionResult
    from snakes_and_ladders.sim.graph import PottsGraph

BOUND_TOLERANCE = 1e-9
"""Float slack on an inequality between two sums of the same terms.

Not a tolerance on the claim: the quantities are the same field and coupling
values added in different orders, so the only disagreement admitted here is
the order of summation.
"""


def _upstream_graph(fixture: PottsLabels) -> "PottsGraph":
    from tests.fixtures import _scaled_graph

    return _scaled_graph(fixture)


def _upstream_icm(fixture: PottsLabels, seed: int = 0) -> tuple[np.ndarray, float]:
    from snakes_and_ladders.search.alpha_expansion import iterated_conditional_modes

    labelling, energy = iterated_conditional_modes(
        _upstream_graph(fixture),
        fixture.field,
        fixture.n_clones,
        np.random.default_rng(seed),
    )
    return np.asarray(labelling), float(energy)


def _upstream_expansion(
    fixture: PottsLabels, start: np.ndarray | None = None
) -> "ExpansionResult":
    from snakes_and_ladders.search.alpha_expansion import alpha_expansion

    return alpha_expansion(
        _upstream_graph(fixture),
        fixture.field,
        fixture.n_clones,
        start=None if start is None else np.asarray(start, dtype=np.int64),
    )


@pytest.fixture(scope="module")
def lattice() -> PottsLabels:
    """A `6 x 6` lattice at the fixture's defaults, three clones."""
    return potts_labels()


@pytest.fixture(scope="module")
def enumerable() -> PottsLabels:
    """Ten nodes and three clones: 59,049 labellings, searched exhaustively."""
    return potts_labels(shape=(5, 2), n_clones=3, signal=0.6, noise=1.0)


@pytest.mark.oracle
@pytest.mark.parametrize("coupling", [0.5, 1.0, 2.0])
def test_alpha_expansion_bounds_the_cnaster_sweep(coupling: float) -> None:
    """**The bound the whole rung rests on**, at three couplings.

    Alpha expansion's move set contains single-site descent's: flipping one
    node to `alpha` is an expansion that changes one node. So its optimum can
    never be worse, and `cnaster`'s sweep landing below it would mean one of
    the two is not minimising the energy it claims to.

    Swept over `coupling` because the gap is what the pairwise term buys: at
    a weak coupling both find nearly the same labelling and the bound is
    nearly tight, and at a strong one single-site descent is the method that
    gets stuck.
    """
    fixture = potts_labels(shape=(6, 6), n_clones=3, coupling=coupling)
    start = np.zeros(fixture.n_nodes, dtype=np.int64)

    sweep, _, _ = cnaster_icm_labelling(fixture, start)
    expansion = _upstream_expansion(fixture)

    swept = upstream_potts_energy(fixture, sweep)
    expanded = upstream_potts_energy(fixture, np.asarray(expansion.labelling))

    assert expanded <= swept + BOUND_TOLERANCE, (
        f"coupling {coupling}: expansion {expanded:.6f} above sweep {swept:.6f}"
    )


@pytest.mark.oracle
def test_no_single_site_move_lowers_what_cnaster_returns(
    lattice: PottsLabels,
) -> None:
    """`cnaster`'s answer is a fixed point of the move set it claims.

    The defining property of iterated conditional modes, and the sharpest
    check on `icm_sweep_deque`'s stopping rule that does not depend on visit
    order: whatever order it used, a labelling it returns must admit no
    single-site improvement. Scored by **upstream's** `energy` at every
    `(node, label)` pair, 36 x 3 here, so the criterion is checked against an
    independent implementation of the objective rather than against the
    solver's own bookkeeping.

    A failure means the deque emptied while a move was still available.

    Note what this does **not** claim, which the next test measures:
    single-site optimal is not expansion optimal.
    """
    start = np.zeros(lattice.n_nodes, dtype=np.int64)
    sweep, _, _ = cnaster_icm_labelling(lattice, start)
    settled = upstream_potts_energy(lattice, sweep)

    worst = 0.0
    for node in range(lattice.n_nodes):
        for label in range(lattice.n_clones):
            if label == sweep[node]:
                continue
            moved = sweep.copy()
            moved[node] = label
            worst = min(worst, upstream_potts_energy(lattice, moved) - settled)

    assert worst >= -BOUND_TOLERANCE, (
        f"a single-site move lowers the returned labelling by {-worst:.6e}"
    )


@pytest.mark.oracle
def test_expansion_improves_on_the_sweep_it_is_started_from(
    lattice: PottsLabels,
) -> None:
    """**Measured: the stronger move set finds 4.62 more, from the same start.**

    `cnaster`'s sweep is single-site optimal -- the test above establishes
    that -- and upstream's alpha expansion, started from the labelling it
    returns, still lowers the energy from -112.657 to -117.274.

    That is not a defect in `cnaster`: it is the gap between the two move
    sets, which is the thing `iterated_conditional_modes` exists upstream to
    make visible. It is recorded here because the gap is the only quantity
    that says what `cnaster`'s choice of solver costs, and #8 is the ticket
    that would act on it.

    The direction is asserted; the size is reported. A run where expansion
    could not improve would mean the instance is too easy to distinguish the
    two, which is a fixture problem rather than a passing test.
    """
    start = np.zeros(lattice.n_nodes, dtype=np.int64)
    sweep, _, _ = cnaster_icm_labelling(lattice, start)

    swept = upstream_potts_energy(lattice, sweep)
    refined = _upstream_expansion(lattice, start=sweep)

    assert refined.energy < swept - BOUND_TOLERANCE, (
        f"expansion found nothing from {swept:.6f}; the instance is too easy"
    )
    assert refined.moves > 0


@pytest.mark.oracle
def test_at_zero_coupling_both_solvers_return_the_field_argmax() -> None:
    """The one instance whose optimum is unique, so equality is assertable.

    With no pairwise term the energy separates over nodes and both solvers
    must return the per-node field argmax. Everything else here is a bound;
    this is the case that would catch two solvers agreeing with each other
    and both being wrong, because the answer is known without either.
    """
    fixture = potts_labels(shape=(6, 6), n_clones=3, coupling=0.0)
    start = np.zeros(fixture.n_nodes, dtype=np.int64)

    sweep, _, _ = cnaster_icm_labelling(fixture, start)
    upstream, _ = _upstream_icm(fixture)
    argmax = np.asarray(fixture.field).reshape(fixture.n_nodes, -1).argmax(axis=1)

    np.testing.assert_array_equal(sweep, argmax)
    np.testing.assert_array_equal(upstream, argmax)


@pytest.mark.oracle
def test_neither_solver_beats_the_exact_minimum(enumerable: PottsLabels) -> None:
    """Enumeration bounds both, which is what makes the chain a chain.

    `oracle` rather than `upstream_oracle`: the expected value is decided by
    exhaustive search over 3^10 labellings, not by upstream. It is here
    because a bound between two approximations says nothing about either
    unless something exact sits underneath.
    """
    _, minimum = enumerate_minimum_energy(enumerable)

    start = np.zeros(enumerable.n_nodes, dtype=np.int64)
    sweep, _, _ = cnaster_icm_labelling(enumerable, start)
    expansion = _upstream_expansion(enumerable)

    swept = upstream_potts_energy(enumerable, sweep)

    assert minimum <= expansion.energy + BOUND_TOLERANCE
    assert minimum <= swept + BOUND_TOLERANCE, (
        f"cnaster {swept:.6f} below the enumerated minimum {minimum:.6f}"
    )


@pytest.mark.oracle
def test_the_sweep_improves_on_the_labelling_it_started_from(
    lattice: PottsLabels,
) -> None:
    """A solver that returned its input would pass every bound above.

    The gap is real and measured rather than assumed: from the all-zero
    labelling the sweep lowers the energy, scored by upstream so the claim is
    not `cnaster` grading its own descent.
    """
    start = np.zeros(lattice.n_nodes, dtype=np.int64)
    sweep, _, _ = cnaster_icm_labelling(lattice, start)

    before = upstream_potts_energy(lattice, start)
    after = upstream_potts_energy(lattice, sweep)

    assert after < before, f"no descent: {before:.6f} -> {after:.6f}"
