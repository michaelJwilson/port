"""`cnaster.icm` refereed by `sal.search` (#140).

Bounds `energy(cnaster) >= energy(alpha_expansion) >= energy(exact)`, all scored by
`upstream_potts_energy`, the negation of `cnaster`'s `calc_assignment_cost`.
"""

from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.adapters import cnaster_icm_labelling, upstream_potts_energy
from tests.fixtures import (
    PottsLabels,
    enumerate_minimum_energy,
    potts_labels,
    scaled_graph,
)

if TYPE_CHECKING:
    from sal.search.alpha_expansion import ExpansionResult

BOUND_TOLERANCE = 1e-9
"""Float slack for summation order on an inequality between sums of the same terms."""


def _upstream_icm(fixture: PottsLabels, seed: int = 0) -> tuple[np.ndarray, float]:
    from sal.search.icm import iterated_conditional_modes

    result = iterated_conditional_modes(
        scaled_graph(fixture), fixture.field, np.random.default_rng(seed)
    )
    return np.asarray(result.labelling), float(result.energy)


def _upstream_expansion(
    fixture: PottsLabels, start: np.ndarray | None = None
) -> "ExpansionResult":
    from sal.backend import Backend
    from sal.search.alpha_expansion import alpha_expansion

    # NB PYTHON was the default before e0aeb19 made it RUST (#410).
    return alpha_expansion(
        scaled_graph(fixture),
        fixture.field,
        start=None if start is None else np.asarray(start, dtype=np.int64),
        backend=Backend.PYTHON,
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
    """Alpha expansion's energy bounds `cnaster`'s sweep from below, at three couplings."""
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
    """No single-site move lowers upstream's `energy` at `cnaster`'s returned labelling."""
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
    """Expansion started from `cnaster`'s labelling lowers the energy (measured -112.657 to
    -117.274; #8).
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
    """At zero coupling both solvers return the per-node field argmax."""
    fixture = potts_labels(shape=(6, 6), n_clones=3, coupling=0.0)
    start = np.zeros(fixture.n_nodes, dtype=np.int64)

    sweep, _, _ = cnaster_icm_labelling(fixture, start)
    upstream, _ = _upstream_icm(fixture)
    argmax = np.asarray(fixture.field).reshape(fixture.n_nodes, -1).argmax(axis=1)

    np.testing.assert_array_equal(sweep, argmax)
    np.testing.assert_array_equal(upstream, argmax)


@pytest.mark.oracle
def test_neither_solver_beats_the_exact_minimum(enumerable: PottsLabels) -> None:
    """Exhaustive search over 3^10 labellings bounds both solvers."""
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
    """The sweep lowers upstream's energy from the all-zero labelling."""
    start = np.zeros(lattice.n_nodes, dtype=np.int64)
    sweep, _, _ = cnaster_icm_labelling(lattice, start)

    before = upstream_potts_energy(lattice, start)
    after = upstream_potts_energy(lattice, sweep)

    assert after < before, f"no descent: {before:.6f} -> {after:.6f}"
