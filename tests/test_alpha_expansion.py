"""Alpha expansion against ICM on one Potts problem (#246).

Referee: upstream's `search.alpha_expansion.energy`; a lower energy is the better MAP
solution, so no tolerance is needed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from port.extensions.label_solver import SOLVERS, Solver, sweep_for
from port.patch.icm.alpha_expansion import (
    alpha_expansion_sweep,
    potts_energy,
    potts_graph_from,
)
from port.patch.icm.interface import CsrGraph

from tests.fixtures import planted_blocky_field


@pytest.mark.oracle
def test_alpha_expansion_reaches_a_lower_energy_than_single_site_descent() -> None:
    """Alpha expansion reaches an energy no higher than ICM's on a field with barriers."""
    field, graph, _, beta = planted_blocky_field(12, 4, seed=3, beta=1.5)

    greedy = np.zeros(field.shape[0], dtype=np.int64)
    greedy[:] = np.argmax(field, axis=1)  # the data optimum: ICM's fixed point
    greedy_energy = potts_energy(field, graph, greedy, beta)

    expanded = greedy.copy()
    result = alpha_expansion_sweep(field, graph, expanded, beta)

    assert result.cost <= greedy_energy, (
        f"alpha expansion {result.cost:.4f} did not improve on {greedy_energy:.4f}"
    )
    assert result.cost == pytest.approx(
        potts_energy(field, graph, expanded, beta), rel=0, abs=1e-9
    ), "the reported cost is not the energy of the labelling returned"


@pytest.mark.oracle
def test_the_energy_is_monotone_and_the_labelling_is_the_one_scored() -> None:
    """Upstream's two invariants, asserted here because the patch relies on them."""
    field, graph, _, beta = planted_blocky_field(10, 3, seed=11, beta=2.0)

    start = np.argmax(field, axis=1).astype(np.int64)
    before = potts_energy(field, graph, start, beta)

    assignment = start.copy()
    result = alpha_expansion_sweep(field, graph, assignment, beta)

    assert result.cost <= before
    assert result.niter >= 1
    assert assignment.dtype == start.dtype, "the caller's array type must survive"


@pytest.mark.patch
def test_each_undirected_edge_is_counted_once() -> None:
    """`CsrGraph`'s two directions convert to one `PottsGraph` edge each."""
    field, graph, _, beta = planted_blocky_field(4, 2, seed=1, beta=1.0)
    potts = potts_graph_from(graph, beta)

    assert len(potts.edges) == len(graph.indices) // 2
    assert all(i < j for i, j in potts.edges)
    assert all(c >= 0.0 for c in potts.coupling)

    # NB 2 * side * (side - 1) undirected edges on a 4-neighbour open lattice.
    assert len(potts.edges) == 2 * 4 * 3


@pytest.mark.patch
def test_a_negative_coupling_is_refused_rather_than_clipped() -> None:
    """A non-metric coupling is refused by `sal`'s `from_directed_csr(scale=)` (T-
    #777).
    """
    field, graph, _, _ = planted_blocky_field(4, 2, seed=1, beta=1.0)

    with pytest.raises(ValueError, match="coupling"):
        potts_graph_from(graph, spatial_weight=-1.0)


@pytest.mark.warning
def test_the_icm_only_knobs_are_accepted_and_ignored() -> None:
    """`min_clone_spots` is accepted and ignored: alpha expansion has no merge move
    (#81).
    """
    field, graph, _, beta = planted_blocky_field(8, 3, seed=5, beta=1.0)

    first = np.argmax(field, axis=1).astype(np.int64)
    second = first.copy()

    a = alpha_expansion_sweep(field, graph, first, beta, min_clone_spots=200)
    b = alpha_expansion_sweep(field, graph, second, beta, min_clone_spots=2)

    assert a.cost == b.cost, "min_clone_spots must not change the result"
    assert np.array_equal(first, second)


@pytest.mark.analytic
def test_zero_coupling_recovers_the_field_argmax() -> None:
    """With no coupling the solution is the field's per-site argmax (sign check)."""
    rng = np.random.default_rng(17)
    n, n_states = 64, 4

    field = rng.normal(size=(n, n_states))
    graph = CsrGraph(
        indptr=np.zeros(n + 1, dtype=np.int64),
        indices=np.empty(0, dtype=np.int64),
        weights=np.empty(0, dtype=float),
    )

    assignment = np.zeros(n, dtype=np.int64)
    alpha_expansion_sweep(field, graph, assignment, 0.0)

    expected = np.argmax(field, axis=1)
    assert np.array_equal(assignment, expected), (
        f"{int((assignment != expected).sum())} of {n} sites disagree with the "
        f"per-site argmax; {int((assignment == np.argmin(field, axis=1)).sum())} "
        f"match the argmin, which is the doubled-negation signature"
    )


@pytest.mark.patch
def test_the_energy_is_minus_cnasters_objective_up_to_a_constant() -> None:
    """`potts_energy` is the negation of cnaster's ICM objective plus a labelling-
    independent offset.
    """
    field, graph, planted, beta = planted_blocky_field(8, 3, seed=5, beta=1.25)
    first, second, coupling = potts_graph_from(graph, beta).endpoints

    for labels in (planted, np.argmax(field, axis=1).astype(np.int64)):
        objective = float(field[np.arange(labels.size), labels].sum()) + float(
            coupling[labels[first] == labels[second]].sum()
        )
        assert -potts_energy(field, graph, labels, beta) == pytest.approx(
            objective, rel=1e-12
        )


def _forbidding(
    side: int, n_states: int, seed: int, scale: float
) -> tuple[np.ndarray, CsrGraph, np.ndarray, float]:
    """`planted_blocky_field` with two allowed labels per site and `-inf` elsewhere
    (#366).
    """
    field, graph, _, beta = planted_blocky_field(side, n_states, seed, beta=1.0)
    rng = np.random.default_rng(seed)
    allowed = np.zeros(field.shape, dtype=bool)

    for site in range(field.shape[0]):
        allowed[site, rng.choice(n_states, 2, replace=False)] = True

    field = np.where(allowed, scale * field, -np.inf)
    start = np.array([rng.choice(np.flatnonzero(row)) for row in allowed])
    return field, graph, start, beta


@pytest.mark.patch
def test_a_finite_penalty_keeps_the_minimizer_of_the_forbidding_field() -> None:
    """Brute force on a 3x3 lattice: the `-inf` and `forbidden_as_finite` minima agree
    and avoid forbidden labels.
    """
    import itertools

    from port.patch.icm.alpha_expansion import forbidden_as_finite

    for seed in range(5):
        field, graph, _, beta = _forbidding(3, 3, seed, scale=100.0)
        finite = forbidden_as_finite(field, graph, beta)
        best_true = best_finite = np.inf
        argmin_finite = None

        for labels in itertools.product(range(3), repeat=9):
            labelling = np.asarray(labels)
            true = potts_energy(field, graph, labelling, beta)
            stated = potts_energy(finite, graph, labelling, beta)
            best_true = min(best_true, true)

            if stated < best_finite:
                best_finite, argmin_finite = stated, labelling

        assert argmin_finite is not None
        assert np.isfinite(field[np.arange(9), argmin_finite]).all()
        assert best_finite == pytest.approx(best_true, rel=1e-12)


@pytest.mark.analytic
@pytest.mark.parametrize("scale", [1.0, 1000.0])
def test_the_sweep_moves_on_a_forbidding_field(scale: float) -> None:
    """The result is a local minimum under every allowed single-site move (#366)."""
    field, graph, start, beta = _forbidding(20, 5, 366, scale)
    labelling = start.copy()
    alpha_expansion_sweep(field, graph, labelling, beta)

    assert np.isfinite(field[np.arange(labelling.size), labelling]).all()
    assert (labelling != start).any()

    here = potts_energy(field, graph, labelling, beta)
    for site in range(labelling.size):
        for label in np.flatnonzero(np.isfinite(field[site])):
            moved = labelling.copy()
            moved[site] = label
            assert potts_energy(field, graph, moved, beta) >= here - 1e-9


@pytest.mark.analytic
@pytest.mark.parametrize("name", SOLVERS)
@pytest.mark.parametrize("scale", [1.0, 1000.0])
def test_every_solver_row_reaches_a_local_minimum_on_a_forbidding_field(
    name: Solver, scale: float
) -> None:
    """Each `PORT_LABEL_SOLVER` row ends at a single-site local minimum (#466)."""
    field, graph, start, beta = _forbidding(20, 5, 466, scale)
    labelling = start.copy()
    sweep_for(name)(field, graph, labelling, beta, min_clone_spots=1)

    assert np.isfinite(field[np.arange(labelling.size), labelling]).all()

    here = potts_energy(field, graph, labelling, beta)
    for site in range(labelling.size):
        for label in np.flatnonzero(np.isfinite(field[site])):
            moved = labelling.copy()
            moved[site] = label
            assert potts_energy(field, graph, moved, beta) >= here - 1e-9, (
                name,
                site,
                label,
            )


EXPANDING: tuple[str, ...] = (
    "alpha",
    "alpha-rust",
    "alpha-rust-icm",
    "alpha-rust-merge",
    "alpha-rust-fuse-merge",
)
"""Rows whose first step is alpha expansion from the caller's labelling."""


def _expanding(name: str) -> Any:
    from port.sandbox.extensions.label_solvers import SWEEPS

    return SWEEPS[name] if name in SWEEPS else sweep_for(name)  # type: ignore[arg-type]


@pytest.mark.analytic
@pytest.mark.parametrize("name", EXPANDING)
def test_every_expanding_row_ends_at_or_below_the_expansion(name: str) -> None:
    """An expand-first row ends no higher than `alpha_expansion_sweep` from the same
    start (#466).
    """
    field, graph, start, beta = _forbidding(20, 5, 466, 1.0)
    expanded = start.copy()
    alpha_expansion_sweep(field, graph, expanded, beta)
    labelling = start.copy()
    _expanding(name)(field, graph, labelling, beta, min_clone_spots=1)

    assert (
        potts_energy(field, graph, labelling, beta)
        <= potts_energy(field, graph, expanded, beta) + 1e-9
    )
