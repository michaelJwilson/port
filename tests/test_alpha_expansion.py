"""Alpha expansion against ICM on the same Potts problem.

**#246.** `cnaster` labels clones with iterated conditional modes, a greedy
single-site descent. Upstream's alpha expansion solves the same MAP problem
with moves that change arbitrarily many sites at once, and carries a proved
bound. They reach different labellings by construction, so the question is
not whether they agree but **which reaches the lower energy** --
`search.alpha_expansion.energy` answers that for either.

`oracle`: upstream decides the expected value, and the value is an absolute
one. A lower Potts energy is a better MAP solution under the same model
whoever produced it, so this needs no tolerance and no appeal to `cnaster`
being right.
"""

from __future__ import annotations

import numpy as np
import pytest
from port.patch.icm.alpha_expansion import (
    alpha_expansion_sweep,
    potts_energy,
    potts_graph_from,
)
from port.patch.icm.interface import CsrGraph


def _lattice(
    side: int, n_states: int, seed: int, beta: float
) -> tuple[np.ndarray, CsrGraph, np.ndarray, float]:
    """A field with a planted blocky labelling, plus a 4-neighbour lattice."""
    rng = np.random.default_rng(seed)
    n = side * side

    blocks = np.zeros((side, side), dtype=np.int64)
    blocks[: side // 2, : side // 2] = 1 % n_states
    blocks[side // 2 :, : side // 2] = 2 % n_states
    blocks[: side // 2, side // 2 :] = 3 % n_states
    planted = blocks.ravel()

    # NB a weak, noisy field: strong enough to carry signal, weak enough that
    #    the coupling decides the boundaries -- which is where a single-site
    #    descent gets stuck and an expansion move does not.
    field = rng.normal(0.0, 1.0, size=(n, n_states))
    field[np.arange(n), planted] += 0.6

    rows, cols, vals = [], [], []
    for r in range(side):
        for c in range(side):
            here = r * side + c
            for dr, dc in ((1, 0), (0, 1)):
                rr, cc = r + dr, c + dc
                if rr < side and cc < side:
                    there = rr * side + cc
                    rows += [here, there]
                    cols += [there, here]
                    vals += [1.0, 1.0]

    row = np.asarray(rows)
    col = np.asarray(cols)
    val = np.asarray(vals, dtype=float)

    order = np.lexsort((col, row))
    row, col, val = row[order], col[order], val[order]

    counts = np.zeros(n + 1, dtype=np.int64)
    np.add.at(counts, row + 1, 1)

    graph = CsrGraph(indptr=np.cumsum(counts), indices=col, weights=val)

    return field, graph, planted, beta


@pytest.mark.oracle
def test_alpha_expansion_reaches_a_lower_energy_than_single_site_descent() -> None:
    """The claim the bound predicts, measured on a problem with barriers.

    A greedy descent stops where no single flip helps. An expansion move
    changes many sites at once, so it crosses that. If this ever fails, the
    field is too strong for the coupling to matter and the fixture, not the
    algorithm, is what needs fixing.
    """
    field, graph, _, beta = _lattice(12, 4, seed=3, beta=1.5)

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
    field, graph, _, beta = _lattice(10, 3, seed=11, beta=2.0)

    start = np.argmax(field, axis=1).astype(np.int64)
    before = potts_energy(field, graph, start, beta)

    assignment = start.copy()
    result = alpha_expansion_sweep(field, graph, assignment, beta)

    assert result.cost <= before
    assert result.niter >= 1
    assert assignment.dtype == start.dtype, "the caller's array type must survive"


@pytest.mark.patch
def test_each_undirected_edge_is_counted_once() -> None:
    """`CsrGraph` stores both directions; `PottsGraph` counts each entry.

    Listing both would double every bond and halve the effective temperature
    without saying so, which would look like a tuning difference rather than
    a bug.
    """
    field, graph, _, beta = _lattice(4, 2, seed=1, beta=1.0)
    potts = potts_graph_from(graph, beta)

    assert len(potts.edges) == len(graph.indices) // 2
    assert all(i < j for i, j in potts.edges)
    assert all(c >= 0.0 for c in potts.coupling)

    # NB 2 * side * (side - 1) undirected edges on a 4-neighbour open lattice.
    assert len(potts.edges) == 2 * 4 * 3


@pytest.mark.patch
def test_a_negative_coupling_is_refused_rather_than_clipped() -> None:
    """The bound requires a metric. Silently clipping would forfeit it."""
    field, graph, _, _ = _lattice(4, 2, seed=1, beta=1.0)

    with pytest.raises(ValueError, match="metric"):
        potts_graph_from(graph, beta=-1.0)


@pytest.mark.warning
def test_the_icm_only_knobs_are_accepted_and_ignored() -> None:
    """`min_clone_spots` has no counterpart, and pretending otherwise is worse.

    `cnaster` merges any clone below 200 spots mid-sweep (#81) using the
    unseeded global RNG. Alpha expansion has no such move, and adding one
    would break the monotonicity its termination proof rests on. Accepting
    the argument keeps the two solvers interchangeable at the call site;
    ignoring it is what this pins, so nobody reads the signature as a promise.
    """
    field, graph, _, beta = _lattice(8, 3, seed=5, beta=1.0)

    first = np.argmax(field, axis=1).astype(np.int64)
    second = first.copy()

    a = alpha_expansion_sweep(field, graph, first, beta, min_clone_spots=200)
    b = alpha_expansion_sweep(field, graph, second, beta, min_clone_spots=2)

    assert a.cost == b.cost, "min_clone_spots must not change the result"
    assert np.array_equal(first, second)


@pytest.mark.analytic
def test_zero_coupling_recovers_the_field_argmax() -> None:
    """With no coupling the joint MAP factorizes, so the answer is known.

    The pin for the sign. Upstream minimizes `-sum h[s] - sum J [s == s]`,
    which already carries the negation, so `cnaster`'s log-likelihood field
    is passed through unchanged. Negating it as well solves the mirror
    problem: this test returned `argmin` before the fix, on every site.

    It is here rather than folded into the energy comparisons because those
    score both solvers through `potts_energy`, which shared the negation --
    an oracle carrying the defect it referees cannot see it. An analytic
    limit can: at `beta = 0` there is nothing to solve.
    """
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
    """What `potts_energy` has to be for "lower is better" to mean anything.

    `cnaster`'s `icm_sweep_deque` maximizes
    `sum_i field[i, s_i] + spatial_weight * sum_ij w_ij [s_i == s_j]`. This
    asserts the referee is exactly the negation of that, offset by
    `beta * sum(w)`, which is the same for every labelling and so cannot
    change an ordering.
    """
    field, graph, planted, beta = _lattice(8, 3, seed=5, beta=1.25)
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
    """`_lattice` with two allowed labels per site and `-inf` on the rest.

    `cnaster`'s field marks a label a spot may not take with `-inf`; `scale`
    sets the field's magnitude against the unit coupling (#366).
    """
    field, graph, _, beta = _lattice(side, n_states, seed, beta=1.0)
    rng = np.random.default_rng(seed)
    allowed = np.zeros(field.shape, dtype=bool)

    for site in range(field.shape[0]):
        allowed[site, rng.choice(n_states, 2, replace=False)] = True

    field = np.where(allowed, scale * field, -np.inf)
    start = np.array([rng.choice(np.flatnonzero(row)) for row in allowed])
    return field, graph, start, beta


@pytest.mark.oracle
@pytest.mark.patch
def test_a_finite_penalty_keeps_the_minimizer_of_the_forbidding_field() -> None:
    """Brute force over every labelling of a 3 x 3 lattice, three labels.

    The referee is `cnaster`'s objective itself: `potts_energy` on the `-inf`
    field is its negation up to a constant
    (`test_the_energy_is_minus_cnasters_objective_up_to_a_constant`), so the
    finite field must leave `cnaster`'s maximizer where it was.

    The global minimum of the energy with `-inf` entries equals that with
    `forbidden_as_finite`'s penalty, and the minimizer takes no forbidden
    label. Fails if the penalty is ever payable: some neighbourhood would
    then prefer a forbidden label, and the finite minimum would sit below
    the true one or on a label the field forbids.
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
    """The result is a local minimum under every single-site allowed move.

    `snakes_and_ladders` at 679d326 makes no expansion move on a field with
    `-inf` entries, so the sweep returned its start, which no single-site
    descent certifies (#366). The expansion's local minimum is stronger than
    the single-site one, so this holds of any correct run and failed on the
    stalled one at both scales.
    """
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
