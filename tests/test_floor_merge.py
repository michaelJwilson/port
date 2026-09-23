"""The clone-size floor and the refinement mask (#348).

`calicost_instance` lost every clone at the read-depth refinement: 16
sub-clones of about 100 spots met `icm_sweep_deque`'s 200-spot floor, which
empties every clone under it at once and moves its spots to a clone drawn at
random from those over it; one was, and took 1,509 of 1,600 spots. What is
pinned here:

- `cnaster`'s floor collapses a problem where one clone clears it (`bug`,
  written to fail when it stops doing so);
- `enforce_floor` merges smallest first into each spot's best clone, stops
  once every clone clears the floor, and never crosses a `-inf` entry
  (`patch`);
- `mask_for` hands the refinement's mask only to the problem it describes
  (`infra`).
"""

from __future__ import annotations

import numpy as np
import pytest


def _problem(sizes: list[int], seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """A field favouring each spot's own clone by 5, over sizes given."""
    rng = np.random.default_rng(seed)
    assignment = np.repeat(np.arange(len(sizes)), sizes)
    field = rng.normal(0.0, 0.1, (assignment.size, len(sizes)))
    field[np.arange(assignment.size), assignment] += 5.0
    return field, assignment


@pytest.mark.bug
def test_cnasters_floor_moves_every_undersized_clone_into_the_one_over_it() -> None:
    """25 spots in one clone, 9 in each of 15: a floor of 20 leaves one clone."""
    from port.patch.icm.interface import CsrGraph, icm_sweep
    from scipy.sparse import csr_matrix

    field, assignment = _problem([25] + [9] * 15)
    graph = CsrGraph.from_matrix(csr_matrix((assignment.size, assignment.size)))
    np.random.seed(0)  # noqa: NPY002 - cnaster's sweep draws from it

    icm_sweep(field, graph, assignment, 0.0, min_clone_spots=20)

    assert np.unique(assignment).size == 1


@pytest.mark.patch
def test_the_floor_merges_smallest_first_and_stops_when_every_clone_clears_it() -> None:
    """The same problem keeps seven clones, each at least 20 spots."""
    from port.patch.icm.floor import enforce_floor

    field, assignment = _problem([25] + [9] * 15)
    emptied = enforce_floor(field, assignment, 20)
    counts = np.bincount(assignment)
    kept = counts[counts > 0]

    assert emptied == 16 - kept.size
    assert kept.min() >= 20
    assert kept.size > 1


@pytest.mark.patch
def test_the_floor_does_not_cross_a_masked_boundary() -> None:
    """Two groups: an undersized clone with no allowed partner keeps its spots."""
    from port.patch.icm.floor import enforce_floor

    field, assignment = _problem([30, 5, 30])
    # NB clone 1 may only be clone 1: every other entry of its spots is -inf.
    field[assignment == 1, 0] = -np.inf
    field[assignment == 1, 2] = -np.inf

    enforce_floor(field, assignment, 20)

    assert (assignment[30:35] == 1).all()


@pytest.mark.infra
def test_the_mask_is_handed_only_to_the_problem_it_describes() -> None:
    from port.patch.hmrf import refinement

    mask = np.zeros((4, 3), dtype=bool)
    mask[:2, :2] = True
    mask[2:, 2] = True
    refinement._KEPT[:] = [mask]

    try:
        assert refinement.mask_for(np.array([0, 1, 2, 2]), 3) is mask
        assert refinement.mask_for(np.array([0, 2, 2, 2]), 3) is None
        assert refinement.mask_for(np.array([0, 1, 1]), 3) is None
        assert refinement.mask_for(np.array([0, 1, 1, 1]), 2) is None
    finally:
        refinement.forget()

    assert refinement.mask_for(np.array([0, 1, 2, 2]), 3) is None


@pytest.mark.cnaster
@pytest.mark.patch
@pytest.mark.usefixtures("cnaster_config")
def test_the_refinement_start_is_upstreams_and_its_mask_is_kept() -> None:
    """The wrapper returns `cnaster`'s three values unchanged and keeps the mask."""
    from types import SimpleNamespace

    import cnaster.spatial as upstream
    from port.patch.hmrf.refinement import (
        forget,
        initialize_rdr_clone_refininement,
        mask_for,
    )

    rng = np.random.default_rng(5)
    coords = np.column_stack(np.unravel_index(np.arange(200), (10, 20)))
    baf = np.repeat([0, 1], 100)
    counts = np.full((50, 200), 30.0)
    config = SimpleNamespace(
        hmrf=SimpleNamespace(n_clones_rdr=2), hmm=SimpleNamespace(gmm_random_state=0)
    )
    del rng

    theirs = upstream.initialize_rdr_clone_refininement(baf, coords, counts, 50, config)
    ours = initialize_rdr_clone_refininement(baf, coords, counts, 50, config)

    try:
        np.testing.assert_array_equal(ours[0], theirs[0])
        np.testing.assert_array_equal(ours[1], theirs[1])
        assert ours[2] == theirs[2]
        assert mask_for(ours[0], ours[2]) is not None
    finally:
        forget()
