"""Pieces of the rebound `pipeline_clone_assignment` (#206, #59) against `cnaster`'s arithmetic.

Whole-run equivalence is in `tests/test_patched_entry_point.py`.
"""

from typing import Any

import cnaster.hmrf
import numpy as np
import pytest
from port.patch.hmrf import clone_assignment
from port.patch.hmrf.clone_assignment import _BOUNDARY, _decoded, boundary
from port.patch.hmrf.invariants import BoundaryInvariants
from port.pipeline import FIGURE_SWAPS, SWAPS, patched


def _cnaster_weight(
    valid_nb: np.ndarray,
    valid_bb: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
) -> np.ndarray:
    """`rel_valid_emision_weight` transcribed from `hmrf.py` (computed inside `njit` there)."""
    n_spots = len(indptr) - 1
    weight = np.ones(n_spots, dtype=np.float64)

    for spot in range(n_spots):
        pooled_nb = pooled_bb = 0.0

        for entry in range(indptr[spot], indptr[spot + 1]):
            neighbour = indices[entry]
            pooled_nb += valid_nb[neighbour]
            pooled_bb += valid_bb[neighbour]

        if pooled_nb > 0 and pooled_bb > 0:
            weight[spot] = pooled_bb / pooled_nb

    return weight


@pytest.mark.patch
@pytest.mark.parametrize("n_spots", [1, 7, 40])
def test_the_channel_weight_is_cnasters_segment_sum(n_spots: int) -> None:
    """CSR `bincount` weight equals `cnaster`'s loop bitwise, including empty neighbourhoods."""
    generator = np.random.default_rng(19)

    degrees = generator.integers(0, 4, n_spots)
    degrees[0] = 0

    indptr = np.concatenate([[0], np.cumsum(degrees)]).astype(np.int64)
    indices = generator.integers(0, n_spots, int(degrees.sum())).astype(np.int64)

    valid_nb = generator.integers(0, 5, n_spots).astype(np.float64)
    valid_bb = generator.integers(0, 5, n_spots).astype(np.float64)

    np.testing.assert_array_equal(
        BoundaryInvariants(valid_nb, valid_bb).relative_channel_weight(indptr, indices),
        _cnaster_weight(valid_nb, valid_bb, indices, indptr),
    )


@pytest.mark.patch
@pytest.mark.parametrize(("n_obs", "n_clones"), [(1, 1), (5, 3), (40, 4)])
def test_the_flat_pred_is_read_as_cnaster_reads_it(n_obs: int, n_clones: int) -> None:
    """`pred[c * n_obs + o]` reads as `decoded[o, c]`, entry by entry."""
    flat = np.arange(n_obs * n_clones)
    decoded = _decoded(flat, n_obs)

    assert decoded.shape == (n_obs, n_clones)

    for clone in range(n_clones):
        for position in range(n_obs):
            assert decoded[position, clone] == flat[clone * n_obs + position]

    square = flat.reshape(n_clones, n_obs).T
    assert _decoded(square, n_obs) is square


@pytest.mark.infra
def test_the_fallback_does_not_call_itself() -> None:
    """The fallback binds `cnaster`'s function at import, so it cannot recurse into itself."""

    captured = clone_assignment.UPSTREAM

    assert captured is not clone_assignment.pipeline_clone_assignment

    with patched():
        assert (
            cnaster.hmrf.pipeline_clone_assignment
            is clone_assignment.pipeline_clone_assignment
        ), "the swap did not install"
        assert (
            clone_assignment.UPSTREAM is not cnaster.hmrf.pipeline_clone_assignment
        ), "the fallback would recurse"


@pytest.mark.infra
def test_the_swap_is_in_the_default_table() -> None:
    """The swap is a default-table row, as it reproduces `cnaster` bitwise."""

    rows: Any = [swap for swap in SWAPS if swap.name == "pipeline_clone_assignment"]

    assert len(rows) == 1
    assert rows[0].ticket == 206
    assert "pipeline_clone_assignment" not in {swap.name for swap in FIGURE_SWAPS}


@pytest.mark.patch
def test_the_boundary_invariants_are_computed_once_per_dataset() -> None:
    """Boundary invariants are cached per dataset and equal a fresh computation (#59 item 4)."""

    generator = np.random.default_rng(13)

    base_nb_mean = generator.integers(0, 3, (20, 9)).astype(np.float64)
    total_bb_RD = generator.integers(0, 3, (20, 9)).astype(np.float64)

    first = boundary(base_nb_mean, total_bb_RD, None)
    second = boundary(base_nb_mean, total_bb_RD, None)

    assert first is second, "the invariants were recomputed for the same arrays"

    assert np.array_equal(
        first.counts.num_valid_nb_spotwise, (base_nb_mean > 0).sum(axis=0)
    )
    assert np.array_equal(
        first.counts.num_valid_bb_spotwise, (total_bb_RD > 0).sum(axis=0)
    )
    assert np.array_equal(first.weight, np.ones(9))


@pytest.mark.smoke
def test_the_invariant_cache_holds_the_arrays_it_is_keyed_on() -> None:
    """The single `id()`-keyed cache entry holds references to its arrays."""

    first = np.ones((4, 3))
    second = np.ones((4, 3))

    entry = boundary(first, second, None)

    assert len(_BOUNDARY) == 1
    assert entry.held[0] is first
    assert entry.held[1] is second

    boundary(np.ones((4, 3)), np.ones((4, 3)), None)

    assert len(_BOUNDARY) == 1, "the cache grew past its one slot"
