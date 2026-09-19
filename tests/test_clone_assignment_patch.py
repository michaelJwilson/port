"""The seam rebound one level up (#206), in the pieces that can be isolated.

`pipeline_clone_assignment` is the name `port` rebinds so that #59's
measured patches install at all -- each of them needs a call-site edit
inside `cnaster.hmrf`, which is read only, and rebinding the caller is what
makes those edits `port`'s to make. The whole-function claim is the
whole-run one in `tests/test_patched_entry_point.py`: two `run_cnaster` runs
compared artifact by artifact.

What is here is the arithmetic that whole-run comparison cannot localize --
the channel weight, the decoded layout, and the delegation that once called
itself.
"""

from typing import Any

import numpy as np
import pytest
from port.patch.clone_assignment import _channel_weight, _decoded


def _cnaster_weight(
    valid_nb: np.ndarray,
    valid_bb: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
) -> np.ndarray:
    """`rel_valid_emision_weight` as `compute_loglike_spot_assignment` writes it.

    Transcribed from `hmrf.py` rather than called, because `cnaster` computes
    it inside an `njit` function that also computes the field and exposes
    neither separately.
    """
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
    """A `bincount` over CSR rows, against the loop it replaces, bitwise.

    The fixture carries the three cases the loop branches on: a spot whose
    neighbourhood pools a zero RDR count, one that pools a zero BAF count,
    and an **empty** neighbourhood -- which is where a `reduceat` would read
    the next row instead of returning zero, and is why this is a `bincount`.
    """
    generator = np.random.default_rng(19)

    degrees = generator.integers(0, 4, n_spots)
    degrees[0] = 0

    indptr = np.concatenate([[0], np.cumsum(degrees)]).astype(np.int64)
    indices = generator.integers(0, n_spots, int(degrees.sum())).astype(np.int64)

    valid_nb = generator.integers(0, 5, n_spots).astype(np.float64)
    valid_bb = generator.integers(0, 5, n_spots).astype(np.float64)

    np.testing.assert_array_equal(
        _channel_weight(valid_nb, valid_bb, indices, indptr),
        _cnaster_weight(valid_nb, valid_bb, indices, indptr),
    )


@pytest.mark.patch
@pytest.mark.parametrize(("n_obs", "n_clones"), [(1, 1), (5, 3), (40, 4)])
def test_the_flat_pred_is_read_as_cnaster_reads_it(n_obs: int, n_clones: int) -> None:
    """`pred[c * n_obs + o]` is `decoded[o, c]`, entry by entry.

    **The live pipeline passes the flat form**, which was found by delegating
    on it and reading the log rather than by reading the call sites -- so the
    layout here is the difference between a patch that runs and one that
    never leaves its fallback.
    """
    flat = np.arange(n_obs * n_clones)
    decoded = _decoded(flat, n_obs)

    assert decoded.shape == (n_obs, n_clones)

    for clone in range(n_clones):
        for position in range(n_obs):
            assert decoded[position, clone] == flat[clone * n_obs + position]

    # A caller that already passed the two-dimensional form gets it back.
    square = flat.reshape(n_clones, n_obs).T
    assert _decoded(square, n_obs) is square


@pytest.mark.infra
def test_the_fallback_does_not_call_itself() -> None:
    """The delegation resolves `cnaster`'s function at import, not at call.

    `port.pipeline.patched()` imports this module to resolve the replacement
    and *then* rebinds `cnaster.hmrf.pipeline_clone_assignment`, so a
    delegation that looked the name up late would find itself. It did, and
    the symptom was a `RecursionError` two minutes into a whole run --
    reachable only from a real pipeline, which is why it is pinned here
    where it costs nothing.
    """
    import cnaster.hmrf
    from port.patch import clone_assignment
    from port.pipeline import patched

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
    """It reproduces `cnaster` bitwise, so it belongs with the rest.

    The whole-run test is what establishes that; this pins that the row is
    where that claim is asserted rather than in `FIGURE_SWAPS`, which is for
    replacements that change their output.
    """
    from port.pipeline import FIGURE_SWAPS, SWAPS

    rows: Any = [swap for swap in SWAPS if swap.name == "pipeline_clone_assignment"]

    assert len(rows) == 1
    assert rows[0].ticket == 206
    assert "pipeline_clone_assignment" not in {swap.name for swap in FIGURE_SWAPS}


@pytest.mark.patch
def test_the_boundary_invariants_are_computed_once_per_dataset() -> None:
    """#59 item 4, hoisted across calls rather than out of a loop body.

    The two valid-segment counts and the channel weight derived from them are
    functions of the input data, which the outer loop never fits, and
    `cnaster` recomputes all three on every iteration.
    `port.patch.clone_assignment.boundary` returns the same object for the
    same arrays, which is what "computed where they are constant" means when
    the loop is inside a dependency this repository cannot edit.

    The values are checked against a fresh computation too: a cache that
    returned the same wrong answer twice would pass an identity check alone.
    """
    from port.patch.clone_assignment import boundary

    generator = np.random.default_rng(13)

    base_nb_mean = generator.integers(0, 3, (20, 9)).astype(np.float64)
    total_bb_RD = generator.integers(0, 3, (20, 9)).astype(np.float64)

    first = boundary(base_nb_mean, total_bb_RD, None)
    second = boundary(base_nb_mean, total_bb_RD, None)

    assert first is second, "the invariants were recomputed for the same arrays"

    assert np.array_equal(first.valid_nb, (base_nb_mean > 0).sum(axis=0))
    assert np.array_equal(first.valid_bb, (total_bb_RD > 0).sum(axis=0))
    assert np.array_equal(first.weight, np.ones(9))


@pytest.mark.smoke
def test_the_invariant_cache_holds_the_arrays_it_is_keyed_on() -> None:
    """Why the entry keeps references, and why there is only ever one.

    The cache is keyed on `id()`, which is unique only while the object is
    alive, so an entry that did not hold its arrays could be handed a
    recycled address and answer with another dataset's counts. The arrays are
    the pipeline's own inputs and outlive the loop regardless, so holding
    them costs nothing.

    One slot, because a run conditions on one dataset: a second entry would
    mean something is calling the seam with data it did not load.
    """
    from port.patch.clone_assignment import _BOUNDARY, boundary

    first = np.ones((4, 3))
    second = np.ones((4, 3))

    entry = boundary(first, second, None)

    assert len(_BOUNDARY) == 1
    assert entry.held[0] is first
    assert entry.held[1] is second

    boundary(np.ones((4, 3)), np.ones((4, 3)), None)

    assert len(_BOUNDARY) == 1, "the cache grew past its one slot"
