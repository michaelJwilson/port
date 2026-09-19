"""`port.patch.spatial` against the `cnaster` functions it replaces (#190).

**13x and 72x less peak on the adjacency at 2,500 spots, 67x and 289x at
10,000; 8.8x and 17x on the partition at the same two sizes.** Both returns
are bitwise what `cnaster` returns, which is what makes the ratios a cost
decision rather than a different answer.

The two are here together because they are the same defect twice: a quantity
computed in a representation larger than it needs. `cnaster` densifies a
k-nearest-neighbour graph to block-diagonalize it, and builds every trial
partition's index lists to measure the sizes it will discard.

**What is not claimed:** that either function is right. Both are `cnaster`'s
algorithm reproduced, and the adjacency's own defects -- an asymmetric kNN
graph, a discarded `maxspots_pooling` -- are #180's and are carried across
unchanged rather than fixed here.
"""

import numpy as np
import pytest
import scipy.sparse as sp

pytestmark = pytest.mark.preprocessing

LATTICES = [(25, 40), (50, 50)]
"""`(rows, columns)` spot lattices: the dev instance, and 2,500 spots."""


def _lattice(rows: int, columns: int) -> np.ndarray:
    """Spot coordinates on a rectangular lattice, as a slide carries them."""
    x_grid, y_grid = np.meshgrid(np.arange(rows), np.arange(columns), indexing="ij")

    return np.stack([x_grid.ravel(), y_grid.ravel()], axis=1).astype(float)


@pytest.mark.patch
@pytest.mark.parametrize(("rows", "columns"), LATTICES)
def test_the_sparse_adjacency_is_the_dense_one(rows: int, columns: int) -> None:
    """**Same graph, same weights, same dtype -- and no `n_spots^2` array.**

    The dtype is asserted because it is the one thing a sparse assembly can
    change silently: `scipy.linalg.block_diag` promotes to the common type of
    its inputs, and a promotion would leave every value equal and every
    downstream buffer twice the size.
    """
    from cnaster.spatial import construct_multislice_lattice_adjacency as upstream
    from port.patch.spatial import construct_multislice_lattice_adjacency as patched

    coords = _lattice(rows, columns)
    sample_ids = np.zeros(len(coords), dtype=int)

    reference = upstream(sample_ids, [0], coords, None, 1, 1, 1)
    realized = patched(sample_ids, [0], coords, None, 1, 1, 1)

    assert (realized.adjacency_mat != reference.adjacency_mat).nnz == 0
    assert (realized.smooth_mat != reference.smooth_mat).nnz == 0
    assert realized.adjacency_mat.dtype == reference.adjacency_mat.dtype
    assert realized.smooth_mat.dtype == reference.smooth_mat.dtype


@pytest.mark.patch
def test_the_sparse_adjacency_joins_slices_the_same_way() -> None:
    """Two slices, so the block diagonal has something to do.

    A one-slice instance is the case where `block_diag` is the identity, and
    the fixtures in this repository are all one slice. The join is where an
    offset can be wrong, so it is exercised deliberately.
    """
    from cnaster.spatial import construct_multislice_lattice_adjacency as upstream
    from port.patch.spatial import construct_multislice_lattice_adjacency as patched

    coords = np.concatenate([_lattice(12, 10), _lattice(9, 8)])
    sample_ids = np.concatenate([np.zeros(120, dtype=int), np.ones(72, dtype=int)])

    reference = upstream(sample_ids, [0, 1], coords, None, 1, 1, 1)
    realized = patched(sample_ids, [0, 1], coords, None, 1, 1, 1)

    assert realized.adjacency_mat.shape == (192, 192)
    assert (realized.adjacency_mat != reference.adjacency_mat).nnz == 0
    assert (realized.smooth_mat != reference.smooth_mat).nnz == 0


@pytest.mark.patch
def test_the_across_slice_term_is_still_added() -> None:
    """The optional inter-slice graph, which the sparse path must not drop."""
    from cnaster.spatial import construct_multislice_lattice_adjacency as upstream
    from port.patch.spatial import construct_multislice_lattice_adjacency as patched

    coords = _lattice(10, 10)
    sample_ids = np.zeros(len(coords), dtype=int)
    across = sp.random(100, 100, density=0.02, format="csr", random_state=5)

    reference = upstream(sample_ids, [0], coords, across, 1, 1, 1)
    realized = patched(sample_ids, [0], coords, across, 1, 1, 1)

    assert (realized.adjacency_mat != reference.adjacency_mat).nnz == 0


@pytest.mark.patch
@pytest.mark.parametrize(("rows", "columns"), LATTICES)
def test_the_partition_keeps_the_trial_cnaster_keeps(rows: int, columns: int) -> None:
    """**The same trial wins, so the same clones come out.**

    `cnaster` compares strictly, so the earliest trial attaining the minimum
    variance wins; a patch that compared with `<=`, or that reordered the
    draws within a trial, would return a different partition from the same
    seed and every clone downstream would differ.

    The index lists are compared and not only the assignment vector, because
    they are separate returns and the caller uses both.
    """
    from cnaster.spatial import best_equal_partition as upstream
    from port.patch.spatial import best_equal_partition as patched

    coords = _lattice(rows, columns)

    reference_index, reference_assignment = upstream(coords, 3, 3, n_trials=200)
    realized_index, realized_assignment = patched(coords, 3, 3, n_trials=200)

    np.testing.assert_array_equal(realized_assignment, reference_assignment)
    assert len(realized_index) == len(reference_index)

    for realized, reference in zip(realized_index, reference_index, strict=True):
        np.testing.assert_array_equal(realized, reference)


@pytest.mark.patch
def test_the_partition_agrees_where_the_summed_area_table_is_declined() -> None:
    """**Scattered coordinates take the fallback, and agree there too.**

    The table is `distinct x` by `distinct y`, which is about `n_spots` for a
    lattice and `n_spots^2` for scattered points, so it is declined for the
    second. That branch is unreachable from any fixture in this repository --
    every one of them is a lattice -- so it is reached directly here.
    """
    from cnaster.spatial import best_equal_partition as upstream
    from port.patch.spatial import _rectangle_counts
    from port.patch.spatial import best_equal_partition as patched

    coords = np.random.default_rng(3).uniform(0, 100, size=(600, 2))

    assert _rectangle_counts(coords) is None

    reference_index, reference_assignment = upstream(coords, 3, 3, n_trials=100)
    realized_index, realized_assignment = patched(coords, 3, 3, n_trials=100)

    np.testing.assert_array_equal(realized_assignment, reference_assignment)

    for realized, reference in zip(realized_index, reference_index, strict=True):
        np.testing.assert_array_equal(realized, reference)


@pytest.mark.patch
def test_the_partition_agrees_under_a_tumour_proportion() -> None:
    """The `single_tumor_prop` branch, which moves the grid and not the spots.

    The threshold selects which spots set the partition's extent, while every
    spot is still assigned to a cell. Reading that the wrong way round would
    leave the tumour spots correctly partitioned and everything else silently
    dropped, so it is pinned separately.
    """
    from cnaster.spatial import best_equal_partition as upstream
    from port.patch.spatial import best_equal_partition as patched

    coords = _lattice(25, 40)
    proportion = np.random.default_rng(7).uniform(0, 1, size=len(coords))

    reference_index, reference_assignment = upstream(
        coords, 3, 3, single_tumor_prop=proportion, n_trials=100
    )
    realized_index, realized_assignment = patched(
        coords, 3, 3, single_tumor_prop=proportion, n_trials=100
    )

    np.testing.assert_array_equal(realized_assignment, reference_assignment)

    for realized, reference in zip(realized_index, reference_index, strict=True):
        np.testing.assert_array_equal(realized, reference)
