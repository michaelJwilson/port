"""`port.patch.spatial` against the `cnaster` functions it replaces, bitwise (#190).

`cnaster`'s own defects (#180) are carried across unchanged.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from tests.adapters import square_coords

pytestmark = pytest.mark.preprocessing

LATTICES = [(25, 40), (50, 50)]
"""`(rows, columns)` spot lattices: the dev instance, and 2,500 spots."""


@pytest.mark.patch
@pytest.mark.parametrize(("rows", "columns"), LATTICES)
def test_the_sparse_adjacency_is_the_dense_one(rows: int, columns: int) -> None:
    """The sparse adjacency equals `cnaster`'s dense one: graph, weights and dtype."""
    from cnaster.spatial import construct_multislice_lattice_adjacency as upstream
    from port.patch.spatial import construct_multislice_lattice_adjacency as patched

    coords = square_coords(rows, columns).astype(float)
    sample_ids = np.zeros(len(coords), dtype=int)

    reference = upstream(sample_ids, [0], coords, None, 1, 1, 1)
    realized = patched(sample_ids, [0], coords, None, 1, 1, 1)

    assert (realized.adjacency_mat != reference.adjacency_mat).nnz == 0
    assert (realized.smooth_mat != reference.smooth_mat).nnz == 0
    assert realized.adjacency_mat.dtype == reference.adjacency_mat.dtype
    assert realized.smooth_mat.dtype == reference.smooth_mat.dtype


@pytest.mark.patch
def test_the_sparse_adjacency_joins_slices_the_same_way() -> None:
    """Two slices join into the same block diagonal as `cnaster`'s."""
    from cnaster.spatial import construct_multislice_lattice_adjacency as upstream
    from port.patch.spatial import construct_multislice_lattice_adjacency as patched

    coords = np.concatenate(
        [square_coords(12, 10).astype(float), square_coords(9, 8).astype(float)]
    )
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

    coords = square_coords(10, 10).astype(float)
    sample_ids = np.zeros(len(coords), dtype=int)
    across = sp.random(100, 100, density=0.02, format="csr", random_state=5)

    reference = upstream(sample_ids, [0], coords, across, 1, 1, 1)
    realized = patched(sample_ids, [0], coords, across, 1, 1, 1)

    assert (realized.adjacency_mat != reference.adjacency_mat).nnz == 0


@pytest.mark.patch
@pytest.mark.parametrize(("rows", "columns"), LATTICES)
def test_the_partition_keeps_the_trial_cnaster_keeps(rows: int, columns: int) -> None:
    """The partition keeps `cnaster`'s winning trial: same indices and assignment."""
    from cnaster.spatial import best_equal_partition as upstream
    from port.patch.spatial import best_equal_partition as patched

    coords = square_coords(rows, columns).astype(float)

    reference_index, reference_assignment = upstream(coords, 3, 3, n_trials=200)
    realized_index, realized_assignment = patched(coords, 3, 3, n_trials=200)

    np.testing.assert_array_equal(realized_assignment, reference_assignment)
    assert len(realized_index) == len(reference_index)

    for realized, reference in zip(realized_index, reference_index, strict=True):
        np.testing.assert_array_equal(realized, reference)


@pytest.mark.patch
def test_the_partition_agrees_where_the_summed_area_table_is_declined() -> None:
    """Scattered coordinates take the non-table fallback and equal `cnaster`'s."""
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
    """Under `single_tumor_prop` the partition equals `cnaster`'s."""
    from cnaster.spatial import best_equal_partition as upstream
    from port.patch.spatial import best_equal_partition as patched

    coords = square_coords(25, 40).astype(float)
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
