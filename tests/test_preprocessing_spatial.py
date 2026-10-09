"""`port.patch.spatial` against the `cnaster` functions it replaces, bitwise (#190).

`cnaster`'s own defects (#180) are carried across unchanged.
"""

import numpy as np
import pytest
from cnaster.spatial import best_equal_partition as cnaster_best_equal_partition
from cnaster.spatial import construct_multislice_lattice_adjacency as upstream
from port.patch.spatial import _rectangle_counts
from port.patch.spatial import best_equal_partition as port_best_equal_partition
from port.patch.spatial import construct_multislice_lattice_adjacency as patched

from tests.adapters import square_coords

pytestmark = pytest.mark.preprocessing

LATTICES = [(25, 40), (50, 50)]
"""`(rows, columns)` spot lattices: the dev instance, and 2,500 spots."""


@pytest.mark.patch
def test_the_sparse_adjacency_joins_slices_the_same_way() -> None:
    """Two slices join into the same block diagonal as `cnaster`'s."""

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
def test_the_partition_agrees_where_the_summed_area_table_is_declined() -> None:
    """Scattered coordinates take the non-table fallback and equal `cnaster`'s."""

    coords = np.random.default_rng(3).uniform(0, 100, size=(600, 2))

    assert _rectangle_counts(coords) is None

    reference_index, reference_assignment = cnaster_best_equal_partition(
        coords, 3, 3, n_trials=100
    )
    realized_index, realized_assignment = port_best_equal_partition(
        coords, 3, 3, n_trials=100
    )

    np.testing.assert_array_equal(realized_assignment, reference_assignment)

    for realized, reference in zip(realized_index, reference_index, strict=True):
        np.testing.assert_array_equal(realized, reference)
