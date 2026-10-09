"""Multi-sample support: the cross-sample adjacency placeholder and plot panels (#328).

`cross_sample_adjacency` is an uninstalled placeholder for
`across_slice_adjacency_mat`; `sample_panels` lays out one panel per sample
for `port.patch.plotting.spatial`'s `sample_layout`.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

__all__ = ["cross_sample_adjacency", "sample_panels"]


def cross_sample_adjacency(sample_label: np.ndarray) -> sp.csr_matrix:
    """Edges between spots of different samples, `(n_spots, n_spots)`: none yet."""
    n_spots = int(np.asarray(sample_label).size)

    return sp.csr_matrix((n_spots, n_spots), dtype=np.float64)


def sample_panels(
    sample_ids: np.ndarray, layout: tuple[int, int]
) -> list[tuple[int, int, np.ndarray]]:
    """`(row, column, spots)` per sample, filling `layout` row by row; `ValueError` if too few panels."""
    samples = np.unique(np.asarray(sample_ids))
    rows, columns = layout

    if rows * columns < samples.size:
        msg = f"a {rows} x {columns} layout holds {rows * columns} of {samples.size} samples"
        raise ValueError(msg)

    return [
        (i // columns, i % columns, np.flatnonzero(sample_ids == sample))
        for i, sample in enumerate(samples)
    ]
