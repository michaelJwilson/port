"""Multi-sample support: the cross-sample adjacency placeholder and plot panels (#328).

`cnaster` fits several samples with shared clones by concatenating their
spots. Spatial edges are within a sample (`construct_multislice_lattice_adjacency`
is block diagonal), and edges between samples enter only through
`across_slice_adjacency_mat`, which `load_input_data` takes from
`get_alignments` and which is `None` without alignment files.

`cross_sample_adjacency` is where those edges will come from. Today it
returns none, and nothing installs it: a run is within-sample only, as the
fixture intends.

`sample_panels` splits a figure's spots into one panel per sample on a
`(rows, columns)` layout, each in its own coordinates, for
`port.patch.plotting.spatial`'s `sample_layout`.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

__all__ = ["cross_sample_adjacency", "sample_panels"]


def cross_sample_adjacency(sample_label: np.ndarray) -> sp.csr_matrix:
    """Edges between spots of different samples, `(n_spots, n_spots)`: none yet.

    The placeholder for `across_slice_adjacency_mat`. An implementation
    returns a symmetric matrix with no entry inside a sample's own block,
    which `tests/test_multisample.py` holds this one to.
    """
    n_spots = int(np.asarray(sample_label).size)

    return sp.csr_matrix((n_spots, n_spots), dtype=np.float64)


def sample_panels(
    sample_ids: np.ndarray, layout: tuple[int, int]
) -> list[tuple[int, int, np.ndarray]]:
    """`(row, column, spots)` per sample, filling `layout` row by row.

    Raises
    ------
    ValueError
        If the layout has fewer panels than there are samples.
    """
    samples = np.unique(np.asarray(sample_ids))
    rows, columns = layout

    if rows * columns < samples.size:
        msg = f"a {rows} x {columns} layout holds {rows * columns} of {samples.size} samples"
        raise ValueError(msg)

    return [
        (i // columns, i % columns, np.flatnonzero(sample_ids == sample))
        for i, sample in enumerate(samples)
    ]
