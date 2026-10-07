"""A labelling recoloured so each label is one connected region (#751).

A Potts sample is a local prior: one colour recurs over disjoint patches, and
`cnaster` would fit each colour as one clone over all of them. `recolour` gives
each connected patch of one label its own label. Moved from
`port.sandbox.wolff_init` (#362), which keeps the name, with
`port.studies.recoloured_clones` as its measurement (#751).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

__all__ = ["recolour"]


def recolour(labels: np.ndarray, adjacency: sp.csr_matrix) -> np.ndarray:
    """Each connected patch of one label, as its own label `0..k-1`.

    `adjacency` is symmetric, `(n, n)`; two spots are in one patch when a path
    of edges joins them through spots of their label. Labels are numbered in
    `scipy.sparse.csgraph.connected_components`' order.
    """
    upper = sp.triu(adjacency, k=1).tocoo()
    same = labels[upper.row] == labels[upper.col]
    bonds = sp.coo_matrix(
        (np.ones(int(same.sum())), (upper.row[same], upper.col[same])),
        shape=adjacency.shape,
    )
    _, patches = connected_components(bonds, directed=False)
    patches_: np.ndarray = patches.astype(np.int64)
    return patches_
