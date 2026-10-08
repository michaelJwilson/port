"""`cnaster.hmrf`'s COO adjacency triple from the CSR arrays, without the Python round trip (#59 item 3).

Replaces `unpack_adjacency(cast_csr(adjacency_mat))`. A simplification;
bitwise equal to `cnaster`'s.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix

__all__ = ["adjacency_coo"]


def adjacency_coo(
    adjacency_mat: csr_matrix,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(adj_spots, adj_neighbors, adj_weights)` as `int64`, `int64`, `float64`, bitwise `cnaster`'s."""
    counts = np.diff(adjacency_mat.indptr)

    return (
        np.repeat(np.arange(adjacency_mat.shape[0]), counts).astype(np.int64),
        adjacency_mat.indices.astype(np.int64),
        adjacency_mat.data.astype(np.float64),
    )
