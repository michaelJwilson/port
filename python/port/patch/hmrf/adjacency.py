"""`cnaster`'s COO adjacency triple, without the Python round trip.

Issue #59 item 3. `cnaster.hmrf` builds one graph in two representations on
every outer iteration:

    adj_list = cast_csr(adjacency_mat)                                 # :284
    adj_spots, adj_neighbors, adj_weights = unpack_adjacency(adj_list)  # :285
    ...
    icm_sweep_deque(adj_indptr=adjacency_mat.indptr, adj_indices=..., ...)

The solver takes the CSR arrays directly. The COO triple exists for
`calc_assignment_cost` and `merge_assignment`, and reaching it costs two pure
Python passes over the non-zeros: `cast_csr` materializes a list of lists of
tuples, and `unpack_adjacency` walks that into three Python lists before
converting to arrays.

The same triple is three `numpy` calls on the CSR arrays already in hand.

**This is a simplification**: two Python loops become three array
expressions, and it lands on its evidence of equivalence alone. Measured:
`docs/measurements.md`, `port.patch.hmrf.adjacency`.

**The graph is also invariant across outer iterations**, so the better change
is to build the triple once outside the loop. That is the call site's to make
and this function is what it would call.

**Referee: `cnaster` itself, bitwise**, on every one of the three arrays.
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
    """`(adj_spots, adj_neighbors, adj_weights)`, from the CSR arrays.

    Byte-for-byte what `unpack_adjacency(cast_csr(adjacency_mat))` returns,
    including the dtypes: `int64`, `int64`, `float64`.

    The row index is `np.repeat` over the per-row non-zero counts, which is
    what `cast_csr`'s outer loop and `unpack_adjacency`'s inner one compute
    between them. `indices` and `data` are already the other two columns and
    are cast rather than rebuilt.
    """
    counts = np.diff(adjacency_mat.indptr)

    return (
        np.repeat(np.arange(adjacency_mat.shape[0]), counts).astype(np.int64),
        adjacency_mat.indices.astype(np.int64),
        adjacency_mat.data.astype(np.float64),
    )
