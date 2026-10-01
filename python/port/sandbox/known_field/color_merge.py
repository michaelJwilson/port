"""The color merge: cnaster's `merge_assignment` loop, in closed form (#556).

`cnaster.hmrf` relabels a whole clone u into another v, the pair that lowers
the energy most, and repeats until no pair does (`cnaster.icm.merge_assignment`,
reached when `merge=True`). Its cost walks every spot's neighbours per round;
here each round is two sparse products. Merging u into v changes the Potts
energy E = -sum_i h_i(s_i) - beta sum_{(i,j)} w_ij [s_i = s_j] by

    delta(u, v) = -(U[u, v] - U[u, u]) - beta (B[u, v] + B[v, u]) / 2,

with U[u, k] the field of u's spots summed at clone k and B[u, v] the directed
edge weight from u's spots to v's. The halving is `potts_graph_from`'s: it
couples an unordered pair at beta (A_ij + A_ji) / 2.
"""

from __future__ import annotations

import numpy as np

__all__ = ["color_merge", "merge_deltas"]


def merge_deltas(
    field: np.ndarray,
    labels: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    beta: float,
) -> np.ndarray:
    """`(q, q)` energy change of relabelling all of u to v; `inf` on the diagonal and at empty clones."""
    import scipy.sparse as sp

    n, q = field.shape
    adjacency = sp.csr_matrix((weights, indices, indptr), shape=(n, n))
    member = sp.csr_matrix((np.ones(n), (np.arange(n), labels)), shape=(n, q))
    summed = np.asarray(member.T @ field)
    boundary = (member.T @ adjacency @ member).toarray()
    delta: np.ndarray = (
        -(summed - np.diag(summed)[:, None]) - beta * (boundary + boundary.T) / 2
    )
    alive = np.bincount(labels, minlength=q) > 0
    delta[~alive, :] = np.inf
    delta[:, ~alive] = np.inf
    np.fill_diagonal(delta, np.inf)
    return delta


def color_merge(
    field: np.ndarray,
    labels: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    beta: float,
) -> tuple[np.ndarray, int]:
    """Merge the best clone pair while the energy drops; the labelling and how many merges."""
    labels = np.asarray(labels, dtype=np.int64).copy()
    merges = 0

    while True:
        delta = merge_deltas(field, labels, indptr, indices, weights, beta)
        u, v = np.unravel_index(int(np.argmin(delta)), delta.shape)
        if not delta[u, v] < 0:
            return labels, merges
        labels[labels == u] = v
        merges += 1
