"""Set aside (#467): the phasing stage's initial clones from Wolff clusters, sized by counts.

**Measured, and worse than cnaster's grid on every sample**, under `--sal`
with #476's clone flags (clone ARI, clones, phase-free exact altered):

| sample | grid | this start |
| --- | --- | --- |
| CalicoST hard | 0.982 (4), 0.472 | 0.858 (5), 0.238 |
| CalicoST easy | 0.986 (4), 0.385 | 0.986 (4), 0.342 |
| `dev_tree` r0 | 0.998 (4), 0.707 | 0.742 (6), 0.701 |

It chose `J = 0.50-0.55` and 9-10 clones where the grid gives 8 over two
slices, and the extra irregular clones survive to the end on r0 and hard.
A q-colour Potts field was tried first and does not serve: Wolff moves order
a 42 x 42 lattice within half a move per spot at every J and q tried (5 to
12). `wolff_init` in this package is the earlier Potts-draw start (#362).

Installed by nothing. To rerun it, record the loader's per-spot SNP UMIs in
`SPOT_COUNTS` and rebind `cnaster.spatial.initialize_clones` to
:func:`initialize_clones`.

`cnaster.spatial.initialize_clones` cuts each slice into an
`npart_phasing x npart_phasing` grid of rectangles; `cnaster.wolff` carries a
Wolff initializer that `run_cnaster.py:257` leaves commented out. A rectangle
is sized by area, which says nothing about how many allele counts it pools,
and its edges are where the grid put them rather than where the tissue is.

`initialize_clones` here grows clusters instead, and sizes them by counts:

1. **Sweep J.** For each coupling on `COUPLINGS`, every spot starts in one
   colour and `sal`'s `wolff_sweep` grows a cluster from each spot not yet
   reached. From one colour a Wolff cluster is a bond-percolation component,
   so one pass realizes a clone label per spot.
2. **Count each component.** Its total SNP-covering UMI, from the loader
   (`SPOT_COUNTS`).
3. **Set J by counts.** The target is the SNP-covering UMI a clone needs to
   read a BAF shift of `DELTA` at `Z` standard errors over an event of
   `EVENT_BP`: `Z^2 p(1-p) / DELTA^2` allele counts on the event, at
   `p = 1/2`, which is `GENOME_BP / EVENT_BP` times that over the genome.
   The largest J at which no component exceeds it is kept: the largest seeds
   short of the percolating one. A q-colour Potts field does not serve: its
   Wolff moves order a 42 x 42 lattice within half a move per spot at every
   J and q tried (5 to 12), leaving one domain or none at the target.
4. **Merge to the target.** Components under it are merged, smallest first,
   into the neighbouring component they share most edges with, until every
   component clears it or one is left.

Without the loader's counts for these spots, or with fewer spots than two
clones' worth, it is `cnaster`'s rectangles.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "COUPLINGS",
    "DELTA",
    "EVENT_BP",
    "GENOME_BP",
    "SPOT_COUNTS",
    "Z",
    "components_at",
    "initialize_clones",
    "neighbour_graph",
    "target_counts",
]

SPOT_COUNTS: list[np.ndarray] = []
"""Per-spot SNP-covering UMI, `(n_spots,)`, recorded by whoever installs this."""

COUPLINGS = np.linspace(0.05, 1.5, 30)
"""The J swept, in units of the adjacency weight."""

DELTA = 0.1
"""The BAF shift a clone must resolve: `(3, 2)` against `(1, 1)`."""

Z = 3.0
"""Standard errors the shift must clear."""

EVENT_BP = 1.0e7
"""The smallest event to resolve: CalicoST hard's `cnasize1e7`."""

GENOME_BP = 2.875e9
"""Autosomal GRCh38."""


def target_counts() -> float:
    """SNP-covering UMI a clone needs: `Z^2 p(1-p) / DELTA^2 * GENOME_BP / EVENT_BP`."""
    return Z**2 * 0.25 / DELTA**2 * GENOME_BP / EVENT_BP


def neighbour_graph(coords: np.ndarray, sample_ids: np.ndarray) -> Any:
    """Unit-weight edges between spots within 1.2 median nearest-neighbour distances, per slice."""
    import scipy.sparse
    from sklearn.neighbors import NearestNeighbors

    n = coords.shape[0]
    rows, cols = [], []

    for sample in np.unique(sample_ids):
        index = np.flatnonzero(sample_ids == sample)

        if index.size < 2:
            continue

        points = np.asarray(coords[index], dtype=np.float64)
        nearest = NearestNeighbors(n_neighbors=2).fit(points)
        spacing = float(np.median(nearest.kneighbors(points)[0][:, 1]))
        graph = NearestNeighbors(radius=1.2 * spacing).fit(points)
        adjacency = graph.radius_neighbors_graph(points, mode="connectivity").tocoo()
        keep = adjacency.row != adjacency.col
        rows.append(index[adjacency.row[keep]])
        cols.append(index[adjacency.col[keep]])

    row = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    col = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)

    return scipy.sparse.csr_matrix((np.ones(row.size), (row, col)), shape=(n, n))


def components_at(
    adjacency: Any, coupling: float, rng: np.random.Generator
) -> np.ndarray:
    """One label per spot: Wolff clusters grown from one colour at `coupling`.

    Two colours, unreached and reached: each cluster is grown by `sal`'s
    `wolff_sweep` from an unreached root and recoloured to reached, and its
    members are the spots that changed. From one colour a Wolff cluster is a
    bond-percolation component, bond probability `1 - exp(-J w)`.
    """
    from sal.sample.potts_mcmc.sweeps import adjacency_lists, wolff_sweep

    n = adjacency.shape[0]
    weights = np.asarray(adjacency.data, dtype=np.float64) * coupling
    lists = adjacency_lists(adjacency.indptr, adjacency.indices, weights)
    state = np.zeros(n, dtype=np.int64)
    field = np.zeros((n, 2))
    labels = np.full(n, -1, dtype=np.int64)
    colour = 0

    for root in rng.permutation(n):
        if state[root] != 0:
            continue

        wolff_sweep(
            state,
            field,
            adjacency.indptr,
            adjacency.indices,
            weights,
            rng,
            root=int(root),
            proposed=1,
            lists=lists,
        )
        labels[(state == 1) & (labels < 0)] = colour
        colour += 1

    return labels


def _merge_small(
    labels: np.ndarray, adjacency: Any, counts: np.ndarray, target: float
) -> np.ndarray:
    """Merge components under `target`, smallest first, into their most-connected neighbour."""
    labels = labels.copy()
    coo = adjacency.tocoo()

    while True:
        totals = np.bincount(labels, weights=counts)
        present = np.flatnonzero(np.bincount(labels) > 0)

        if present.size <= 1:
            return labels

        small = present[totals[present] < target]

        if small.size == 0:
            return labels

        victim = int(small[np.argmin(totals[small])])
        inside = labels[coo.row] == victim
        across = labels[coo.col[inside]]
        across = across[across != victim]

        if across.size == 0:
            # NB an island: into the component nearest in index order, so the
            #    loop ends; islands are rare on a connected tissue.
            others = present[present != victim]
            labels[labels == victim] = int(others[0])
            continue

        labels[labels == victim] = int(np.bincount(across).argmax())


def initialize_clones(
    coords: Any,
    sample_ids: Any,
    x_part: int,
    y_part: int,
    single_tumor_prop: Any = None,
    threshold: Any = None,
    random_state: Any = None,
    config: Any = None,
) -> list[np.ndarray]:
    """`cnaster.spatial.initialize_clones`'s signature; Wolff clusters sized by counts."""
    from cnaster.spatial import initialize_clones as upstream

    coords = np.asarray(coords)
    sample_ids = np.asarray(sample_ids)
    n = coords.shape[0]
    counts = SPOT_COUNTS[-1] if SPOT_COUNTS else None
    target = target_counts()

    if counts is None or counts.size != n or counts.sum() < 2 * target:
        return upstream(  # type: ignore[no-any-return]
            coords,
            sample_ids,
            x_part,
            y_part,
            single_tumor_prop=single_tumor_prop,
            threshold=threshold,
            random_state=random_state,
            config=config,
        )

    adjacency = neighbour_graph(coords, sample_ids)
    rng = np.random.default_rng(0 if random_state is None else int(random_state))
    chosen = (float(COUPLINGS[0]), components_at(adjacency, float(COUPLINGS[0]), rng))

    for coupling in COUPLINGS[1:]:
        labels = components_at(adjacency, float(coupling), rng)

        if np.bincount(labels, weights=counts).max() > target:
            break

        chosen = (float(coupling), labels)

    labels = _merge_small(chosen[1], adjacency, counts, target)
    _, labels = np.unique(labels, return_inverse=True)
    CHOSEN.append((chosen[0], target, int(labels.max()) + 1))

    return [np.flatnonzero(labels == c) for c in range(int(labels.max()) + 1)]


CHOSEN: list[tuple[float, float, int]] = []
"""`(J, target counts, clones)` per call, for the run's record."""
