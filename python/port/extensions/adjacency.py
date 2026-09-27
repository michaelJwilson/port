"""The fixtures' lattices as adjacencies: symmetric, no self loops, boundaries reinforced (#417).

`cnaster` builds its spatial graph as eight nearest neighbours per spot. That
is a directed graph -- spot `i` naming `j` does not make `j` name `i` -- and
its ICM sums a spot's own row, so the Potts term is not a symmetric coupling.
On the dev instance 344 of 12,800 entries have no transpose, and `--sal`'s
conversion kept the upper triangle and dropped them silently.

**For now**, the lattices the fixtures plant are built directly:

| lattice | coordinates | neighbours | `z` |
| --- | --- | --- | --- |
| square | integer `(row, col)` | `(0, ±1)`, `(±1, 0)`, `(±1, ±1)` | 8 |
| triangular | Visium array: `col` steps by 2, odd rows offset | `(0, ±2)`, `(±1, ±1)` | 6 |

The square lattice keeps its diagonals because `cnaster`'s eight nearest
neighbours on a square grid are exactly these eight: the interior graph is
`cnaster`'s, and what changes is symmetry and the boundary. Dropping them
(`cnaster.adjacency.lattice_map`'s `square: 4`) halves the coupling at a
fixed `spatial_weight`; measured on the dev instance, clone ARI 0.3851.

**Boundary reinforcement.** A spot on the boundary has fewer neighbours, so
the smoothing it feels is weaker. Each edge carries
`w_ij = (z / d_i + z / d_j) / 2`, with `d` the spot's neighbour count: `1`
between two interior spots, above `1` wherever an end is on the boundary,
and symmetric by construction. Exact balancing -- every weighted degree `z`
-- converges for the triangular lattice and fails for the four-neighbour
square one, which is bipartite; the closed form is used for both kinds.

Coordinates that are neither lattice are refused rather than approximated;
the general construction is #417's future work.

`port.patch.spatial.lattice_multislice_adjacency` is the swap that installs
these over `cnaster`'s builder; what is here replaces nothing, so it lives
under `extensions/` (#274).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import scipy.sparse as sp

__all__ = [
    "COORDINATION",
    "RECIPROCATED",
    "AdjacencyError",
    "adjacency_setting",
    "knn_adjacency",
    "lattice_adjacency",
    "lattice_kind",
    "neighbourhood_for",
    "reinforced_weights",
    "validate_adjacency",
]

Lattice = Literal["square", "triangular"]
"""The geometry the coordinates are on, read off them by `lattice_kind`."""

Neighbourhood = Literal["square", "moore", "triangular"]
"""Which spots count as neighbours: a square grid's four or eight, or a hexagon's six."""

COORDINATION: dict[str, int] = {"square": 4, "moore": 8, "triangular": 6}
"""Interior neighbour count `z` per neighbourhood."""

OFFSETS: dict[str, tuple[tuple[int, int], ...]] = {
    "square": ((0, 1), (0, -1), (1, 0), (-1, 0)),
    "moore": ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)),
    "triangular": ((0, 2), (0, -2), (1, 1), (1, -1), (-1, 1), (-1, -1)),
}

Construction = Literal["knn", "lattice"]
"""`knn`: each spot's `z` nearest, `cnaster`'s directed construction with `k`
set by the neighbourhood rather than fixed at eight -- a boundary spot reaches
farther to keep `k`, which is its reinforcement. `lattice`: the
neighbourhood's offsets exactly, symmetric, with boundary edges reinforced."""

ENVIRONMENT = {"construction": "PORT_ADJACENCY", "square": "PORT_SQUARE_NEIGHBOURHOOD"}
DEFAULTS: dict[str, str] = {"construction": "knn", "square": "moore"}
CHOICES: dict[str, tuple[str, ...]] = {
    "construction": ("knn", "lattice"),
    "square": ("moore", "square"),
}


def adjacency_setting(name: str) -> str:
    """The construction or the square grid's neighbourhood, refusing a typo."""
    import os

    value = os.environ.get(ENVIRONMENT[name], DEFAULTS[name])

    if value not in CHOICES[name]:
        msg = f"{ENVIRONMENT[name]} is {value!r}, not one of {CHOICES[name]}"
        raise AdjacencyError(msg)

    return value


def neighbourhood_for(kind: Lattice) -> Neighbourhood:
    """A triangular layout's six, or the square grid's setting (Moore by default)."""
    if kind == "triangular":
        return "triangular"

    return "moore" if adjacency_setting("square") == "moore" else "square"


AXIS: tuple[tuple[int, int], ...] = ((0, 1), (0, -1), (1, 0), (-1, 0))
"""What only a square grid has: a triangular layout's diagonals are its own too."""

RECIPROCATED = 0.6
"""Of a kNN graph's edges, the share whose transpose is also an edge must be
at least this: asymmetry belongs at the boundary, where a spot reaches
farther to keep `k`. Counted over edges rather than spots because a small
slice is mostly boundary: per spot the fixtures' 6 x 5 lattices realize
0.40, per edge 0.85. Realized per edge, Moore: 0.987 at 40 x 40, 0.942 at
12 x 10, 0.781 at 4 x 4; the least over every fixture shape and
neighbourhood is 0.775 (4 x 5, square)."""

TOLERANCE = 1e-12
"""On a reinforced weight: the rule is exact arithmetic on small integers."""


class AdjacencyError(ValueError):
    """An adjacency the HMRF must not be given."""


def _integer_coords(coords: np.ndarray) -> np.ndarray:
    values = np.asarray(coords, dtype=np.float64)

    if values.ndim != 2 or values.shape[1] != 2:
        msg = f"coordinates must be (n_spots, 2), got {values.shape}"
        raise AdjacencyError(msg)

    rounded = np.rint(values)

    if not np.array_equal(rounded, values):
        msg = "coordinates are not integer array positions, so no lattice is defined"
        raise AdjacencyError(msg)

    return rounded.astype(np.int64)


def _neighbours(
    grid: np.ndarray, offsets: tuple[tuple[int, int], ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Every `(i, j)` whose positions differ by one of `offsets`, both directions."""
    low = grid.min(axis=0)
    shifted = grid - low + 2
    width = int(shifted[:, 1].max()) + 3
    keys = shifted[:, 0] * width + shifted[:, 1]
    order = np.argsort(keys)
    sorted_keys = keys[order]

    if np.unique(sorted_keys).size != sorted_keys.size:
        msg = "two spots share a position"
        raise AdjacencyError(msg)

    rows, cols = [], []

    for d_row, d_col in offsets:
        target = (shifted[:, 0] + d_row) * width + (shifted[:, 1] + d_col)
        slot = np.clip(np.searchsorted(sorted_keys, target), 0, keys.size - 1)
        found = sorted_keys[slot] == target
        rows.append(np.flatnonzero(found))
        cols.append(order[slot[found]])

    return np.concatenate(rows), np.concatenate(cols)


def lattice_kind(coords: np.ndarray) -> Lattice:
    """`square` if any spot has an axis neighbour at unit distance, else `triangular`.

    A triangular (Visium) layout has no pair at `(0, ±1)` or `(±1, 0)`: columns
    step by two within a row and change parity between rows. A square grid
    has both kinds of offset -- its diagonals are triangular ones -- so the
    axis test is the one that decides.
    """
    grid = _integer_coords(coords)

    if grid.shape[0] < 2:
        msg = "a lattice needs at least two spots"
        raise AdjacencyError(msg)

    if _neighbours(grid, AXIS)[0].size:
        return "square"

    if _neighbours(grid, OFFSETS["triangular"])[0].size:
        return "triangular"

    msg = "no two spots are lattice neighbours: neither square nor triangular"
    raise AdjacencyError(msg)


def reinforced_weights(
    rows: np.ndarray, cols: np.ndarray, degree: np.ndarray, coordination: int
) -> np.ndarray:
    """`(z / d_i + z / d_j) / 2` per edge: 1 in the interior, above 1 at a boundary."""
    inverse = coordination / degree.astype(np.float64)
    weights: np.ndarray = 0.5 * (inverse[rows] + inverse[cols])
    return weights


def _reinforced(
    rows: np.ndarray, cols: np.ndarray, n_spots: int, neighbourhood: str
) -> Any:
    degree = np.bincount(rows, minlength=n_spots)

    if np.any(degree == 0):
        spot = int(np.argmax(degree == 0))
        msg = f"spot {spot} has no {neighbourhood} neighbour"
        raise AdjacencyError(msg)

    weights = reinforced_weights(rows, cols, degree, COORDINATION[neighbourhood])
    adjacency = sp.csr_matrix((weights, (rows, cols)), shape=(n_spots, n_spots))
    adjacency.sort_indices()

    return adjacency


def knn_adjacency(coords: np.ndarray, neighbourhood: Neighbourhood) -> Any:
    """Each spot's `z` nearest as unit edges, directed, no self loop.

    `cnaster`'s construction (`construct_lattice_adjacency`) with `k` the
    neighbourhood's coordination instead of a fixed eight, measured in the
    lattice's own embedding: unit spacing on a square grid, a unit hexagon
    for Visium array positions. Every row carries exactly `k` edges, so a
    boundary spot is not under-coupled: it reaches farther instead. On a
    square grid with `k = 8` this is `cnaster`'s graph entry for entry.
    """
    from scipy.spatial import cKDTree

    grid = _integer_coords(coords).astype(np.float64)

    if neighbourhood == "triangular":
        grid = np.stack([grid[:, 0] * np.sqrt(3.0) / 2.0, grid[:, 1] / 2.0], axis=1)

    n_spots = grid.shape[0]
    k = COORDINATION[neighbourhood]

    if n_spots <= k:
        msg = f"{n_spots} spots cannot each have {k} neighbours"
        raise AdjacencyError(msg)

    _, nearest = cKDTree(grid).query(grid, k=k + 1)
    rows = np.repeat(np.arange(n_spots), k)
    # NB the query's first column is the spot itself at distance zero, as
    #    `cnaster` drops it; positions are distinct, so it is never a tie.
    cols = np.asarray(nearest)[:, 1:].reshape(-1)
    adjacency = sp.csr_matrix(
        (np.ones(rows.size), (rows, cols)), shape=(n_spots, n_spots)
    )
    adjacency.sort_indices()

    return adjacency


def lattice_adjacency(
    coords: np.ndarray, neighbourhood: Neighbourhood | None = None
) -> Any:
    """One slice's reinforced lattice adjacency, as CSR `float64`."""
    grid = _integer_coords(coords)
    neighbourhood = (
        neighbourhood_for(lattice_kind(grid))
        if neighbourhood is None
        else neighbourhood
    )
    rows, cols = _neighbours(grid, OFFSETS[neighbourhood])
    n_spots = grid.shape[0]

    return _reinforced(rows, cols, n_spots, neighbourhood)


def validate_adjacency(
    adjacency: Any,
    coordination: int | None = None,
    *,
    construction: Construction = "lattice",
) -> None:
    """Raise unless the graph is one the HMRF may be given.

    Both constructions: square, and no self loop. `knn` may be asymmetric --
    it is directed by construction -- but only at the edges of the lattice:
    at least `RECIPROCATED` of its edges have their transpose, and every
    row carries exactly `z` edges of unit weight, so `spatial_weight` is the
    coupling (#420) and no spot is under-coupled. `lattice` must be symmetric, and each weight the
    reinforced `(z / d_i + z / d_j) / 2` for the neighbour counts the pattern
    gives, to `TOLERANCE`. `z` is the largest row count when not given.
    """
    matrix = sp.csr_matrix(adjacency, dtype=np.float64)
    matrix.eliminate_zeros()

    if matrix.shape[0] != matrix.shape[1]:
        msg = f"adjacency is {matrix.shape}, not square"
        raise AdjacencyError(msg)

    if np.any(matrix.diagonal() != 0.0):
        spot = int(np.argmax(matrix.diagonal() != 0.0))
        msg = f"adjacency has a self loop at spot {spot}"
        raise AdjacencyError(msg)

    coo = matrix.tocoo()
    degree = np.bincount(coo.row, minlength=matrix.shape[0])
    z = int(degree.max()) if coordination is None else coordination

    if construction == "knn":
        if np.any(degree != z):
            spot = int(np.argmax(degree != z))
            msg = f"spot {spot} has {degree[spot]} neighbours, not the k = {z}"
            raise AdjacencyError(msg)

        if np.any(coo.data != 1.0):
            msg = "a kNN adjacency carries unit weights, so spatial_weight is J"
            raise AdjacencyError(msg)

        symmetric = matrix.multiply(matrix.T).nnz / max(matrix.nnz, 1)

        if symmetric < RECIPROCATED:
            msg = (
                f"only {symmetric:.3f} of edges are reciprocated; a kNN lattice "
                f"is symmetric away from its boundary, at least {RECIPROCATED}"
            )
            raise AdjacencyError(msg)

        return

    asymmetric = abs(matrix - matrix.T)
    asymmetric.eliminate_zeros()

    if asymmetric.nnz:
        msg = (
            f"adjacency is not symmetric: {asymmetric.nnz} of {matrix.nnz} entries "
            "differ from their transpose"
        )
        raise AdjacencyError(msg)

    expected = reinforced_weights(coo.row, coo.col, np.maximum(degree, 1), z)
    error = np.abs(coo.data - expected)

    if coo.nnz and error.max() > TOLERANCE:
        worst = int(np.argmax(error))
        msg = (
            f"edge ({coo.row[worst]}, {coo.col[worst]}) weighs {coo.data[worst]:.6g}, "
            f"not the reinforced {expected[worst]:.6g} (z = {z})"
        )
        raise AdjacencyError(msg)
