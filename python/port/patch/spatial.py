"""`cnaster.spatial`, without the dense round trip and the repeated partition.

**Proposed for `cnaster`, written here.** #190. Two functions on the
preprocessing path allocate or recompute what their answers do not need:

*   `construct_multislice_lattice_adjacency` builds each slice's adjacency and
    pooling matrices **sparse**, calls `.toarray()` on both, block-diagonalizes
    the dense copies with `scipy.linalg.block_diag`, and converts the result
    back to CSR. The graph is a k-nearest-neighbour lattice with eight edges
    per spot, so the dense form is `n_spots^2` entries to carry `8 * n_spots`
    of them: at 5,000 spots that is 200 MB per matrix to hold 0.3 MB of graph.
*   `best_equal_partition` draws `n_trials` random grid partitions and keeps
    the one whose clone sizes vary least. It builds every trial's index lists
    to measure them, when the variance needs only the sizes -- `x_part *
    y_part` calls to `np.where` per trial, of which all but the winner's are
    discarded.

`port` cannot land either (`CLAUDE.md`, **Working against a repository you do
not own**), so both are written here with their referees beside them in
`tests/test_preprocessing_sweep.py`.

Both returns are **bitwise** what `cnaster` returns.
"""

from __future__ import annotations

from collections import namedtuple
from typing import Any

import numpy as np
import scipy.sparse as sp
from cnaster.config import start_time
from cnaster.logger import get_logger
from cnaster.spatial import construct_lattice_adjacency, rectangle_partition

logger = get_logger(__name__, start_time=start_time)

Adjacency = namedtuple("Adjacency", ["adjacency_mat", "smooth_mat"])
"""`cnaster`'s own return shape, declared here because it declares it inline."""


def _block_diagonal(blocks: list[Any]) -> Any:
    """The block diagonal of sparse blocks, as CSR, without densifying.

    `scipy.linalg.block_diag` takes dense arrays, so `cnaster` pays
    `sum(n_i)^2` entries to place `sum(nnz_i)` of them. The sparse form places
    the same entries by offsetting their indices.

    The dtype is taken from the blocks explicitly. `scipy.linalg.block_diag`
    promotes to the common type of its inputs and `scipy.sparse.block_diag`
    does not always agree with it, and the two returns are compared bitwise,
    which a silent promotion would break.
    """
    dtype = np.result_type(*[block.dtype for block in blocks])

    return sp.block_diag(blocks, format="csr", dtype=dtype)


def construct_multislice_lattice_adjacency(
    sample_ids: np.ndarray,
    sample_list: Any,
    coords: np.ndarray,
    across_slice_adjacency_mat: Any,
    maxspots_pooling: int,
    unit_xsquared: int = 9,
    unit_ysquared: int = 3,
) -> Any:
    """What `cnaster`'s returns, built sparse throughout.

    Same graph, same weights, same order: the per-slice matrices come from
    `cnaster`'s own `construct_lattice_adjacency`, and only how they are
    assembled differs.
    """
    logger.info("Solving for multi-slice adjacency (and spot-pooling) matrix.")

    adjacency_blocks, smooth_blocks = [], []

    for index, _ in enumerate(sample_list):
        this_coords = np.array(coords[np.where(sample_ids == index)[0], :])

        smooth_mat, adjacency_mat = construct_lattice_adjacency(
            this_coords,
            maxspots_pooling=maxspots_pooling,
            unit_xsquared=unit_xsquared,
            unit_ysquared=unit_ysquared,
        )

        adjacency_blocks.append(adjacency_mat)
        smooth_blocks.append(smooth_mat)

    adjacency_mat = _block_diagonal(adjacency_blocks)
    smooth_mat = _block_diagonal(smooth_blocks)

    if across_slice_adjacency_mat is not None:
        adjacency_mat += across_slice_adjacency_mat

    return Adjacency(adjacency_mat=adjacency_mat, smooth_mat=smooth_mat)


def _rectangle_counts(coords: np.ndarray) -> Any:
    """A summed-area table over the distinct coordinates, or `None`.

    Every trial counts the spots in each cell of an axis-aligned grid: the
    grid changes and the spots do not. Built once, the inclusive
    two-dimensional cumulative count answers any rectangle in four lookups, so
    a trial costs `x_part * y_part` reads rather than a pass over the spots --
    O(1) in the spot count where `cnaster` is O(n) twice over.

    `None` where the table would be larger than the data it summarizes. A
    slide's coordinates are a lattice, so the distinct values are about
    `sqrt(n_spots)` per axis and the table is about `n_spots`; scattered
    coordinates have as many distinct values as spots and the table would be
    `n_spots^2`, which is the case this declines rather than the case it is
    for.
    """
    (x_values, x_index), (y_values, y_index) = (
        np.unique(coords[:, axis], return_inverse=True) for axis in (0, 1)
    )

    if x_values.size * y_values.size > 4 * coords.shape[0] + 1024:
        return None

    grid = np.zeros((x_values.size + 1, y_values.size + 1), dtype=np.int64)
    np.add.at(grid, (x_index + 1, y_index + 1), 1)

    return x_values, y_values, grid.cumsum(axis=0).cumsum(axis=1)


def _trial_edges(
    range_coords: np.ndarray,
    axis: int,
    parts: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """One axis's cell boundaries for one trial, on the data's own scale.

    Drawn in `cnaster`'s order -- x before y, from one generator -- because
    the two draws share a stream, so swapping them would be a different
    partition at the same seed.
    """
    edges = np.sort(generator.uniform(0, 1, parts))
    edges[-1] = 1.01

    low, high = np.min(range_coords[:, axis]), np.max(range_coords[:, axis])

    return np.asarray(low + (high - low) * edges)


def partition_sizes(
    coords: np.ndarray,
    x_part: int,
    y_part: int,
    single_tumor_prop: np.ndarray | None,
    threshold: float,
    trial: int,
    table: Any = None,
) -> np.ndarray:
    """How many spots fall in each cell of one trial's grid, in clone order.

    `cnaster` gets these by building the index array of every cell and taking
    its length, which is `x_part * y_part` passes over the spots per trial.
    With a summed-area table it is four lookups per cell and none; without one
    it is a single `bincount` over the two digitizations.

    A spot whose digitization lands past the last cell belongs to no clone in
    `cnaster`'s double loop, so it is dropped here too rather than folded into
    the last cell. The partition's last edge is set beyond the data, so this
    cannot happen -- matching it costs nothing and relying on it would be a
    claim about a constant in someone else's code.
    """
    if single_tumor_prop is not None:
        range_coords = coords[np.where(single_tumor_prop >= threshold)[0]]
    else:
        range_coords = coords

    generator = np.random.default_rng(trial)
    edges = [
        _trial_edges(range_coords, axis, parts, generator)
        for axis, parts in ((0, x_part), (1, y_part))
    ]

    if table is not None:
        x_values, y_values, cumulative = table
        upper = [
            np.searchsorted(values, axis_edges, side="right")
            for values, axis_edges in zip((x_values, y_values), edges, strict=True)
        ]
        lower = [np.concatenate(([0], span[:-1])) for span in upper]

        counts = (
            cumulative[np.ix_(upper[0], upper[1])]
            - cumulative[np.ix_(lower[0], upper[1])]
            - cumulative[np.ix_(upper[0], lower[1])]
            + cumulative[np.ix_(lower[0], lower[1])]
        )

        return np.asarray(counts).reshape(-1)

    digits = [
        np.digitize(coords[:, axis], axis_edges, right=True)
        for axis, axis_edges in enumerate(edges)
    ]
    inside = (digits[0] < x_part) & (digits[1] < y_part)

    return np.bincount(
        digits[0][inside] * y_part + digits[1][inside], minlength=x_part * y_part
    )


def _all_trial_edges(
    range_coords: np.ndarray, x_part: int, y_part: int, n_trials: int
) -> tuple[np.ndarray, np.ndarray]:
    """Every trial's cell boundaries, as two `(n_trials, parts)` arrays.

    The draws stay in a Python loop because each trial seeds its own generator
    and `cnaster`'s partition at a seed is what is being reproduced. Nothing
    else does: with the boundaries in one array, the counting and the variance
    run once across all trials rather than once per trial, which is what the
    per-trial `numpy` call overhead was.
    """
    x_edges = np.empty((n_trials, x_part))
    y_edges = np.empty((n_trials, y_part))

    for trial in range(n_trials):
        generator = np.random.default_rng(trial)

        for edges, parts in ((x_edges, x_part), (y_edges, y_part)):
            drawn = np.sort(generator.uniform(0, 1, parts))
            drawn[-1] = 1.01
            edges[trial] = drawn

    scaled = []

    for axis, edges in enumerate((x_edges, y_edges)):
        low, high = np.min(range_coords[:, axis]), np.max(range_coords[:, axis])
        scaled.append(low + (high - low) * edges)

    return scaled[0], scaled[1]


def _all_trial_variances(
    x_part: int,
    y_part: int,
    range_coords: np.ndarray,
    n_trials: int,
    table: Any,
) -> np.ndarray:
    """Every trial's clone-size variance, in one pass over the trials.

    The counting is `n_trials * x_part * y_part` lookups into the summed-area
    table and no pass over the spots at all, so the cost stops depending on
    how many spots there are. The intermediate is the counts themselves --
    `n_trials * x_part * y_part` integers, 72 KB at a thousand trials on a
    3x3 grid.
    """
    x_edges, y_edges = _all_trial_edges(range_coords, x_part, y_part, n_trials)
    x_values, y_values, cumulative = table

    upper, lower = [], []

    for values, edges in ((x_values, x_edges), (y_values, y_edges)):
        span = np.searchsorted(values, edges, side="right")
        upper.append(span)
        lower.append(
            np.concatenate([np.zeros((n_trials, 1), dtype=int), span[:, :-1]], axis=1)
        )

    def corner(rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
        return np.asarray(cumulative[rows[:, :, None], columns[:, None, :]])

    counts = (
        corner(upper[0], upper[1])
        - corner(lower[0], upper[1])
        - corner(upper[0], lower[1])
        + corner(lower[0], lower[1])
    )

    return np.asarray(counts.reshape(n_trials, -1).var(axis=1))


def best_equal_partition(
    coords: np.ndarray,
    x_part: int,
    y_part: int,
    single_tumor_prop: np.ndarray | None = None,
    threshold: float = 0.5,
    n_trials: int = 10_000,
) -> tuple[Any, Any]:
    """`cnaster`'s partition, measured before it is built.

    The trial kept is the same one: the comparison is strict, so the earliest
    trial attaining the minimum variance wins, and the winner's index lists
    come from `cnaster`'s own `rectangle_partition` at that seed rather than
    from a second implementation of it.
    """
    table = _rectangle_counts(coords)

    if single_tumor_prop is not None:
        range_coords = coords[np.where(single_tumor_prop >= threshold)[0]]
    else:
        range_coords = coords

    if table is not None:
        variances = _all_trial_variances(x_part, y_part, range_coords, n_trials, table)
    else:
        variances = np.array(
            [
                np.var(
                    partition_sizes(
                        coords, x_part, y_part, single_tumor_prop, threshold, trial
                    )
                )
                for trial in range(n_trials)
            ]
        )

    best = int(np.argmin(variances))

    logger.info(
        f"Best partition variance after {n_trials} trials: {variances[best]:.2f} "
        f"at trial {best}."
    )

    index, assignment = rectangle_partition(
        coords,
        x_part,
        y_part,
        single_tumor_prop=single_tumor_prop,
        threshold=threshold,
        random_state=best,
    )

    return index, assignment


RECTANGLE_TRIES = 1_000
"""Block assignments tried before the block boundaries are redrawn.

`cnaster` draws the boundaries once and then retries only the assignment,
which cannot succeed when a block is too small: with `n_clones` a perfect
square every clone takes exactly one block, so a small block is a small
clone on every try (#304).
"""


def initialize_rectangular_clones(
    coords: np.ndarray, n_clones: int, random_state: int = 0
) -> tuple[list[np.ndarray], np.ndarray]:
    """`cnaster.spatial.initialize_rectangular_clones`, which terminates.

    **`cnaster`'s never returns on some inputs.** It dices the coordinates
    into `p x p` blocks at Dirichlet-drawn boundaries, then loops `while
    True` assigning blocks to clones at random until every clone holds more
    than 20 per cent of an equal share of spots. The boundaries are drawn
    once, before the loop. When one block holds fewer spots than that and
    `n_clones = p ** 2`, so that each clone takes exactly one block, no
    assignment passes and the loop spins forever. #298's normal clone makes
    a 12-row band on the dev instance that does exactly this.

    Here the same boundaries, the same random stream and the same test, with
    one change: after :data:`RECTANGLE_TRIES` failed assignments the
    boundaries are redrawn from the same stream. Wherever `cnaster` returns
    within that many tries this returns the same, bitwise; where it would
    not return, this does.
    """
    # NB the legacy global stream, deliberately: `cnaster` draws from it, and
    #    the same draws in the same order are what makes this bitwise.
    np.random.seed(random_state)  # noqa: NPY002

    p = int(np.ceil(np.sqrt(n_clones)))

    if n_clones <= 1:
        clone_id = np.zeros(len(coords), dtype=int)

        return [np.where(clone_id == i)[0] for i in range(n_clones)], clone_id

    def blocks() -> np.ndarray:
        digits = []

        for axis in (0, 1):
            share = np.random.dirichlet(np.ones(p) * 10)  # noqa: NPY002
            share[-1] += 1e-4

            low = np.percentile(coords[:, axis], 5)
            high = np.percentile(coords[:, axis], 95)

            boundary = low + (high - low) * np.cumsum(share)
            boundary[-1] = np.max(coords[:, axis]) + 1

            digits.append(np.digitize(coords[:, axis], boundary, right=True))

        block_id: np.ndarray = digits[0] * p + digits[1]

        return block_id

    block_id = blocks()
    tries = 0

    while True:
        if tries == RECTANGLE_TRIES:
            block_id = blocks()
            tries = 0

        tries += 1
        block_clone_map = np.random.randint(low=0, high=n_clones, size=p**2)  # noqa: NPY002

        while len(np.unique(block_clone_map)) < n_clones:
            counts = np.bincount(block_clone_map, minlength=n_clones)
            block_clone_map[np.where(block_clone_map == np.argmax(counts))[0][0]] = (
                np.where(counts == 0)[0][0]
            )

        clone_id = block_clone_map[block_id]
        initial_clone_index = [np.where(clone_id == i)[0] for i in range(n_clones)]

        if min(len(x) for x in initial_clone_index) > 0.2 * coords.shape[0] / n_clones:
            return initial_clone_index, clone_id
