"""Replaces `cnaster.spatial`'s adjacency, partition and rectangular-clone functions (#190).

`construct_multislice_lattice_adjacency` and `best_equal_partition` are built
sparse and counted without index lists, bitwise `cnaster`'s.
`lattice_multislice_adjacency` (#417) and `initialize_rectangular_clones`
(T- #692) are stated departures.
"""

from __future__ import annotations

from collections import namedtuple
from typing import Any

import numpy as np
import scipy.sparse as sp
from cnaster.config import start_time
from cnaster.logger import get_logger
from cnaster.spatial import construct_lattice_adjacency, rectangle_partition
from sal.opt.termination import Stop, Termination

logger = get_logger(__name__, start_time=start_time)

Adjacency = namedtuple("Adjacency", ["adjacency_mat", "smooth_mat"])
"""`cnaster`'s return shape, which it declares inline."""


def _block_diagonal(blocks: list[Any]) -> Any:
    """The block diagonal of sparse blocks as CSR, dtype taken explicitly from the blocks."""
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
    """`cnaster`'s `construct_multislice_lattice_adjacency`, built sparse; bitwise.

    Not installed (T- #617): kept as the referee of `_block_diagonal`.
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
    """A summed-area table over the distinct coordinates, or `None` where it would exceed ~`n_spots`."""
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
    """One axis's cell boundaries for one trial; drawn x before y as `cnaster` does."""
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
    """Spots in each cell of one trial's grid, in clone order; spots past the last cell dropped."""
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
    """Every trial's cell boundaries, as two `(n_trials, parts)` arrays, one seed per trial."""
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
    """Every trial's clone-size variance from the summed-area table."""
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
    """`cnaster`'s partition: the earliest minimum-variance trial, built by `rectangle_partition`."""
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


RECTANGLE_REDRAWS = 10
"""Boundary redraws after an infeasible draw before the partition is banded (T- #692, #248)."""


class RectangularClones(tuple[list[np.ndarray], np.ndarray]):
    """`cnaster`'s `(initial_clone_index, clone_id)` two-tuple, with `termination` and `redraws` (T- #692)."""

    termination: Termination
    redraws: int

    def __new__(
        cls,
        initial_clone_index: list[np.ndarray],
        clone_id: np.ndarray,
        termination: Termination,
        redraws: int,
    ) -> RectangularClones:
        result = super().__new__(cls, (initial_clone_index, clone_id))
        result.termination = termination
        result.redraws = redraws

        return result


def admits_assignment(block_sizes: np.ndarray, n_clones: int, floor: float) -> bool:
    """Whether some assignment of blocks to clones gives every clone `> floor` spots.

    Exactly when `cnaster`'s loop returns. Depth-first search, largest block first.
    """
    sizes = sorted((int(size) for size in block_sizes), reverse=True)
    need = int(np.floor(floor)) + 1
    remaining = [*np.cumsum(sizes[::-1])[::-1].tolist(), 0]
    totals = [0] * n_clones

    def place(block: int) -> bool:
        deficits = [need - total for total in totals if total < need]
        if not deficits:
            return True
        if block == len(sizes) or len(deficits) > len(sizes) - block:
            return False
        if sum(deficits) > remaining[block]:
            return False

        tried = set()
        for clone in range(n_clones):
            if totals[clone] in tried:
                continue
            tried.add(totals[clone])
            totals[clone] += sizes[block]
            if place(block + 1):
                return True
            totals[clone] -= sizes[block]

        return False

    return place(0)


def _banded(coords: np.ndarray, n_clones: int) -> tuple[list[np.ndarray], np.ndarray]:
    """`n_clones` equal-count bands along the axis with the most distinct values, deterministic."""
    n_spots = len(coords)
    axis = int(np.argmax([np.unique(coords[:, a]).size for a in (0, 1)]))
    order = np.lexsort((np.arange(n_spots), coords[:, 1 - axis], coords[:, axis]))
    clone_id = np.empty(n_spots, dtype=int)
    clone_id[order] = np.arange(n_spots) * n_clones // n_spots

    return [np.where(clone_id == i)[0] for i in range(n_clones)], clone_id


def initialize_rectangular_clones(
    coords: np.ndarray, n_clones: int, random_state: int = 0
) -> RectangularClones:
    """`cnaster.spatial.initialize_rectangular_clones`, which terminates.

    Departure (T- #692, #304, #248): a draw that admits no passing assignment
    (:func:`admits_assignment`) is redrawn up to :data:`RECTANGLE_REDRAWS` times,
    then banded (`Stop.INFEASIBLE`). Where `cnaster` returns, the result is
    bitwise. `termination.iterations` counts boundary draws.
    """
    # NB the legacy global stream: `cnaster` draws from it, so this is bitwise.
    np.random.seed(random_state)  # noqa: NPY002

    p = int(np.ceil(np.sqrt(n_clones)))

    if n_clones <= 1:
        clone_id = np.zeros(len(coords), dtype=int)

        return RectangularClones(
            [np.where(clone_id == i)[0] for i in range(n_clones)],
            clone_id,
            Termination.after(0, converged=True),
            0,
        )

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

    floor = 0.2 * coords.shape[0] / n_clones

    draws = 0

    while draws <= RECTANGLE_REDRAWS:
        block_id = blocks()
        sizes = np.bincount(block_id, minlength=p**2)
        draws += 1

        if admits_assignment(sizes, n_clones, floor):
            break

        logger.info(
            f"Rectangular blocks {sizes.tolist()} admit no {n_clones} clones above "
            f"{floor:.2f} spots; redrawing (T- #692)."
        )
    else:
        termination = Termination(
            converged=False, iterations=draws, reason=Stop.INFEASIBLE
        )
        logger.info(
            f"Rectangular clone initialization {termination.reason}: "
            f"{n_clones} equal-count bands (T- #692)."
        )

        return RectangularClones(*_banded(coords, n_clones), termination, draws)

    while True:
        block_clone_map = np.random.randint(low=0, high=n_clones, size=p**2)  # noqa: NPY002

        while len(np.unique(block_clone_map)) < n_clones:
            counts = np.bincount(block_clone_map, minlength=n_clones)
            block_clone_map[np.where(block_clone_map == np.argmax(counts))[0][0]] = (
                np.where(counts == 0)[0][0]
            )

        clone_id = block_clone_map[block_id]
        initial_clone_index = [np.where(clone_id == i)[0] for i in range(n_clones)]

        if min(len(x) for x in initial_clone_index) > floor:
            return RectangularClones(
                initial_clone_index,
                clone_id,
                Termination.after(draws, converged=True),
                draws - 1,
            )


def lattice_multislice_adjacency(
    sample_ids: np.ndarray,
    sample_list: Any,
    coords: np.ndarray,
    across_slice_adjacency_mat: Any,
    maxspots_pooling: int,
    unit_xsquared: int = 9,  # noqa: ARG001 -- a lattice has no metric to scale
    unit_ysquared: int = 3,  # noqa: ARG001 -- a lattice has no metric to scale
) -> Adjacency:
    """`construct_multislice_lattice_adjacency`'s signature on the lattice graph (#417).

    Identity pooling; raises `AdjacencyError` if `validate_adjacency` rejects the graph.
    """
    from port.extensions.adjacency import (
        COORDINATION,
        AdjacencyError,
        adjacency_setting,
        knn_adjacency,
        lattice_adjacency,
        lattice_kind,
        neighbourhood_for,
        validate_adjacency,
    )

    if maxspots_pooling != 1:
        msg = f"maxspots_pooling={maxspots_pooling}; the lattice path pools nothing"
        raise AdjacencyError(msg)

    construction = adjacency_setting("construction")
    build = knn_adjacency if construction == "knn" else lattice_adjacency
    blocks, kinds = [], []

    for index, _ in enumerate(sample_list):
        this_coords = np.asarray(coords[np.flatnonzero(sample_ids == index), :])
        kind = neighbourhood_for(lattice_kind(this_coords))
        kinds.append(kind)
        blocks.append(build(this_coords, kind))

    if len(set(kinds)) != 1:
        msg = f"slices are on different lattices: {kinds}"
        raise AdjacencyError(msg)

    adjacency_mat = _block_diagonal(blocks)
    smooth_mat = sp.identity(adjacency_mat.shape[0], dtype=np.int8, format="csr")

    if across_slice_adjacency_mat is not None:
        adjacency_mat = adjacency_mat + across_slice_adjacency_mat

    validate_adjacency(
        adjacency_mat,
        COORDINATION[kinds[0]],
        construction=construction,  # type: ignore[arg-type]
    )
    logger.info(
        f"{construction} {kinds[0]} adjacency: {adjacency_mat.nnz} entries over "
        f"{adjacency_mat.shape[0]} spots, validated."
    )

    return Adjacency(adjacency_mat=adjacency_mat, smooth_mat=smooth_mat)
