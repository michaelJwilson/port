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

**`lattice_multislice_adjacency` is not bitwise by contract** (#417): it is
the swap the run installs over `construct_multislice_lattice_adjacency`,
builds each slice from `port.extensions.adjacency` -- `knn` by default, which
on a square grid is `cnaster`'s graph entry for entry, or `lattice` -- and
refuses any graph `validate_adjacency` rejects.
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

    **Not installed** (T- #617): `SWAPS` binds `lattice_multislice_adjacency`
    over this name (#417). It is kept as the bitwise referee of the sparse
    assembly, `_block_diagonal`, which the installed row shares
    (`tests/test_preprocessing_spatial.py`).
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


RECTANGLE_REDRAWS = 10
"""Boundary redraws, after an infeasible draw, before the partition is banded.

A draw is infeasible when no assignment of its blocks to clones passes
`cnaster`'s test (:func:`admits_assignment`), and `cnaster`'s loop then never
returns (T- #692). A redraw cannot help when the coordinates themselves leave
a block empty on every draw: a one-row strip puts every spot in one band of
the other axis, so two of four blocks are always empty (#248).
"""


class RectangularClones(tuple[list[np.ndarray], np.ndarray]):
    """`cnaster`'s `(initial_clone_index, clone_id)`, with how the search ended.

    A two-tuple, so every call site that unpacks `cnaster`'s return unpacks
    this one; :attr:`termination` is `snakes_and_ladders`' `Termination`
    (T- #692), :attr:`redraws` the boundary draws discarded as infeasible.
    """

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

    `cnaster`'s loop draws block-to-clone maps from `randint` and repairs any
    that leave a clone empty. Every surjective map is a `randint` draw with
    probability `n_clones ** -n_blocks > 0`, and the repair returns only
    surjective maps, so the loop reaches exactly the surjections. With
    `floor >= 0` a passing map is surjective, so the loop returns, with
    probability one, exactly when this is `True`.

    A depth-first search, largest block first, placing each block on a clone
    and pruning on the spots and blocks still needed. A block is tried on one
    clone per distinct current total, since clones of equal total are
    interchangeable. `n_blocks` is `ceil(sqrt(n_clones)) ** 2`.
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
    """`n_clones` equal-count bands along the axis with the most distinct values.

    Spots are ordered along that axis, ties by the other axis then by index,
    so the result is deterministic. Every band holds `n // n_clones` or one
    more spots, which passes `cnaster`'s 20 per cent test whenever
    `n >= n_clones`.
    """
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

    **Contract.** `cnaster`'s signature, defaults and return: the spots split
    into `n_clones` clones by `p x p` rectangular blocks, `p =
    ceil(sqrt(n_clones))`, at Dirichlet-drawn boundaries, every clone holding
    more than `0.2 * n_spots / n_clones` spots. The return is a two-tuple
    that also carries a `Termination`.

    **Departure (T- #692, #304, #248).** `cnaster` draws the boundaries once
    and loops `while True` over block-to-clone assignments. When the blocks
    admit no passing assignment it never returns: on dev (`07b82e92`) BAF
    clone 2's 297 spots fall in blocks of [194, 3, 77, 23] against a floor of
    14.85 spots at four clones. Here each draw is first tested by
    :func:`admits_assignment`, which draws nothing:

    *   admitted, `cnaster`'s own loop runs, uncapped, on the same stream.
        `cnaster` returns exactly on these draws, so wherever it returns this
        is its result, bitwise; `Stop.CONVERGED`, `iterations=1`.
    *   refused, the boundaries are redrawn from the same stream, up to
        :data:`RECTANGLE_REDRAWS` times; an admitted redraw is
        `Stop.CONVERGED` after that many draws plus one.
    *   every draw refused, the partition is :func:`_banded`'s,
        `Stop.INFEASIBLE` after `RECTANGLE_REDRAWS + 1` draws.

    `iterations` counts boundary draws. Where `cnaster` returns this result
    equals it; where it does not, `cnaster` has no result to compare.
    """
    # NB the legacy global stream, deliberately: `cnaster` draws from it, and
    #    the same draws in the same order are what makes this bitwise.
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
    """`construct_multislice_lattice_adjacency`'s signature, on the lattice graph.

    Per slice, in `sample_list` order as `cnaster` assembles them, then block
    diagonal. The pooling matrix is `cnaster`'s identity. The result is
    validated before it is returned, so the HMRF never receives a graph that
    fails `validate_adjacency`.
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
