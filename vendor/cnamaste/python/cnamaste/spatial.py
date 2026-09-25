from collections import namedtuple

import numpy as np
import scipy.linalg
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

# from cnamaste.annotation import get_clone_label_annotation
from cnamaste.config import start_time
from cnamaste.logger import get_logger
from typing import Any
import scipy.sparse as sp

logger = get_logger(__name__, start_time=start_time)


def log_sparse_matrix_stats(matrix, name):
    # NB see https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.getnnz.html
    num_neighbors = matrix.getnnz(axis=1)

    unique_vals, counts = np.unique(num_neighbors, return_counts=True)
    fractions = counts / len(num_neighbors)
    sort_idx = np.argsort(fractions)[::-1]

    fraction_strs = [
        f"{unique_vals[i]} neighbors\t{fractions[i]:.1%}" for i in sort_idx
    ]

    if matrix.nnz > 0:
        median_weight = np.median(matrix.data)
    else:
        median_weight = 0.0

    logger.info(
        f"{name} | Median edge weight: {median_weight:.2f} | "
        f"Neighbor fractions:\n{'\n'.join(fraction_strs)}"
    )


# TODO respect alignment.
def rectangle_partition(
    coords, x_part, y_part, single_tumor_prop=None, threshold=0.5, random_state=None
):
    if single_tumor_prop is not None:
        idx_tumor = np.where(single_tumor_prop >= threshold)[0]
        range_coords = coords[idx_tumor]
    else:
        range_coords = coords

    if random_state is not None:
        rng = np.random.default_rng(random_state)
        px = np.sort(rng.uniform(0, 1, x_part))
        px[-1] = 1.01
    else:
        px = np.linspace(0, 1, 1 + x_part)
        px[-1] += 0.01
        px = px[1:]

    # NB min. to max. x values of all spots (meeting tumor threshold).
    xrange = [np.min(range_coords[:, 0]), np.max(range_coords[:, 0])]

    # NB bin x values into positions and return an appropriate indexing.
    xdigit = np.digitize(
        coords[:, 0], xrange[0] + (xrange[1] - xrange[0]) * px, right=True
    )

    # NB same for y.
    if random_state is not None:
        py = np.sort(rng.uniform(0, 1, y_part))
        py[-1] = 1.01
    else:
        py = np.linspace(0, 1, y_part + 1)
        py[-1] += 0.01
        py = py[1:]

    yrange = [np.min(range_coords[:, 1]), np.max(range_coords[:, 1])]
    ydigit = np.digitize(
        coords[:, 1], yrange[0] + (yrange[1] - yrange[0]) * py, right=True
    )

    initial_clone_index = []
    clone_id = 0

    clone_assignment = np.full(coords.shape[0], -1)

    for xid in range(x_part):
        for yid in range(y_part):
            idx = np.where((xdigit == xid) & (ydigit == yid))[0]

            initial_clone_index.append(idx)
            clone_assignment[idx] = clone_id
            clone_id += 1

    # NB initial clones assigned according to grid partitioning given x_part, y_part; list of lists.
    return initial_clone_index, clone_assignment


Adjacency = namedtuple("Adjacency", ["adjacency_mat", "smooth_mat"])
"""`cnamaste`'s own return shape, declared here because it declares it inline."""


def _block_diagonal(blocks: list[Any]) -> Any:
    """The block diagonal of sparse blocks, as CSR, without densifying.

    `scipy.linalg.block_diag` takes dense arrays, so `cnamaste` pays
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
    """What `cnamaste`'s returns, built sparse throughout.

    Same graph, same weights, same order: the per-slice matrices come from
    `cnamaste`'s own `construct_lattice_adjacency`, and only how they are
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
    O(1) in the spot count where `cnamaste` is O(n) twice over.

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

    Drawn in `cnamaste`'s order -- x before y, from one generator -- because
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

    `cnamaste` gets these by building the index array of every cell and taking
    its length, which is `x_part * y_part` passes over the spots per trial.
    With a summed-area table it is four lookups per cell and none; without one
    it is a single `bincount` over the two digitizations.

    A spot whose digitization lands past the last cell belongs to no clone in
    `cnamaste`'s double loop, so it is dropped here too rather than folded into
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
    and `cnamaste`'s partition at a seed is what is being reproduced. Nothing
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
    """`cnamaste`'s partition, measured before it is built.

    The trial kept is the same one: the comparison is strict, so the earliest
    trial attaining the minimum variance wins, and the winner's index lists
    come from `cnamaste`'s own `rectangle_partition` at that seed rather than
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

`cnamaste` draws the boundaries once and then retries only the assignment,
which cannot succeed when a block is too small: with `n_clones` a perfect
square every clone takes exactly one block, so a small block is a small
clone on every try (#304).
"""


def initialize_rectangular_clones(
    coords: np.ndarray, n_clones: int, random_state: int = 0
) -> tuple[list[np.ndarray], np.ndarray]:
    """`cnamaste.spatial.initialize_rectangular_clones`, which terminates.

    **`cnamaste`'s never returns on some inputs.** It dices the coordinates
    into `p x p` blocks at Dirichlet-drawn boundaries, then loops `while
    True` assigning blocks to clones at random until every clone holds more
    than 20 per cent of an equal share of spots. The boundaries are drawn
    once, before the loop. When one block holds fewer spots than that and
    `n_clones = p ** 2`, so that each clone takes exactly one block, no
    assignment passes and the loop spins forever. #298's normal clone makes
    a 12-row band on the dev instance that does exactly this.

    Here the same boundaries, the same random stream and the same test, with
    one change: after :data:`RECTANGLE_TRIES` failed assignments the
    boundaries are redrawn from the same stream. Wherever `cnamaste` returns
    within that many tries this returns the same, bitwise; where it would
    not return, this does.
    """
    # NB the legacy global stream, deliberately: `cnamaste` draws from it, and
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


def initialize_clones(
    coords,
    sample_ids,
    x_part,
    y_part,
    single_tumor_prop=None,
    threshold=None,
    random_state=None,
    config=None,
):
    """
    if config is not None:
        # NB assumes the known clone labels for phasing.
        if config.annotation.clone_label is not None:
            clone_annotation, _ = get_clone_label_annotation(config)

            # NB assumes the known clone labels for phasing.
            return clone_annotation
    """
    logger.info(
        f"Initializing clones given fixed grid partitions and max. sample_id={np.max(sample_ids)}"
    )

    initial_clone_index = []

    # NB for all slices.
    for s in range(1 + np.max(sample_ids)):
        logger.debug(f"Solving for sample_id={s}")

        # NB sample_ids idx for all spots in this slice.
        index = np.where(sample_ids == s)[0]

        if len(index) == 0:
            logger.error(f"Expected at least one spot in slice {s}.")
            raise RuntimeError()

        # NB tumor_proportion for each spot in this slice.
        this_tumor_prop = (
            single_tumor_prop[index] if single_tumor_prop is not None else None
        )

        tmp_clone_index, _ = rectangle_partition(
            coords[index, :],
            x_part,
            y_part,
            this_tumor_prop,
            threshold=threshold,
            random_state=random_state,
        )

        for x in tmp_clone_index:
            initial_clone_index.append(index[x])

    logger.info(
        f"Initialized {len(initial_clone_index)} clones given x_part,y_part={x_part},{y_part}."
    )

    return initial_clone_index


# TODO!! spatially contigous clones?


def construct_lattice_adjacency(
    coords,
    maxspots_pooling=7,
    unit_xsquared=9,
    unit_ysquared=3,
    coordination_num=8,
):
    logger.info(
        f"Assigning lattice adjacency matrix with coordination_num={coordination_num}, "
        f"assuming unit_xsquared, unit_ysquared={unit_xsquared},{unit_ysquared}."
    )

    n_spots = coords.shape[0]

    scaled_coords = coords.copy().astype(float)
    scaled_coords[:, 0] *= np.sqrt(unit_xsquared)
    scaled_coords[:, 1] *= np.sqrt(unit_ysquared)

    logger.info(f"Building kd-tree for efficient nearest neighbor search")

    tree = cKDTree(scaled_coords)

    # NB query (k+1) nearest neighbors, as includes self.
    _, indices = tree.query(scaled_coords, k=1 + coordination_num)

    indices = indices[:, 1:]

    logger.info(f"Constructed nearest neighbor indices via KD-tree")

    logger.warning(f"Assuming identity smooth mat.")

    # NB smooth matrix: identity (each spot pools only itself)
    smooth_mat = scipy.sparse.identity(n_spots, dtype=np.int8, format="csr")

    logger.info(f"Constructing adjacency matrix.")

    # nearest_indices = np.argpartition(pairwise_squared_dist, coordination_num, axis=1)[:, :coordination_num]

    rows = np.repeat(np.arange(n_spots), coordination_num)
    cols = indices.flatten()

    # NB uniform (unit) edge weight, we will apply global spatial_weight in hmrf.
    data = np.ones(len(rows), dtype=np.float64)

    adjacency_mat = csr_matrix((data, (rows, cols)), shape=(n_spots, n_spots))

    log_sparse_matrix_stats(smooth_mat, "smooth_mat")
    log_sparse_matrix_stats(adjacency_mat, "adjacency_mat")

    return smooth_mat, adjacency_mat


# @cacher("adjacency.hdf5")


def initialize_rdr_clone_refininement(
    merged_baf_assignment, coords, single_total_bb_RD, n_obs, config
):
    n_spots = len(merged_baf_assignment)
    n_baf_clones = len(np.unique(merged_baf_assignment))

    # TODO HACK assert (0, ..., N-1) for clone labels.

    splits_per_baf, total_rdr_clones = [], 0

    for bafc in range(n_baf_clones):
        idx_spots = np.where(merged_baf_assignment == bafc)[0]

        sufficient_snp_umi = np.sum(single_total_bb_RD[:, idx_spots]) >= 20 * n_obs

        n_splits = config.hmrf.n_clones_rdr if sufficient_snp_umi else 1
        splits_per_baf.append(n_splits)
        total_rdr_clones += n_splits

    logger.info(
        f"Global hmrf will optimize {total_rdr_clones} rdr-refined clones given {n_baf_clones} baf-identified clones."
    )

    global_initial_assignment = np.zeros(n_spots, dtype=np.int32)

    # NB one-hot allowed (rdr-refined) clones.
    allowed_clones = np.zeros((n_spots, total_rdr_clones), dtype=bool)
    global_clone_offset = 0

    for bafc, n_splits in enumerate(splits_per_baf):
        idx_spots = np.where(merged_baf_assignment == bafc)[0]

        # TODO HACK define initializer; define seed.
        initial_clone_index, _ = initialize_rectangular_clones(
            coords[idx_spots],
            n_splits,
            random_state=config.hmm.gmm_random_state,
        )

        for local_c, local_idx in enumerate(initial_clone_index):
            global_c = global_clone_offset + local_c

            global_spot_ids = idx_spots[local_idx]
            global_initial_assignment[global_spot_ids] = global_c

        allowed_clones[
            idx_spots, global_clone_offset : global_clone_offset + n_splits
        ] = True
        global_clone_offset += n_splits

    return global_initial_assignment, allowed_clones, total_rdr_clones


def renormalize_adjacency_mat(adjacency_mat):
    num_edges, total_edge_weight = [], []

    for row in list(adjacency_mat.tolil()):
        num_edges.append(row.nnz)
        total_edge_weight.append(row.sum())

    num_edges = np.array(num_edges)
    total_edge_weight = np.array(total_edge_weight)

    us, cnts = np.unique(num_edges, return_counts=True)
    med_num_edges = np.median(num_edges)

    logger.info(
        f"Found node degree distribution with median {med_num_edges}:\n{us}\n{cnts}"
    )

    us, cnts = np.unique(total_edge_weight, return_counts=True)
    med_edge_weight = np.median(total_edge_weight)

    logger.info(
        f"Found edge weight distribution with median {med_edge_weight}:\n{us}\n{cnts}"
    )

    adj = adjacency_mat.tocsr().astype(np.float64)
    row_sums = np.asarray(adj.sum(axis=1)).ravel()  # shape (n_rows,)

    indptr = adj.indptr
    data = adj.data
    n_rows = adj.shape[0]

    for i in range(n_rows):
        start, end = indptr[i], indptr[i + 1]

        if start == end:
            # empty row
            continue

        rs = row_sums[i]

        if rs == 0.0:
            continue

        scale = med_edge_weight / rs
        data[start:end] *= scale

    adj.eliminate_zeros()

    logger.info(f"Normalized adjacency_mat:\n{adj}")

    return adj
