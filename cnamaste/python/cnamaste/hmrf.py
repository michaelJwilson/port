import copy
import time

import numpy as np
import scipy.special
from numba import njit, prange
from sklearn.metrics import adjusted_rand_score

from cnamaste.config import get_global_config, start_time
from cnamaste.hmm import pipeline_baum_welch

# from cnamaste.wolff import wolff_sweep
from cnamaste.hmm_initialize import cna_mixture_init, gmm_init
from cnamaste.hmm_phased import hmm_phased
from cnamaste.hmrf_utils import cast_csr, clone_stack_obs
from cnamaste.icm import icm_sweep_deque, merge_assignment, unpack_adjacency
from cnamaste.logger import get_logger
from cnamaste.pseudobulk import merge_pseudobulk_by_index_mix
from cnamaste.hmm_nophasing import get_log_transmat
from cnamaste.hmm_nophasing import _bb_logpmf_1d, _nb_logpmf_1d
from dataclasses import dataclass
from scipy.sparse import csr_matrix
from typing import Any

logger = get_logger(__name__, start_time=start_time)


@njit
def logsumexp(x):
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(x - x_max)))


@njit(parallel=False, cache=True, fastmath=False, error_model="numpy")
def pool_spatio_genomic_counts(
    single_X,
    single_base_nb_mean,
    single_total_bb_RD,
    smooth_indices,
    smooth_indptr,
    single_tumor_prop=None,
    is_tumor_mixed=False,
):
    """
    Aggregate X, nb_baseline and bb_read_depth by smooth mat. to downsize for hmrf inference.
    """
    # NB no logger comments in jit compiled.
    n_obs, n_comp, N = single_X.shape

    pooled_X = np.zeros((n_obs, n_comp, N), dtype=single_X.dtype)
    pooled_base_nb_mean = np.zeros((n_obs, N), dtype=single_base_nb_mean.dtype)
    pooled_total_bb_RD = np.zeros((n_obs, N), dtype=single_total_bb_RD.dtype)

    mean_tumor_prop, weighted_tp = None, None

    # NB valid neighbors for this spot.
    for i in prange(N):
        start_idx, end_idx = smooth_indptr[i], smooth_indptr[i + 1]

        # NB vaild neighbors have finite tumor proportion if is_tumor_mixed.
        valid_neighbors = []

        # TODO tumor prop. is not currently supported.
        for k in range(start_idx, end_idx):
            col = smooth_indices[k]

            if is_tumor_mixed and single_tumor_prop is not None:
                if not np.isnan(single_tumor_prop[col]):
                    valid_neighbors.append(col)
            else:
                valid_neighbors.append(col)

        num_valid_neighbors = len(valid_neighbors)

        # NB assigned zero to pooled_X, pooled_base_nb_mean, pooled_total_bb_RD
        #    if no valid neighbors.
        if num_valid_neighbors == 0:
            continue

        # NB for all segments, and spots, we aggregate the X, base_nb_mean, and total_bb_RD
        #    of all valid neighbors.
        #
        #    valid as updated the counts and the baseline will be accounted for in the likelihood (TBC).
        for obs_idx in range(n_obs):
            for neighbor_idx in valid_neighbors:
                pooled_X[obs_idx, 0, i] += single_X[obs_idx, 0, neighbor_idx]
                pooled_X[obs_idx, 1, i] += single_X[obs_idx, 1, neighbor_idx]

                pooled_base_nb_mean[obs_idx, i] += single_base_nb_mean[
                    obs_idx, neighbor_idx
                ]

                pooled_total_bb_RD[obs_idx, i] += single_total_bb_RD[
                    obs_idx, neighbor_idx
                ]

    return (
        pooled_X,
        pooled_base_nb_mean,
        pooled_total_bb_RD,
        mean_tumor_prop,
        weighted_tp,
    )


@njit(parallel=True, cache=True)
def compute_loglike_spot_assignment(
    n_spots,
    num_valid_nb_spotwise,
    num_valid_bb_spotwise,
    single_tumor_prop,
    is_tumor_mixed,
    log_emission_rdr,
    log_emission_baf,
    pred,
    n_obs,
    n_clones,
    smooth_indices=None,
    smooth_indptr=None,
    non_zero_weight=True,
):
    """As `cnamaste`'s, with the spot loop innermost.

    Every argument, every shape and the returned `(n_spots, n_clones)` array
    are `cnamaste`'s. The relative-channel weight is carried unchanged rather
    than fixed: it is a separate finding (#58), and changing two things at
    once would make the bitwise comparison meaningless.

    `prange` moves to the clone loop because each clone writes its own column
    and its own accumulators, so there is no reduction across threads. The
    spot loop cannot carry it -- it is the vectorized one.
    """
    loglike_spot_clone_assignment = np.zeros((n_spots, n_clones))
    rel_valid_emision_weight = np.ones(n_spots, dtype=np.float64)

    if non_zero_weight and smooth_indices is not None and smooth_indptr is not None:
        for i in prange(n_spots):
            start_idx, end_idx = smooth_indptr[i], smooth_indptr[i + 1]

            pooled_num_valid_nb_spotwise, pooled_num_valid_bb_spotwise = 0.0, 0.0

            for k in range(start_idx, end_idx):
                neighbor = smooth_indices[k]

                if is_tumor_mixed and np.isnan(single_tumor_prop[neighbor]):
                    continue

                pooled_num_valid_nb_spotwise += num_valid_nb_spotwise[neighbor]
                pooled_num_valid_bb_spotwise += num_valid_bb_spotwise[neighbor]

            if pooled_num_valid_nb_spotwise > 0 and pooled_num_valid_bb_spotwise > 0:
                rel_valid_emision_weight[i] = (
                    pooled_num_valid_bb_spotwise / pooled_num_valid_nb_spotwise
                )

    is_1d_pred = pred.ndim == 1

    for c in prange(n_clones):
        accumulated_rdr = np.zeros(n_spots)
        accumulated_baf = np.zeros(n_spots)

        for o in range(n_obs):
            copy_state = pred[c * n_obs + o] if is_1d_pred else pred[o, c]

            # NB the contiguous axis, walked in stride order: a vector
            #    accumulation rather than a reduction to a scalar.
            for spot in range(n_spots):
                accumulated_rdr[spot] += log_emission_rdr[copy_state, o, spot]
                accumulated_baf[spot] += log_emission_baf[copy_state, o, spot]

        for spot in range(n_spots):
            loglike_spot_clone_assignment[spot, c] = (
                rel_valid_emision_weight[spot] * accumulated_rdr[spot]
                + accumulated_baf[spot]
            )

    return loglike_spot_clone_assignment


# NB the per-bin kernels are `cnamaste`'s own, imported rather than restated:
#    a patch that reimplemented them would be comparing two implementations
#    of the emission as well as two of the field, and the bitwise claim below
#    would then be about the wrong thing. Inside `cnamaste` this is a local
#    import from the same package.


@njit(nogil=True, cache=True, parallel=True, error_model="numpy")
def fused_spot_clone_field(
    counts_nb,
    base_nb_mean,
    counts_bb,
    total_bb_RD,
    log_mu,
    alphas,
    p_binom,
    taus,
    pred,
    rel_valid_emision_weight,
    out=None,
):
    """The `(n_spots, n_clones)` field, without an emission array.

    Replaces the pair at `cnamaste.hmrf`'s call site rather than either
    function alone, so it takes what the emission was built from:
    `(n_obs, n_spots)` counts and exposures per channel, the per-state
    parameters, the decoded profiles, and the relative channel weight
    `compute_loglike_spot_assignment` would have applied.

    `prange` runs over clones: each writes its own column and its own
    accumulators, so nothing reduces across threads. The spot loop is the
    vectorized one, as in item 1.

    `out` is an `(n_spots, n_clones)` buffer to write into, or `None` to
    allocate one. This is upstream's shape --
    `oxi_snakes_and_ladders.external_field(..., field)` writes in place --
    and it is carried here for that correspondence rather than for the
    bytes: **the buffer is 400 KB at the declared scale**, against the 8 GB
    the two-step materialized, so a caller that reuses it across outer
    iterations saves an allocation and not a footprint. Every entry is
    written before it is read, so a reused buffer needs no clearing.
    """
    n_obs, n_spots = counts_nb.shape
    n_clones = pred.shape[1]

    # NB SIM108's ternary would ask `numba` to unify `none` with an array
    #    rather than specialize on which was passed, which is the whole
    #    mechanism of the optional buffer.
    if out is None:  # noqa: SIM108
        field = np.zeros((n_spots, n_clones))
    else:
        field = out

    for c in prange(n_clones):
        accumulated_rdr = np.zeros(n_spots)
        accumulated_baf = np.zeros(n_spots)
        scratch = np.zeros(n_spots)

        for o in range(n_obs):
            copy_state = pred[o, c]

            # NB one row per bin, contiguous, and only the state this clone
            #    decoded to -- which is the cut: n_clones of n_states.
            _nb_logpmf_1d(
                counts_nb[o, :],
                base_nb_mean[o, :],
                np.exp(log_mu[copy_state]),
                alphas[copy_state],
                scratch,
            )
            accumulated_rdr += scratch

            _bb_logpmf_1d(
                counts_bb[o, :],
                total_bb_RD[o, :],
                p_binom[copy_state],
                taus[copy_state],
                scratch,
            )
            accumulated_baf += scratch

        for spot in range(n_spots):
            field[spot, c] = (
                rel_valid_emision_weight[spot] * accumulated_rdr[spot]
                + accumulated_baf[spot]
            )

    return field


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


def _channel_weight(
    valid_nb: np.ndarray,
    valid_bb: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
) -> np.ndarray:
    """`rel_valid_emision_weight`, as a segment sum rather than two loops.

    `cnamaste` computes it inside `compute_loglike_spot_assignment`: pooled
    valid BAF segments over pooled valid RDR segments, across the spot's
    smoothed neighbourhood, and one where either is zero. The neighbourhood
    is a CSR row, so the pooling is `np.add.reduceat` over the non-zeros --
    the same arithmetic, without the Python loop over spots.

    What the factor *is* remains #58's: the paper's field carries no such
    term, and this reproduces it rather than endorsing it.
    """
    n_spots = len(indptr) - 1
    weight = np.ones(n_spots, dtype=np.float64)

    if indices.size == 0:
        return weight

    # NB `reduceat` needs the start of each row and misbehaves on an empty
    #    one, so rows are summed by segment id instead -- which also gives
    #    an empty row a pooled count of zero, as the loop does.
    rows = np.repeat(np.arange(n_spots), np.diff(indptr))

    pooled_nb = np.bincount(rows, weights=valid_nb[indices], minlength=n_spots)
    pooled_bb = np.bincount(rows, weights=valid_bb[indices], minlength=n_spots)

    live = (pooled_nb > 0) & (pooled_bb > 0)
    weight[live] = pooled_bb[live] / pooled_nb[live]

    return weight


@dataclass
class _Boundary:
    """What the seam recomputes per outer iteration and need not (#59 item 4).

    `num_valid_nb_spotwise`, `num_valid_bb_spotwise` and the relative channel
    weight derived from them are properties of the **input data**.
    `single_base_nb_mean` and `single_total_bb_RD` are read by
    `load_input_data` and conditioned on throughout -- `cnamaste` never fits
    them -- so none of the three can change while the outer loop runs, and
    `cnamaste` recomputes all three on every iteration anyway.

    `held` is the point of the dataclass rather than an afterthought. The
    cache is keyed on `id()`, and an `id()` is only unique while its object
    is alive, so the entry keeps a reference to every array it was keyed on.
    Those arrays are the pipeline's own inputs and outlive the loop regardless,
    so this costs nothing and closes the one way an identity cache goes
    wrong.
    """

    valid_nb: np.ndarray
    valid_bb: np.ndarray
    weight: np.ndarray
    held: tuple[Any, ...]


_BOUNDARY: dict[tuple[int, ...], _Boundary] = {}
"""One slot. A run conditions on one dataset, so a second entry is a bug."""


def boundary(
    single_base_nb_mean: np.ndarray,
    single_total_bb_RD: np.ndarray,
    smooth_mat: Any,
) -> _Boundary:
    """The seam's loop invariants, computed once per dataset.

    Measured as a **simplification** rather than a speedup: the two count
    passes are 25.2 ms at 3,000 x 5,000 and under two tenths of a per cent of
    the boundary (#59 item 4). What it buys is that a quantity which cannot
    change stops being recomputed, `max_iter_outer` times.
    """
    key = (
        id(single_base_nb_mean),
        id(single_total_bb_RD),
        id(smooth_mat) if smooth_mat is not None else 0,
    )

    cached = _BOUNDARY.get(key)

    if cached is not None:
        return cached

    valid_nb = (single_base_nb_mean > 0).sum(axis=0)
    valid_bb = (single_total_bb_RD > 0).sum(axis=0)

    weight = (
        _channel_weight(valid_nb, valid_bb, smooth_mat.indices, smooth_mat.indptr)
        if smooth_mat is not None
        else np.ones(single_base_nb_mean.shape[1], dtype=np.float64)
    )

    _BOUNDARY.clear()
    _BOUNDARY[key] = _Boundary(
        valid_nb=valid_nb,
        valid_bb=valid_bb,
        weight=weight,
        held=(single_base_nb_mean, single_total_bb_RD, smooth_mat),
    )

    return _BOUNDARY[key]


def _decoded(pred: np.ndarray, n_obs: int) -> np.ndarray:
    """`pred` as `(n_obs, n_clones)`, whichever form the caller passed.

    `cnamaste` accepts both and reads `pred[c * n_obs + o]` from the flat one,
    so the clone-major reshape is the transpose of `(n_clones, n_obs)`. **The
    flat form is what the live pipeline passes** -- found by delegating on it
    and reading the log, not by reading the call sites -- so supporting it is
    the difference between a patch that runs and one that does not.
    """
    if pred.ndim == 2:
        return pred

    return np.ascontiguousarray(pred.reshape(-1, n_obs).T)


def _delegates(single_tumor_prop: Any) -> str | None:
    """Why this call goes to `cnamaste`'s function, or `None` if it does not."""
    if single_tumor_prop is not None:
        return "the tumour-mixed field (#135)"

    return None


def _clone_shifts(
    hmmclass: Any, res: Any, decoded: np.ndarray, single_base_nb_mean: np.ndarray
) -> np.ndarray | None:
    """`log sum_g lambda_g mu_{s_c(g)}` per clone, or `None` when unshifted.

    Only when the fit was shifted (#276, #293): scoring spots with the shift
    off against rates fitted with it on is the inconsistency this repairs.
    `lambda` is built as `hmrf.py:476` builds `normal_lambda` -- the baseline
    summed over spots, normalized -- because that is what the fit's shift
    was taken against, and `decoded` is the `(n_obs, n_clones)` path the
    field reads.
    """
    import scipy.special

    from cnamaste.clone_paths import state_vector

    if not getattr(hmmclass, "apply_logmu_shift", False):
        return None

    profile = np.asarray(single_base_nb_mean, dtype=np.float64).sum(axis=1)
    total = profile.sum()

    if total <= 0.0:
        return None

    with np.errstate(divide="ignore"):
        log_lambda = np.log(profile / total)

    rates = state_vector(res["new_log_mu"])
    terms = rates[np.asarray(decoded, dtype=np.int64)] + log_lambda[:, None]

    shifts: np.ndarray = scipy.special.logsumexp(terms, axis=0)

    return shifts


def pipeline_clone_assignment(
    single_X: np.ndarray,
    single_base_nb_mean: np.ndarray,
    single_total_bb_RD: np.ndarray,
    res: Any,
    pred: np.ndarray,
    adjacency_mat: Any,
    prev_assignment: np.ndarray,
    sample_ids: Any,
    spatial_weight: float,
    smooth_mat: Any = None,
    log_persample_weights: Any = None,
    single_tumor_prop: Any = None,
    hmmclass: Any = None,
    merge: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """What `cnamaste.hmrf.pipeline_clone_assignment` returns, computed leaner."""
    from cnamaste.icm_interface import CsrGraph, fold_unary, icm_sweep
    from cnamaste.clone_paths import state_vector

    reason = _delegates(single_tumor_prop)

    if reason is not None:
        logger.info_once(f"Delegating clone assignment to cnamaste: {reason}.")

        return pipeline_clone_assignment_reference(
            single_X,
            single_base_nb_mean,
            single_total_bb_RD,
            res,
            pred,
            adjacency_mat,
            prev_assignment,
            sample_ids,
            spatial_weight,
            smooth_mat=smooth_mat,
            log_persample_weights=log_persample_weights,
            single_tumor_prop=single_tumor_prop,
            hmmclass=hmmclass,
            merge=merge,
        )

    n_obs, _, n_spots = single_X.shape
    n_states = res["new_p_binom"].shape[0]

    decoded = _decoded(pred, n_obs)
    n_clones = decoded.shape[1]

    started = time.time()
    new_assignment = copy.copy(prev_assignment)

    logger.info(
        f"Solving (pooled) emission likelihood for X.shape={single_X.shape}, "
        f"n_states={n_states} and {n_clones} clones with {hmmclass.__name__}, "
        f"is_tumor_mixed=False and merge={merge}."
    )

    logger.info("Pooling hmrf data by smooth mat. (reduces necessary computation).")

    if smooth_mat is not None:
        pooled_X, pooled_base_nb_mean, pooled_total_bb_RD, _, _ = (
            pool_spatio_genomic_counts(
                single_X,
                single_base_nb_mean,
                single_total_bb_RD,
                smooth_mat.indices,
                smooth_mat.indptr,
                None,
                False,
            )
        )
    else:
        pooled_X = single_X.copy()
        pooled_base_nb_mean = single_base_nb_mean.copy()
        pooled_total_bb_RD = single_total_bb_RD.copy()

    # NB hoisted: all three are functions of the input data, which the outer
    #    loop never fits, and `cnamaste` recomputes them per iteration (#59
    #    item 4).
    invariants = boundary(single_base_nb_mean, single_total_bb_RD, smooth_mat)

    # NB the two steps `cnamaste` runs here -- build (n_states, n_obs, n_spots)
    #    per channel, then read one decoded state per (bin, clone) out of it --
    #    in one pass that materializes neither.
    shifts = _clone_shifts(hmmclass, res, decoded, single_base_nb_mean)

    if shifts is None:
        field = fused_spot_clone_field(
            pooled_X[:, 0, :],
            pooled_base_nb_mean,
            pooled_X[:, 1, :],
            pooled_total_bb_RD,
            # NB `(n_states,)`, normalized at the edge. The kernel indexes by
            #    state alone, because a state parameter has no second axis to
            #    index (#278).
            state_vector(res["new_log_mu"]),
            state_vector(res["new_alphas"]),
            state_vector(res["new_p_binom"]),
            state_vector(res["new_taus"]),
            decoded,
            invariants.weight,
            # NB the buffer is the caller's, which is upstream's shape --
            #    `external_field(..., field)` writes in place. It is allocated
            #    per call rather than reused, and that is deliberate: this
            #    function *returns* the field, so a reused buffer would
            #    rewrite an array its caller still holds, which is the defect
            #    `tests/test_seam_defects.py` pins on the solver. At 400 KB at
            #    the declared scale there is nothing to win by taking that
            #    risk.
            np.empty((n_spots, n_clones)),
        )
    else:
        # NB the shift is the **candidate** clone's, not the spot's current
        #    one: a spot scored against clone `c` is scored under `c`'s
        #    normalizer. It enters the mean as `base * exp(-shift_c)`, so each
        #    clone's column is the same kernel over a rescaled exposure --
        #    the same work as the one call, split by clone.
        #    Both factors are taken relative to the shifts' mean: the rates
        #    have no scale under the shift and drift along it (to near -7,024
        #    on #292's realization 3), so `exp(-shift)` and `exp(log_mu)` are
        #    each out of range while their product is not.
        field = np.empty((n_spots, n_clones))
        centre = float(np.mean(shifts))

        for clone in range(n_clones):
            column = fused_spot_clone_field(
                pooled_X[:, 0, :],
                pooled_base_nb_mean * np.exp(-(shifts[clone] - centre)),
                pooled_X[:, 1, :],
                pooled_total_bb_RD,
                state_vector(res["new_log_mu"]) - centre,
                state_vector(res["new_alphas"]),
                state_vector(res["new_p_binom"]),
                state_vector(res["new_taus"]),
                np.ascontiguousarray(decoded[:, clone : clone + 1]),
                invariants.weight,
                np.empty((n_spots, 1)),
            )
            field[:, clone] = column[:, 0]

    if get_global_config().hmrf.fixed_assignment:
        logger.warning("Assuming a fixed clone assignment")
    else:
        # NB `port`'s `--sal` selects another solver here; the copy carries
        #    the ICM alone, which is the default (#392).
        logger.info("Solving for updated clone assignment with icm.")

        # NB the solver takes the problem -- a unary field, one graph, one
        #    coupling -- rather than its call site (#59 item 5). The
        #    per-sample weights fold into the field, which is where they were
        #    added anyway, once per visit instead of once per sweep; the
        #    three adjacency arrays travel as the one graph they are.
        result = icm_sweep(
            fold_unary(field, log_persample_weights, sample_ids),
            CsrGraph.from_matrix(adjacency_mat),
            new_assignment,
            spatial_weight,
        )

        niter, new_cost = result.niter, result.cost

        logger.info(f"Ready for potential merging of clones?  {merge}.")

        # NB the COO triple is built here rather than above, and by three
        #    array expressions rather than two pure-Python passes over every
        #    non-zero (#59 item 3). `merge_assignment` is its only consumer,
        #    so on a run with `merge=False` `cnamaste` computes the round trip
        #    and discards it.
        if merge:
            adj_spots, adj_neighbors, adj_weights = adjacency_coo(adjacency_mat)

        while merge:
            new_cost, best_merge_cost, best_merge_pair = merge_assignment(
                field,
                adj_spots,
                adj_neighbors,
                adj_weights,
                new_assignment,
                spatial_weight,
                log_persample_weights=log_persample_weights,
                sample_ids=sample_ids,
            )

            if best_merge_cost > new_cost:
                first, second = best_merge_pair
                merged = int((new_assignment == first).sum())

                new_assignment[new_assignment == first] = second

                logger.info(
                    f"Merged {merged} spots from clone {first} into clone {second} "
                    f"with new cost={best_merge_cost} given original "
                    f"cost={new_cost:.6e}."
                )

                new_cost = best_merge_cost
            else:
                logger.info(
                    f"No more beneficial merges available (best merge "
                    f"cost={best_merge_cost} given original cost={new_cost:.6e})."
                )
                break

        _, counts = np.unique(new_assignment, return_counts=True)

        logger.info(
            f"Found new clone assignment with new cost {new_cost:.6e} in {niter} "
            f"iterations ({time.time() - started:.2f}s with clone breakdown=\n"
            f"{[f'{share:.3f}' for share in counts / counts.sum()]})."
        )

    logger.info("Computing ln likelihood for hmrf.")

    log_likelihood = float(
        np.sum(np.take_along_axis(field, new_assignment.astype(int)[:, None], axis=1))
    )

    rows, columns = adjacency_mat.nonzero()
    upper = rows < columns

    log_likelihood += spatial_weight * np.sum(
        new_assignment[rows[upper]] == new_assignment[columns[upper]]
    )

    return new_assignment, field, log_likelihood


# NB aggregate by smooth mat. with tumor/normal mix, spot reassignment, concatenated by clone?
def pipeline_clone_assignment_reference(
    single_X,
    single_base_nb_mean,
    single_total_bb_RD,
    res,
    pred,
    adjacency_mat,
    prev_assignment,
    sample_ids,
    spatial_weight,
    smooth_mat=None,
    log_persample_weights=None,
    single_tumor_prop=None,
    hmmclass=None,
    merge=False,
):
    # NB n_obs is the number of genomic segments, N is the number of spots.
    n_obs, _, N = single_X.shape
    n_states = res["new_p_binom"].shape[0]

    # NB pred is the argmax posterior by genome, potentially __concatenated__ across clones.
    if pred.ndim == 1:
        n_clones = len(pred) // n_obs
    else:
        n_clones = pred.shape[1]

    start_time = time.time()

    # NB clone assignment for all spots.
    new_assignment = copy.copy(prev_assignment)

    # NB utilize tumor mixture model?
    is_tumor_mixed = single_tumor_prop is not None

    # NB compute lambda, i.e. normalized baseline expression, for mixture model
    # lambd = (
    #     np.sum(single_base_nb_mean, axis=1) / np.sum(single_base_nb_mean)
    #     if is_tumor_mixed
    #     else None  # TODO BUG?
    # )

    logger.info(
        f"Solving (pooled) emission likelihood for X.shape={single_X.shape}, n_states={n_states} and {n_clones} clones with {hmmclass.__name__}, is_tumor_mixed={is_tumor_mixed} and merge={merge}."
    )

    logger.info("Pooling hmrf data by smooth mat. (reduces necessary computation).")

    if smooth_mat is not None:
        # NB   pool (sum) data according to smooth (adjacency) matrix, for X, nb_baseline, bb read depth and mean tumor proportion:
        pooled_X, pooled_base_nb_mean, pooled_total_bb_RD, _, _ = (
            pool_spatio_genomic_counts(
                single_X,
                single_base_nb_mean,
                single_total_bb_RD,
                smooth_mat.indices,
                smooth_mat.indptr,
                single_tumor_prop,
                is_tumor_mixed,
            )
        )
    else:
        # TODO copies necessary?
        pooled_X, pooled_base_nb_mean, pooled_total_bb_RD, _, _ = (
            single_X.copy(),
            single_base_nb_mean.copy(),
            single_total_bb_RD.copy(),
            None,
            None,
        )

    # NB emission shape: (n_states, n_obs, n_spots)
    (
        tmp_log_emission_rdr,
        tmp_log_emission_baf,
    ) = hmmclass.compute_emission_probability_nb_betabinom(
        pooled_X,
        pooled_base_nb_mean,
        res["new_log_mu"],
        res["new_alphas"],
        pooled_total_bb_RD,
        res["new_p_binom"],
        res["new_taus"],
    )

    _tumor_prop = single_tumor_prop if single_tumor_prop is not None else np.empty(0)

    # NB For all spots, the number of valid genomic segments for a given emission type,
    #    as per nb_baseline and bb_read depth.
    num_valid_nb_spotwise = (single_base_nb_mean > 0).sum(axis=0)
    num_valid_bb_spotwise = (single_total_bb_RD > 0).sum(axis=0)

    # NB computes the log likelihood for each spot, for all clones, given the "pooling" strategy,
    #    no longer IID and erroneously weights rdr and baf according to number of non-zero segments.
    loglike_spot_clone_assignment = compute_loglike_spot_assignment(
        N,
        num_valid_nb_spotwise,
        num_valid_bb_spotwise,
        _tumor_prop,
        is_tumor_mixed,
        tmp_log_emission_rdr,
        tmp_log_emission_baf,
        pred,
        n_obs,
        n_clones,
        smooth_indices=smooth_mat.indices if smooth_mat is not None else None,
        smooth_indptr=smooth_mat.indptr if smooth_mat is not None else None,
    )

    # assert np.allclose(single_llf, new_single_llf), "BUG: single_llf mismatch"

    adj_list = cast_csr(adjacency_mat)
    adj_spots, adj_neighbors, adj_weights = unpack_adjacency(adj_list)

    if get_global_config().hmrf.fixed_assignment:
        logger.warning(f"Assuming a fixed clone assignment")
    else:
        logger.info(f"Solving for updated clone assignment with icm_sweep_dequeue.")

        # NB updates new_assignment and posterior in place given log emission likelihood.
        """
        niter, new_cost = icm_sweep_deque(
            loglike_spot_clone_assignment,
            adj_spots,
            adj_neighbors,
            adj_weights,
            new_assignment,
            spatial_weight,
            posterior,
            # tol=0.1,  # MAGIC TODO
            log_persample_weights=log_persample_weights,
            sample_ids=sample_ids,
        )
        """
        niter, new_cost = icm_sweep_deque(
            single_llf=loglike_spot_clone_assignment,
            adj_indptr=adjacency_mat.indptr,
            adj_indices=adjacency_mat.indices,
            adj_weights=adjacency_mat.data,
            new_assignment=new_assignment,
            spatial_weight=spatial_weight,
            posterior=None,
            onehot_allowed_clones=None,
            # tol=0.1,  # MAGIC TODO
            log_persample_weights=log_persample_weights,
            sample_ids=sample_ids,
        )

        logger.info(f"Ready for potential merging of clones?  {merge}.")

        while merge:
            # NB merge_assignment returns the original cost, the best new cost after merging this clone pair, and the clone pair.
            new_cost, best_merge_cost, best_merge_pair = merge_assignment(
                loglike_spot_clone_assignment,
                adj_spots,
                adj_neighbors,
                adj_weights,
                new_assignment,
                spatial_weight,
                log_persample_weights=log_persample_weights,
                sample_ids=sample_ids,
            )

            # NB only merge the clone pair if the cost is improved.
            if best_merge_cost > new_cost:
                u, v = best_merge_pair
                num_merged_spots = 0

                # TODO ensure clone "v" has a larger index than clone "u" to minimize downstream reindexing issues.
                # NB assigns clone "u" to clone "v"
                for i in range(len(new_assignment)):
                    if new_assignment[i] == u:
                        new_assignment[i] = v
                        num_merged_spots += 1

                logger.info(
                    f"Merged {num_merged_spots} spots from clone {u} into clone {v} with new cost={best_merge_cost} given original cost={new_cost:.6e}."
                )

                # TODO DEPRECATE?
                new_cost = best_merge_cost
            else:
                logger.info(
                    f"No more beneficial merges available (best merge cost={best_merge_cost} given original cost={new_cost:.6e})."
                )
                break

        # NB counts per clone in the final (potentially merged) assignment.
        _, cnts = np.unique(new_assignment, return_counts=True)

        logger.info(
            f"Found new clone assignment with new cost {new_cost:.6e} in {niter} iterations ({time.time() - start_time:.2f}s with clone breakdown=\n{[f'{xx:.3f}' for xx in cnts / cnts.sum()]})."
        )

    logger.info(f"Computing ln likelihood for hmrf.")

    # NB loglike_spot_clone_assignment was the log likelihood of each spot given that its label is each clone, i.e. unary Potts term;
    #    sum this assuming iid given new assignment.
    log_likelihood = np.sum(
        np.take_along_axis(
            loglike_spot_clone_assignment, new_assignment.astype(int)[:, None], axis=1
        )
    )

    # NB add the pairwise cost for this assignment, according to the (weighted) number of neighbors with the same assignment.
    #    does __not__account for any edge weighting, i.e. assumes all edges are equal.
    #
    #
    # TODO double counts edges.
    # for i in range(N):
    #     log_likelihood += np.sum(
    #         spatial_weight
    #         * np.sum(
    #             new_assignment[adjacency_mat[i, :].nonzero()[1]] == new_assignment[i]
    #         )
    #     )

    adj_rows, adj_cols = adjacency_mat.nonzero()

    # NB mask to prevent double counting (upper triangle)
    unique_edges_mask = adj_rows < adj_cols

    select_adj_rows = adj_rows[unique_edges_mask]
    select_adj_cols = adj_cols[unique_edges_mask]

    num_aligned = np.sum(
        new_assignment[select_adj_rows] == new_assignment[select_adj_cols]
    )

    log_likelihood += spatial_weight * num_aligned

    return new_assignment, loglike_spot_clone_assignment, log_likelihood


def run_core_inference(
    single_X,
    lengths,
    single_base_nb_mean,
    single_total_bb_RD,
    single_tumor_prop,
    initial_clone_index,
    n_states,
    log_sitewise_transmat,
    # prefix="clones",
    # coords=None,
    smooth_mat=None,
    adjacency_mat=None,
    sample_ids=None,
    sample_list=None,
    max_iter_outer=5,
    # nodepotential="max",
    hmmclass=hmm_phased,  # hmm_sitewise
    hmm_initializer=gmm_init,  # {cna_mixture_init, gmm_init}
    params="stmp",
    t=1 - 1e-6,
    random_state=0,
    init_log_mu=None,
    init_p_binom=None,
    init_alphas=None,
    init_taus=None,
    fix_NB_dispersion=False,
    shared_NB_dispersion=True,
    fix_BB_dispersion=False,
    shared_BB_dispersion=True,
    is_diag=True,
    max_iter=100,
    tol=1e-4,
    # unit_xsquared=9,
    # unit_ysquared=3,
    spatial_weight=1.0 / 6.0,
    tumorprop_threshold=0.5,
    propagate_hmm_param_errors=False,
    deconcatenate_clones=False,
):
    # NB num. of genomic bins, num. pseudobulk (clones, spots, ...)
    n_obs, _, _ = single_X.shape

    # NB num. of clones in initial assignment.
    n_clones = len(initial_clone_index)

    # NB map sample_ids to integer enum, i.e. per slice.
    unique_sample_ids = np.unique(sample_ids)
    n_samples = len(unique_sample_ids)

    logger.info(
        f"Running hmrfmix_concatenate_pipeline for {n_clones} clones and {n_samples} samples/slices."
    )

    tmp_map_index = {unique_sample_ids[i]: i for i in range(len(unique_sample_ids))}
    sample_ids = np.array([tmp_map_index[x] for x in sample_ids])

    has_normal_lambda = np.count_nonzero(single_base_nb_mean > 0.0)
    normal_lambda = None

    # NB baseline expression by summing over all clones; should be zero for BAF only.
    if not has_normal_lambda:
        logger.warning(
            f"Found ill-defined normal baseline; corresponds to baf only run."
        )
    else:
        # NB expect the normal baseline, scaled by the total number of transcripts per spot.
        with np.errstate(divide="ignore", invalid="ignore"):
            # TBC sum over all spots, lamba x total sample transcripts.
            normal_lambda = np.sum(single_base_nb_mean, axis=1)

            # NB lambda.
            normal_lambda /= np.sum(single_base_nb_mean)

    # NB aggregation to pseudobulk based on current clone assignment of spots.
    X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        initial_clone_index,
        single_tumor_prop,
        threshold=tumorprop_threshold,
    )

    # validation_summary(lengths, X, base_nb_mean, total_bb_RD, tumor_prop)

    # NB transform (n_obs, 2, n_clones) to (n_obs * n_clones, 2, 1) for HMM processing.
    #    i.e. stack bins per clone lengthwise, useful for fitting shared copy state.
    (
        clone_stack_X,
        clone_stack_base_nb_mean,
        clone_stack_total_bb_RD,
        clone_stack_lengths,
        clone_stack_sitewise_transmat,
        stack_tumor_prop,
    ) = clone_stack_obs(
        X, base_nb_mean, total_bb_RD, lengths, log_sitewise_transmat, tumor_prop
    )

    merge = False

    if (init_log_mu is None) or (init_p_binom is None):
        log_transmat = get_log_transmat(n_states, t)

        new_init_log_mu, new_init_p_binom, _, _ = hmm_initializer(
            n_states,
            clone_stack_X,
            clone_stack_base_nb_mean,
            clone_stack_total_bb_RD,
            params,
            clone_stack_lengths,
            log_transmat,
            clone_stack_sitewise_transmat,
            random_state=random_state,
            in_log_space=False,
            only_minor=False,  # NB with no phasing, we need states > 0.5;
        )

        new_init_alphas, new_init_taus = init_alphas, init_taus

        if init_log_mu is None:
            init_log_mu = new_init_log_mu
            init_alphas = new_init_alphas

        if init_p_binom is None:
            init_p_binom = new_init_p_binom
            init_taus = new_init_taus

        logger.info(
            f"Solved for hmm initialized parameters:\n{init_log_mu}\n{init_p_binom}"
        )

        # n_states = init_p_binom.shape[0]

    last_log_mu = init_log_mu if "m" in params else None
    last_p_binom = init_p_binom if "p" in params else None
    last_alphas = init_alphas
    last_taus = init_taus
    last_assignment = np.zeros(single_X.shape[2], dtype=int)

    for c, idx in enumerate(initial_clone_index):
        last_assignment[idx] = c

    # NB inertia to spot clone change: log(1 / n_clones) per spot, i.e. uniform prior over clones.
    inertia = bool(get_global_config().hmrf.inertia)
    log_persample_weights = (
        np.ones((n_clones, n_samples)) * (-np.log(n_clones)) if inertia else None
    )

    logger.info(f"Assuming hmrf inertia={inertia} and {hmmclass.__name__} instance.")

    # TODO HACK this scratches res input.
    # NB res required for remain_kwargs construction;
    res = {}
    r = 0

    # NB [num_segments, num_segments ..., num_segments] of length num_clones.
    clone_lengths = X.shape[0] * np.ones(X.shape[2], dtype=int)

    # NB convoluted loop logic to achieve merge on last iteration.
    while r <= max_iter_outer:
        logger.info(
            f"----****  Solving iteration {r}/{max_iter_outer} of copy number state fitting & clone assignment (HMM + HMRF) ****----"
        )

        res = pipeline_baum_welch(
            None,
            clone_stack_X,
            clone_stack_lengths,
            n_states,
            clone_stack_base_nb_mean,
            clone_stack_total_bb_RD,
            clone_stack_sitewise_transmat,
            stack_tumor_prop,
            hmmclass=hmmclass,
            params=params,
            t=t,
            random_state=random_state,
            fix_NB_dispersion=fix_NB_dispersion,
            shared_NB_dispersion=shared_NB_dispersion,
            fix_BB_dispersion=fix_BB_dispersion,
            shared_BB_dispersion=shared_BB_dispersion,
            is_diag=is_diag,
            init_log_mu=last_log_mu,
            init_p_binom=last_p_binom,
            init_alphas=last_alphas,
            init_taus=last_taus,
            max_iter=max_iter,
            tol=tol,
            normal_lambda=normal_lambda,
            clone_lengths=clone_lengths,
            init_log_gamma=None,  # res.get("log_gamma", None)
        )

        # NB MAP copy state, irrespective of phasing. contrast to "pred_cnv".
        pred = np.argmax(res["log_gamma"], axis=0)

        # NB TODO 'max' clone assignment.
        new_assignment, _, total_llf = pipeline_clone_assignment(
            single_X,
            single_base_nb_mean,
            single_total_bb_RD,
            res,
            pred,
            adjacency_mat,
            last_assignment,
            sample_ids,
            smooth_mat=smooth_mat,
            spatial_weight=spatial_weight,
            log_persample_weights=log_persample_weights,
            single_tumor_prop=single_tumor_prop,
            hmmclass=hmmclass,
            merge=merge,
        )
        """
        # NB new assignment did not populate an input clone.
        if len(np.unique(new_assignment)) < X.shape[2]:
            # DEPRECATE
            # res["assignment_before_reindex"] = new_assignment

            # NB imposes new order, if not previously sorted, rather than skip only.
            remaining_clones = np.sort(np.unique(new_assignment))

            # NB map clone id -> new (0,...,N-1) enumeration.
            re_indexing = {c: i for i, c in enumerate(remaining_clones)}

            logger.warning(
                f"Detected clone loss on iteration {r}:  re-indexing clones with {re_indexing}"
            )

            # NB re-index new_assignment to be consecutive given a missing clone.
            # TODO faster way?
            new_assignment = np.array([re_indexing[x] for x in new_assignment])

            concat_idx = np.concatenate(
                [np.arange(c * n_obs, c * n_obs + n_obs) for c in remaining_clones]
            )

            # NB log_gamma and pred_cnv by new clone order (concatenated).
            res["log_gamma"] = res["log_gamma"][:, concat_idx]
            res["pred_cnv"] = res["pred_cnv"][concat_idx]
        """
        remaining_clones, new_assignment_reindexed = np.unique(
            new_assignment, return_inverse=True
        )

        if len(remaining_clones) < X.shape[2]:
            logger.warning(f"Detected clone loss on iteration {r}: re-indexing clones.")

            new_assignment = new_assignment_reindexed
            concat_idx = (remaining_clones[:, None] * n_obs + np.arange(n_obs)).ravel()

            # NB log_gamma and pred_cnv by new clone order (concatenated).
            res["log_gamma"] = res["log_gamma"][:, concat_idx]
            res["pred_cnv"] = res["pred_cnv"][concat_idx]

        res["prev_assignment"] = last_assignment
        res["new_assignment"] = new_assignment
        res["total_llf"] = total_llf

        clone_index = [
            np.where(res["new_assignment"] == c)[0]
            for c in np.unique(res["new_assignment"])
        ]

        X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(
            single_X,
            single_base_nb_mean,
            single_total_bb_RD,
            clone_index,
            single_tumor_prop,
            threshold=tumorprop_threshold,
        )

        # TODO clone stack can be an arg. to merge_pseudobulk_by_index_mix
        (
            clone_stack_X,
            clone_stack_base_nb_mean,
            clone_stack_total_bb_RD,
            clone_stack_lengths,
            clone_stack_sitewise_transmat,
            stack_tumor_prop,
        ) = clone_stack_obs(
            X, base_nb_mean, total_bb_RD, lengths, log_sitewise_transmat, tumor_prop
        )

        state_counts = np.bincount(pred, minlength=n_states)
        state_usage = state_counts / len(pred)

        logger.info(
            f"{np.count_nonzero(last_assignment != res['new_assignment'])}/{len(last_assignment)} assignment changes with ARI to last assignment: {adjusted_rand_score(last_assignment, res['new_assignment']):.4f}"
        )

        with np.printoptions(linewidth=np.inf):
            logger.info(f"Copy number state usage [%]:\n{100. * state_usage}")

        # NB potential conflict with GOTO logic below.
        r += 1

        if (
            # TODO config.hmrf.assignment_ari_tolerance: 0.9?
            adjusted_rand_score(res["prev_assignment"], res["new_assignment"])
            >= get_global_config().hmrf.ari_tolerance
            or len(np.unique(res["new_assignment"])) == 1  # NB single clone assigned.
            or r
            == (
                max_iter_outer - 2
            )  # NB we merge on the iteration before last, facilitating assignment to merged clones.
        ):
            if not merge:
                # NB next round we merge; and the one after fit parameters to the merged clone.
                #    skip ahead (GOTO) between iterations.
                r = max_iter_outer - 1
                merge = True

        last_log_mu = res["new_log_mu"]
        last_p_binom = res["new_p_binom"]
        last_alphas = res["new_alphas"]
        last_taus = res["new_taus"]
        last_assignment = res["new_assignment"]

        # NB X.shape[2] is the current inferred number of clones.
        if inertia:
            log_persample_weights = np.ones((X.shape[2], n_samples)) * (
                -np.log(X.shape[2])
            )

            for sidx in range(n_samples):
                index = np.where(sample_ids == sidx)[0]

                this_persample_weight = np.bincount(
                    res["new_assignment"][index], minlength=X.shape[2]
                ) / len(index)

                log_persample_weights[:, sidx] = np.where(
                    this_persample_weight > 0, np.log(this_persample_weight), -50
                )

                log_persample_weights[:, sidx] = log_persample_weights[
                    :, sidx
                ] - scipy.special.logsumexp(log_persample_weights[:, sidx])

    # TODO FINAL
    # NB after last (merged) assignment, we calculated the clone stacks,
    #    preserve assignment keys, but update baum welch related.
    final_bm_res = pipeline_baum_welch(
        None,
        clone_stack_X,
        clone_stack_lengths,
        n_states,
        clone_stack_base_nb_mean,
        clone_stack_total_bb_RD,
        clone_stack_sitewise_transmat,
        stack_tumor_prop,
        hmmclass=hmmclass,
        params=params,
        t=t,
        random_state=random_state,
        fix_NB_dispersion=fix_NB_dispersion,
        shared_NB_dispersion=shared_NB_dispersion,
        fix_BB_dispersion=fix_BB_dispersion,
        shared_BB_dispersion=shared_BB_dispersion,
        is_diag=is_diag,
        init_log_mu=last_log_mu,
        init_p_binom=last_p_binom,
        init_alphas=last_alphas,
        init_taus=last_taus,
        max_iter=max_iter,
        tol=tol,
        normal_lambda=normal_lambda,
        clone_lengths=clone_lengths,
        init_log_gamma=None,  # res.get("log_gamma", None)
        propagate_errors=propagate_hmm_param_errors,
    )

    # TODO llf should also technically be updated.
    res.params = final_bm_res.params
    res.param_errors = final_bm_res.param_errors if propagate_hmm_param_errors else None
    res.profile = final_bm_res.profile

    if deconcatenate_clones:
        # NB shape=(state, segment, clone)
        res["log_gamma"] = np.stack(
            [
                res["log_gamma"][:, (c * n_obs) : (c * n_obs + n_obs)]
                for c in range(len(np.unique(res["new_assignment"])))
            ],
            axis=-1,
        )

        res["pred_cnv"] = np.argmax(res["log_gamma"], axis=0)

    return res


def reindex_clones(res_combine, posterior=None, single_tumor_prop=None):
    assert single_tumor_prop is None, "single_tumor_prop must be None"

    EPS_BAF = 0.05  # MAGIC
    new_res_combine = copy.copy(res_combine)

    assignments = res_combine["new_assignment"]
    clone_labels = np.unique(assignments)
    n_clones = len(clone_labels)

    pred_cnv = res_combine["pred_cnv"]

    is_concatenated = pred_cnv.ndim == 1

    if is_concatenated:
        n_obs = len(pred_cnv) // n_clones
    else:
        n_obs = pred_cnv.shape[0]

    assert res_combine["new_p_binom"].shape[1] == 1

    baf_profile_list = []
    for c in range(n_clones):
        if is_concatenated:
            clone_path = pred_cnv[c * n_obs : (c + 1) * n_obs]
        else:
            clone_path = pred_cnv[:, c]

        baf_profile_list.append(res_combine["new_p_binom"][clone_path, 0])

    baf_profiles = np.column_stack(baf_profile_list).T

    # NB normal clone minimizes deviation from 0.5 (outside the EPS_BAF deadband)
    baf_penalty = np.maximum(np.abs(baf_profiles - 0.5) - EPS_BAF, 0)
    cid_normal = int(np.argmin(np.sum(baf_penalty, axis=1)))

    unique_clones, spot_counts = np.unique(assignments, return_counts=True)

    mask_rest = unique_clones != cid_normal
    cid_rest = unique_clones[mask_rest]
    counts_rest = spot_counts[mask_rest]

    cid_rest_sorted = cid_rest[np.argsort(counts_rest)]

    reidx = np.concatenate(([cid_normal], cid_rest_sorted)).astype(int)

    logger.info(
        f"Remapping clone index: {cid_normal} (normal) to 0, otherwise sorted by spot count."
    )

    max_id = np.max(unique_clones)
    palette = np.zeros(max_id + 1, dtype=int)

    for new_idx, old_idx in enumerate(reidx):
        palette[old_idx] = new_idx

    new_res_combine["new_assignment"] = palette[assignments]

    for key in ["new_log_mu", "new_alphas", "new_p_binom", "new_taus"]:
        if res_combine[key].shape[1] > 1:
            new_res_combine[key] = res_combine[key][:, reidx]

    if is_concatenated:
        concat_idx = np.concatenate(
            [np.arange(c * n_obs, c * n_obs + n_obs) for c in reidx]
        )

        new_res_combine["pred_cnv"] = pred_cnv[concat_idx]

        if "log_gamma" in res_combine.keys():
            new_res_combine["log_gamma"] = res_combine["log_gamma"][:, concat_idx]

    else:
        if pred_cnv.shape[1] > 1:
            new_res_combine["pred_cnv"] = pred_cnv[:, reidx]

        if "log_gamma" in res_combine.keys():
            log_gamma = res_combine["log_gamma"]
            if log_gamma.ndim == 3 and log_gamma.shape[2] > 1:
                new_res_combine["log_gamma"] = log_gamma[:, :, reidx]

    if posterior is not None and posterior.shape[1] > 1:
        new_posterior = copy.copy(posterior)[:, reidx]
    else:
        new_posterior = posterior

    return new_res_combine, new_posterior


# TODO FINAL
def merge_by_minspots(
    assignment,
    res,
    single_total_bb_RD,
    min_spots_thresholds=50,
    min_umicount_thresholds=0,
    single_tumor_prop=None,
    threshold=0.5,
    adjacency_mat=None,
):
    if adjacency_mat is not None:
        raise NotImplementedError()
    else:
        logger.warning_once("TODO: adjacency_mat not queried by merge_by_minspots.")

    n_clones = len(np.unique(assignment))
    if n_clones == 1:
        merged_groups = [[assignment[0]]]
        return merged_groups, res

    # NB genomic axis is concatenated across clones.
    n_obs = int(len(res["pred_cnv"]) / n_clones)
    new_assignment = copy.copy(assignment)
    if single_tumor_prop is None:
        tmp_single_tumor_prop = np.array([1] * len(assignment))
    else:
        tmp_single_tumor_prop = single_tumor_prop

    unique_assignment = np.unique(new_assignment)

    # NB find entries in unique_assignment such that either:
    #    i) min_spots_thresholds
    #    ii) (SNP) min_umicount_thresholds are not satisfied
    # NB find clones failing min_spots_thresholds
    insufficient_spots_clones = [
        c
        for c in unique_assignment
        if np.sum(new_assignment[tmp_single_tumor_prop > threshold] == c)
        < min_spots_thresholds
    ]

    # NB find clones failing min_umicount_thresholds
    insufficient_umi_clones = [
        c
        for c in unique_assignment
        if np.sum(
            single_total_bb_RD[
                :, (new_assignment == c) & (tmp_single_tumor_prop > threshold)
            ]
        )
        < min_umicount_thresholds
    ]

    # NB log each condition separately
    logger.info(
        f"Found {len(insufficient_spots_clones)} clones with < {min_spots_thresholds} spots: {insufficient_spots_clones}"
    )
    logger.info(
        f"Found {len(insufficient_umi_clones)} clones with < {min_umicount_thresholds:_} SNP UMIs: {insufficient_umi_clones}"
    )

    # TODO
    # failed_clones = list(set(insufficient_spots_clones) | set(insufficient_umi_clones))
    failed_clones = [
        c
        for c in unique_assignment
        if (
            np.sum(new_assignment[tmp_single_tumor_prop > threshold] == c)
            < min_spots_thresholds
        )
        or (
            np.sum(
                single_total_bb_RD[
                    :, (new_assignment == c) & (tmp_single_tumor_prop > threshold)
                ]
            )
            < min_umicount_thresholds
        )
    ]
    logger.info(
        f"Found {len(failed_clones)} new clones failing thresholds on min. spots or min. SNP umis."
    )

    # NB find the remaining unique_assigment that satisfies both thresholds
    successful_clones = [c for c in unique_assignment if not c in failed_clones]

    if len(successful_clones) == 0:
        logger.error(
            f"All clones failed min. spots or min. SNP UMIs thresholds; cannot proceed with merging."
        )
        raise RuntimeError()

    # NB initial merging groups: each successful clone is its own group
    merging_groups = [[i] for i in successful_clones]

    if len(failed_clones) > 0:
        for c in failed_clones:
            # NB assigns failed clone to that with largest snp umis.
            idx_max = np.argmax(
                [
                    np.sum(
                        single_total_bb_RD[
                            :,
                            (new_assignment == c_prime)
                            & (tmp_single_tumor_prop > threshold),
                        ]
                    )
                    for c_prime in successful_clones
                ]
            )
            logger.warning(
                f"Assigning failed clone {c} to clone {[successful_clones[idx_max]]} (with largest SNP UMIs)."
            )

            merging_groups[idx_max].append(c)

    # NB re-map new_assignment according to merging_groups.
    map_clone_id = {}
    for i, x in enumerate(merging_groups):
        for z in x:
            map_clone_id[z] = i
    new_assignment = np.array([map_clone_id[x] for x in new_assignment])

    merged_res = copy.copy(res)
    merged_res["new_assignment"] = new_assignment
    merged_res["total_llf"] = np.nan

    # Extract the representative clone IDs to keep
    rep_clones = [c[0] for c in merging_groups]

    # NB expect an array of clones concatenated along the genomic axis,
    if res["pred_cnv"].ndim == 1:
        n_obs = len(res["pred_cnv"]) // n_clones
        merged_res["pred_cnv"] = np.concatenate(
            [res["pred_cnv"][(c * n_obs) : (c * n_obs + n_obs)] for c in rep_clones]
        )
        merged_res["log_gamma"] = np.hstack(
            [res["log_gamma"][:, (c * n_obs) : (c * n_obs + n_obs)] for c in rep_clones]
        )
    # NB expect an independent clone axis.
    else:
        merged_res["pred_cnv"] = res["pred_cnv"][:, rep_clones]
        merged_res["log_gamma"] = res["log_gamma"][:, :, rep_clones]

    return merging_groups, merged_res
