"""`cnaster.hmrf` at the pin, its clone assignment `port`'s (T- #670 PR7).

`docs/port-forward.md` rows 31-34, moved in below the marked seam:

- row 31, `compute_loglike_spot_assignment`: the spot loop innermost, a
  vector accumulation; bitwise `cnaster`'s (#59 item 1);
- row 32, `pipeline_clone_assignment`: `port.patch.hmrf.clone_assignment`,
  the field fused and tabulated (`cnamaste.spot_clone_field`, #59 items
  2-4), the solver behind one interface (`cnamaste.label_solver`), the
  floor met smallest first (#348), and the read-depth refinement's mask
  (row 24, #348, #467). `cnaster`'s is `_cnaster_pipeline_clone_assignment`,
  which the tumour-mixed field still takes (#135);
- row 33, `run_core_inference`: the initializer's options (#348, #489,
  #540), the sample-id check (#418), and with the shift the neutral state
  pinned to `mu = 1` and each clone's `log Z_c` recorded (#293, #299, #362);
- row 34, `reindex_clones`: every fitted parameter one value per state
  (#278), the clones' shifts permuted with them (#362, #501).

**Departures from `port`'s, stated.** The options are keywords of
`run_core_inference`, which hands the clone assignment's on, where `port`
binds each row's at install. The refinement mask travels as an argument
(`RefinementMask`), where `port` holds it in a module global; the boundary's
invariants are computed per call, where `port` caches them by `id()`. Both
compute the same; neither outlives the call. The field is the log-space one
alone (`cnamaste`'s kernels since PR4). The reindexed decode and shifts that
`port` keeps for its integer decode are not kept: that decode moves at PR8.

**Defects fixed (#483).** The merge's boundary gain counts both sides of
each edge, and the reported `total_llf`'s spatial term is the energy the
solvers minimize (`cnamaste.label_solver.merge_assignment`,
`spatial_log_prior`). `port`'s rows carry both defects still.
"""

import copy
import functools
import time
from typing import Any

import numpy as np
import scipy.special
from numba import njit, prange
from sklearn.metrics import adjusted_rand_score

from cnamaste.config import get_global_config, start_time
from cnamaste.hmm import pipeline_baum_welch

# from cnaster.wolff import wolff_sweep
from cnamaste.hmm_initialize import cna_mixture_init, gmm_init
from cnamaste.hmm_phased import hmm_phased
from cnamaste.hmrf_utils import cast_csr, clone_stack_obs
from cnamaste.icm import icm_sweep_deque, unpack_adjacency
from cnamaste.label_solver import merge_assignment, spatial_log_prior
from cnamaste.logger import get_logger
from cnamaste.pseudobulk import merge_pseudobulk_by_index_mix
from cnamaste.hmm_nophasing import get_log_transmat

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
    non_zero_weight=True,  # NB if False, rdr out-weighs the baf signal, which has many zero read_depth segments.
):
    """
    Calculates the (log) emission likelihood for each spot, for all clones.
    Optionally, applies a relative weighting to the rdr and baf likelihoods,
    based on the number of valid segments for each emission type.
    """
    # NB compute the log likelihood for each spot, for all clones.
    loglike_spot_clone_assignment = np.zeros((n_spots, n_clones))
    rel_valid_emision_weight = np.ones(n_spots, dtype=np.float64)

    if non_zero_weight and smooth_indices is not None and smooth_indptr is not None:
        for i in prange(n_spots):
            start_idx, end_idx = smooth_indptr[i], smooth_indptr[i + 1]

            # NB calculate the nb and bb baseline for this spot.
            pooled_num_valid_nb_spotwise, pooled_num_valid_bb_spotwise = 0.0, 0.0

            # NB loop over pooled neighbors of spot i, skipping those with nan tumor proportion.
            for k in range(start_idx, end_idx):
                neighbor = smooth_indices[k]

                if is_tumor_mixed:
                    if np.isnan(single_tumor_prop[neighbor]):
                        continue

                # NB num_valid_nb_spotwise, num_valid_bb_spotwise contain the number of valid genomic segments for the given emission type;
                #    pooled for this across spots.
                pooled_num_valid_nb_spotwise += num_valid_nb_spotwise[neighbor]
                pooled_num_valid_bb_spotwise += num_valid_bb_spotwise[neighbor]

            # NB both normal and baf signals available.
            if pooled_num_valid_nb_spotwise > 0 and pooled_num_valid_bb_spotwise > 0:
                rel_valid_emision_weight[i] = (
                    pooled_num_valid_bb_spotwise / pooled_num_valid_nb_spotwise
                )

    # NB Numba evaluates .ndim at compile_time. This creates a zero-cost branch.
    is_1d_pred = pred.ndim == 1

    # NB T- #670 PR7, row 31 (#59 item 1): `port.patch.hmrf.field`'s order.
    #    The spot loop is innermost and contiguous, a vector accumulation; the
    #    additions per `(spot, clone)` run over `o` as `cnaster`'s did, so
    #    the result is bitwise its. `prange` moves to the clone loop, where
    #    each clone writes its own column.
    for c in prange(n_clones):
        accumulated_rdr = np.zeros(n_spots)
        accumulated_baf = np.zeros(n_spots)

        for o in range(n_obs):
            copy_state = pred[c * n_obs + o] if is_1d_pred else pred[o, c]

            for spot in range(n_spots):
                accumulated_rdr[spot] += log_emission_rdr[copy_state, o, spot]
                accumulated_baf[spot] += log_emission_baf[copy_state, o, spot]

        for spot in range(n_spots):
            loglike_spot_clone_assignment[spot, c] = (
                rel_valid_emision_weight[spot] * accumulated_rdr[spot]
                + accumulated_baf[spot]
            )

    return loglike_spot_clone_assignment


# NB aggregate by smooth mat. with tumor/normal mix, spot reassignment, concatenated by clone?
# NB T- #670 PR7: `cnaster`'s, which `pipeline_clone_assignment` below hands
#    the tumour-mixed field to (#135).
def _cnaster_pipeline_clone_assignment(
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

    # NB T- #670 PR7 (#483 defect 2): the energy the solvers minimize, each
    #    stored entry at `spatial_weight * w / 2`, where `cnaster` counted
    #    the entries with `i < j`, unweighted.
    log_likelihood += spatial_log_prior(new_assignment, adjacency_mat, spatial_weight)

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
    *,
    hmm_start=None,
    distinct_init=False,
    baf_start=None,
    label_solver="icm",
    floor_merge=False,
    onehot_allowed_clones=None,
):
    """`cnaster`'s inference, with `port`'s row 33 (T- #670 PR7).

    Options, each `cnaster`'s behaviour by default, which `run_cnamaste`
    binds as `run_cnaster_port --sal` does: `distinct_init` (#348), and
    `hmm_start` / `baf_start`, each stage's copy-state start (#489, #540),
    are `gmm_init`'s; `label_solver`, `floor_merge` and
    `onehot_allowed_clones`, the read-depth refinement's mask (row 24), are
    the clone assignment's. With a shifted `hmmclass` and a read-depth fit,
    the result is pinned once after the optimization (`pin_neutral`,
    `clone_shifts`).
    """
    # NB passed rather than rebound: `cnaster` binds the initializer as a
    #    default argument (#348). `gmm_init` holds every start.
    if hmm_initializer is gmm_init and (
        hmm_start is not None or baf_start is not None or distinct_init
    ):
        hmm_initializer = functools.partial(
            gmm_init, start=hmm_start, distinct=distinct_init, baf_start=baf_start
        )

    identity_remap(sample_ids, sample_list)

    refinement = RefinementMask(onehot_allowed_clones)

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
            label_solver=label_solver,
            floor_merge=floor_merge,
            refinement=refinement,
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

    # NB T- #670 PR7, row 33 (#293, #362): the shifted likelihood sets no
    #    scale, so the result is pinned once, after the whole optimization.
    if shifted(hmmclass) and "m" in str(params):
        pin_neutral(res)

        try:
            decoded = res["pred_cnv"] is not None
        except KeyError:
            decoded = False

        if decoded:
            clone_shifts(res, np.asarray(single_base_nb_mean), ZERO_NORMAL_SHIFT)

    return res


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


# --- `port.patch.hmrf`'s rows 32-34, moved in by T- #670 PR7 --------------------------

from cnamaste.clone_paths import parameter_by_path, state_vector  # noqa: E402
from cnamaste.hmm_nophasing import (  # noqa: E402
    NEUTRAL_BAF_TOLERANCE,
    neutral_state,
    shifted,
)

ZERO_NORMAL_SHIFT = True
"""The normal clone's shift is set to zero (#362)."""

MASK_PENALTY = 100.0
"""Nats a spot's field loses on a sub-clone of another BAF clone (#467).

Finite, so a strong read-depth preference can still correct a spot the BAF
stage misplaced; `-inf` made the BAF stage's boundary final. Measured in
`port` with `--sal --refinement-mask --floor-merge` (clone ARI; `-inf` in
brackets): `dev` 1.000 (0.868), CalicoST hard 0.982 (0.982), easy 0.986.
"""


class RefinementMask:
    """The read-depth refinement's allowed-clone mask, for one inference (row 24, #348).

    `initialize_rdr_clone_refininement` returns it and `cnaster`'s call to
    `run_core_inference` drops it (`# onehot_allowed_clones=None`). Without
    it the refinement is an unconstrained `n_baf * n_clones_rdr`-label
    problem, which `cnaster`'s floor finishes by moving spots at random
    (#348: 1,509 of 1,600 spots into one clone, ARI 0.000). Applied to a
    problem of its shape whose labelling keeps to it, and narrowed to the
    surviving clones as `run_core_inference` relabels them.
    """

    def __init__(self, mask=None):
        self.mask = None if mask is None else np.asarray(mask, dtype=bool)

    def kept(self):
        return self.mask is not None

    def for_problem(self, assignment, n_clones):
        """The mask, if it describes this problem and this labelling."""
        if self.mask is None:
            return None

        labels = np.asarray(assignment, dtype=np.int64)

        if self.mask.shape != (labels.size, n_clones):
            logger.info(
                f"Refinement mask {self.mask.shape} not applied to "
                f"{labels.size} x {n_clones}."
            )
            return None
        if (
            labels.max(initial=-1) >= n_clones
            or not self.mask[np.arange(labels.size), labels].all()
        ):
            logger.info("Refinement mask not applied: the assignment already crosses it.")
            return None

        logger.info(f"Applying the refinement mask over {n_clones} clones (#348).")
        return self.mask

    def compact(self, assignment):
        """Keep the columns of the clones `assignment` still uses, ascending (`hmrf.py:648`)."""
        if self.mask is not None:
            survivors = np.unique(np.asarray(assignment, dtype=np.int64))
            if survivors.size < self.mask.shape[1]:
                self.mask = self.mask[:, survivors]


def identity_remap(sample_ids, sample_list=None):
    """Refuse `sample_ids` that `run_core_inference`'s re-map would renumber (#418).

    The plots read `sample_list` by position, so ids and names agree only
    where the ranks are `0..n-1` and `sample_list` has `n` entries.
    """
    if sample_ids is None:
        return

    unique = np.unique(np.asarray(sample_ids))

    if not np.array_equal(unique, np.arange(unique.size)):
        msg = (
            f"sample_ids {unique.tolist()} are not 0..{unique.size - 1}: "
            "run_core_inference would renumber them"
        )
        raise ValueError(msg)

    if sample_list is not None and len(sample_list) != unique.size:
        msg = f"{len(sample_list)} names in sample_list for {unique.size} sample ids"
        raise ValueError(msg)


def _relocked(result, key, value):
    """`result[key] = value`, unlocking a locked `CnaHMRFResult` for the one assignment."""
    locked = bool(getattr(result, "_locked", False))

    if locked:
        result.unlock()

    try:
        result[key] = value
    finally:
        if locked:
            result.lock()


def pin_neutral(result):
    """Set the neutral state's `mu` to 1 in `result`, in place; return the state (#293, #299).

    The shifted mean `lambda_g T_n mu / sum_g lambda_g mu` is unchanged by
    `mu -> c mu`, so the rates carry an arbitrary common factor until this.
    """
    column = np.asarray(result["new_log_mu"])
    rates = state_vector(column)
    try:
        path = np.asarray(result["pred_cnv"])
    except KeyError:
        path = None

    neutral = neutral_state(
        rates,
        state_vector(result["new_p_binom"]),
        path if path is not None and path.ndim == 2 else None,
    )
    _relocked(result, "new_log_mu", (rates - rates[neutral]).reshape(column.shape))

    return neutral


def clone_shifts(res, base_nb_mean, zero_normal=True):
    """Record each clone's `log Z_c` over the pinned rates in `new_log_mu_shift`; return them (#362).

    The normal clone is the one with the largest share of bins in balanced
    states; with `zero_normal` its shift is 0.
    """
    rates = state_vector(np.asarray(res["new_log_mu"]))
    balanced = (
        np.abs(state_vector(np.asarray(res["new_p_binom"])) - 0.5)
        <= NEUTRAL_BAF_TOLERANCE
    )
    path = np.asarray(res["pred_cnv"], dtype=np.int64)
    path = path.reshape(path.shape[0], -1)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_lambda = np.log(np.sum(base_nb_mean, axis=1) / np.sum(base_nb_mean))

    kept = np.isfinite(log_lambda)
    shifts = np.array(
        [
            float(scipy.special.logsumexp(rates[path[kept, c]] + log_lambda[kept]))
            for c in range(path.shape[1])
        ]
    )

    if zero_normal:
        shifts[int(np.argmax(balanced[path].mean(axis=0)))] = 0.0

    _relocked(res, "new_log_mu_shift", shifts)

    return shifts


PARAMETERS = ("new_log_mu", "new_alphas", "new_p_binom", "new_taus")
"""The four the M step fits, each one value per state."""

EPS_BAF = 0.05
"""`cnaster`'s dead band around a balanced BAF, kept at its value."""


def reindex_clones(res_combine, posterior=None, single_tumor_prop=None):
    """`cnaster`'s reindex, every parameter one value per state, the shifts permuted (row 34).

    `port.patch.hmrf.reindex` and `core_inference.reindex_clones`: every
    fitted parameter is checked to be one value per state (`state_vector`,
    #278) and the reorder of a second column `cnaster` keeps for parameters
    that cannot have one is gone; `new_log_mu_shift`, which `cnaster` does
    not permute, moves with the clones (#362, #501). The normal clone,
    least far outside the dead band, is 0; the rest by spot count.
    """
    if single_tumor_prop is not None:  # invariant
        msg = "single_tumor_prop must be None"
        raise AssertionError(msg)

    for key in PARAMETERS:
        state_vector(res_combine[key], key)

    new_res_combine = copy.copy(res_combine)

    assignments = res_combine["new_assignment"]
    n_clones = len(np.unique(assignments))

    pred_cnv = np.asarray(res_combine["pred_cnv"])
    is_concatenated = pred_cnv.ndim == 1
    n_obs = len(pred_cnv) // n_clones if is_concatenated else pred_cnv.shape[0]

    baf_profiles = np.stack(
        [
            parameter_by_path(
                res_combine["new_p_binom"],
                pred_cnv[c * n_obs : (c + 1) * n_obs]
                if is_concatenated
                else pred_cnv[:, c],
            )
            for c in range(n_clones)
        ]
    )

    baf_penalty = np.maximum(np.abs(baf_profiles - 0.5) - EPS_BAF, 0)
    cid_normal = int(np.argmin(np.sum(baf_penalty, axis=1)))

    unique_clones, spot_counts = np.unique(assignments, return_counts=True)

    mask_rest = unique_clones != cid_normal
    cid_rest = unique_clones[mask_rest]
    counts_rest = spot_counts[mask_rest]

    cid_rest_sorted = cid_rest[np.argsort(counts_rest)]
    reidx = np.concatenate(([cid_normal], cid_rest_sorted)).astype(int)

    logger.info(
        f"Remapping clone index: {cid_normal} (normal) to 0, "
        f"otherwise sorted by spot count."
    )

    palette = np.zeros(int(np.max(unique_clones)) + 1, dtype=int)

    for new_idx, old_idx in enumerate(reidx):
        palette[old_idx] = new_idx

    new_res_combine["new_assignment"] = palette[assignments]

    if is_concatenated:
        concat_idx = np.concatenate(
            [np.arange(c * n_obs, c * n_obs + n_obs) for c in reidx]
        )

        new_res_combine["pred_cnv"] = pred_cnv[concat_idx]

        # NB `.keys()`: `CnaHMRFResult` defines no `__contains__`.
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

    # NB by `__getitem__`: `CnaHMRFResult` has no `get` (#501).
    try:
        shifts = res_combine["new_log_mu_shift"]
    except (KeyError, TypeError):
        shifts = None

    if shifts is None or np.ndim(shifts) != 1:
        return new_res_combine, new_posterior

    # NB the permutation by matching columns, as `port` recovers it.
    old = pred_cnv.reshape(pred_cnv.shape[0], -1)
    after = np.asarray(new_res_combine["pred_cnv"])
    new = after.reshape(after.shape[0], -1)
    order = []

    for column in range(new.shape[1]):
        matches = [
            c
            for c in range(old.shape[1])
            if c not in order and np.array_equal(old[:, c], new[:, column])
        ]
        order.append(matches[0] if matches else column)

    _relocked(
        new_res_combine,
        "new_log_mu_shift",
        np.asarray(shifts, dtype=np.float64)[order],
    )

    return new_res_combine, new_posterior


class PooledSmoothing(ValueError):
    """A `smooth_mat` that pools a spot with any spot but itself (#513)."""


def require_unpooled(smooth_mat):
    """Refuse a `smooth_mat` other than `None` or the identity (#513).

    The counts are read unpooled; `cnaster` only ever builds the identity.
    """
    if smooth_mat is None:
        return

    import scipy.sparse as sp

    matrix = sp.csr_matrix(smooth_mat)
    n_spots = matrix.shape[0]

    if (
        matrix.shape == (n_spots, n_spots)
        and np.array_equal(matrix.indptr, np.arange(n_spots + 1))
        and np.array_equal(matrix.indices, np.arange(n_spots))
        and np.all(matrix.data == 1)
    ):
        return

    msg = (
        "smooth_mat pools spots with their neighbours; the clone assignment "
        "reads counts unpooled and supports only the identity (#513)"
    )
    raise PooledSmoothing(msg)


def _decoded(pred, n_obs):
    """`pred` as `(n_obs, n_clones)`: the flat form is clone-major, `pred[c * n_obs + o]`."""
    if pred.ndim == 2:
        return pred

    return np.ascontiguousarray(pred.reshape(-1, n_obs).T)


def pipeline_clone_assignment(
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
    *,
    label_solver="icm",
    floor_merge=False,
    refinement=None,
) -> Any:
    """`port.patch.hmrf.pipeline_clone_assignment` (#206, #59): what `cnaster`'s returns, computed leaner.

    The field fused and tabulated, one pass that materializes no emission
    array (`cnamaste.spot_clone_field`), and with a shifted `hmmclass` each
    clone's column under its own `log Z_c` (#276, #293). `label_solver`
    names the solver (`cnamaste.label_solver.SOLVERS`; `"icm"` is
    `cnaster`'s), `floor_merge` meets the floor smallest first (#348), and
    `refinement` carries the read-depth refinement's mask (#348, #467). The
    floor is `hmrf.min_spots_per_clone` where the configuration states it
    (#468), `cnaster`'s 200 where it does not.

    The tumour-mixed field is `cnaster`'s (`_cnaster_pipeline_clone_assignment`,
    #135), and a `smooth_mat` that pools is refused (#513).
    """
    from cnamaste.label_solver import (
        CsrGraph,
        configured_floor,
        enforce_floor,
        fold_unary,
        solver_for,
        sweep_for,
    )
    from cnamaste.logmu_shift import clone_log_normalizers
    from cnamaste.spot_clone_field import (
        adjacency_coo,
        boundary_invariants,
        field_kernel,
        spot_clone_field,
    )

    if single_tumor_prop is not None:
        logger.info("Delegating clone assignment to cnaster: the tumour-mixed field (#135).")

        dropped = [
            flag
            for flag, on in (
                ("the refinement mask", refinement is not None and refinement.kept()),
                ("the floor merge", floor_merge),
                ("the shift", shifted(hmmclass)),
            )
            if on
        ]
        if dropped:
            logger.warning(
                f"{' and '.join(dropped)} not applied: clone assignment "
                "delegates to cnaster for the tumour-mixed field (#135)."
            )

        return _cnaster_pipeline_clone_assignment(
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

    # NB no pooling (#513): `cnaster` builds `smooth_mat` as the identity.
    require_unpooled(smooth_mat)

    # NB functions of the input data alone (#59 item 4).
    invariants = boundary_invariants(single_base_nb_mean, single_total_bb_RD)
    weight = (
        invariants.relative_channel_weight(np.arange(n_spots + 1), np.arange(n_spots))
        if smooth_mat is not None
        else np.ones(n_spots, dtype=np.float64)
    )

    log_mu = state_vector(res["new_log_mu"])
    alphas = state_vector(res["new_alphas"])
    p_binom = state_vector(res["new_p_binom"])
    taus = state_vector(res["new_taus"])

    shifts = (
        clone_log_normalizers(log_mu, decoded, single_base_nb_mean)
        if shifted(hmmclass)
        else None
    )

    if shifts is None:
        field = spot_clone_field(
            single_X[:, 0, :],
            single_base_nb_mean,
            single_X[:, 1, :],
            single_total_bb_RD,
            log_mu,
            alphas,
            p_binom,
            taus,
            decoded,
            weight,
            np.empty((n_spots, n_clones)),
        )
    else:
        # NB the candidate clone's shift: a spot scored against clone `c` is
        #    scored under `c`'s normalizer, as `base * exp(-shift_c)`. Both
        #    factors relative to the shifts' mean, which keeps each in range.
        field = np.empty((n_spots, n_clones))
        centre = float(np.mean(shifts))
        kernel = field_kernel(single_X[:, 0, :], single_X[:, 1, :], single_total_bb_RD)
        scaled = np.empty_like(single_base_nb_mean)

        for clone in range(n_clones):
            np.multiply(
                single_base_nb_mean, np.exp(-(shifts[clone] - centre)), out=scaled
            )
            column = kernel(
                single_X[:, 0, :],
                scaled,
                single_X[:, 1, :],
                single_total_bb_RD,
                log_mu - centre,
                alphas,
                p_binom,
                taus,
                np.ascontiguousarray(decoded[:, clone : clone + 1]),
                weight,
                np.empty((n_spots, 1)),
            )
            field[:, clone] = column[:, 0]

    if get_global_config().hmrf.fixed_assignment:
        logger.warning("Assuming a fixed clone assignment")
    else:
        solver = solver_for(label_solver)
        sweep = sweep_for(solver)

        logger.info(f"Solving for updated clone assignment with {solver}.")

        # NB the refinement's mask into the field, less `MASK_PENALTY` (#467),
        #    and to the solver and floor as the knob.
        mask = None if refinement is None else refinement.for_problem(new_assignment, n_clones)
        knobs = {} if mask is None else {"onehot_allowed_clones": mask}

        if mask is not None:
            unmasked = field
            field = np.where(mask, field, field - MASK_PENALTY)

        folded = fold_unary(field, log_persample_weights, sample_ids)
        graph = CsrGraph.from_matrix(adjacency_mat)

        # NB with the floor merge the sweep runs floorless and the floor is
        #    met after it, smallest first (#348).
        knobs["min_clone_spots"] = 0 if floor_merge else configured_floor()

        result = sweep(folded, graph, new_assignment, spatial_weight, **knobs)

        if floor_merge:
            emptied = enforce_floor(folded, new_assignment, configured_floor())

            if emptied:
                logger.info(
                    f"Merged {emptied} clones under {configured_floor()} spots, "
                    "smallest first (#348)."
                )
                result = sweep(folded, graph, new_assignment, spatial_weight, **knobs)
                enforce_floor(folded, new_assignment, configured_floor())

        niter, new_cost = result.niter, result.cost

        logger.info(f"Ready for potential merging of clones?  {merge}.")

        # NB the COO triple only where it is consumed (#59 item 3).
        if merge:
            adj_spots, adj_neighbors, adj_weights = adjacency_coo(adjacency_mat)

        while merge:
            # NB both sides of each boundary edge (#483 defect 1).
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

        # NB `run_core_inference` relabels the survivors ascending; the mask
        #    follows, so it still describes the next iteration's problem.
        if mask is not None:
            refinement.compact(new_assignment)
            field = unmasked

    logger.info("Computing ln likelihood for hmrf.")

    log_likelihood = float(
        np.sum(np.take_along_axis(field, new_assignment.astype(int)[:, None], axis=1))
    )

    # NB the energy the solvers minimize (#483 defect 2).
    log_likelihood += spatial_log_prior(new_assignment, adjacency_mat, spatial_weight)

    return new_assignment, field, log_likelihood
