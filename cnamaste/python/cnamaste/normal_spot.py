import ast

import anndata
import numpy as np
import scanpy as sc
import scipy
import scipy.stats
from numba import njit
from sklearn.cluster import KMeans

from cnamaste.config import get_global_config, start_time
from cnamaste.hmm_emission import Weighted_BetaBinom
from cnamaste.hmm_utils import get_em_solver_params
from cnamaste.logger import get_logger
from cnamaste.spatio_genomic_counts import SpatioGenomicCounts
from collections.abc import Iterator
from typing import Any
import scipy.special

logger = get_logger(__name__, start_time=start_time)


def determine_normal_candidates(
    config,
    res,
    baf_profiles,
    single_X,
    single_X_rdr,
    smooth_mat,
    single_tumor_prop=None,
):
    """
    Determine normal candidate spots based on BAF profiles, tumor proportion, or provided indices.
    Returns a boolean array normal_candidate.
    """
    logger.info(f"Determining normal spots based on BAF-only clones.")

    # NB no input files for barcodes of normal spots, or tumor proportion per spot.
    if (config.preprocessing.normalidx_file is None) and (
        config.preprocessing.tumorprop_file is None
    ):
        EPS_BAF = 0.05  # MAGIC
        PERCENT_NORMAL = 40  # MAGIC

        logger.info(
            f"Identifying normal spots based on estimated BAF given EPS_BAF={EPS_BAF} and PERCENT_NORMAL={PERCENT_NORMAL}."
        )

        # NB sum deviations > EPS_BAF from 0.5 along the genome for each clone; pick normal as minimum deviation.
        baf_deviations = np.sum(
            np.maximum(np.abs(baf_profiles - 0.5) - EPS_BAF, 0), axis=1
        )
        id_nearnormal_clone = np.argmin(baf_deviations)

        logger.info(
            f"Found clone {id_nearnormal_clone} to be the most normal-like given BAF deviations."
        )

        # NB measure the standard deviation of log-transformed, smoothed transcript counts for each spot.
        if smooth_mat is not None:
            vec_stds = np.std(np.log1p(single_X_rdr @ smooth_mat), axis=0)
        else:
            vec_stds = np.std(np.log1p(single_X_rdr), axis=0)

        prior_stdthreshold = np.inf

        while True:
            stdthreshold = np.percentile(
                vec_stds[res["new_assignment"] == id_nearnormal_clone],
                PERCENT_NORMAL,
            )
            normal_candidate = (vec_stds < stdthreshold) & (
                res["new_assignment"] == id_nearnormal_clone
            )

            # DEPRECATE config.run_legacy.
            if False and (
                np.sum(single_X_rdr[:, (normal_candidate == True)])
                > 200 * single_X.shape[0]  # MAGIC
            ):
                logger.info(
                    f"Assumed legacy normal spot allocation for {PERCENT_NORMAL}[%] normal spots"
                )
                break

            elif stdthreshold > 1.5 * prior_stdthreshold:  # MAGIC
                logger.info(
                    f"Determined {PERCENT_NORMAL}% normal spots with sufficient UMIs, assigned to normal like clone."
                )
                logger.info(
                    f"BAF-clone breakdown:\n{np.unique(res['new_assignment'][normal_candidate], return_counts=True)}"
                )
                break
            elif PERCENT_NORMAL == 100:
                logger.warning(
                    f"All {np.count_nonzero(res['new_assignment'] == id_nearnormal_clone)} spots for clone {id_nearnormal_clone} considered to be normal."
                )
                break

            PERCENT_NORMAL += 10
        return normal_candidate

    elif config.preprocessing.normalidx_file is not None:
        # single_base_nb_mean has already been added in loading data step.
        if config.preprocessing.tumorprop_file is not None:
            logger.warning(
                f"Found mixed sources for normal spot definition, assuming {config.preprocessing.normalidx_file}."
            )
        # You may want to load normal_candidate from file here
        return None

    else:
        assert single_tumor_prop is not None

        logger.info(f"Identifying normal spots based on provided tumor proportion.")

        for prop_threshold in np.arange(0.05, 0.6, 0.05):
            # NB suggests 0 is perfectly normal and otherwise measures tumor proportion, sensibly!
            normal_candidate = single_tumor_prop < prop_threshold

            if (
                np.sum(single_X_rdr[:, (normal_candidate == True)])
                > 200 * single_X.shape[0]  # MAGIC
            ):
                logger.info(
                    f"Determined normal spots with sufficient UMIs based on input tumor proportion @ prop_threshold={prop_threshold}"
                )
                break
        else:
            logger.warning(
                f"Failed to determine normal spots with sufficient UMIs based on input tumor proportion."
            )
        return normal_candidate


def determine_normal_baseline(single_X_rdr, normal_candidate, config=None):
    if config is None:
        config = get_global_config()

    logger.info(
        f"Found sparsity of normal spot set={100. * np.mean(single_X_rdr[:, (normal_candidate == True)]) == 0.:.3f}%"
    )

    # NB normal baseline transcript count; unnormalized.
    # TODO normal_candidate clause
    rdr_normal = np.sum(single_X_rdr[:, (normal_candidate == True)], axis=1)

    bidx_inconfident = np.where(rdr_normal < config.quality.min_normal_count_perbin)[0]

    logger.info(
        f"Found {100. * np.mean(rdr_normal >= config.quality.min_normal_count_perbin):.3f}% of segments with confident normal baseline for MIN_NORMAL_COUNT_PERBIN={config.quality.min_normal_count_perbin}"
    )

    # NB where normal transcript count < config.quality.min_normal_count_perbin, zero.
    rdr_normal[bidx_inconfident] = 0
    rdr_normal = rdr_normal / np.sum(rdr_normal)

    # TODO copy?
    # NB avoid ill-defined distributions if normal has 0 count in that bin.
    single_X_rdr[bidx_inconfident, :] = 0

    # NB replicate and normalize rdr_normal to the per-spot total transcripts, T_n, after zero'ing.
    spots_coverage = np.sum(single_X_rdr, axis=0)

    single_base_nb_mean = rdr_normal.reshape(-1, 1) @ spots_coverage.reshape(1, -1)

    return rdr_normal, single_X_rdr, single_base_nb_mean


@njit
def compute_local_normal_mask(
    baf_profiles,  # (n_clones, n_bins)
    spot_assignments,  # (n_spots,) mapping each spot to a clone index
    single_X_rdr,  # (n_bins, n_spots) raw RNA counts
    window_size,  # int: The "n_segments" to look left and right
    eps_baf=0.05,  # float: allowed BAF wiggle room around 0.5
    baf_margin=0.01,  # float: margin to allow multiple identical clones to pass
    start_percentile=0.4,  # float: start relaxing from 40% of variance distribution
    percentile_step=0.05,  # float: step size (+5%) for relaxing transcript variance
    var_ratio_thresh=1.1,  # float: Required factor of SS_between / SS_within to stop
    amp_min=0.8,  # float: lower bound for amplitude regression
    amp_max=1.25,  # float: upper bound for amplitude regression
):
    """
    Creates a (n_bins x n_spots) boolean mask tracking which spots act as the optimal
    normal baseline for a given genomic segment.

    Identifies a single global BAF-normal clone that is unconditionally anchored.
    Additional localized clones are checked for regression amplitude consistency
    with the anchor, and iteratively merged if they fit within the variance ratio threshold.
    """
    n_clones, n_bins = baf_profiles.shape
    n_spots = single_X_rdr.shape[1]

    local_mask = np.zeros((n_bins, n_spots), dtype=np.bool_)

    # -------------------------------------------------------------------------
    # GLOBAL PRE-COMPUTE: Find global normal clone & normalize clone expressions
    # -------------------------------------------------------------------------

    # 1A. Find single global BAF normal clone
    global_baf_devs = np.zeros(n_clones)
    for c in range(n_clones):
        d = 0.0
        for i in range(n_bins):
            diff = np.abs(baf_profiles[c, i] - 0.5) - eps_baf
            if diff > 0:
                d += diff
        global_baf_devs[c] = d

    global_anchor_clone = 0
    min_global_dev = global_baf_devs[0]
    for c in range(1, n_clones):
        if global_baf_devs[c] < min_global_dev:
            min_global_dev = global_baf_devs[c]
            global_anchor_clone = c

    # 1B. Aggregate spots into pure clone pseudo-bulks and get library sizes
    clone_X_rdr = np.zeros((n_bins, n_clones), dtype=np.float64)
    clone_spot_counts = np.zeros(n_clones, dtype=np.int32)
    clone_lib_size = np.zeros(n_clones, dtype=np.float64)

    for s in range(n_spots):
        c = spot_assignments[s]
        clone_spot_counts[c] += 1
        s_sum = 0.0
        for i in range(n_bins):
            val = single_X_rdr[i, s]
            clone_X_rdr[i, c] += val
            s_sum += val
        clone_lib_size[c] += s_sum

    # Precompute mean log1p of the clone expressions
    # and normalize clone_X_rdr raw counts by library size for fast regression
    log1p_clone_rdr = np.empty_like(clone_X_rdr)
    norm_clone_rdr = np.empty_like(clone_X_rdr)

    for c in range(n_clones):
        if clone_spot_counts[c] > 0 and clone_lib_size[c] > 0:
            scale_factor = 1.0 / clone_lib_size[c]
        else:
            scale_factor = 0.0

        for i in range(n_bins):
            if clone_spot_counts[c] > 0:
                log1p_clone_rdr[i, c] = np.log1p(
                    clone_X_rdr[i, c] / clone_spot_counts[c]
                )
            else:
                log1p_clone_rdr[i, c] = 0.0

            norm_clone_rdr[i, c] = clone_X_rdr[i, c] * scale_factor

    # -------------------------------------------------------------------------
    # MAIN LOOP: Segment-by-segment localized clustering
    # -------------------------------------------------------------------------
    total_accepted_clones = 0
    for b in range(n_bins):
        w_start = max(0, b - window_size)
        w_end = min(n_bins, b + window_size + 1)
        w_len = w_end - w_start

        # 1. Start with the guaranteed global anchor
        accepted_clones = np.zeros(n_clones, dtype=np.bool_)
        if clone_spot_counts[global_anchor_clone] > 0:
            accepted_clones[global_anchor_clone] = True

        # 2. Find LOCAL diploid clones based on BAF variance (initial candidates)
        local_baf_devs = np.zeros(n_clones)
        min_local_dev = 1e9

        for c in range(n_clones):
            if clone_spot_counts[c] == 0:
                continue
            dev = 0.0
            for i in range(w_start, w_end):
                diff = np.abs(baf_profiles[c, i] - 0.5) - eps_baf
                if diff > 0:
                    dev += diff
            dev_avg = dev / w_len
            local_baf_devs[c] = dev_avg
            if dev_avg < min_local_dev:
                min_local_dev = dev_avg

        # 3. Filter candidates: check BAF margin AND check Negative Binomial Regression Proxy Amplitude
        candidate_clones = np.zeros(n_clones, dtype=np.int32)
        n_candidates = 0

        for c in range(n_clones):
            if clone_spot_counts[c] > 0 and c != global_anchor_clone:
                if local_baf_devs[c] <= min_local_dev + baf_margin:

                    # Compute pseudo-Poisson/NB weighted regression of candidate vs Global Anchor
                    # y = Candidate, x = Anchor.
                    # In count models, variance scales with mean. A 1st-order WLS sets weights ~ 1/mean.
                    sum_wxy = 0.0
                    sum_wxx = 0.0

                    for i in range(w_start, w_end):
                        x_val = norm_clone_rdr[i, global_anchor_clone]
                        y_val = norm_clone_rdr[i, c]

                        # Weight proportional to inverse expected variance
                        weight = 1.0 / (x_val + 1e-8)

                        sum_wxy += weight * x_val * y_val
                        sum_wxx += weight * x_val * x_val

                    if sum_wxx > 1e-12:
                        beta = sum_wxy / sum_wxx
                    else:
                        beta = 0.0

                    # Only accept as a candidate if amplitude difference is within 20% copy number bounds
                    if amp_min <= beta <= amp_max:
                        candidate_clones[n_candidates] = c
                        n_candidates += 1

        # 4. Calculate transcript variance specifically for surviving CLONES
        if n_candidates > 0:
            clone_stds = np.zeros(n_candidates)
            for idx in range(n_candidates):
                c = candidate_clones[idx]
                mean_val = 0.0
                for i in range(w_start, w_end):
                    mean_val += log1p_clone_rdr[i, c]
                mean_val /= w_len

                var_val = 0.0
                for i in range(w_start, w_end):
                    diff = log1p_clone_rdr[i, c] - mean_val
                    var_val += diff * diff
                var_val /= w_len
                clone_stds[idx] = np.sqrt(var_val)

            sorted_idx = np.argsort(clone_stds)

            # 5. Iteratively relax the threshold to absorb ordered clones
            total_pool_size = n_candidates + 1
            pool_stds = np.zeros(total_pool_size)

            # Get anchor variance
            mean_anchor = 0.0
            for i in range(w_start, w_end):
                mean_anchor += log1p_clone_rdr[i, global_anchor_clone]
            mean_anchor /= w_len
            var_anchor = 0.0
            for i in range(w_start, w_end):
                diff = log1p_clone_rdr[i, global_anchor_clone] - mean_anchor
                var_anchor += diff * diff
            var_anchor /= w_len

            pool_stds[0] = np.sqrt(var_anchor)
            for i in range(n_candidates):
                pool_stds[i + 1] = clone_stds[sorted_idx[i]]

            mean_all_stds = np.mean(pool_stds)

            if total_pool_size > 3:
                best_keep = total_pool_size
                percentile = start_percentile

                while percentile <= 0.95:
                    thresh_idx = int(percentile * total_pool_size)
                    if thresh_idx < 1:
                        thresh_idx = 1
                    if thresh_idx >= total_pool_size - 1:
                        thresh_idx = total_pool_size - 2

                    n_in = thresh_idx + 1
                    sum_in = 0.0
                    for i in range(n_in):
                        sum_in += pool_stds[i]
                    mean_in = sum_in / n_in

                    n_out = total_pool_size - n_in
                    sum_out = 0.0
                    for i in range(thresh_idx + 1, total_pool_size):
                        sum_out += pool_stds[i]
                    mean_out = sum_out / n_out

                    ss_w = 0.0
                    for i in range(n_in):
                        d = pool_stds[i] - mean_in
                        ss_w += d * d
                    for i in range(thresh_idx + 1, total_pool_size):
                        d = pool_stds[i] - mean_out
                        ss_w += d * d

                    d_in = mean_in - mean_all_stds
                    d_out = mean_out - mean_all_stds
                    ss_b = (n_in * d_in * d_in) + (n_out * d_out * d_out)

                    if ss_w > 1e-12:
                        if ss_b > var_ratio_thresh * ss_w:
                            best_keep = n_in
                            break

                    percentile += percentile_step

                for i in range(best_keep - 1):
                    c = candidate_clones[sorted_idx[i]]
                    accepted_clones[c] = True
            else:
                for i in range(n_candidates):
                    c = candidate_clones[i]
                    accepted_clones[c] = True

        # 6. Lock in spots from accepted clones for this segment
        accepted_count = 0
        for s in range(n_spots):
            c_idx = spot_assignments[s]
            if accepted_clones[c_idx]:
                local_mask[b, s] = True

        for c in range(n_clones):
            if accepted_clones[c]:
                accepted_count += 1

        total_accepted_clones += accepted_count

    avg_clones_included = total_accepted_clones / n_bins
    return local_mask, avg_clones_included


@njit
def _compute_baseline_core(single_X_rdr, local_normal_mask, min_normal_count_perbin):
    """
    JIT compiled core for computing baseline stats and zeroing out low-confidence
    segments in-place without generating massive intermediate matrices.
    """
    n_bins, n_spots = single_X_rdr.shape
    rdr_normal = np.zeros(n_bins, dtype=np.float64)

    # 1. Sum up normal spots per bin utilizing the 2D sliding-window mask
    for i in range(n_bins):
        b_sum = 0.0
        for j in range(n_spots):
            if local_normal_mask[i, j]:
                b_sum += single_X_rdr[i, j]
        rdr_normal[i] = b_sum

    # 2. Filter unconfident bins directly in place
    for i in range(n_bins):
        if rdr_normal[i] < min_normal_count_perbin:
            rdr_normal[i] = 0.0
            # Avoid ill-defined distributions if normal baseline falls short
            for j in range(n_spots):
                single_X_rdr[i, j] = 0.0

    # 3. Normalize rdr_normal to create a probability distribution across genomic segments
    rdr_sum = 0.0
    for i in range(n_bins):
        rdr_sum += rdr_normal[i]

    if rdr_sum > 0:
        for i in range(n_bins):
            rdr_normal[i] /= rdr_sum

    # 4. Calculate total spot coverage AFTER zeroing out unconfident segments
    spots_coverage = np.zeros(n_spots, dtype=np.float64)
    for j in range(n_spots):
        s_sum = 0.0
        for i in range(n_bins):
            s_sum += single_X_rdr[i, j]
        spots_coverage[j] = s_sum

    return rdr_normal, spots_coverage


def determine_local_normal_baseline(
    config,
    res,
    baf_profiles,
    single_X,
    single_X_rdr,
    smooth_mat=None,  # Kept to preserve exact caller parameter index compatibility
    single_tumor_prop=None,
    window_size=100,
):
    """
    Unified entrypoint replacing determine_normal_candidates and determine_normal_baseline.
    Derives segment-specific normal cells dynamically and solves for the global expected
    baseline transcripts. Returns the derived rdr matrices alongside the completed 2D mask.
    """
    logger.info("Determining segment-specific normal spots...")
    n_bins, n_spots = single_X_rdr.shape

    # 1. Override Check: Handle user-provided masks first
    if config.preprocessing.normalidx_file is not None:
        if config.preprocessing.tumorprop_file is not None:
            logger.warning(
                f"Found mixed sources for normal spot definition, assuming {config.preprocessing.normalidx_file}."
            )
        # Standard fallback to whatever downstream logic previously caught 'None'
        return None, None, None, None

    elif single_tumor_prop is not None:
        logger.info("Identifying normal spots based on provided tumor proportion.")
        global_mask = np.zeros(n_spots, dtype=np.bool_)
        for prop_threshold in np.arange(0.05, 0.6, 0.05):
            global_mask = single_tumor_prop < prop_threshold
            if np.sum(single_X_rdr[:, global_mask]) > 200 * single_X.shape[0]:
                logger.info(
                    f"Determined normal spots with sufficient UMIs based on input tumor proportion @ prop_threshold={prop_threshold}"
                )
                break
        else:
            logger.warning(
                "Failed to determine normal spots with sufficient UMIs based on input tumor proportion."
            )

        # Broadcast user's 1-dimensional array into 2-dimensional uniform mask
        local_normal_mask = np.tile(global_mask, (n_bins, 1))

    # 2. Dynamic Discovery: Run Numba optimized sliding-window
    else:
        logger.info(
            "Determining normal mask dynamically based on intra/inter-cluster statistical bounds."
        )
        local_normal_mask, avg_clones_included = compute_local_normal_mask(
            baf_profiles=baf_profiles,
            spot_assignments=np.array(res["new_assignment"], dtype=np.int32),
            single_X_rdr=single_X_rdr,
            window_size=window_size,
        )
        logger.info(
            f"Average effective number of clones included per segment: {avg_clones_included:.2f}"
        )

    # 3. Base Compute execution
    mean_sparsity = np.mean(np.sum(local_normal_mask, axis=1) == 0)
    logger.info(
        f"Found sparsity of segment-normal spot set = {100. * mean_sparsity:.3f}%"
    )

    min_norm = config.quality.min_normal_count_perbin
    rdr_normal, spots_coverage = _compute_baseline_core(
        single_X_rdr, local_normal_mask, min_norm
    )

    # Calculate normalization outer-product and report
    single_base_nb_mean = rdr_normal.reshape(-1, 1) @ spots_coverage.reshape(1, -1)
    confident_fraction = np.mean(rdr_normal > 0)

    logger.info(
        f"Found {100. * confident_fraction:.3f}% of segments with confident normal baseline for "
        f"MIN_NORMAL_COUNT_PERBIN={min_norm}"
    )

    return rdr_normal, single_X_rdr, single_base_nb_mean, local_normal_mask


'''
def filter_normal_diffexp(
    exp_counts,
    df_bininfo,
    normal_candidate,
    sample_list=None,
    sample_ids=None,
    logfcthreshold_u=2, # LogFC threshold for dropping a gene between Unsure and Normal spots.
    logfcthreshold_t=4, # LogFC threshold for dropping a gene between the Tumor and Normal spots
    quantile_threshold=80, # percentile of total UMI counts a gene must exceed to be considered for dropping.
):
    """
    Cluster input transcripts per slice into "normal" vs "tumor" spots based on pca + kmeans,
    utilizing pre-labeled "normal" candidates to identify the "normal" cluster.

    Drop gene transcripts that are differentially expressed between this "normal" cluster
˚   and the "tumor" spots based on log fold change.

    Returns new counts structure of (genomic bins x spots) after filtering genes with estimated
    differential expression, namely new_single_X_rdr.
    """
    adata = anndata.AnnData(exp_counts)
    adata.layers["count"] = exp_counts.values
    adata.obs["normal_candidate"] = normal_candidate

    map_gene_adatavar, map_gene_umi = {}, {}

    # NB gene_umis summed over spots.
    list_gene_umi = np.sum(adata.layers["count"], axis=0)

    # NB map of unique integer per gene.
    for i, x in enumerate(adata.var.index):
        map_gene_adatavar[x] = i
        map_gene_umi[x] = list_gene_umi[i]

    if sample_list is None:
        sample_list = [None]

    filtered_out_set = set()

    # NB loop over slices.
    for s, sname in enumerate(sample_list):
        if sname is None:
            index = np.arange(adata.shape[0])
        else:
            index = np.where(sample_ids == s)[0]

        # NB adata for this slice.
        tmpadata = adata[index, :].copy()

        # NB insufficient normal spot umis for this slice.
        if (
            np.sum(tmpadata.layers["count"][tmpadata.obs["normal_candidate"], :])
            < tmpadata.shape[1] * 10  # MAGIC
        ):
            logger.warning(f"TODO!")
            continue

        umi_threshold = np.percentile(
            np.sum(tmpadata.layers["count"], axis=0), quantile_threshold
        )

        # NB  filter genes based on number of cells or counts.
        #     see https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.filter_genes.html
        sc.pp.filter_genes(tmpadata, min_cells=10)

        # NB median number of umis per spot?
        med = np.median(np.sum(tmpadata.layers["count"], axis=1))

        # NB normalize such that every spot has the same total count after normalization.
        #    see https://scanpy.readthedocs.io/en/1.9.x/generated/scanpy.pp.normalize_total.html
        sc.pp.normalize_total(tmpadata, target_sum=med)

        # NB log(1 + x) transform.
        #    see https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.log1p.html
        sc.pp.log1p(tmpadata)

        # NB adds PCA representation of data: adata.obsm['X_pca' with shape (adata.n_obs, n_comps)
        #    see https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.pca.html
        sc.pp.pca(tmpadata, n_comps=4)

        # NB fit two clusters to PCA representation of data.
        kmeans = KMeans(n_clusters=2, random_state=0).fit(tmpadata.obsm["X_pca"])
        kmeans_labels = kmeans.predict(tmpadata.obsm["X_pca"])

        # NB determine which cluster corresponds to normal candidates.
        idx_kmeans_label = np.argmax(
            np.bincount(kmeans_labels[tmpadata.obs["normal_candidate"]], minlength=2)
        )

        # NB all normal candidates are "normal".
        clone = np.array(["normal"] * tmpadata.shape[0])
        clone[
            (kmeans_labels != idx_kmeans_label) & (~tmpadata.obs["normal_candidate"])
        ] = "tumor"

        # NB spots that are in the same kmeans cluster as normal candidates but not pre-labeled as "normal candidates"
        clone[
            (kmeans_labels == idx_kmeans_label) & (~tmpadata.obs["normal_candidate"])
        ] = "unsure"

        tmpadata.obs["clone"] = clone

        # NB aggregate counts per normal/tumor designation.
        agg_counts = np.vstack(
            [
                np.sum(
                    tmpadata.layers["count"][tmpadata.obs["clone"] == label, :], axis=0
                )
                for label in ["normal", "unsure", "tumor"]
            ]
        )
        agg_counts = agg_counts / np.sum(agg_counts, axis=1, keepdims=True) * 1e6

        # NB total umis per gene for genes corresponding to adata.var.index
        geneumis = np.array([map_gene_umi[x] for x in tmpadata.var.index])

        # TODO divide-by-zero errors >>>>
        # NB log fold change normal vs unsure
        logfc_u = np.where(
            ((agg_counts[1, :] == 0) | (agg_counts[0, :] == 0)),
            10,
            np.log2(agg_counts[1, :] / agg_counts[0, :]),
        )

        # NB log fold change normal vs tumor
        logfc_t = np.where(
            ((agg_counts[2, :] == 0) | (agg_counts[0, :] == 0)),
            10,
            np.log2(agg_counts[2, :] / agg_counts[0, :]),
        )
        # <<<<<
        this_filtered_out_set = set(
            list(
                tmpadata.var.index[
                    (np.abs(logfc_u) > logfcthreshold_u) & (geneumis > umi_threshold)
                ]
            )
        ) | set(
            list(
                tmpadata.var.index[
                    (np.abs(logfc_t) > logfcthreshold_t) & (geneumis > umi_threshold)
                ]
            )
        )
        filtered_out_set = filtered_out_set | this_filtered_out_set

        logger.info(
            f"Removed {len(filtered_out_set)} genes with differential expression based on normal spots."
        )

    new_single_X_rdr = np.zeros((df_bininfo.shape[0], adata.shape[0]))
    total_counts, retained_counts = 0, 0

    for b, genestr in enumerate(df_bininfo.INCLUDED_GENES.values):
        # RDR (genes)
        bin_genes = set(genestr.split(" "))
        involved_genes = bin_genes - filtered_out_set

        total_counts += np.sum(
            adata.layers["count"][:, adata.var.index.isin(bin_genes)]
        )
        retained_counts += np.sum(
            adata.layers["count"][:, adata.var.index.isin(involved_genes)]
        )

        new_single_X_rdr[b, :] = np.sum(
            adata.layers["count"][:, adata.var.index.isin(involved_genes)], axis=1
        )

    logger.info(f"Retained {100. * retained_counts / total_counts:.3f}% of bin UMIs.")

    return new_single_X_rdr, filtered_out_set
'''


def filter_normal_diffexp(
    exp_counts,
    df_bininfo,
    normal_candidate,
    sample_list=None,
    sample_ids=None,
    logfcthreshold_u=2,
    logfcthreshold_t=4,
    quantile_threshold=80,
    use_kmeans=True,
):
    """
    Cluster input transcripts per slice into "normal" vs "tumor" spots based on pca + kmeans
    (or directly via pre-defined normal candidates), dropping differentially expressed genes.

    Returns new counts structure of (genomic bins x spots) after filtering genes with estimated
    differential expression, namely new_single_X_rdr.
    """
    adata = anndata.AnnData(exp_counts)
    adata.layers["count"] = exp_counts.values
    adata.obs["normal_candidate"] = normal_candidate

    map_gene_adatavar, map_gene_umi = {}, {}

    list_gene_umi = np.sum(adata.layers["count"], axis=0)

    for i, x in enumerate(adata.var.index):
        map_gene_adatavar[x] = i
        map_gene_umi[x] = list_gene_umi[i]

    if sample_list is None:
        sample_list = [None]

    filtered_out_set = set()

    for s, sname in enumerate(sample_list):
        if sname is None:
            index = np.arange(adata.shape[0])
        else:
            index = np.where(sample_ids == s)[0]

        tmpadata = adata[index, :].copy()

        if (
            np.sum(tmpadata.layers["count"][tmpadata.obs["normal_candidate"], :])
            < tmpadata.shape[1] * 10
        ):
            logger.warning(
                f"Insufficient normal spot UMIs for slice {sname}. Skipping."
            )
            continue

        umi_threshold = np.percentile(
            np.sum(tmpadata.layers["count"], axis=0), quantile_threshold
        )

        sc.pp.filter_genes(tmpadata, min_cells=10)
        med = np.median(np.sum(tmpadata.layers["count"], axis=1))
        sc.pp.normalize_total(tmpadata, target_sum=med)
        sc.pp.log1p(tmpadata)

        # ---------------------------------------------------------
        # CLUSTERING LOGIC: K-Means vs Pre-Defined
        # ---------------------------------------------------------
        if use_kmeans:
            sc.pp.pca(tmpadata, n_comps=4)
            kmeans = KMeans(n_clusters=2, random_state=0).fit(tmpadata.obsm["X_pca"])
            kmeans_labels = kmeans.predict(tmpadata.obsm["X_pca"])

            # Log cluster proportions
            n_spots_slice = len(kmeans_labels)
            counts = np.bincount(kmeans_labels, minlength=2)
            logger.info(
                f"K-means clustering for slice {sname}: "
                f"Cluster 0: {100.0 * counts[0] / n_spots_slice:.1f}% | "
                f"Cluster 1: {100.0 * counts[1] / n_spots_slice:.1f}%"
            )

            idx_kmeans_label = np.argmax(
                np.bincount(
                    kmeans_labels[tmpadata.obs["normal_candidate"]], minlength=2
                )
            )

            # Log overlap with normal candidates
            normal_mask = tmpadata.obs["normal_candidate"].values
            n_candidates = np.sum(normal_mask)
            overlap = np.sum(kmeans_labels[normal_mask] == idx_kmeans_label)

            if n_candidates > 0:
                logger.info(
                    f"Overlap: {overlap}/{n_candidates} ({100.0 * overlap / n_candidates:.1f}%) "
                    f"of pre-defined normal candidates fell into the designated 'normal' Cluster {idx_kmeans_label}."
                )

            clone = np.array(["normal"] * tmpadata.shape[0], dtype=object)
            clone[(kmeans_labels != idx_kmeans_label) & (~normal_mask)] = "tumor"
            clone[(kmeans_labels == idx_kmeans_label) & (~normal_mask)] = "unsure"

        else:
            # Bypass K-means entirely, rely on pre-labeled normal candidates
            logger.info(
                f"Skipping K-means for slice {sname}. Using pre-defined normal candidates directly."
            )
            normal_mask = tmpadata.obs["normal_candidate"].values

            clone = np.array(["tumor"] * tmpadata.shape[0], dtype=object)
            clone[normal_mask] = "normal"
            # Note: There are no "unsure" spots in this mode.

        tmpadata.obs["clone"] = clone

        # ---------------------------------------------------------
        # DIFFERENTIAL EXPRESSION CALCULATION
        # ---------------------------------------------------------
        agg_counts = np.vstack(
            [
                np.sum(
                    tmpadata.layers["count"][tmpadata.obs["clone"] == label, :], axis=0
                )
                for label in ["normal", "unsure", "tumor"]
            ]
        )
        # Avoid division by zero on empty slices/clusters
        row_sums = np.sum(agg_counts, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        agg_counts = agg_counts / row_sums * 1e6

        geneumis = np.array([map_gene_umi[x] for x in tmpadata.var.index])

        has_unsure = np.sum(tmpadata.obs["clone"] == "unsure") > 0
        has_tumor = np.sum(tmpadata.obs["clone"] == "tumor") > 0

        # Safe logFC for unsure
        if has_unsure:
            logfc_u = np.where(
                ((agg_counts[1, :] == 0) | (agg_counts[0, :] == 0)),
                10,
                np.log2(agg_counts[1, :] / agg_counts[0, :]),
            )
        else:
            logfc_u = np.zeros(tmpadata.shape[1])

        # Safe logFC for tumor
        if has_tumor:
            logfc_t = np.where(
                ((agg_counts[2, :] == 0) | (agg_counts[0, :] == 0)),
                10,
                np.log2(agg_counts[2, :] / agg_counts[0, :]),
            )
        else:
            logfc_t = np.zeros(tmpadata.shape[1])

        this_filtered_out_set = set(
            list(
                tmpadata.var.index[
                    (np.abs(logfc_u) > logfcthreshold_u) & (geneumis > umi_threshold)
                ]
            )
        ) | set(
            list(
                tmpadata.var.index[
                    (np.abs(logfc_t) > logfcthreshold_t) & (geneumis > umi_threshold)
                ]
            )
        )
        filtered_out_set = filtered_out_set | this_filtered_out_set

        logger.info(
            f"Removed {len(this_filtered_out_set)} genes with differential expression for slice {sname}. "
            f"Total unique removed across run: {len(filtered_out_set)}."
        )

    new_single_X_rdr = np.zeros((df_bininfo.shape[0], adata.shape[0]))
    total_counts, retained_counts = 0, 0

    for b, genestr in enumerate(df_bininfo.INCLUDED_GENES.values):
        bin_genes = set(genestr.split(" "))
        involved_genes = bin_genes - filtered_out_set

        total_counts += np.sum(
            adata.layers["count"][:, adata.var.index.isin(bin_genes)]
        )
        retained_counts += np.sum(
            adata.layers["count"][:, adata.var.index.isin(involved_genes)]
        )

        new_single_X_rdr[b, :] = np.sum(
            adata.layers["count"][:, adata.var.index.isin(involved_genes)], axis=1
        )

    logger.info(
        f"Retained {100. * retained_counts / max(total_counts, 1):.3f}% of bin UMIs."
    )

    return new_single_X_rdr


MIN_BETABINOM_TAU = 30
"""`cnamaste`'s floor on the fitted concentration, carried across unchanged.

The fit's success probability is discarded and replaced by 0.5 and its
concentration is floored here, both in `cnamaste` and in this patch: #38 owns
whether that is the right prior, and a patch that changed it would be
answering a different question from the one it is measuring.
"""


def _log_mass(
    index: np.ndarray, totals: np.ndarray, alpha: float, beta: float
) -> np.ndarray:
    """`scipy`'s own beta-binomial log mass function, evaluated elementwise.

    Written out rather than called because `scipy` reaches it one
    distribution at a time; the formula is `betabinom._logpmf`, unchanged.
    """
    return np.asarray(
        -np.log(totals + 1)
        - scipy.special.betaln(totals - index + 1, index + 1)
        + scipy.special.betaln(index + alpha, totals - index + beta)
        - scipy.special.betaln(alpha, beta)
    )


TERM_BUDGET = 1 << 16
"""How many mass-function terms are held at once: 65,536, about 5 MB of peak.

The summation covers every bin, so its intermediate is the whole ragged
evaluation -- `sum(min(k, n - k))` terms, which at a slide's read depth is
tens of millions and hundreds of megabytes. Bins are taken in groups under
this budget instead, which bounds the peak at a constant and leaves the
arithmetic identical: `np.add.reduceat` never sums across bins, so where the
groups fall cannot change a result, and the values are bitwise the same at
every budget measured.

**The small budget is also the fast one.** At 20,000 reads per bin: 112 ms and
167 MB at four million terms, 72.7 ms and 4.6 MB at this one. The working set
fits in cache, so chunking buys time rather than trading it for memory --
which is why the budget is set here rather than at the largest size that fits.

One bin whose own range exceeds the budget is still evaluated whole.
Splitting it would mean summing its parts and adding them, which is a
different association from what the values are reported against.
"""


def _chunks(lengths: np.ndarray) -> Iterator[np.ndarray]:
    """Bin indices, in groups whose summation stays under `TERM_BUDGET`."""
    start, running = 0, 0

    for index, length in enumerate(lengths):
        if running and running + int(length) > TERM_BUDGET:
            yield np.arange(start, index)
            start, running = index, 0

        running += int(length)

    if start < len(lengths):
        yield np.arange(start, len(lengths))


def _log_tables(max_total: int, alpha: float, beta: float) -> tuple[np.ndarray, ...]:
    """Log-factorial and log-rising-factorial tables up to `max_total`.

    Every term of a beta-binomial mass function is four log-gamma values at
    integer offsets from `1`, `alpha` and `beta`:

    ```
    log pmf(i) = logGamma(n + 1)  - logGamma(i + 1)     - logGamma(n - i + 1)
               + logGamma(i + a)  + logGamma(n - i + b) - logGamma(n + a + b)
               - betaln(a, b)
    ```

    Those offsets are consecutive integers, so each family is a cumulative sum
    of logarithms built once in `O(max_total)` and read thereafter by index.
    `scipy` evaluates `betaln` twice per term instead, which is six log-gamma
    calls where this is four gathers.

    The tables are `(max_total + 2)` doubles each, three of them -- half a
    megabyte at a slide's read depth, against the sum itself.
    """
    steps = np.arange(1, max_total + 2, dtype=float)

    log_factorial = np.concatenate(([0.0], np.cumsum(np.log(steps))))
    rising = [
        np.concatenate(([0.0], np.cumsum(np.log(shift + np.arange(max_total + 1)))))
        for shift in (alpha, beta)
    ]

    return log_factorial, rising[0], rising[1]


def _log_mass_tabulated(
    index: np.ndarray,
    totals: np.ndarray,
    alpha: float,
    beta: float,
    tables: tuple[np.ndarray, ...],
) -> np.ndarray:
    """`_log_mass`, read off the tables rather than evaluated.

    `logGamma(a) + logGamma(b) - betaln(a, b)` is `logGamma(a + b)`, which is
    why neither appears below: the two constants cancel into one.
    """
    log_factorial, rising_alpha, rising_beta = tables
    complement = totals - index

    return np.asarray(
        log_factorial[totals]
        - log_factorial[index]
        - log_factorial[complement]
        + rising_alpha[index]
        + rising_beta[complement]
        + scipy.special.gammaln(alpha + beta)
        - scipy.special.gammaln(totals + alpha + beta)
    )


def _ragged_index(
    starts: np.ndarray, lengths: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """One flat index array over ragged ranges, and which range each came from.

    `[start_j, start_j + length_j)` laid end to end, so the whole ragged
    evaluation is one `numpy` call and one segmented sum instead of one call
    per bin.
    """
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    segment = np.repeat(np.arange(len(lengths)), lengths)
    flat = (
        np.arange(offsets[-1], dtype=np.int64) - offsets[:-1][segment] + starts[segment]
    )

    return flat, segment


def cumulative_and_mass(
    counts: np.ndarray, totals: np.ndarray, alpha: float, beta: float
) -> tuple[np.ndarray, np.ndarray]:
    """`cdf(counts)` and `pmf(counts)` for every bin, in one vectorized sweep.

    **`scipy` has no vectorized beta-binomial distribution function.**
    `betabinom.cdf` goes through `_cdf_single`, which sums the mass function
    from zero for one element at a time under `np.vectorize`, so a call over
    `B` bins is `B` Python-level calls. After #175 removed the quantile
    inversion, these two calls are **90 per cent of what the filter costs**:
    2.50 s of 2.76 s at 2,500 spots and 400 bins.

    Three things change and none of them is the arithmetic:

    *   **One call.** Every bin's summation range is laid end to end, the mass
        function is evaluated once over the concatenation, and the sums come
        back from `np.add.reduceat`.
    *   **Tabulated log-gammas.** The mass function's four log-gamma values
        sit at integer offsets from `1`, `alpha` and `beta`, so each family is
        one cumulative sum of logarithms and every term is four gathers. That
        is what makes the ratio hold as the read depth grows, where the single
        call alone does not.
    *   **A bounded intermediate.** The bins are taken in groups under
        `TERM_BUDGET`, so the peak is a constant rather than the whole ragged
        evaluation. `scipy` holds one bin's range at a time, and a vectorized
        rewrite that held every bin's at once would buy time with memory.
    *   **The shorter tail.** `cdf(k) = 1 - sf(k)`, so a bin sums
        `min(k + 1, n - k)` terms rather than `k + 1`. The B-allele count of a
        diploid bin sits near `n / 2`, which is exactly where the saving is
        least and where it is still a factor of two on any bin above it.
    *   **One sweep for both.** The caller needs `cdf(k)` and `cdf(k - 1)`,
        which differ by `pmf(k)`, so the second comes from the first for the
        cost of one term rather than a second summation.

    **This is a tolerance and not an identity.** The terms are summed in a
    different order and, on the upper branch, subtracted from one, so the
    values agree to floating point rather than bitwise. The **mask** the
    caller builds from them is asserted bitwise against `cnamaste`; the values
    are asserted to `1e-12`.
    """
    counts = np.asarray(counts, dtype=np.int64)
    totals = np.asarray(totals, dtype=np.int64)

    tables = _log_tables(int(totals.max(initial=0)), alpha, beta)

    mass = np.where(
        (counts >= 0) & (counts <= totals),
        np.exp(
            _log_mass_tabulated(np.clip(counts, 0, totals), totals, alpha, beta, tables)
        ),
        0.0,
    )

    # NB sum whichever tail is shorter; `upper` sums (k, n] and is subtracted
    #    from one, `lower` sums [0, k].
    upper = counts + 1 > totals - counts
    starts = np.where(upper, counts + 1, 0)
    lengths = np.where(upper, totals - counts, counts + 1)
    lengths = np.clip(lengths, 0, None)

    sums = np.zeros(len(counts), dtype=float)

    for group in _chunks(lengths):
        nonempty = group[lengths[group] > 0]

        if not nonempty.size:
            continue

        spans = lengths[nonempty]
        flat, segment = _ragged_index(starts[nonempty], spans)
        terms = np.exp(
            _log_mass_tabulated(flat, totals[nonempty][segment], alpha, beta, tables)
        )

        sums[nonempty] = np.add.reduceat(
            terms, np.concatenate(([0], np.cumsum(spans)[:-1]))
        )

    cumulative = np.where(upper, 1.0 - sums, sums)

    return np.clip(cumulative, 0.0, 1.0), mass


DECISION_MARGIN = 1.0e-8
"""How close to a threshold a bin has to be before `scipy` decides it.

`cumulative_and_mass` sums the mass function in a different order from
`scipy`, so its values agree to floating point rather than bitwise --
`3e-9` at the deepest size measured. Both comparisons below are **strict**,
so a bin whose distribution function sits within that of a threshold could
fall either way on rounding alone, and the mask is what decides whether a
genomic bin survives.

The margin is wider than the largest error measured, and the bins inside it
are recomputed with `scipy` itself, so the answer is `cnamaste`'s wherever the
decision is close and the fast path's wherever it is not. On the fixtures here
the margin catches nothing; the case it exists for is an exact tie, which a
symmetric beta-binomial reaches at its midpoint against a threshold of `0.5`.
"""


def _settle_near_thresholds(
    counts: np.ndarray,
    totals: np.ndarray,
    alpha: float,
    beta: float,
    confidence_interval: tuple[float, float],
    below: np.ndarray,
    at_or_below: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-decide the bins within `DECISION_MARGIN` of a threshold, with `scipy`.

    One `scipy` call over the uncertain bins, which is empty on every instance
    measured -- so this costs a comparison and buys back the one thing the
    faster summation could change.
    """
    uncertain = np.flatnonzero(
        (np.abs(below - confidence_interval[0]) <= DECISION_MARGIN)
        | (np.abs(at_or_below - confidence_interval[1]) <= DECISION_MARGIN)
    )

    if not uncertain.size:
        return below, at_or_below

    logger.info(
        f"Deciding {uncertain.size} bin(s) within {DECISION_MARGIN} of a "
        "confidence threshold with scipy."
    )

    below, at_or_below = below.copy(), at_or_below.copy()
    below[uncertain] = scipy.stats.betabinom.cdf(
        counts[uncertain], totals[uncertain], alpha, beta
    )
    at_or_below[uncertain] = scipy.stats.betabinom.cdf(
        counts[uncertain] - 1, totals[uncertain], alpha, beta
    )

    return below, at_or_below


def removal_indicator(
    counts: np.ndarray,
    totals: np.ndarray,
    alpha: float,
    beta: float,
    confidence_interval: tuple[float, float],
) -> np.ndarray:
    """Which bins fall outside the interval, without inverting the quantile.

    `cnamaste` writes

    ```python
    counts < scipy.stats.betabinom.ppf(lo, totals, alpha, beta)
    counts > scipy.stats.betabinom.ppf(hi, totals, alpha, beta)
    ```

    and `scipy` has no closed-form inverse for a beta-binomial, so each element
    goes through `_drv2_ppfsingle`, a bisection whose every step evaluates the
    distribution function by summing the mass function from zero. On the dev
    instance's 40 bins the two calls make **1,219** mass-function evaluations
    and cost 1.44 s.

    The two comparisons below are the same predicates. For a discrete law
    `ppf(q) = min{k : cdf(k) >= q}`, so `x < ppf(q)` is exactly `cdf(x) < q`,
    and `x <= ppf(q)` is exactly `cdf(x - 1) < q`, whose negation is the
    second. **Both are equalities rather than approximations**, which is why
    the mask is bitwise `cnamaste`'s and not within a tolerance -- the
    equivalence test asserts exactly that.
    """
    below, mass = cumulative_and_mass(counts, totals, alpha, beta)
    at_or_below = np.clip(below - mass, 0.0, 1.0)

    below, at_or_below = _settle_near_thresholds(
        counts, totals, alpha, beta, confidence_interval, below, at_or_below
    )

    return np.asarray(
        (below < confidence_interval[0]) | (at_or_below >= confidence_interval[1]),
        dtype=bool,
    )


def normal_baf_bin_filter(
    df_gene_snp: Any,
    single_X: np.ndarray,
    single_base_nb_mean: np.ndarray,
    single_total_bb_RD: np.ndarray,
    nu: float,  # noqa: ARG001
    logphase_shift: float,  # noqa: ARG001
    index_normal: np.ndarray,
    geneticmap_file: Any,  # noqa: ARG001
    confidence_interval: tuple[float, float] | None = None,
    min_betabinom_tau: int = MIN_BETABINOM_TAU,
) -> tuple[Any, SpatioGenomicCounts]:
    """What `cnamaste`'s returns, with the quantile inversion replaced.

    Everything but `removal_indicator` is `cnamaste`'s, in its order: the pooled
    counts, the one-state fit, the two patched parameters, the renumbering of
    the survivors and the per-chromosome lengths. `nu`, `logphase_shift` and
    `geneticmap_file` are accepted and unused, as upstream -- the docstring
    promises a `log_sitewise_transmat` the function has never returned, and
    `run_cnamaste` calls `get_sitewise_transmat` itself on the next line.
    """
    if confidence_interval is None:
        confidence_interval = ast.literal_eval(
            get_global_config().quality.normal_allele_specific_confidence
        )

    assert confidence_interval is not None

    pooled_counts = np.sum(single_X[:, 1, index_normal], axis=1)
    pooled_totals = np.sum(single_total_bb_RD[:, index_normal], axis=1)

    model = Weighted_BetaBinom(
        pooled_counts,
        np.ones(len(pooled_counts)),
        weights=np.ones(len(pooled_counts)),
        exposure=pooled_totals,
    )
    fitted = model.fit(**get_em_solver_params())

    # NB the fitted success probability is discarded for 0.5 and the
    #    concentration floored, both as upstream. #38 owns whether it should be.
    fitted.params[0] = 0.5
    fitted.params[-1] = max(fitted.params[-1], min_betabinom_tau)

    alpha = fitted.params[0] * fitted.params[1]
    beta = (1.0 - fitted.params[0]) * fitted.params[1]

    removal = removal_indicator(
        pooled_counts, pooled_totals, alpha, beta, confidence_interval
    )

    index_removal = np.where(removal)[0]
    index_remaining = np.where(~removal)[0]

    logger.info(
        f"Removing {np.count_nonzero(removal)} [{100.0 * np.mean(removal):.4f}]% "
        f"genomic segments with potential allele-specific expression, based on "
        f"normal candidates --- confidence={confidence_interval} and "
        f"min_betabinom_tau={min_betabinom_tau}."
    )

    column = np.where(df_gene_snp.columns == "bin_id")[0][0]
    df_gene_snp.iloc[np.where(df_gene_snp.bin_id.isin(index_removal))[0], column] = None

    df_gene_snp["bin_id"] = df_gene_snp["bin_id"].map(
        {x: i for i, x in enumerate(index_remaining)}
    )
    df_gene_snp.bin_id = df_gene_snp.bin_id.astype("Int64")

    if df_gene_snp.bin_id.isnull().any():
        logger.warning(
            f"Detected {df_gene_snp.bin_id.isnull().sum()} bin_ids with NaN value."
        )

    single_X = single_X[index_remaining, :, :]
    single_base_nb_mean = single_base_nb_mean[index_remaining, :]
    single_total_bb_RD = single_total_bb_RD[index_remaining, :]

    lengths = np.zeros(len(df_gene_snp.CHR.unique()), dtype=int)

    for i, contig in enumerate(df_gene_snp.CHR.unique()):
        lengths[i] = len(
            df_gene_snp[
                (contig == df_gene_snp.CHR) & (~df_gene_snp.bin_id.isnull())
            ].bin_id.unique()
        )

    assert df_gene_snp["bin_id"].nunique(dropna=True) == single_X.shape[0]
    assert df_gene_snp["bin_id"].nunique(dropna=True) == sum(lengths)

    return df_gene_snp, SpatioGenomicCounts(
        lengths, single_X, single_base_nb_mean, single_total_bb_RD
    )


