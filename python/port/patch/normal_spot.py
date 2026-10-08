"""Replaces `cnaster.normal_spot.normal_baf_bin_filter` and `filter_normal_diffexp`.

The bin filter tests `cdf(x) < q` and `cdf(x - 1) >= q` in place of comparing
against `betabinom.ppf(q)`: equal for a discrete law, so the mask is bitwise
(#174). `filter_normal_diffexp` is fixed and reconnected (#440).
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from typing import Any

import cnaster.normal_spot
import numpy as np
import scipy.special
import scipy.stats
from cnaster.config import get_global_config, start_time
from cnaster.hmm_emission import Weighted_BetaBinom
from cnaster.hmm_utils import get_em_solver_params
from cnaster.logger import get_logger
from cnaster.spatio_genomic_counts import SpatioGenomicCounts

from port.extensions.segments import observe

_UPSTREAM_CANDIDATES = cnaster.normal_spot.determine_normal_candidates
"""`cnaster`'s own, bound at import, before `patched()` rebinds the name (#479)."""

logger = get_logger(__name__, start_time=start_time)

MIN_BETABINOM_TAU = 30
"""`cnaster`'s floor on the fitted concentration, unchanged (#38)."""


def _log_mass(
    index: np.ndarray, totals: np.ndarray, alpha: float, beta: float
) -> np.ndarray:
    """`scipy`'s `betabinom._logpmf`, evaluated elementwise."""
    return np.asarray(
        -np.log(totals + 1)
        - scipy.special.betaln(totals - index + 1, index + 1)
        + scipy.special.betaln(index + alpha, totals - index + beta)
        - scipy.special.betaln(alpha, beta)
    )


TERM_BUDGET = 1 << 16
"""Mass-function terms held at once; bins are grouped under it, results unchanged.

A bin whose own range exceeds the budget is evaluated whole.
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
    """Log-factorial and log-rising-factorial (from `alpha`, `beta`) tables to `max_total`."""
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
    """`_log_mass`, read off `_log_tables`."""
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
    """Concatenated `[start_j, start_j + length_j)` ranges, and each element's range."""
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

    Sums the shorter tail from tabulated log-gammas, bins grouped under
    `TERM_BUDGET`. Agrees with `scipy` to `1e-12`, not bitwise.
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
"""Bins whose distribution function is this close to a threshold are decided by `scipy`."""


def _settle_near_thresholds(
    counts: np.ndarray,
    totals: np.ndarray,
    alpha: float,
    beta: float,
    confidence_interval: tuple[float, float],
    below: np.ndarray,
    at_or_below: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-decide the bins within `DECISION_MARGIN` of a threshold, with `scipy`."""
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
    """Which bins fall outside `confidence_interval`, without inverting the quantile.

    Bitwise `cnaster`'s mask; a closed end (`lo <= 0`, `hi >= 1`) removes
    nothing on that side, as `scipy`'s support does (#332).
    """
    below, mass = cumulative_and_mass(counts, totals, alpha, beta)
    at_or_below = np.clip(below - mass, 0.0, 1.0)

    below, at_or_below = _settle_near_thresholds(
        counts, totals, alpha, beta, confidence_interval, below, at_or_below
    )

    lo, hi = confidence_interval
    low = below < lo if lo > 0.0 else np.zeros(counts.shape, dtype=bool)
    high = at_or_below >= hi if hi < 1.0 else np.zeros(counts.shape, dtype=bool)

    return np.asarray(low | high, dtype=bool)


def determine_normal_candidates(
    config: Any,
    res: Any,
    baf_profiles: Any,
    single_X: Any,
    single_X_rdr: Any,
    smooth_mat: Any,
    single_tumor_prop: Any = None,
) -> Any:
    """`cnaster`'s, returning the named spots when `normalidx_file` is set (#479).

    Upstream returns `None` there; this returns `port.patch.io.NORMAL_SPOTS`.
    """
    if config.preprocessing.normalidx_file is None:
        return _UPSTREAM_CANDIDATES(
            config,
            res,
            baf_profiles,
            single_X,
            single_X_rdr,
            smooth_mat,
            single_tumor_prop=single_tumor_prop,
        )

    from port.patch.io import NORMAL_SPOTS

    if not NORMAL_SPOTS:
        msg = (
            "preprocessing.normalidx_file is set but the loader recorded no "
            "normal spots: was the data loaded through port's load_input_data?"
        )
        raise RuntimeError(msg)

    return NORMAL_SPOTS[0].copy()


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
    """`cnaster`'s, with the quantile inversion replaced by `removal_indicator`.

    Departures: a chromosome with every bin removed is absent from `lengths`
    rather than 0 (#466); a removed bin's genes get `is_interval = False` (#105).
    `nu`, `logphase_shift` and `geneticmap_file` are unused, as upstream.
    """
    if confidence_interval is None:
        confidence_interval = ast.literal_eval(
            get_global_config().quality.normal_allele_specific_confidence
        )

    if confidence_interval is None:  # invariant
        msg = "expected confidence_interval is not None"
        raise AssertionError(msg)

    pooled_counts = np.sum(single_X[:, 1, index_normal], axis=1)
    pooled_totals = np.sum(single_total_bb_RD[:, index_normal], axis=1)

    model = Weighted_BetaBinom(
        pooled_counts,
        np.ones(len(pooled_counts)),
        weights=np.ones(len(pooled_counts)),
        exposure=pooled_totals,
    )
    fitted = model.fit(**get_em_solver_params())

    # NB success probability set to 0.5 and concentration floored, as upstream (#38).
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
    removed = np.where(df_gene_snp.bin_id.isin(index_removal))[0]
    df_gene_snp.iloc[removed, column] = None

    if removed.size and "is_interval" in df_gene_snp.columns:
        interval = np.where(df_gene_snp.columns == "is_interval")[0][0]
        df_gene_snp.iloc[removed, interval] = False

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

    # NB a contig with every bin removed is absent rather than zero (#438).
    lengths = observe(df_gene_snp, "bin_id", "bins-filtered").lengths

    if df_gene_snp["bin_id"].nunique(dropna=True) != single_X.shape[0]:  # invariant
        msg = 'expected df_gene_snp["bin_id"].nunique(dropna=True) == single_X.shape[0]'
        raise AssertionError(msg)
    if df_gene_snp["bin_id"].nunique(dropna=True) != sum(lengths):  # invariant
        msg = 'expected df_gene_snp["bin_id"].nunique(dropna=True) == sum(lengths)'
        raise AssertionError(msg)

    return df_gene_snp, SpatioGenomicCounts(
        lengths, single_X, single_base_nb_mean, single_total_bb_RD
    )


# -- `filter_normal_diffexp` (#440) ----------------------------------------
#
# Fixes `cnaster`'s two defects: genes split on `" "` where the binner joins
# with `","` (#165), and a result `run_cnaster` overwrites unread (#177). The
# flagged genes are recorded on the run's lineage, which excludes them from
# later bin sums.


def flagged_genes(
    exp_counts: Any,
    normal_candidate: Any,
    sample_list: Any = None,
    sample_ids: Any = None,
    logfcthreshold_u: float = 2,
    logfcthreshold_t: float = 4,
    quantile_threshold: float = 80,
    use_kmeans: bool = True,
) -> set[str]:
    """The genes `cnaster`'s filter removes, as `normal_spot.py:727-887` selects them."""
    import anndata
    import scanpy as sc
    from sklearn.cluster import KMeans

    adata = anndata.AnnData(exp_counts)
    adata.layers["count"] = exp_counts.values
    adata.obs["normal_candidate"] = normal_candidate

    gene_umi = dict(
        zip(adata.var.index, np.sum(adata.layers["count"], axis=0), strict=True)
    )

    if sample_list is None:
        sample_list = [None]

    flagged: set[str] = set()

    for s, name in enumerate(sample_list):
        index = (
            np.arange(adata.shape[0]) if name is None else np.where(sample_ids == s)[0]
        )
        sample: Any = adata[index, :].copy()
        normal = sample.obs["normal_candidate"]

        if np.sum(sample.layers["count"][normal, :]) < sample.shape[1] * 10:
            continue

        umi_threshold = np.percentile(
            np.sum(sample.layers["count"], axis=0), quantile_threshold
        )

        sc.pp.filter_genes(sample, min_cells=10)
        median = np.median(np.sum(sample.layers["count"], axis=1))
        sc.pp.normalize_total(sample, target_sum=median)
        sc.pp.log1p(sample)

        normal_mask = sample.obs["normal_candidate"].to_numpy()

        if use_kmeans:
            sc.pp.pca(sample, n_comps=4)
            kmeans = KMeans(n_clusters=2, random_state=0).fit(sample.obsm["X_pca"])
            labels = kmeans.predict(sample.obsm["X_pca"])
            normal_label = np.argmax(np.bincount(labels[normal_mask], minlength=2))

            clone = np.array(["normal"] * sample.shape[0], dtype=object)
            clone[(labels != normal_label) & (~normal_mask)] = "tumor"
            clone[(labels == normal_label) & (~normal_mask)] = "unsure"
        else:
            clone = np.array(["tumor"] * sample.shape[0], dtype=object)
            clone[normal_mask] = "normal"

        aggregated = np.vstack(
            [
                np.sum(sample.layers["count"][clone == label, :], axis=0)
                for label in ["normal", "unsure", "tumor"]
            ]
        )
        totals = np.sum(aggregated, axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        aggregated = aggregated / totals * 1e6

        umis = np.array([gene_umi[x] for x in sample.var.index])

        logfc_u = _logfc(aggregated, 1, present=bool(np.any(clone == "unsure")))
        logfc_t = _logfc(aggregated, 2, present=bool(np.any(clone == "tumor")))

        flagged |= set(
            sample.var.index[
                (np.abs(logfc_u) > logfcthreshold_u) & (umis > umi_threshold)
            ]
        ) | set(
            sample.var.index[
                (np.abs(logfc_t) > logfcthreshold_t) & (umis > umi_threshold)
            ]
        )

    return flagged


def _logfc(aggregated: np.ndarray, row: int, *, present: bool) -> np.ndarray:
    """`log2` of row `row` over the normal row, 10 where either is zero; zeros if absent."""
    if not present:
        return np.zeros(aggregated.shape[1])

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio: np.ndarray = np.where(
            (aggregated[row, :] == 0) | (aggregated[0, :] == 0),
            10,
            np.log2(aggregated[row, :] / aggregated[0, :]),
        )
    return ratio


def _genes_of(text: str) -> list[str]:
    """A bin's genes from `INCLUDED_GENES`, `","`-joined as the binner writes it."""
    return [gene for gene in text.replace(" ", ",").split(",") if gene]


def filter_normal_diffexp(
    exp_counts: Any,
    df_bininfo: Any,
    normal_candidate: Any,
    sample_list: Any = None,
    sample_ids: Any = None,
    logfcthreshold_u: float = 2,
    logfcthreshold_t: float = 4,
    quantile_threshold: float = 80,
    use_kmeans: bool = True,
) -> np.ndarray:
    """`(n_bins, n_spots)` read depth without the flagged genes; the genes recorded.

    `cnaster`'s signature and return. Genes are recorded only inside
    `port.extensions.segments.recording()` (#177, #617); outside it the
    filter is inert, as in `cnaster`.
    """
    import scipy.sparse as sp

    from port.extensions.segments import current

    flagged = flagged_genes(
        exp_counts,
        normal_candidate,
        sample_list=sample_list,
        sample_ids=sample_ids,
        logfcthreshold_u=logfcthreshold_u,
        logfcthreshold_t=logfcthreshold_t,
        quantile_threshold=quantile_threshold,
        use_kmeans=use_kmeans,
    )

    lineage = current()
    if lineage is not None:
        lineage.excluded_genes |= flagged

    genes = np.asarray(exp_counts.columns)
    column = {gene: i for i, gene in enumerate(genes)}
    kept = np.array([gene not in flagged for gene in genes])

    rows, columns = [], []
    for b, text in enumerate(df_bininfo.INCLUDED_GENES.to_numpy()):
        for gene in _genes_of(text):
            if gene in column:
                rows.append(b)
                columns.append(column[gene])

    membership = sp.csr_matrix(
        (np.ones(len(rows)), (rows, columns)), shape=(len(df_bininfo), genes.size)
    )
    counts = np.asarray(exp_counts.to_numpy(), dtype=np.float64) * kept
    retained: np.ndarray = np.asarray(membership @ counts.T)

    return retained
