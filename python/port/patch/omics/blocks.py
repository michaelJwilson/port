"""Replaces `cnaster.omics`' gene/SNP table and block/bin summaries, vectorized (#190).

Python walks and per-block slices become window searches, `searchsorted` and
sparse indicator products; outputs are bitwise `cnaster`'s except as stated
per function (#466, #105, #551). `form_gene_snp_table` is undecorated (no `@cacher`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from cnaster.config import start_time
from cnaster.logger import get_logger
from cnaster.omics import assign_initial_blocks as _UPSTREAM_ASSIGN_INITIAL_BLOCKS
from cnaster.omics import create_bin_ranges as _UPSTREAM_CREATE_BIN_RANGES
from cnaster.spatio_genomic_counts import SpatioGenomicCounts

from port.extensions.segments import Segmentation, current, observe
from port.patch.reference import get_reference_genes

logger = get_logger(__name__, start_time=start_time)


def preceding_gene(
    chromosome: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    is_interval: np.ndarray,
    unassigned: np.ndarray,
    num_preceeding_rows: int,
) -> np.ndarray:
    """For each unassigned row, the nearest preceding gene containing it, or `-1`.

    One pass per offset up to `num_preceeding_rows`; the smallest offset wins.
    Requires the table sorted by `(CHR, START)`.
    """
    rows = np.flatnonzero(unassigned)
    found = np.full(rows.size, -1, dtype=np.int64)

    if not rows.size:
        return found

    position = start[rows]

    for offset in range(1, num_preceeding_rows + 1):
        candidate = rows - offset
        open_rows = (candidate >= 0) & (found < 0)

        if not open_rows.any():
            break

        safe = np.where(open_rows, candidate, 0)
        hit = (
            open_rows
            & is_interval[safe]
            & (chromosome[safe] == chromosome[rows])
            & (start[safe] <= position)
            & (end[safe] > position)
        )
        found = np.where(hit, safe, found)

    return found


def form_gene_snp_table(
    unique_snp_ids: np.ndarray,
    hgtable_file: str,
    adata: Any,
    verbose: bool = False,  # noqa: ARG001 -- upstream's signature, and unused there too
    num_preceeding_rows: int = 100,
) -> Any:
    """`cnaster.omics.form_gene_snp_table` without the per-SNP walk; `verbose` is unused, as upstream."""
    logger.info("Forming gene & snp meta data.")

    # NB `port.patch.reference`: bitwise the same frame, read with `polars` (#185).
    df_gene = get_reference_genes(hgtable_file)

    common_genes = set(df_gene.gene) & set(adata.var.index)

    logger.info(
        f"Found {100.0 * len(common_genes) / len(adata.var.index):.2f}% of "
        "visium genes to be in reference."
    )

    df_gene = df_gene[df_gene.gene.isin(adata.var.index)]

    # NB `{contig}_{pos}_{ref}_{alt}`, parsed as upstream parses it.
    parsed = np.array(
        [identifier.split("_")[:2] for identifier in unique_snp_ids], dtype=np.int64
    ).reshape(-1, 2)
    snp_chr, snp_pos = parsed[:, 0], parsed[:, 1]

    df_gene_snp = pd.concat(
        [
            df_gene,
            pd.DataFrame(
                {
                    "CHR": snp_chr,
                    "START": snp_pos,
                    "END": snp_pos + 1,
                    "snp_id": unique_snp_ids,
                    "gene": None,
                    "is_interval": False,
                }
            ),
        ],
        ignore_index=True,
    )

    df_gene_snp = df_gene_snp.sort_values(by=["CHR", "START"])

    unassigned = df_gene_snp.gene.isna().to_numpy()
    found = preceding_gene(
        df_gene_snp.CHR.to_numpy(),
        df_gene_snp.START.to_numpy(),
        df_gene_snp.END.to_numpy(),
        df_gene_snp.is_interval.to_numpy(),
        unassigned,
        num_preceeding_rows,
    )

    genes = df_gene_snp.gene.to_numpy(copy=True)
    rows = np.flatnonzero(unassigned)
    matched = found >= 0

    genes[rows[matched]] = genes[found[matched]]
    df_gene_snp["gene"] = genes

    isin = ~df_gene_snp.gene.isna()

    logger.info(
        f"Retaining {100.0 * np.mean(isin[~df_gene_snp.is_interval]):.3f}% of snps "
        "with matched gene (given reference filtered by visium panel) for "
        f"num_preceeding_rows={num_preceeding_rows}."
    )

    return df_gene_snp[isin]


def merged_gene_intervals(
    chromosome: np.ndarray, start: np.ndarray, end: np.ndarray
) -> np.ndarray:
    """Where each run of overlapping gene intervals begins, sorted by `(CHR, START)`.

    A sweep against the running maximum of ends, reset per chromosome.
    """
    if not chromosome.size:
        return np.zeros(0, dtype=np.int64)

    new_chromosome = np.concatenate(([True], chromosome[1:] != chromosome[:-1]))
    bounds = np.flatnonzero(new_chromosome)

    reach = np.empty_like(end)

    for first, last in zip(bounds, np.append(bounds[1:], end.size), strict=True):
        reach[first:last] = np.maximum.accumulate(end[first:last])

    disjoint = np.concatenate(([True], start[1:] >= reach[:-1]))

    return np.flatnonzero(new_chromosome | disjoint)


def block_of_row(
    chromosome: np.ndarray, start: np.ndarray, interval_row: np.ndarray
) -> np.ndarray:
    """Which merged interval each row of the sorted table falls in: one `searchsorted`."""
    keys = chromosome.astype(np.int64) * (1 << 40) + start.astype(np.int64)
    edges = keys[interval_row]

    # NB a row on an interval's edge belongs to that interval (genes sort first).
    placed = np.searchsorted(edges, keys, side="right") - 1

    return np.clip(placed, 0, None)


def assign_initial_blocks(
    df_gene_snp: Any,
    adata: Any,
    cell_snp_Aallele: Any,
    cell_snp_Ballele: Any,
    unique_snp_ids: np.ndarray,
    initial_min_umi: int,
) -> Any:
    """`cnaster.omics.assign_initial_blocks` via `searchsorted` and prefix sums.

    Logs via `port.patch.omics.summaries.summarize_blocks` (#191). Departure
    (#466): a block never spans two chromosomes, where `cnaster` may close one
    across a chromosome boundary.
    """
    from port.patch.omics.summaries import summarize_blocks

    if "known_id" in df_gene_snp.columns:
        # NB upstream as imported: under `patched()` the module name is this
        #    function, and calling it recursed (#466).
        return _UPSTREAM_ASSIGN_INITIAL_BLOCKS(
            df_gene_snp,
            adata,
            cell_snp_Aallele,
            cell_snp_Ballele,
            unique_snp_ids,
            initial_min_umi,
        )

    chromosome = df_gene_snp.CHR.to_numpy()
    start = df_gene_snp.START.to_numpy()
    end = df_gene_snp.END.to_numpy()
    is_interval = df_gene_snp.is_interval.to_numpy().astype(bool)

    gene_rows = np.flatnonzero(is_interval)
    opens = merged_gene_intervals(
        chromosome[gene_rows], start[gene_rows], end[gene_rows]
    )
    interval_row = gene_rows[opens]

    logger.info(
        f"Merged {100.0 * (1.0 - opens.size / max(gene_rows.size, 1)):.3f}% of "
        "genes to ranges as overlapping."
    )

    initial_block_id = block_of_row(chromosome, start, interval_row)
    df_gene_snp["initial_block_id"] = initial_block_id

    block_starts = np.searchsorted(initial_block_id, np.arange(opens.size), side="left")
    block_stops = np.searchsorted(initial_block_id, np.arange(opens.size), side="right")

    summarize_blocks(
        df_gene_snp,
        adata,
        cell_snp_Aallele,
        cell_snp_Ballele,
        unique_snp_ids,
        block_key="initial_block_id",
    )

    # NB per-SNP then per-block UMI totals; the loop reads their prefix sums.
    per_snp = (
        np.asarray(cell_snp_Aallele.sum(axis=0)).ravel()
        + np.asarray(cell_snp_Ballele.sum(axis=0)).ravel()
    )

    snp_index = {site: index for index, site in enumerate(unique_snp_ids)}
    snp_rows = np.flatnonzero(df_gene_snp.snp_id.notna().to_numpy())
    snp_columns = np.array(
        [snp_index[site] for site in df_gene_snp.snp_id.to_numpy()[snp_rows]],
        dtype=np.int64,
    )

    per_block = np.bincount(
        initial_block_id[snp_rows],
        weights=per_snp[snp_columns],
        minlength=opens.size,
    )
    cumulative = np.concatenate(([0.0], np.cumsum(per_block)))
    block_chr = chromosome[interval_row]

    block_ranges_new: list[tuple[int, int]] = []
    lower = 0

    while lower < opens.size:
        same_chromosome = np.flatnonzero(block_chr[lower:] != block_chr[lower])
        last_on_chromosome = (
            opens.size if not same_chromosome.size else lower + int(same_chromosome[0])
        )

        reached = np.searchsorted(
            cumulative[lower + 1 : last_on_chromosome + 1],
            cumulative[lower] + initial_min_umi,
            side="left",
        )
        upper = min(lower + 1 + int(reached), last_on_chromosome)
        totals = cumulative[upper] - cumulative[lower]

        merge_back = (
            totals < initial_min_umi
            and lower > 0
            and block_chr[lower - 1] == block_chr[lower]
        )
        span = (int(block_starts[lower]), int(block_stops[upper - 1]))

        if merge_back:
            block_ranges_new[-1] = (block_ranges_new[-1][0], span[1])
        else:
            block_ranges_new.append(span)

        lower = upper

    block_id = np.zeros(len(df_gene_snp), dtype=np.int64)

    for index, (first, last) in enumerate(block_ranges_new):
        block_id[first:last] = index

    df_gene_snp["block_id"] = block_id

    logger.info(
        "Updated genome segmentation given (population phased) genotypes and "
        f"min. snp-covering umi={initial_min_umi} per segment."
    )

    summarize_blocks(
        df_gene_snp,
        adata,
        cell_snp_Aallele,
        cell_snp_Ballele,
        unique_snp_ids,
        block_key="block_id",
    )

    return df_gene_snp.drop(columns=["initial_block_id"])


def _positions(
    values: np.ndarray, vocabulary: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Where each of `values` sits in `vocabulary`, and which ones are in it."""
    order = np.argsort(vocabulary)
    ordered = vocabulary[order]

    slot = np.clip(np.searchsorted(ordered, values), 0, ordered.size - 1)
    known = ordered[slot] == values

    return order[slot], known


def _group_indicator(
    rows: np.ndarray, groups: np.ndarray, n_rows: int, n_groups: int
) -> Any:
    """A sparse `(n_rows, n_groups)` 0/1 matrix marking each row's group.

    Duplicate `(row, group)` pairs count once, as upstream's set membership.
    """
    if not rows.size:
        return sp.csr_matrix((n_rows, n_groups), dtype=np.int64)

    # NB integer keys: cheaper than `np.unique(..., axis=0)`.
    keys = np.unique(rows.astype(np.int64) * n_groups + groups.astype(np.int64))
    row, group = np.divmod(keys, n_groups)

    return sp.csr_matrix(
        (np.ones(keys.size, dtype=np.int64), (row, group)),
        shape=(n_rows, n_groups),
    )


SPOT_BLOCK = 64
"""Spots per grouped-sum block: bounds scipy's C-contiguous copy so peak memory
does not exceed `cnaster`'s; values are independent of it."""


def _grouped_column_sums(matrix: Any, indicator: Any) -> np.ndarray:
    """`matrix`'s columns summed by group, dense `(n_groups, n_rows_of_matrix)`.

    Densified because `np.asarray` of a sparse product is an object array;
    `matrix` may be dense or sparse (#186).
    """
    counts = _as_matrix(matrix)
    n_rows = counts.shape[0]

    out = np.zeros((indicator.shape[1], n_rows), dtype=np.int64)

    # NB row blocks bound scipy's copy of the non-contiguous `counts.T`.
    for start in range(0, n_rows, SPOT_BLOCK):
        block = counts[start : start + SPOT_BLOCK]
        product = indicator.T @ (block.toarray() if sp.issparse(block) else block).T

        out[:, start : start + SPOT_BLOCK] = (
            product.toarray() if sp.issparse(product) else product
        )

    return out


def _as_matrix(counts: Any) -> Any:
    """`counts` as something that can be multiplied, sparse or dense."""
    return counts if sp.issparse(counts) else np.asarray(counts)


def summarize_counts_for_blocks(
    df_gene_snp: Any,
    adata: Any,
    cell_snp_Aallele: Any,
    cell_snp_Ballele: Any,
    unique_snp_ids: np.ndarray,
) -> Any:
    """`cnaster.omics.summarize_counts_for_blocks` as three indicator products."""
    logger.info("Aggregating (snp, umi) counts for genome segmentation.")

    block_id = df_gene_snp.block_id.to_numpy()
    n_blocks = df_gene_snp.block_id.nunique()
    n_spots = adata.shape[0]

    snp_rows = np.flatnonzero(df_gene_snp.snp_id.notna().to_numpy())
    snp_columns, _ = _positions(
        df_gene_snp.snp_id.to_numpy()[snp_rows], np.asarray(unique_snp_ids)
    )
    by_snp = _group_indicator(
        snp_columns, block_id[snp_rows], len(unique_snp_ids), n_blocks
    )

    gene_names = adata.var.index.to_numpy()
    is_gene = df_gene_snp.is_interval.to_numpy().astype(bool)
    columns, known = _positions(df_gene_snp.gene.to_numpy(), gene_names)
    gene_rows = np.flatnonzero(is_gene & known)

    by_gene = _group_indicator(
        columns[gene_rows], block_id[gene_rows], len(gene_names), n_blocks
    )

    a_allele = _grouped_column_sums(cell_snp_Aallele, by_snp)

    single_X = np.zeros((n_blocks, 2, n_spots), dtype=int)
    single_X[:, 1, :] = a_allele
    single_X[:, 0, :] = _grouped_column_sums(adata.layers["count"], by_gene)

    # NB the A sum again, where upstream recomputes it.
    single_total_bb_RD = (
        a_allele + _grouped_column_sums(cell_snp_Ballele, by_snp)
    ).astype(int)

    # NB lengths from the blocks as a labelling of the genes (#438).
    lengths = observe(df_gene_snp, "block_id", "blocks").lengths

    return SpatioGenomicCounts(
        lengths, single_X, np.zeros((n_blocks, n_spots)), single_total_bb_RD
    )


def summarize_counts_for_bins(
    df_gene_snp: Any,
    adata: Any,
    single_X: np.ndarray,
    single_total_bb_RD: np.ndarray,
    phase_indicator: np.ndarray,
    nu: float,  # noqa: ARG001 -- upstream takes it and never reads it
    logphase_shift: float,  # noqa: ARG001 -- likewise
    geneticmap_file: Any,  # noqa: ARG001 -- likewise
) -> Any:
    """`cnaster.omics.summarize_counts_for_bins` as two indicator products.

    `nu`, `logphase_shift`, `geneticmap_file` are unread, as upstream (#196).
    """
    logger.info("Summarizing counts for bins.")

    assigned = df_gene_snp.bin_id.notna().to_numpy()
    bin_id = df_gene_snp.bin_id.to_numpy()

    logger.info(
        f"Retaining {100.0 * np.mean(assigned):.2f}% of gene/snps with assigned bin."
    )

    # NB upstream groups by sorted `bin_id`: a bin's row is its rank.
    bins, bin_rank = np.unique(bin_id[assigned], return_inverse=True)
    n_bins, n_spots = bins.size, adata.shape[0]

    block_of_row = df_gene_snp.block_id.to_numpy()[assigned]
    has_block = pd.notna(block_of_row)

    by_block = _group_indicator(
        np.asarray(block_of_row[has_block], dtype=np.int64),
        bin_rank[has_block],
        single_X.shape[0],
        n_bins,
    )

    gene_names = adata.var.index.to_numpy()
    columns, known = _positions(df_gene_snp.gene.to_numpy()[assigned], gene_names)

    # NB genes flagged by the differential-expression filter leave the read
    #    depth (#440, #177).
    lineage = current()
    if lineage is not None and lineage.excluded_genes:
        flagged = np.isin(gene_names[columns], list(lineage.excluded_genes))
        known = known & ~flagged

    by_gene = _group_indicator(columns[known], bin_rank[known], len(gene_names), n_bins)

    # NB phasing is per block, independent of the bins.
    phased = np.where(
        np.asarray(phase_indicator).reshape(-1, 1),
        single_X[:, 1, :],
        single_total_bb_RD - single_X[:, 1, :],
    )

    bin_single_X = np.zeros((n_bins, 2, n_spots), dtype=int)
    bin_single_X[:, 1, :] = by_block.T @ phased
    bin_single_X[:, 0, :] = _grouped_column_sums(adata.layers["count"], by_gene)

    bin_single_total_bb_RD = np.asarray(by_block.T @ single_total_bb_RD, dtype=int)

    # NB lengths from the bins (#438); no zero-length contigs, unlike `cnaster` (D5).
    lengths = observe(df_gene_snp, "bin_id", "bins").lengths

    return SpatioGenomicCounts(
        lengths, bin_single_X, np.zeros((n_bins, n_spots)), bin_single_total_bb_RD
    )


def create_bin_ranges(
    df_gene_snp: Any,
    adata: Any,
    cell_snp_Aallele: Any,
    cell_snp_Ballele: Any,
    unique_snp_ids: Any,
    single_X: Any,
    single_total_bb_RD: Any,
    refined_lengths: Any,
    secondary_min_umi: Any,
    secondary_min_snp_umi: Any,
    secondary_min_normal_umi: Any,
    normal_candidates: Any = None,
    max_binlength: float = 5e6,
    key: str = "block_id",
    *,
    min_segment_normal_umi: float | None = None,
) -> Any:
    """`cnaster.omics.create_bin_ranges`, without the rows its merge leaves unbinned (#438 D8, #105).

    Rows with missing `bin_id` (removed by `normal_baf_bin_filter`) are dropped,
    since `run_cnaster` would index out of bounds on them. Departure (T- #617):
    bins are re-cut by `floor_bins` where `segment_floor` is on (#551);
    `min_segment_normal_umi` is the normal floor where no key is stated.
    """
    table = _UPSTREAM_CREATE_BIN_RANGES(
        df_gene_snp,
        adata,
        cell_snp_Aallele,
        cell_snp_Ballele,
        unique_snp_ids,
        single_X,
        single_total_bb_RD,
        refined_lengths,
        secondary_min_umi,
        secondary_min_snp_umi,
        secondary_min_normal_umi,
        normal_candidates=normal_candidates,
        max_binlength=max_binlength,
        key=key,
    )

    if key != "bin_id":
        return table

    unbinned = table["bin_id"].isna().to_numpy()

    if unbinned.any():
        logger.info(
            f"Dropping {int(unbinned.sum())} rows whose bins the normal-BAF "
            "filter removed, so the gene-level output can index them (#105)."
        )
        table = table.loc[~unbinned]

    min_segment, min_normal = segment_floor(min_segment_normal_umi)

    if min_segment is not None or min_normal is not None:
        table = floor_bins(
            table,
            adata,
            normal_candidates,
            min_length=(min_segment or 0.0) * 1e6,
            min_normal_umi=max(float(secondary_min_normal_umi), min_normal or 0.0),
        )

    return table


MIN_SEGMENT_MB = 0.75
"""The read-depth segment floor `quality.min_segment_mb: true` sets, in Mb (#551)."""

MIN_SEGMENT_NORMAL_UMI = 300.0
"""The normal-UMI floor `quality.min_segment_normal_umi: true` sets, and
`run_cnaster_port`'s default where no key is stated (#551, #547)."""


def segment_floor(
    normal_umi: float | None = None,
) -> tuple[float | None, float | None]:
    """`(Mb, normal UMI)` floors from `cnaster`'s config `quality` section.

    Per key: `false`/`none` off, `true` its default, a number itself; absent
    is off, except the normal floor, which is then `normal_umi`.
    """
    from cnaster.config import get_global_config

    section = getattr(get_global_config(), "quality", None)

    def read(key: str, default: float, unstated: float | None) -> float | None:
        if not hasattr(section, key):
            return unstated
        value = getattr(section, key)
        if value is None or value is False:
            return None
        return default if value is True else float(value)

    return (
        read("min_segment_mb", MIN_SEGMENT_MB, None),
        read("min_segment_normal_umi", MIN_SEGMENT_NORMAL_UMI, normal_umi),
    )


def floor_bins(
    table: Any,
    adata: Any,
    normal_candidates: Any,
    *,
    min_length: float,
    min_normal_umi: float,
) -> Any:
    """`table`'s bins merged within each contig to `min_length` bp and `min_normal_umi` normal-spot UMIs (#551).

    Unlike `cnaster`, guarantees the floor across BAF breakpoint runs. Excludes
    filtered genes (#440); records `bins-floored` and the floor on the lineage.
    """
    lineage = current()
    genes = None if lineage is None else lineage.genes
    bins = Segmentation.from_table(table, "bin_id", genes)

    counts = adata.layers["count"]
    normal = np.zeros(adata.shape[0], dtype=bool)
    if normal_candidates is not None:
        normal[np.asarray(normal_candidates)] = True
    per_column = np.asarray(counts[normal].sum(axis=0), dtype=np.float64).ravel()

    names = table["gene"].reindex(bins.genes.key).to_numpy()
    gene_names = adata.var.index.to_numpy()
    known = pd.notna(names)
    columns = np.zeros(names.size, dtype=np.int64)
    columns[known], found = _positions(names[known], gene_names)
    known[known] = found
    if lineage is not None and lineage.excluded_genes:
        known &= ~np.isin(names, list(lineage.excluded_genes))

    weight = np.where(known, per_column[columns], 0.0)
    floored = bins.floored(min_length, weight, min_normal_umi, name="bins-floored")
    parent = dict(
        zip(bins.ids.tolist(), floored.label[bins.first].tolist(), strict=True)
    )

    logger.info(
        f"Floored {bins.n_segments} bins to {floored.n_segments}: each at least "
        f"{min_length:_.0f} bp and {min_normal_umi:g} normal UMI (#551)."
    )

    table = table.copy()
    table["bin_id"] = table["bin_id"].map(parent)

    if lineage is not None:
        lineage.floor = (min_length, weight, min_normal_umi)
        lineage.record(floored, "bins-floored")

    return table
