from collections import namedtuple

import numpy as np
import pandas as pd

from cnamaste.config import start_time
from cnamaste.logger import get_logger
from cnamaste.recomb import get_sitewise_transmat
from cnamaste.reference import get_reference_genes
from cnamaste.spatio_genomic_counts import SpatioGenomicCounts
from cnamaste.utils import cacher
from typing import Any
import scipy.sparse as sp

logger = get_logger(__name__, start_time=start_time)


def greedy_binning_nobreak(
    block_lengths,
    block_umi,
    block_snp_umi,
    block_normal_umi,
    secondary_min_umi,
    secondary_min_snp_umi,
    secondary_min_normal_umi,
    max_binlength,
):
    """
    Given a set of blocks, find new bins that meet requirements on:
    - minimum total UMIs
    - minimum SNP-covering UMIs
    - minimum normal UMIs
    - maximum bin length
    """
    assert (
        len(block_lengths)
        == len(block_umi)
        == len(block_snp_umi)
        == len(block_normal_umi)
    ), (
        f"Block array length mismatch: "
        f"lengths={len(block_lengths)}, "
        f"umi={len(block_umi)}, "
        f"snp_umi={len(block_snp_umi)}, "
        f"normal_umi={len(block_normal_umi)}"
    )

    # NB (start, end) indices of new bins that aggregate old blocks to
    #    meet umi, length, etc. requirements.
    bin_ranges = []
    s = 0

    while s < len(block_lengths):
        t = s + 1

        # NB extend included blocks until meets required umi count.
        while t < len(block_lengths):
            total_umi = np.sum(block_umi[s:t])
            snp_umi = np.sum(block_snp_umi[s:t])
            normal_umi = np.sum(block_normal_umi[s:t])
            length = np.sum(block_lengths[s:t])

            # NB check if all min requirements are met
            meets_umi = total_umi >= secondary_min_umi
            meets_snp = snp_umi >= secondary_min_snp_umi
            meets_normal = normal_umi >= secondary_min_normal_umi
            all_criteria_met = meets_umi and meets_snp and meets_normal

            # NB break if bin is too long but meets UMI requirements
            if length >= max_binlength and all_criteria_met:
                logger.warning(
                    f"Solved for bin length={length/max_binlength:>6.2f} [max_binlength] "
                    f"(umi={total_umi:>8}, snp-umi={snp_umi:>8}, normal-umi={normal_umi:>8})"
                )
                t = max(t - 1, s + 1)
                break

            # NB continue if criteria not met and not too long
            if all_criteria_met:
                break

            t += 1

        # NB final counts for bin [s:t]
        total_umi = np.sum(block_umi[s:t])
        snp_umi = np.sum(block_snp_umi[s:t])
        normal_umi = np.sum(block_normal_umi[s:t])
        length = np.sum(block_lengths[s:t])

        # NB check if it's a small bin at the end that doesn't meet criteria
        if s > 0 and t == len(block_lengths):
            if (
                total_umi < secondary_min_umi
                or snp_umi < secondary_min_snp_umi
                or normal_umi < secondary_min_normal_umi
            ):
                logger.debug(
                    f"Last bin failed thresholds "
                    f"(UMI={total_umi:>8}/{secondary_min_umi:<8}, "
                    f"SNP-UMI={snp_umi:>8}/{secondary_min_snp_umi:<8}, "
                    f"normal-UMI={normal_umi:>8}/{secondary_min_normal_umi:<8}), "
                    f"merging with previous."
                )
                bin_ranges[-1][1] = t
            else:
                bin_ranges.append([s, t])
        else:
            bin_ranges.append([s, t])

        s = t

    bin_ids = np.zeros(len(block_lengths), dtype=int)

    for i, x in enumerate(bin_ranges):
        bin_ids[x[0] : x[1]] = i

    # NB return new bin ids for each block, where new bins meet umi, length, etc. requirements.
    return bin_ids


# TODO assumes reference gene contains all those present in Visium anndata.

GENE_COLUMN = 4
"""`cnamaste` writes the gene by position -- `df_gene_snp.iloc[i, 4]`.

Recorded rather than used: this patch writes the column by name. It is here
because the position is what makes `cnamaste`'s write fragile (#189), and a
reader comparing the two should be able to see that they address the same
column.
"""


def preceding_gene(
    chromosome: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    is_interval: np.ndarray,
    unassigned: np.ndarray,
    num_preceeding_rows: int,
) -> np.ndarray:
    """For each unassigned row, the nearest preceding gene containing it.

    `-1` where there is none. `cnamaste` finds this by walking backwards from
    each row and breaking at the first gene interval that contains the SNP's
    position, stopping at `num_preceeding_rows` rows, at the start of the
    table, or at a change of chromosome.

    **The chromosome test is a stop and not a skip**, and it does not need to
    be reproduced as one: the table is sorted by `(CHR, START)`, so every row
    before the first one on another chromosome is on another chromosome too.
    Requiring a match is therefore the same condition, and it is the one that
    vectorizes.

    One pass per offset, and a row keeps the match from the **smallest**
    offset, which is what breaking out of a backwards walk means. That is
    `num_preceeding_rows` passes over the table rather than one Python
    iteration and two `pandas` scalar accesses per SNP.
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
    """What `cnamaste.omics.form_gene_snp_table` returns, without the walk.

    `verbose` is upstream's and gates a log of the genes absent from the
    reference. It is kept in the signature so a caller can be pointed at
    either function, and does nothing here for the same reason it does little
    there.
    """
    logger.info("Forming gene & snp meta data.")

    # NB `port.patch.reference`, not `cnamaste.reference`: the read is 13.3x
    #    with `polars` at a human reference's size (#185), and the frame it
    #    returns is bitwise the same.
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
    """Where each run of overlapping gene intervals begins.

    `cnamaste` merges them by walking the gene rows and extending the last
    interval whenever the next one overlaps it. Sorted by `(CHR, START)`, that
    is the classic sweep: a new run begins exactly where a gene's start is at
    or past the running maximum of the ends before it.

    Within a chromosome the running maximum may be taken over the whole
    chromosome rather than over the current run, because the two agree: if a
    start were below a maximum attained in an **earlier** run, that run would
    not have ended where it did, so the maximum is always attained inside the
    current one.

    **Across chromosomes it may not**, and that is the one thing the sweep has
    to be told. Positions restart at each chromosome, so a running maximum
    carried over from the last chromosome exceeds every start on the next one
    and swallows it whole. The accumulation is therefore reset per
    chromosome -- twenty-two short calls rather than one long one.
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
    """Which merged interval each row of the table falls in.

    `cnamaste` finds this the other way round -- for each merged interval, a
    full-table `np.where` over the overlap condition, then the first and last
    row it matched. That is `n_intervals` passes over `n_rows`, which at a
    slide's scale is the product of two large numbers where the table is
    already sorted.

    The intervals tile the rows in order, which `cnamaste` asserts, so the
    interval a row belongs to is the last one that starts at or before it.
    One `searchsorted` gives every row at once.
    """
    keys = chromosome.astype(np.int64) * (1 << 40) + start.astype(np.int64)
    edges = keys[interval_row]

    # NB genes open their own interval, so a gene row sitting exactly on an
    #    edge belongs to that interval rather than to the one before it; a SNP
    #    at the same key does too, since the gene precedes it in sort order.
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
    """What `cnamaste.omics.assign_initial_blocks` returns, in three passes.

    `cnamaste` takes two quadratic loops to get here:

    *   one full-table `np.where` per merged gene interval, to find the rows
        that interval covers -- `n_intervals` passes over `n_rows`;
    *   one re-count of every SNP-covering UMI in `[s, t)` for **every**
        candidate upper bound `t`, which its own comment marks
        `TODO recalculates for every upper bound`.

    Both answer questions the sort order has already answered. The intervals
    tile the rows, so a row's interval is a `searchsorted`; and the UMI total
    over a run of blocks is a difference of prefix sums, so the smallest `t`
    meeting the threshold is another one.

    **The two `summarize_blocks` calls stay**, because their log lines are
    the only thing they produce and removing them is a behaviour change. They
    come from `port.patch.omics.summaries` instead (#191), which logs the same lines
    from two passes and a `bincount` rather than a fancy-indexed slice of the
    count matrix per block -- 39% of this function, computed rather than
    looped.
    """

    if "known_id" in df_gene_snp.columns:

        return assign_initial_blocks_reference(
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

    # NB one pass for every SNP's UMI total, then one per initial block. The
    #    loop below reads prefix sums of these rather than re-summing the
    #    allele matrices for every candidate upper bound.
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


def assign_initial_blocks_reference(
    df_gene_snp,
    adata,
    cell_snp_Aallele,
    cell_snp_Ballele,
    unique_snp_ids,
    initial_min_umi,
):
    """
    Initially assigns snps to blocks along the genome, based on merging overlapping gene intervals
    & requiring blocks have a minimm number of snp-covering reads; these can be aggregated up to the
    scale of switch errors in the population-based phasing.

    Returns
    ----------
    df_gene_snp : data frame, names: (CHR, START, END, snp_id, gene, is_interval, block_id)
         Gene and SNP info combined into a single dataframe sorted by (CHR, START).
        "is_interval"=True is a gene, otherwise SNP.
        "gene" contains the name of a gene, or the gene a SNP belongs.
    """
    if "known_id" in df_gene_snp.columns:
        logger.warning(
            "Assuming known genomic segmentation given 'known_id' in df_gene_snp."
        )

        df_gene_snp["known_id"] = df_gene_snp["known_id"].astype("Int64")

        unique_ids = df_gene_snp["known_id"].dropna().unique()
        unique_ids = np.sort(unique_ids)  # Ensure order is consistent

        mapping = {old_id: i for i, old_id in enumerate(unique_ids)}

        df_gene_snp["block_id"] = (
            df_gene_snp["known_id"].map(mapping).fillna(-1).astype(int)
        )

        return df_gene_snp

    logger.info(
        f"Creating initial genome segmentation based solely on overlapping genes."
    )

    # NB first level: partition of genome by gene range (if two genes overlap, they are grouped to one range);
    # NB == is_gene.
    is_interval = df_gene_snp.is_interval

    # NB merge overlapping genes.
    tmp_block_genome_intervals = list(
        zip(
            df_gene_snp[is_interval].CHR.to_numpy(),
            df_gene_snp[is_interval].START.to_numpy(),
            df_gene_snp[is_interval].END.to_numpy(),
        )
    )

    # NB (chr, start, end) for first gene.
    first_interval = tmp_block_genome_intervals[0]

    # NB list of intervals for initial block definition (merged overlapping genes).
    block_genome_intervals = [first_interval]
    merged = 0

    # NB called snps are limited to transcripts, ergo limited to genes.
    #
    #    initial intervals are gene ranges merged based on overlap.
    for next_interval in tmp_block_genome_intervals[1:]:
        contig, start, end = next_interval

        # NB check whether overlap with previous block
        if contig == block_genome_intervals[-1][0] and max(
            start, block_genome_intervals[-1][1]
        ) < min(end, block_genome_intervals[-1][2]):
            block_genome_intervals[-1] = (
                contig,
                min(start, block_genome_intervals[-1][1]),
                max(end, block_genome_intervals[-1][2]),
            )

            # TODO warn on excessive length;
            merged += 1
        else:
            block_genome_intervals.append(next_interval)

    # NB TODO 20%?
    logger.info(
        f"Merged {100.0 * merged / len(tmp_block_genome_intervals):.3f}% of genes to ranges as overlapping."
    )

    # NB map block_genome_intervals to block_ranges for rows of df_gene_snp.
    block_ranges = []

    for x in block_genome_intervals:
        # NB overlap of df_gene_snp (genes & sites) with block_genome_interval.
        indexes = np.where(
            (df_gene_snp.CHR.to_numpy() == x[0])
            & (
                np.maximum(df_gene_snp.START.to_numpy(), x[1])
                < np.minimum(df_gene_snp.END.to_numpy(), x[2])
            )
        )[0]

        # index of rows into df_gene_snp that overlap each interval.
        # TODO can fail?
        block_ranges.append((indexes[0], indexes[-1] + 1))

    assert np.all(
        np.array([x[1] for x in block_ranges[:-1]])
        == np.array([x[0] for x in block_ranges[1:]])
    )

    # NB record the initial block id in df_gene_snps
    # BUG previously 0, spuriously assigned to the zeroth block - safe as discarded all snps that don't overlap a gene (merged to block).
    df_gene_snp["initial_block_id"] = -1

    for i, x in enumerate(block_ranges):
        df_gene_snp.iloc[x[0] : x[1], -1] = i

    assert (
        np.all(df_gene_snp["initial_block_id"].values) >= 0
    ), "Found genes/sites with no assigned block."

    logger.info(
        "Assigned snps to initial genome segments (segments == genes, merged on overlap)."
    )

    summarize_blocks(
        df_gene_snp,
        adata,
        cell_snp_Aallele,
        cell_snp_Ballele,
        unique_snp_ids,
        block_key="initial_block_id",
    )

    logger.info(
        f"Updating genome segmentation to ensure min. snp-covering umi={initial_min_umi} threshold is satisfied for the new segments."
    )

    # NB second level: extend the first level blocks based on haplotype-aggregated counts such that the min. snp-covering umi counts >= initial_min_umi.
    #    maps site_id, {chr}_{pos}_{ref}_{alt} to integer index.
    map_snp_index = {site: index for index, site in enumerate(unique_snp_ids)}
    initial_block_chr = df_gene_snp.CHR.to_numpy()[
        np.array([x[0] for x in block_ranges])
    ]
    block_ranges_new = []
    s = 0

    # NB s is the "lower" initial_block_id and t is the "upper" initial_block_id.
    while s < len(block_ranges):
        t = s

        while t <= len(block_ranges):
            t += 1

            reach_end = t == len(block_ranges)
            change_chr = initial_block_chr[s] != initial_block_chr[t - 1]

            # NB count SNP-covering UMI
            # TODO recalculates for every upper bound.
            involved_snps_ids = df_gene_snp[
                (df_gene_snp.initial_block_id >= s) & (df_gene_snp.initial_block_id < t)
            ].snp_id

            # NB drop genes.
            involved_snps_ids = involved_snps_ids[~involved_snps_ids.isnull()]
            involved_snp_idx = np.array([map_snp_index[x] for x in involved_snps_ids])

            # NB num. of snp-covering umis for initial block ids s to t.
            this_snp_umis = (
                0
                if len(involved_snp_idx) == 0
                else np.sum(cell_snp_Aallele[:, involved_snp_idx])
                + np.sum(cell_snp_Ballele[:, involved_snp_idx])
            )

            if reach_end:
                logger.warning(
                    f"Reached last block with {this_snp_umis}/{initial_min_umi} required snp umis."
                )
                break

            if change_chr:
                t -= 1

                # re-count snp-covering UMIs
                involved_snps_ids = df_gene_snp.snp_id.iloc[
                    block_ranges[s][0] : block_ranges[t - 1][1]
                ]
                involved_snps_ids = involved_snps_ids[~involved_snps_ids.isnull()]

                involved_snp_idx = np.array(
                    [map_snp_index[x] for x in involved_snps_ids]
                )

                this_snp_umis = (
                    0
                    if len(involved_snp_idx) == 0
                    else np.sum(cell_snp_Aallele[:, involved_snp_idx])
                    + np.sum(cell_snp_Ballele[:, involved_snp_idx])
                )

                logger.warning(
                    f"Reached contig end with {this_snp_umis}/{initial_min_umi} required snp umis."
                )

                break

            if this_snp_umis >= initial_min_umi:
                break

        # NB goal is to have assigned this_snp_umis, s and t.
        if (
            this_snp_umis < initial_min_umi
            and s > 0
            and initial_block_chr[s - 1] == initial_block_chr[s]
        ):
            indexes = np.where(df_gene_snp.initial_block_id.isin(np.arange(s, t)))[0]
            block_ranges_new[-1] = (block_ranges_new[-1][0], indexes[-1] + 1)
        else:
            indexes = np.where(df_gene_snp.initial_block_id.isin(np.arange(s, t)))[0]
            block_ranges_new.append((indexes[0], indexes[-1] + 1))

        # NB fast-forward lower block id to upper.
        s = t

    # NB record the block id in df_gene_snp
    df_gene_snp["block_id"] = 0

    for i, x in enumerate(block_ranges_new):
        df_gene_snp.iloc[x[0] : x[1], -1] = i

    logger.info(
        f"Updated genome segmentation given (population phased) genotypes and min. snp-covering umi={initial_min_umi} per segment."
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
    """Where each of `values` sits in `vocabulary`, and which ones are in it.

    A dictionary comprehension over a column is a Python loop per row, which
    is the cost the products below exist to remove -- so the lookup is a
    `searchsorted` against the sorted vocabulary instead.
    """
    order = np.argsort(vocabulary)
    ordered = vocabulary[order]

    slot = np.clip(np.searchsorted(ordered, values), 0, ordered.size - 1)
    known = ordered[slot] == values

    return order[slot], known


def _group_indicator(
    rows: np.ndarray, groups: np.ndarray, n_rows: int, n_groups: int
) -> Any:
    """A sparse `(n_rows, n_groups)` 0/1 matrix marking each row's group.

    Summing a matrix's columns by group is a product with this. `cnamaste`
    does it the other way -- one fancy-indexed slice of the whole matrix per
    group, each one `n_spots` deep -- which is `n_groups` passes over the data
    where the product is one.

    Duplicate `(row, group)` pairs are dropped rather than summed, because
    upstream gathers each group's members as a **set** and a repeated member
    contributes once.
    """
    if not rows.size:
        return sp.csr_matrix((n_rows, n_groups), dtype=np.int64)

    # NB one key per pair rather than `np.unique(..., axis=0)`, which sorts
    #    the rows of a two-column array and costs several times sorting the
    #    integers those rows encode.
    keys = np.unique(rows.astype(np.int64) * n_groups + groups.astype(np.int64))
    row, group = np.divmod(keys, n_groups)

    return sp.csr_matrix(
        (np.ones(keys.size, dtype=np.int64), (row, group)),
        shape=(n_rows, n_groups),
    )


SPOT_BLOCK = 64
"""How many spots the grouped sum multiplies at once.

`scipy` wants its dense operand C-contiguous and the transpose it is handed is
not, so it copies. Copying the whole count matrix makes the peak **larger**
than the per-slice loop this replaces, which would be buying time with memory.
In row blocks the copy is bounded at `SPOT_BLOCK` rows, and the block size is
a real trade rather than a free choice -- at 2,500 spots, against `cnamaste`'s
229.5 ms and 56.4 MB:

| `SPOT_BLOCK` | time | peak |
| ---: | ---: | ---: |
| unchunked | 1.91x | 0.64x |
| 1024 | 2.85x | 0.78x |
| 256 | 4.14x | 0.94x |
| **64** | **3.60x** | **0.99x** |
| 32 | 3.45x | 1.00x |

64 is where the peak stops regressing against what it replaces. 256 is 15%
faster and 6% above upstream's peak, which is the wrong side of a patch whose
point is to allocate less.

Where the blocks fall cannot change a value: a spot's column sums do not
depend on any other spot's, and the sweep above is bitwise identical at every
size.
"""


def _grouped_column_sums(matrix: Any, indicator: Any) -> np.ndarray:
    """`matrix`'s columns summed by group, as `(n_groups, n_rows_of_matrix)`.

    `indicator.T @ matrix.T` rather than `matrix @ indicator`, so the sparse
    operand leads and the result is the orientation the caller stores, without
    a transpose of a large array.

    **The product is densified deliberately.** Two sparse operands give a
    sparse result, and `np.asarray` of one is a zero-dimensional object array
    rather than its values -- so a caller that stored it would store garbage
    instead of failing. The answer is `(n_groups, n_spots)`, which is the
    dense shape the caller writes into either way. The loader returns dense
    counts today and sparse under `sparse_counts` (#186), so both reach here.
    """
    counts = _as_matrix(matrix)
    n_rows = counts.shape[0]

    out = np.zeros((indicator.shape[1], n_rows), dtype=np.int64)

    # NB in row blocks, because `scipy` needs the dense operand C-contiguous
    #    and `counts.T` is not: it copies, and an unchunked copy is the whole
    #    matrix. The column sums are independent per row, so where the blocks
    #    fall cannot change a value.
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
    """What `cnamaste.omics.summarize_counts_for_blocks` returns, in three products.

    `cnamaste` loops the blocks and, for each one, slices every spot's column
    out of three matrices:

    ```python
    single_X[block_id, 1, :] = cell_snp_Aallele[:, snp_idx].sum(axis=1)
    single_total_bb_RD[block_id, :] = (cell_snp_Aallele[:, snp_idx].sum(axis=1)
                                       + cell_snp_Ballele[:, snp_idx].sum(axis=1))
    gene_mask = np.isin(gene_names, genes)
    single_X[block_id, 0, :] = gene_counts[:, gene_mask].sum(axis=1)
    ```

    Three things there, and the first is free: **the A-allele sum is computed
    twice**, once for its own row and once inside the total. The second is
    `np.isin` over every gene name per block, which is `n_blocks * n_genes`.
    The third is the loop itself -- a grouped column sum is a product with a
    0/1 indicator, and three products replace all of it.
    """
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

    lengths = df_gene_snp.groupby("CHR")["block_id"].nunique().to_numpy()

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
    """What `cnamaste.omics.summarize_counts_for_bins` returns, in two products.

    The same defect one level up: `cnamaste` loops the bins, gathers each one's
    blocks, phases them and sums, then slices the count matrix per bin for the
    genes. **The phasing does not depend on the bin** -- it is a `where` over
    every block at once -- and what remains is two grouped sums.

    **Three of the parameters are upstream's and unread**, here and there:
    `nu`, `logphase_shift` and `geneticmap_file`. Its docstring promises a
    `log_sitewise_transmat` return computed from the genetic map, and the
    function returns a `SpatioGenomicCounts` with no such field. They are kept
    in the signature so a caller can be pointed at either function; #196 is
    where the signature belongs.
    """
    logger.info("Summarizing counts for bins.")

    assigned = df_gene_snp.bin_id.notna().to_numpy()
    bin_id = df_gene_snp.bin_id.to_numpy()

    logger.info(
        f"Retaining {100.0 * np.mean(assigned):.2f}% of gene/snps with assigned bin."
    )

    # NB upstream groups by bin_id with sort=True, so the output row for a bin
    #    is its rank among the sorted ids rather than the id itself.
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

    by_gene = _group_indicator(columns[known], bin_rank[known], len(gene_names), n_bins)

    # NB every block's phased B count at once; the choice is per block, not
    #    per bin, so the bins never enter it.
    phased = np.where(
        np.asarray(phase_indicator).reshape(-1, 1),
        single_X[:, 1, :],
        single_total_bb_RD - single_X[:, 1, :],
    )

    bin_single_X = np.zeros((n_bins, 2, n_spots), dtype=int)
    bin_single_X[:, 1, :] = by_block.T @ phased
    bin_single_X[:, 0, :] = _grouped_column_sums(adata.layers["count"], by_gene)

    bin_single_total_bb_RD = np.asarray(by_block.T @ single_total_bb_RD, dtype=int)

    chr_order = df_gene_snp.CHR.unique()
    lengths = (
        df_gene_snp.loc[assigned]
        .groupby("CHR")["bin_id"]
        .nunique()
        .reindex(chr_order, fill_value=0)
        .to_numpy()
    )

    return SpatioGenomicCounts(
        lengths, bin_single_X, np.zeros((n_bins, n_spots)), bin_single_total_bb_RD
    )


MAX_ROWS = 25
"""How many blocks the breakdown lists. `cnamaste`'s own magic number."""


def _segment_first(values: np.ndarray, block: np.ndarray, blocks: int) -> np.ndarray:
    """The first value in each block, in block order."""
    first = np.zeros(blocks, dtype=values.dtype)
    order = np.argsort(block, kind="stable")[::-1]
    first[block[order]] = values[order]

    return first


def block_summary(
    gene_snp_table: Any,
    adata: Any,
    cell_snp_Aallele: Any,
    cell_snp_Ballele: Any,
    unique_snp_ids: np.ndarray,
    block_key: str,
    normal_candidates: Any = None,
    sort_key: str | None = "total_umi",
) -> Any:
    """`cnamaste`'s `block_summary` frame, without the per-block slice."""
    block = gene_snp_table[block_key].to_numpy()
    keys = np.unique(block)
    position = np.searchsorted(keys, block)
    blocks = keys.size

    is_interval = gene_snp_table.is_interval.to_numpy().astype(bool)
    snp_id = gene_snp_table.snp_id.to_numpy()
    gene = gene_snp_table.gene.to_numpy()
    chromosome = gene_snp_table.CHR.to_numpy()
    start = gene_snp_table.START.to_numpy()
    end = gene_snp_table.END.to_numpy()

    # NB an object column carrying both `None` and `NaN`, so the test is
    #    `pd.notna` rather than a comparison -- upstream's `notna().sum()`
    #    and `[s for s in x if s is not None]` differ on a `NaN`, and this
    #    reproduces each where it is used.
    has_snp = pd.notna(snp_id)

    summary = pd.DataFrame(
        {
            "num_snps": np.bincount(position[has_snp], minlength=blocks),
            "num_genes": np.bincount(
                position, weights=is_interval.astype(float), minlength=blocks
            ).astype(np.int64),
            "chr": _segment_first(chromosome, position, blocks),
            "start": np.minimum.reduceat(
                start, np.searchsorted(position, np.arange(blocks))
            )
            if blocks
            else np.zeros(0),
            "end": np.maximum.reduceat(
                end, np.searchsorted(position, np.arange(blocks))
            )
            if blocks
            else np.zeros(0),
        },
        index=pd.Index(keys, name=block_key),
    )

    summary["length"] = 1_000.0 * (summary["end"] - summary["start"])

    count_matrix = adata.layers["count"]
    gene_totals = np.asarray(np.sum(count_matrix, axis=0)).ravel()
    gene_index = {name: index for index, name in enumerate(adata.var.index.to_numpy())}

    # NB a set per block upstream, so a gene name on two rows counts once.
    present = np.array([name in gene_index for name in gene], dtype=bool)
    pairs = (
        np.unique(
            np.stack(
                [
                    position[present],
                    np.array(
                        [gene_index[name] for name in gene[present]], dtype=np.int64
                    ),
                ],
                axis=1,
            ),
            axis=0,
        )
        if present.any()
        else np.zeros((0, 2), dtype=np.int64)
    )

    summary["total_umi"] = np.bincount(
        pairs[:, 0], weights=gene_totals[pairs[:, 1]], minlength=blocks
    ).astype(np.int64)

    snp_index = {site: index for index, site in enumerate(unique_snp_ids)}
    named = has_snp & np.array([bool(value) for value in snp_id], dtype=bool)
    columns = np.array([snp_index[value] for value in snp_id[named]], dtype=np.int64)
    snp_totals = (
        np.asarray(cell_snp_Aallele.sum(axis=0)).ravel()
        + np.asarray(cell_snp_Ballele.sum(axis=0)).ravel()
    )

    summary["snp_umi"] = np.bincount(
        position[named], weights=snp_totals[columns], minlength=blocks
    ).astype(np.int64)

    summary["normal_umi"] = np.zeros(blocks, dtype=np.int64)
    summary["normal_snp_umi"] = np.zeros(blocks, dtype=np.int64)

    if normal_candidates is not None:
        normal_gene = np.asarray(
            np.sum(count_matrix[normal_candidates, :], axis=0)
        ).ravel()
        summary["normal_umi"] = np.bincount(
            pairs[:, 0], weights=normal_gene[pairs[:, 1]], minlength=blocks
        ).astype(np.int64)

        normal_snp = (
            np.asarray(cell_snp_Aallele[normal_candidates, :].sum(axis=0)).ravel()
            + np.asarray(cell_snp_Ballele[normal_candidates, :].sum(axis=0)).ravel()
        )
        summary["normal_snp_umi"] = np.bincount(
            position[named], weights=normal_snp[columns], minlength=blocks
        ).astype(np.int64)

    if sort_key is not None:
        summary = summary.sort_values(sort_key, ascending=False)

    return summary


def summarize_blocks(
    gene_snp_table: Any,
    adata: Any,
    cell_snp_Aallele: Any,
    cell_snp_Ballele: Any,
    unique_snp_ids: np.ndarray,
    block_key: str | None = None,
    normal_candidates: Any = None,
    sort_key: str | None = "total_umi",
) -> None:
    """What `cnamaste.omics.summarize_blocks` logs, line for line."""
    logger.info("Summarizing blocks ...")

    assert block_key is not None, "block_key must be specified"
    assert block_key in gene_snp_table.columns, f"{block_key} not in DataFrame"

    summary = block_summary(
        gene_snp_table,
        adata,
        cell_snp_Aallele,
        cell_snp_Ballele,
        unique_snp_ids,
        block_key,
        normal_candidates,
        sort_key,
    )

    logger.info(
        f"Breakdown of genes/snps/umi per {block_key} sorted by {sort_key} "
        f"(top {MAX_ROWS}):"
    )
    logger.info(
        f"{'block id':<10}\t{'chr':>4}\t{'start':>12}\t{'length':>12} [Kbp]\t"
        f"{'snps':>8}\t{'genes':>8}\t{'total umi':>12}\t{'snp umi':>12}\t"
        f"{'normal umi':>12}\t{'normal snp umi':>12}"
    )
    logger.info("-" * 136)

    # NB `itertuples`, not `iterrows`: a row Series takes one dtype for the
    #    whole row, so an all-numeric frame prints its integers as floats.
    #    Upstream's frame carries two list columns and is therefore object,
    #    which keeps them integers -- the same lines, from a different
    #    accident. This is the deliberate version of it.
    for index, row in enumerate(summary.itertuples(index=True)):
        logger.info(
            f"{row.Index:<10}\t{row.chr:>4}\t{row.start:>12}\t"
            f"{row.length / 1.0e6:>12}\t{row.num_snps:>8}\t"
            f"{row.num_genes:>8}\t{row.total_umi:>12}\t{row.snp_umi:>12}\t"
            f"{row.normal_umi:>12}\t{row.normal_snp_umi:>12}"
        )

        if index > MAX_ROWS:
            break

    logger.info(
        f"\n"
        f"median block length: {summary['length'].median() / 1.0e6:.1f} [Kbp],\n"
        f"mean block length: {summary['length'].mean() / 1.0e6:.1f} [Kbp],\n"
        f"median snps/block: {summary['num_snps'].median():.1f},\n"
        f"median genes/block: {summary['num_genes'].median():.1f},\n"
        f"median umis/block: {summary['total_umi'].median():.1f},\n"
        f"median snp-umis/block: {summary['snp_umi'].median():.1f},\n"
        f"total blocks: {len(summary):_},\n"
        f"total umis: {summary['total_umi'].sum():_},\n"
        f"total snp-umis: {summary['snp_umi'].sum():_},\n"
        f"total normal umis: {summary['normal_umi'].sum():_},\n"
        f"total normal snp-umis: {summary['normal_snp_umi'].sum():_},\n"
        f"blocks with 0 umis: {(summary['total_umi'] == 0).mean():.1%},\n"
        f"blocks with <100 umis: {(summary['total_umi'] < 100).mean():.1%},\n"
        "blocks with snp-umis, but no gene-umis: "
        f"{((summary['snp_umi'] > 0) & (summary['total_umi'] == 0)).mean():.1%}\n"
    )

    if summary.index.isna().any():
        logger.warning(f"Found ill-defined group:/n{summary.loc[np.nan]}")


def binned_gene_snp(df_gene_snp, key="bin_id"):
    # NB table with contig range + set of genes + snp_ids,
    #    only for those defined with key bin_id.
    table_bininfo = (
        # df_gene_snp[~df_gene_snp.bin_id.isnull()]
        df_gene_snp[~getattr(df_gene_snp, key).isnull()]
        .groupby(key)
        .agg(
            {
                "CHR": "first",
                "START": "first",
                "END": "last",
                "gene": set,
                "snp_id": set,
            }
        )
        .reset_index()  # TBC (0, ..., N-1).
    )
    # table_bininfo["ARM"] = "."
    table_bininfo["INCLUDED_GENES"] = [
        ",".join([x for x in y if not x is None]) for y in table_bininfo.gene.values
    ]
    table_bininfo["INCLUDED_SNP_IDS"] = [
        ",".join([x for x in y if not x is None]) for y in table_bininfo.snp_id.values
    ]
    table_bininfo["NORMAL_COUNT"] = np.nan
    table_bininfo["N_SNPS"] = [
        len([x for x in y if not x is None]) for y in table_bininfo.snp_id.values
    ]

    table_bininfo.drop(columns=["gene", "snp_id"], inplace=True)

    logger.info(
        f"Finalizing genomic segment annotation for {table_bininfo.shape[0]} bins, given initial {len(df_gene_snp.gene.unique())} genes ({table_bininfo.shape[0] / len(df_gene_snp.gene.unique()):.2f} sampling)."
    )

    return table_bininfo


# @cacher("blocked_counts.hdf5")


# @cacher("blocked_gene_snp_table.tsv")


# @cacher("binned_gene_snp_table.tsv")
def create_bin_ranges(
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
    normal_candidates=None,
    max_binlength=5e6,
    key="block_id",
):
    """
    Aggregate haplotype blocks to bins with multiple UMI constraints.
    Cannot aggregate blocks separated by __refined_lengths__

    Parameters
    ----------
    df_gene_snp : pd.DataFrame
        Gene and SNP info with block_id assignments.
    adata : AnnData
        Annotated data object.
    cell_snp_Aallele, cell_snp_Ballele : array, (n_spots, n_snps)
        Allele counts.
    unique_snp_ids : list
        SNP identifiers.
    single_X : array, (n_blocks, 2, n_spots)
        Block-level transcript and SNP counts.
    single_total_bb_RD : array, (n_blocks, n_spots)
        Total SNP-covering reads per block.
    refined_lengths : array
        Number of blocks before each phase switch.
    secondary_min_umi : int
        Minimum total UMIs per bin.
    secondary_min_snp_umi : int
        Minimum SNP-covering UMIs per bin.
    secondary_min_normal_umi : int
        Minimum normal-spot UMIs per bin.
    normal_candidates : array-like or None
        Boolean mask or integer indices for normal spots.
    max_binlength : int
        Maximum genomic length per bin.

    Returns
    -------
    df_gene_snp : pd.DataFrame
        Updated with bin_id column.
    """
    logger.info(
        f"Aggregating blocks to bins given baf-inferred phasing to satisfy umi, length, etc. constraints."
    )

    # TODO BUG dropna?
    # NB block intervals: by key (e.g. block_id)
    sorted_chr_pos_both = df_gene_snp.groupby(key).agg(
        {"CHR": "first", "START": "first", "END": "last"}
    )

    # NB block intervals
    block_lengths = (
        sorted_chr_pos_both.END.to_numpy() - sorted_chr_pos_both.START.to_numpy()
    )
    n_blocks = len(block_lengths)

    # NB total umi per block (summed across spots)
    block_umi = np.sum(single_X[:, 0, :], axis=1)

    # NB total snp-covering umi per block
    block_snp_umi = np.sum(single_total_bb_RD, axis=1)

    # NB normal-spot umi per block
    if normal_candidates is not None:
        if (
            isinstance(normal_candidates, (np.ndarray, pd.Series))
            and normal_candidates.dtype == bool
        ):
            normal_idx = np.flatnonzero(normal_candidates)
        else:
            normal_idx = np.asarray(normal_candidates, dtype=int)

        block_normal_umi = np.sum(single_X[:, 0, normal_idx], axis=1)
    else:
        block_normal_umi = np.zeros(n_blocks, dtype=int)
        secondary_min_normal_umi = 0

    assert (
        len(block_lengths)
        == len(block_umi)
        == len(block_snp_umi)
        == len(block_normal_umi)
    ), (
        f"Block array length mismatch: "
        f"lengths={len(block_lengths)}, "
        f"umi={len(block_umi)}, "
        f"snp_umi={len(block_snp_umi)}, "
        f"normal_umi={len(block_normal_umi)}"
    )

    frac_normal = normal_candidates.mean() if normal_candidates is not None else np.nan

    logger.info(
        f"Creating (phased) bin ranges: max_length={max_binlength:_}, "
        f"min_umi={secondary_min_umi}, "
        f"min_snp_umi={secondary_min_snp_umi}, "
        f"min_normal_umi={secondary_min_normal_umi}, "
        f"fraction normal={frac_normal:.3f}"
    )

    # NB breakpoints defined by jump in minor baf, and oversized blocks.
    breakpoints = np.concatenate(
        [
            np.cumsum(refined_lengths),
            np.where(block_lengths > max_binlength)[0],
            np.where(block_lengths > max_binlength)[0] + 1,
        ]
    )

    # NB sorted, unique.
    breakpoints = np.sort(np.unique(breakpoints))

    if breakpoints[0] != 0:
        breakpoints = np.append([0], breakpoints)

    assert np.all(breakpoints[:-1] < breakpoints[1:])

    # NB assign each block to a bin (that meets umi, length, etc. requirements)
    bin_ids = np.zeros(n_blocks, dtype=int)
    offset = 0

    for i in range(len(breakpoints) - 1):
        b1, b2 = breakpoints[i], breakpoints[i + 1]

        if b2 - b1 == 1:
            bin_ids[b1:b2] = offset
            offset += 1
        else:
            this_bin_ids = greedy_binning_nobreak(
                block_lengths[b1:b2],
                block_umi[b1:b2],
                block_snp_umi[b1:b2],
                block_normal_umi[b1:b2],
                secondary_min_umi,
                secondary_min_snp_umi,
                secondary_min_normal_umi,
                max_binlength,
            )

            bin_ids[b1:b2] = offset + this_bin_ids
            offset += np.max(this_bin_ids) + 1

    if "bin_id" in df_gene_snp.columns:
        logger.warning(f"Overwriting bin_id column, storing in block_id.")
        df_gene_snp["block_id"] = df_gene_snp["bin_id"]

    # NB map each block_id to its bin id
    df_gene_snp["bin_id"] = getattr(df_gene_snp, key).map(
        {i: x for i, x in enumerate(bin_ids)}
    )

    # NB return df_gene_snp with an updated bin_id column, and potentially deifned block_id column if it was overwritten.
    return df_gene_snp


# @cacher("binned_counts.hdf5")
