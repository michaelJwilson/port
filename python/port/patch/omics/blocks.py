"""`cnaster.omics`, with the Python walks over the genome vectorized.

**Proposed for `cnaster`, written here.** #190. `form_gene_snp_table` assigns
each SNP to the gene interval containing it by walking backwards through the
sorted table in Python, up to `num_preceeding_rows` at a time, and writing the
result with a `pandas` scalar assignment per SNP:

```python
for i in np.where(df_gene_snp.gene.isna())[0]:
    ...
    df_gene_snp.iloc[i, 4] = df_gene_snp.iloc[j]["gene"]
```

Those two `iloc` calls are **66 per cent of the function** at 782 SNPs -- 782
of them, 260 microseconds each. A Visium slide carries 500,000 SNPs, where the
same line is over two minutes on its own.

The assignment is a window search over a sorted table, so it vectorizes
exactly: one pass per offset rather than one Python iteration per SNP, and the
first match at the smallest offset wins, which is what walking backwards and
breaking means.

`port` cannot land the change (`CLAUDE.md`, **Working against a repository you
do not own**), so it is written here with its referee beside it in
`tests/test_preprocessing_omics.py`. The return is **bitwise** what `cnaster`
returns, column for column and row for row.

**Not carried across:** `cnaster` decorates the function with
`@cacher("gene_snp_table.tsv")`. The cache is orthogonal to what is being
measured and writing one from a patch would put a second file under the same
name, so this is the undecorated function.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from cnaster.config import start_time
from cnaster.logger import get_logger
from cnaster.spatio_genomic_counts import SpatioGenomicCounts

from port.patch.reference import get_reference_genes

logger = get_logger(__name__, start_time=start_time)

GENE_COLUMN = 4
"""`cnaster` writes the gene by position -- `df_gene_snp.iloc[i, 4]`.

Recorded rather than used: this patch writes the column by name. It is here
because the position is what makes `cnaster`'s write fragile (#189), and a
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

    `-1` where there is none. `cnaster` finds this by walking backwards from
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
    """What `cnaster.omics.form_gene_snp_table` returns, without the walk.

    `verbose` is upstream's and gates a log of the genes absent from the
    reference. It is kept in the signature so a caller can be pointed at
    either function, and does nothing here for the same reason it does little
    there.
    """
    logger.info("Forming gene & snp meta data.")

    # NB `port.patch.reference`, not `cnaster.reference`: the read is 13.3x
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

    `cnaster` merges them by walking the gene rows and extending the last
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

    `cnaster` finds this the other way round -- for each merged interval, a
    full-table `np.where` over the overlap condition, then the first and last
    row it matched. That is `n_intervals` passes over `n_rows`, which at a
    slide's scale is the product of two large numbers where the table is
    already sorted.

    The intervals tile the rows in order, which `cnaster` asserts, so the
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
    """What `cnaster.omics.assign_initial_blocks` returns, in three passes.

    `cnaster` takes two quadratic loops to get here:

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
    from port.patch.omics.summaries import summarize_blocks

    if "known_id" in df_gene_snp.columns:
        from cnaster.omics import assign_initial_blocks as upstream

        return upstream(
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


def _positions(
    values: np.ndarray, vocabulary: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Where each of `values` sits in `vocabulary`, and which ones are in it.

    A dictionary comprehension over a column is a Python loop per row, which
    is the cost the products below exist to remove -- so the lookup is a
    `searchsorted` against the sorted vocabulary instead.

    A missing value -- a SNP row with no gene, `None` -- is unknown. It is
    swapped for `""` before the search, because `None` does not order against
    a string and `searchsorted` raises on it where upstream's lookup skips it.
    """
    order = np.argsort(vocabulary)
    ordered = vocabulary[order]

    missing = pd.isna(values)
    if missing.any():
        values = np.where(missing, "", values)

    slot = np.clip(np.searchsorted(ordered, values), 0, ordered.size - 1)
    known = (ordered[slot] == values) & ~missing

    return order[slot], known


def _group_indicator(
    rows: np.ndarray, groups: np.ndarray, n_rows: int, n_groups: int
) -> Any:
    """A sparse `(n_rows, n_groups)` 0/1 matrix marking each row's group.

    Summing a matrix's columns by group is a product with this. `cnaster`
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
a real trade rather than a free choice -- at 2,500 spots, against `cnaster`'s
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
    """What `cnaster.omics.summarize_counts_for_blocks` returns, in three products.

    `cnaster` loops the blocks and, for each one, slices every spot's column
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
    """What `cnaster.omics.summarize_counts_for_bins` returns, in two products.

    The same defect one level up: `cnaster` loops the bins, gathers each one's
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
