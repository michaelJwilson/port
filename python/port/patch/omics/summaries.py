"""`cnaster.omics.summarize_blocks`, computed rather than looped (#191).

**0.465 s of `assign_initial_blocks`' 1.183 s, twice over, and the function
returns nothing: every figure it computes exists to be logged.**

`cnaster` builds its summary with a `groupby(...).agg(...)` over four Python
lambdas and then loops the blocks, taking a fancy-indexed column slice of the
full count matrix per block:

```python
block_sum = count_matrix[:, gene_idx].sum()
snp_umis[idx] = cell_snp_Aallele[:, snp_idx].sum() + cell_snp_Ballele[:, snp_idx].sum()
```

Each of those is a segment sum of per-gene and per-SNP column totals, so all
of them together are two passes over the matrices and a `bincount`.

**The log lines are the contract here, not a return value**, so that is what
the equivalence test compares: `tests/test_summaries_patch.py` captures both
functions' output and asserts the lines are identical.

Two details that a faster summary gets wrong if it is not looking for them:

*   the genes of a block are taken as a **set**, so a gene name appearing on
    two rows of one block contributes its UMIs once;
*   the SNPs are taken as a **list**, so a repeated `snp_id` contributes
    twice. The two are not symmetric and upstream's aggregation says so.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from cnaster.config import start_time
from cnaster.logger import get_logger

logger = get_logger(__name__, start_time=start_time)

MAX_ROWS = 25
"""How many blocks the breakdown lists. `cnaster`'s own magic number."""


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
    """`cnaster`'s `block_summary` frame, without the per-block slice."""
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
    """What `cnaster.omics.summarize_blocks` logs, line for line."""
    logger.info("Summarizing blocks ...")

    if block_key is None:  # invariant
        msg = "block_key must be specified"
        raise AssertionError(msg)
    if block_key not in gene_snp_table.columns:  # invariant
        msg = f"{block_key} not in DataFrame"
        raise AssertionError(msg)

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
