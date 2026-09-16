"""Take a binned fixture back to the gene and SNP level it was binned from.

Issue #68. `run_cnaster` starts at SNPs and genes; `tests/fixtures.py` plants
at bins. Inverting `cnaster`'s binning statistically is not possible -- it
sums -- but it does not have to be: a **pre-image** is enough. `unsegment`
constructs gene-level and block-level counts that `cnaster`'s own aggregation
carries back to exactly the bins they came from, so the round trip is the
referee and it is bitwise.

What `summarize_counts_for_bins` does, and therefore what has to be inverted
(`omics.py:912-1020`):

*   **Expression.** `bin_single_X[b, 0, :]` is the sum of
    `adata.layers["count"]` over the genes assigned to bin `b`.
*   **Alleles.** For the blocks in bin `b`, the B count is
    `where(phase_indicator, X[:, 1], total - X[:, 1])` summed over blocks, and
    the total is the blocks' totals summed.
*   **Lengths.** Bins per chromosome, in order of first appearance.
*   **Baseline.** `bin_single_base_nb_mean` is returned as **zeros**. It is
    not derived from anything this function reads, so it is not part of the
    round trip and `base_nb_mean` is not claimed to survive one.

The split across genes and blocks is deliberately non-trivial. One gene per
bin would make the aggregation a copy, and a copy round-trips under an
implementation that picks rather than sums.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from tests.fixtures import CoreInferenceTruth

GENES_PER_BIN = (1, 6)
"""Genes per bin, drawn in this half-open range rather than fixed.

A constant count makes every bin the same shape, so an aggregation that
struck the right total by construction -- summing a fixed stride, say --
would round-trip. Variable counts mean the grouping has to be read from the
table.
"""

BLOCKS_PER_BIN = (1, 4)
"""Blocks per bin, likewise. At least one: a bin with no block aggregates to
zero alleles, which could not carry a fixture's non-zero counts back."""

UNASSIGNED_GENES = 25
"""Genes carrying counts and **no** `bin_id`.

`summarize_counts_for_bins` filters on `~df_gene_snp.bin_id.isnull()` before
grouping, so these must never reach a bin. They are the half of the contract
a partition alone cannot test: a binner that summed every gene in `adata`
rather than the genes the table assigns would pass every other assertion here
and fail on these.
"""


@dataclass(frozen=True)
class Unsegmented:
    """A pre-image of a binned fixture, in the arguments the binner takes.

    Parameters
    ----------
    df_gene_snp : pd.DataFrame
        One row per gene and per SNP, carrying `CHR`, `START`, `END`, `gene`,
        `snp_id`, `block_id` and `bin_id` -- the columns
        `summarize_counts_for_bins` groups by.
    adata : object
        An `AnnData` whose `layers["count"]` is `(n_spots, n_genes)` and whose
        `var.index` names the genes `df_gene_snp` refers to.
    block_single_X, block_single_total_bb_RD : np.ndarray
        Block-level allele counts, `(n_blocks, 2, n_spots)` and
        `(n_blocks, n_spots)`.
    phase_indicator : np.ndarray
        Whether each block's B count is already on the H0 haplotype. **Not all
        true**: the flipped blocks are what make the round trip exercise the
        phasing branch rather than skip it.
    """

    df_gene_snp: pd.DataFrame
    adata: Any
    block_single_X: np.ndarray
    block_single_total_bb_RD: np.ndarray
    phase_indicator: np.ndarray


def _split(
    total: np.ndarray, weights: np.ndarray, rng: np.random.Generator
) -> list[np.ndarray]:
    """Split a non-negative integer array into uneven parts that sum back to it.

    Proportional by `weights`, with the floor's residual handed out one unit
    at a time in a seeded order. Exact for every entry, and no float survives
    into the result -- a rounding residual is indistinguishable from an
    aggregation defect once the round trip is asserted bitwise.
    """
    share = weights / weights.sum()
    pieces = [(total * fraction).astype(np.int64) for fraction in share]

    residual = total - sum(pieces)
    order = rng.permutation(len(pieces))
    for step in range(len(pieces)):
        take = (residual > 0) & (step < residual + (residual <= 0))
        pieces[order[step]] += np.where(residual > step, 1, 0) * take.astype(np.int64)

    residual = total - sum(pieces)
    pieces[order[0]] += residual
    return pieces


def _capped_split(
    amount: np.ndarray, caps: list[np.ndarray], rng: np.random.Generator
) -> list[np.ndarray]:
    """Split `amount` across parts, never exceeding each part's `cap`.

    The allele channel needs it: a block's B count cannot exceed that block's
    total, or the flipped form `total - B` is negative and outside the
    support. Proportional to the caps first, which respects them by
    construction since `amount <= sum(caps)`, then the residual to whichever
    parts still have room.
    """
    ceiling = sum(caps)
    with np.errstate(divide="ignore", invalid="ignore"):
        pieces = [
            np.where(ceiling > 0, (amount * cap) // np.maximum(ceiling, 1), 0)
            for cap in caps
        ]

    for index in rng.permutation(len(pieces)):
        residual = amount - sum(pieces)
        room = caps[index] - pieces[index]
        pieces[index] += np.minimum(np.maximum(residual, 0), room)

    return pieces


def unsegment(
    truth: CoreInferenceTruth,
    *,
    genes_per_bin: tuple[int, int] = GENES_PER_BIN,
    blocks_per_bin: tuple[int, int] = BLOCKS_PER_BIN,
    unassigned_genes: int = UNASSIGNED_GENES,
    flip_every: int = 3,
) -> Unsegmented:
    """Build gene and block counts that bin back to `truth` exactly.

    Parameters
    ----------
    flip_every : int
        Every `flip_every`-th block is stored on the opposite haplotype with
        `phase_indicator` false, so the binner has to apply `total - B` to
        recover it. A fixture with no flipped block would pass under a binner
        that ignored `phase_indicator` entirely.

        **Zero flips none.** The files `run_cnaster` reads carry allele counts
        and not a phase, and the phase is `cnaster`'s to infer, so a fixture
        written out for that path stores the true B count everywhere. The
        flipped form is exercised where it belongs, at the binner
        (`tests/test_unsegment_round_trip.py`).

    Raises
    ------
    ValueError
        If a chromosome would carry no bin, which `lengths` could not express.
    """
    import anndata

    n_obs, n_spots = truth.n_obs, truth.n_spots
    if truth.lengths.min() < 1:
        msg = f"lengths {truth.lengths} leaves a chromosome with no bin"
        raise ValueError(msg)

    # NB which chromosome each bin belongs to, read off `lengths` rather than
    #    divided out of it. The chromosomes are unequal since #667 gave the
    #    fixture a shape that can say so, and `bin_id // bins_per_chromosome`
    #    silently re-cut a ragged genome into equal pieces -- it returned six
    #    chromosomes of [45, 45, 45, 45, 45, 15] for a planted [45, 96, 35, 64].
    chromosome_of_bin = np.repeat(np.arange(1, truth.lengths.size + 1), truth.lengths)

    counts_nb = truth.counts_nb.astype(np.int64)
    counts_bb = truth.counts_bb.astype(np.int64)
    trials = truth.total_bb_RD.astype(np.int64)

    rng = np.random.default_rng([truth.seed, n_obs])
    gene_counts_per_bin = rng.integers(*genes_per_bin, n_obs)
    blocks_per = rng.integers(*blocks_per_bin, n_obs)

    # --- expression ------------------------------------------------------
    gene_names: list[str] = []
    gene_columns: list[np.ndarray] = []
    gene_of_bin: list[list[str]] = []

    for bin_id in range(n_obs):
        parts = int(gene_counts_per_bin[bin_id])
        weights = rng.uniform(1.0, 4.0, parts)
        names = [f"gene_{bin_id}_{part}" for part in range(parts)]
        gene_of_bin.append(names)
        for name, piece in zip(
            names, _split(counts_nb[bin_id], weights, rng), strict=True
        ):
            gene_names.append(name)
            gene_columns.append(piece)

    for extra in range(unassigned_genes):
        gene_names.append(f"unassigned_{extra}")
        gene_columns.append(rng.integers(1, 50, n_spots).astype(np.int64))

    gene_counts = np.stack(gene_columns, axis=1)
    adata = anndata.AnnData(
        X=gene_counts.astype(np.float64),
        var=pd.DataFrame(index=pd.Index(gene_names, name="gene")),
        obs=pd.DataFrame(index=pd.Index([f"spot_{s}" for s in range(n_spots)])),
    )
    adata.layers["count"] = gene_counts

    # --- alleles ---------------------------------------------------------
    n_blocks = int(blocks_per.sum())
    block_X = np.zeros((n_blocks, 2, n_spots), dtype=np.int64)
    block_total = np.zeros((n_blocks, n_spots), dtype=np.int64)
    phase_indicator = np.ones(n_blocks, dtype=bool)
    block_of_bin: list[list[int]] = []

    block = 0
    for bin_id in range(n_obs):
        parts = int(blocks_per[bin_id])
        totals = _split(trials[bin_id], rng.uniform(1.0, 4.0, parts), rng)
        b_counts = _capped_split(counts_bb[bin_id], totals, rng)
        blocks: list[int] = []

        for part in range(parts):
            block_total[block] = totals[part]
            if flip_every > 0 and block % flip_every == 0:
                phase_indicator[block] = False
                block_X[block, 1, :] = totals[part] - b_counts[part]
            else:
                block_X[block, 1, :] = b_counts[part]
            blocks.append(block)
            block += 1

        block_of_bin.append(blocks)

    # --- the table the binner groups by -----------------------------------
    rows: list[dict[str, object]] = []
    for bin_id in range(n_obs):
        chromosome = f"chr{chromosome_of_bin[bin_id]}"
        for part, name in enumerate(gene_of_bin[bin_id]):
            rows.append(
                {
                    "CHR": chromosome,
                    "START": bin_id * 1000 + part,
                    "END": bin_id * 1000 + part + 1,
                    "gene": name,
                    "snp_id": None,
                    "block_id": None,
                    "bin_id": bin_id,
                }
            )
        for part, block_id in enumerate(block_of_bin[bin_id]):
            rows.append(
                {
                    "CHR": chromosome,
                    "START": bin_id * 1000 + 500 + part,
                    "END": bin_id * 1000 + 500 + part + 1,
                    "gene": None,
                    "snp_id": f"snp_{bin_id}_{part}",
                    "block_id": block_id,
                    "bin_id": bin_id,
                }
            )
    for extra in range(unassigned_genes):
        rows.append(
            {
                # On an existing chromosome, not a new one: `lengths` counts
                # distinct bins per chromosome over every row, so an
                # unassigned gene on a chromosome of its own would add a
                # zero-length entry the fixture never declared.
                "CHR": "chr1",
                "START": 10**6 + extra,
                "END": 10**6 + extra + 1,
                "gene": f"unassigned_{extra}",
                "snp_id": None,
                "block_id": None,
                "bin_id": None,
            }
        )

    table = pd.DataFrame(rows)
    # `summarize_counts_for_bins` filters block ids with `x is not None`
    # (`omics.py:982`), which only holds in an **object** column: pandas
    # represents a missing integer in a numeric column as `NaN`, which is not
    # `None`, and `np.fromiter(..., dtype=int)` then raises on it. Forced here
    # so the pre-image matches what that guard expects rather than what a
    # default `DataFrame` infers.
    for column in ("block_id", "bin_id"):
        missing = table[column].isna()
        table[column] = table[column].astype(object)
        table.loc[missing, column] = None

    return Unsegmented(
        df_gene_snp=table,
        adata=adata,
        block_single_X=block_X,
        block_single_total_bb_RD=block_total,
        phase_indicator=phase_indicator,
    )
