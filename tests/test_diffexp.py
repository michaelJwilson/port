"""The differential-expression filter, fixed and reconnected (#440).

Three referees. `cnaster`'s own filter on bins of one gene each, where its
separator defect cannot bite (`patch`, bitwise, with genes planted to be
flagged). A brute-force sum over each multi-gene bin's unflagged genes
(`oracle`). And the reconnection: the bins summed after the filter read
depth from every gene but the flagged ones (`oracle`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

N_SPOTS = 400
N_GENES = 300
FOLD = 300.0
"""A tumour-spot fold that stays above `logfcthreshold_t = 4` after normalization.

The filter compares library-normalized expression, so a gene raised in the
tumour spots also raises their library size; two genes at 300x against a
300-gene background leave a normalized log fold change of about 5.
"""


def _expression(seed: int = 7) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Spots x genes, a normal half and a tumour half, two high genes up `FOLD` in tumour."""
    rng = np.random.default_rng(seed)
    genes = [f"gene_{i}" for i in range(N_GENES)]
    normal = np.arange(N_SPOTS) < N_SPOTS // 2

    rate = rng.uniform(2.0, 8.0, N_GENES)
    rate[:2] = 25.0  # NB above the 80th UMI percentile before the fold.
    means = np.tile(rate, (N_SPOTS, 1))
    planted = genes[:2]
    means[np.ix_(~normal, np.arange(2))] *= FOLD

    counts = rng.poisson(means)
    return pd.DataFrame(counts, columns=genes), normal, planted


def _bins(genes: list[str], per_bin: int) -> pd.DataFrame:
    groups = [genes[i : i + per_bin] for i in range(0, len(genes), per_bin)]
    return pd.DataFrame({"INCLUDED_GENES": [",".join(g) for g in groups]})


@pytest.mark.cnaster
@pytest.mark.patch
def test_one_gene_bins_are_cnasters_bitwise_with_genes_flagged() -> None:
    """On one-gene bins `cnaster`'s filter is right; the drop-in agrees to the bit, and flags the planted genes."""
    from cnaster.normal_spot import filter_normal_diffexp as upstream
    from port.patch.normal_spot import filter_normal_diffexp, flagged_genes

    counts, normal, planted = _expression()
    bins = _bins(list(counts.columns), 1)

    theirs = np.asarray(upstream(counts, bins, normal))
    ours = filter_normal_diffexp(counts, bins, normal)

    np.testing.assert_array_equal(ours, theirs)
    assert set(planted) <= flagged_genes(counts, normal)


@pytest.mark.oracle
def test_a_multi_gene_bin_keeps_its_unflagged_genes() -> None:
    """Each bin's read depth is the sum of its genes' counts, the flagged ones left out."""
    from port.patch.normal_spot import filter_normal_diffexp, flagged_genes

    counts, normal, _ = _expression()
    bins = _bins(list(counts.columns), 4)
    flagged = flagged_genes(counts, normal)

    ours = filter_normal_diffexp(counts, bins, normal)

    for b, text in enumerate(bins.INCLUDED_GENES):
        kept = [g for g in text.split(",") if g not in flagged]
        np.testing.assert_array_equal(ours[b], counts[kept].sum(axis=1).to_numpy())

    assert np.all(ours.sum(axis=1) > 0), "no bin may be summed over nothing"


@pytest.mark.oracle
def test_the_bins_summed_after_the_filter_leave_the_flagged_genes_out() -> None:
    """`summarize_counts_for_bins` inside a recording reads depth from every gene but the flagged."""
    import anndata
    from port.extensions.segments import recording
    from port.patch.omics import summarize_counts_for_bins

    counts, _, _ = _expression()
    genes = list(counts.columns)
    starts = np.arange(len(genes)) * 1_000
    table = pd.DataFrame(
        {
            "CHR": 1,
            "START": starts,
            "END": starts + 500,
            "gene": genes,
            "snp_id": None,
            "is_interval": True,
            "block_id": np.arange(len(genes)),
            "bin_id": np.arange(len(genes)) // 4,
        }
    )
    adata = anndata.AnnData(counts.to_numpy().astype(float))
    adata.var.index = genes
    adata.layers["count"] = counts.to_numpy()
    blocks_X = np.zeros((len(genes), 2, N_SPOTS), dtype=int)
    blocks_total = np.zeros((len(genes), N_SPOTS), dtype=int)

    excluded = {"gene_1", "gene_6", "gene_7"}

    with recording() as lineage:
        lineage.excluded_genes |= excluded
        summed: Any = summarize_counts_for_bins(
            table,
            adata,
            blocks_X,
            blocks_total,
            np.ones(len(genes), dtype=bool),
            1.0,
            0.0,
            None,
        )

    for b in range(len(genes) // 4):
        kept = [g for g in genes[4 * b : 4 * b + 4] if g not in excluded]
        np.testing.assert_array_equal(
            summed.X[b, 0, :], counts[kept].sum(axis=1).to_numpy()
        )
