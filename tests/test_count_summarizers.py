"""`port.patch.omics.blocks`'s two count summarizers against `cnaster`'s (#198).

**2.02x and 1.51x, bitwise.** Both build the same thing the same way wrong:
a grouped column sum done one group at a time, each pass slicing every spot's
column out of a full matrix. A grouped column sum is a product with a 0/1
indicator, and three products replace the first loop, two the second.

What the referee has to hold, beyond the numbers agreeing:

*   **the grouping is by set**, so a gene named twice in one block counts its
    UMIs once — a product with a plain incidence matrix would count it twice;
*   **a column in no group contributes nowhere**, which an indicator built
    from the wrong side would silently place in group zero;
*   **the phasing in the bin summary is per block, not per bin**, so lifting
    it out of the loop has to give the same answer for every phase vector --
    not just the all-true one a fixture happens to produce.
"""

from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp

from tests.test_load_input_data_patch import (
    gate_config,  # noqa: F401  -- used by name, and it needs the one below
    planted_instance,  # noqa: F401  -- `gate_config` resolves it in this module
)

pytestmark = pytest.mark.preprocessing

PHASES = ["all-true", "alternating", "all-false"]
"""Phase vectors the bin summary is compared under.

`initial_phase_given_partition` produces one of these in principle and the
dev fixture produces the first in practice, so the other two are supplied
directly: the `where` that chooses between a block's B count and its
complement is the one place lifting the loop could invert a haplotype.
"""


@pytest.fixture(scope="module")
def blocked(
    planted_instance: tuple[Any, Any, Any, Any],  # noqa: F811
    gate_config: Any,  # noqa: F811
) -> tuple[Any, Any, Any]:
    """The instance, its gene-SNP table with blocks, and the block counts."""
    from cnaster.io import load_input_data
    from cnaster.omics import (
        assign_initial_blocks,
        form_gene_snp_table,
        summarize_counts_for_blocks,
    )

    _, _, written, _ = planted_instance
    loaded = load_input_data(gate_config)
    alleles = (loaded.cell_snp_Aallele, loaded.cell_snp_Ballele)

    table = form_gene_snp_table(
        loaded.unique_snp_ids, str(written.hgtable), loaded.adata
    )
    table = assign_initial_blocks(
        table, loaded.adata, *alleles, loaded.unique_snp_ids, 1
    )
    counts = summarize_counts_for_blocks(
        table.copy(), loaded.adata, *alleles, loaded.unique_snp_ids
    )

    return loaded, table, counts


def _fields_equal(realized: Any, reference: Any) -> None:
    """Every field of a `SpatioGenomicCounts`, bitwise.

    `.keys()` and not iteration: `SpatioGenomicCounts` mimics a dictionary for
    item access but defines `__iter__` to yield its **values**, so `for key in
    counts` would hand back arrays. #196 owns that inconsistency.
    """
    for key in reference.keys():  # noqa: SIM118 -- not a dict; see above
        np.testing.assert_array_equal(
            np.asarray(realized[key]), np.asarray(reference[key]), err_msg=key
        )


@pytest.mark.patch
def test_the_block_counts_are_cnasters(blocked: tuple[Any, Any, Any]) -> None:
    """**Every field of the block summary, bitwise.**

    Including `single_X[:, 1, :]` and `single_total_bb_RD`, which upstream
    builds from the same A-allele sum computed twice. Taking it once is the
    only arithmetic this patch removes rather than reorders, so the two rows
    agreeing is what says it was the same sum.
    """
    from cnaster.omics import summarize_counts_for_blocks as upstream
    from port.patch.omics.blocks import summarize_counts_for_blocks as patched

    loaded, table, _ = blocked
    alleles = (loaded.cell_snp_Aallele, loaded.cell_snp_Ballele)
    arguments = (loaded.adata, *alleles, loaded.unique_snp_ids)

    _fields_equal(patched(table.copy(), *arguments), upstream(table.copy(), *arguments))


@pytest.mark.patch
@pytest.mark.parametrize("phase", PHASES)
def test_the_bin_counts_are_cnasters(blocked: tuple[Any, Any, Any], phase: str) -> None:
    """**Every field of the bin summary, bitwise, under three phase vectors.**"""
    from cnaster.omics import create_bin_ranges
    from cnaster.omics import summarize_counts_for_bins as upstream
    from port.patch.omics.blocks import summarize_counts_for_bins as patched

    loaded, table, counts = blocked
    alleles = (loaded.cell_snp_Aallele, loaded.cell_snp_Ballele)

    binned = create_bin_ranges(
        table.copy(),
        loaded.adata,
        *alleles,
        loaded.unique_snp_ids,
        counts.X,
        counts.total_bb_RD,
        counts.lengths,
        secondary_min_umi=1,
        secondary_min_snp_umi=1,
        secondary_min_normal_umi=0,
    )

    n_blocks = counts.X.shape[0]
    indicator = {
        "all-true": np.ones(n_blocks, dtype=bool),
        "alternating": np.arange(n_blocks) % 2 == 0,
        "all-false": np.zeros(n_blocks, dtype=bool),
    }[phase]

    def call(implementation: Any) -> Any:
        return implementation(
            binned.copy(),
            loaded.adata,
            counts.X,
            counts.total_bb_RD,
            indicator,
            nu=1.0,
            logphase_shift=0.0,
            geneticmap_file=None,
        )

    _fields_equal(call(patched), call(upstream))


@pytest.mark.patch
def test_the_bin_counts_are_cnasters_on_snps_outside_a_gene() -> None:
    """**Bitwise with SNP rows whose `gene` is `None`**, which upstream skips.

    The prep chain names a gene on every row it bins, so the test above never
    carries one; `unsegment`'s pre-image carries 112 of 327. The patch raised
    `TypeError` on them -- `searchsorted` cannot order `None` against a string
    -- until `_positions` read them as unknown (#392, found by
    `cnamaste/tests/test_unsegment_round_trip.py`).
    """
    from cnaster.omics import summarize_counts_for_bins as upstream
    from port.patch.omics.blocks import summarize_counts_for_bins as patched

    from tests.fixtures import core_inference_truth
    from tests.unsegment import unsegment

    truth = core_inference_truth(
        n_clones=2, n_states=3, lattice=(6, 5), n_obs=60, n_segments=2
    )
    pre_image = unsegment(truth)
    assert pre_image.df_gene_snp.gene.isna().any(), "no row without a gene"

    def call(implementation: Any) -> Any:
        return implementation(
            pre_image.df_gene_snp.copy(),
            pre_image.adata,
            pre_image.block_single_X,
            pre_image.block_single_total_bb_RD,
            pre_image.phase_indicator,
            nu=1.0,
            logphase_shift=0.0,
            geneticmap_file=None,
        )

    _fields_equal(call(patched), call(upstream))


@pytest.mark.analytic
def test_the_indicator_sums_each_group_and_nothing_else() -> None:
    """**The grouped sum against an explicit loop, on a matrix built to trip it.**

    Column 3 belongs to no group, column 0 is named twice in group 0, and
    group 2 is empty. A product with a plain incidence matrix double-counts
    the second; an indicator built from the wrong side places the first in
    group zero; an implementation that skipped empty groups would shorten the
    answer.
    """
    from port.patch.omics.blocks import _group_indicator, _grouped_column_sums

    matrix = np.arange(12, dtype=np.int64).reshape(3, 4)
    rows = np.array([0, 0, 1, 2])
    groups = np.array([0, 0, 0, 1])

    indicator = _group_indicator(rows, groups, 4, 3)
    realized = _grouped_column_sums(matrix, indicator)

    expected = np.stack(
        [
            matrix[:, [0, 1]].sum(axis=1),
            matrix[:, [2]].sum(axis=1),
            np.zeros(3, dtype=np.int64),
        ]
    )

    assert realized.shape == (3, 3)
    np.testing.assert_array_equal(realized, expected)


@pytest.mark.analytic
def test_the_grouped_sum_agrees_sparse_and_dense() -> None:
    """The same answer whether the counts are a matrix or an array.

    The loader returns dense today and sparse under `sparse_counts` (#186),
    and the product is `O(nnz)` on the second -- 19.7x at a slide's density.
    Which container it gets must not change the number.
    """
    from port.patch.omics.blocks import _group_indicator, _grouped_column_sums

    dense = np.arange(40, dtype=np.int64).reshape(5, 8)
    dense[dense % 3 == 0] = 0

    indicator = _group_indicator(np.arange(8), np.array([0, 0, 1, 1, 2, 2, 2, 2]), 8, 3)

    np.testing.assert_array_equal(
        _grouped_column_sums(sp.csr_matrix(dense), indicator),
        _grouped_column_sums(dense, indicator),
    )


@pytest.mark.end2end
def test_the_block_counts_are_the_planted_counts(
    planted_instance: tuple[Any, Any, Any, Any],  # noqa: F811
    blocked: tuple[Any, Any, Any],
) -> None:
    """**What the blocks carry is what the fixture planted, summed per block.**

    Two implementations agreeing says nothing about whether either is right,
    so the transcript row is taken back to the planted per-gene counts and
    re-summed over each block's genes from the table itself.
    """
    _, pre_image, written, _ = planted_instance
    loaded, table, counts = blocked

    planted = np.asarray(pre_image.adata.layers["count"])
    names = np.asarray(pre_image.adata.var.index)
    column_of = {str(name): index for index, name in enumerate(names)}
    spot_of = {str(barcode): spot for spot, barcode in enumerate(written.barcodes)}

    rows = np.array([spot_of[str(barcode)] for barcode in loaded.barcodes])
    genes = table[table.is_interval.to_numpy().astype(bool)]

    expected = np.zeros_like(counts.X[:, 0, :])

    for block, frame in genes.groupby("block_id"):
        columns = sorted(
            {column_of[str(name)] for name in frame.gene if name in column_of}
        )

        if columns:
            expected[int(block)] = planted[np.ix_(rows, columns)].sum(axis=1)

    np.testing.assert_array_equal(counts.X[:, 0, :], expected)
