"""`port.patch.omics.blocks.form_gene_snp_table` against `cnaster`'s (#190).

**The same table, bitwise, without the per-SNP `pandas` write.** `cnaster`
assigns each SNP to its gene by walking backwards through the sorted table in
Python and writing the result with `df_gene_snp.iloc[i, 4] = ...`, which is
two `pandas` scalar accesses per SNP and 66% of the function at 782 of them.

The window search is what the referee has to pin, because it is the part with
semantics rather than mechanics: the nearest **preceding** gene, at most
`num_preceeding_rows` rows back, on the same chromosome, whose interval
contains the SNP. Four ways to get that wrong -- the nearest following gene,
an unbounded window, across a chromosome, or a containment test that is
inclusive at the end -- and each of them changes which SNPs survive.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.test_load_input_data_patch import (
    gate_config,  # noqa: F401  -- used by name, and it needs the one below
    planted_instance,  # noqa: F401  -- `gate_config` resolves it in this module
)

pytestmark = pytest.mark.preprocessing


@pytest.fixture(scope="module")
def both_tables(
    planted_instance: tuple[Any, Any, Any, Any],  # noqa: F811
    gate_config: Any,  # noqa: F811
) -> tuple[Any, Any]:
    """Both implementations run once on the planted instance."""
    from cnaster.io import load_input_data
    from cnaster.omics import form_gene_snp_table as upstream
    from port.patch.omics.blocks import form_gene_snp_table as patched

    _, _, written, _ = planted_instance
    loaded = load_input_data(gate_config)
    arguments = (loaded.unique_snp_ids, str(written.hgtable), loaded.adata)

    return upstream(*arguments), patched(*arguments)


@pytest.mark.patch
def test_the_table_is_cnasters_table(both_tables: tuple[Any, Any]) -> None:
    """**Every column, every row, in order.**

    Compared as a frame rather than column by column so the index comes with
    it: the rows are sorted by `(CHR, START)` and not renumbered, and a patch
    that reset the index would leave every value right and every downstream
    `iloc` range wrong.
    """
    reference, realized = both_tables

    pd.testing.assert_frame_equal(realized, reference)


@pytest.mark.patch
def test_every_snp_kept_got_the_gene_that_contains_it(
    both_tables: tuple[Any, Any],
) -> None:
    """**The assignment, checked against the intervals rather than each other.**

    Two implementations agreeing is not the claim; the claim is the window
    search. So each surviving SNP is taken back to the gene rows of the table
    and the result recomputed from the containment condition alone.
    """
    _, realized = both_tables

    snps = realized[~realized.is_interval]
    genes = realized[realized.is_interval]

    assert len(snps) > 0

    for _, snp in snps.iterrows():
        candidates = genes[
            (genes.CHR == snp.CHR)
            & (genes.START <= snp.START)
            & (genes.END > snp.START)
            & (genes.gene == snp.gene)
        ]

        assert len(candidates) > 0, (
            f"{snp.snp_id} assigned {snp.gene}, which does not contain it"
        )


@pytest.mark.analytic
def test_the_window_takes_the_nearest_preceding_gene_and_stops_at_the_edges() -> None:
    """**The four ways the search can be wrong, on a table built to catch them.**

    Rows in sorted order, with `is_interval` marking the genes:

    | row | CHR | START | END | gene |
    | --- | --- | --- | --- | --- |
    | 0 | 1 | 100 | 400 | outer |
    | 1 | 1 | 200 | 300 | inner |
    | 2 | 1 | 250 | 251 | *the SNP* |
    | 3 | 1 | 260 | 900 | later |
    | 4 | 2 | 250 | 251 | *a SNP on the next chromosome* |

    Row 2 is inside `outer`, `inner` and nothing else preceding; the nearest
    preceding is `inner`, so an implementation taking the first match forwards
    would say `outer` and one taking any match would be ambiguous. Row 4 is
    inside nothing on its own chromosome, and `later` contains its position
    numerically -- so an implementation that did not stop at the chromosome
    would assign it.
    """
    from port.patch.omics.blocks import preceding_gene

    chromosome = np.array([1, 1, 1, 1, 2])
    start = np.array([100, 200, 250, 260, 250])
    end = np.array([400, 300, 251, 900, 251])
    is_interval = np.array([True, True, False, True, False])
    unassigned = ~is_interval

    found = preceding_gene(chromosome, start, end, is_interval, unassigned, 100)

    np.testing.assert_array_equal(found, [1, -1])


@pytest.mark.analytic
def test_the_window_does_not_reach_past_its_own_length() -> None:
    """A gene further back than `num_preceeding_rows` is not found.

    `cnaster`'s window is a row count and not a distance, so a SNP separated
    from its gene by many other rows loses it. That is upstream's behaviour
    and the patch reproduces it: the alternative would be a different table,
    not a faster one.
    """
    from port.patch.omics.blocks import preceding_gene

    filler = 8
    chromosome = np.ones(filler + 2, dtype=int)
    start = np.concatenate(([100], np.arange(1_000, 1_000 + filler), [150]))
    end = np.concatenate(([400], np.arange(1_000, 1_000 + filler) + 1, [151]))
    is_interval = np.zeros(filler + 2, dtype=bool)
    is_interval[0] = True

    reachable = preceding_gene(
        chromosome, start, end, is_interval, ~is_interval, filler + 1
    )
    unreachable = preceding_gene(chromosome, start, end, is_interval, ~is_interval, 3)

    assert reachable[-1] == 0
    assert unreachable[-1] == -1


MIN_UMIS = [1, 500, 500_000, 5_000_000]
"""SNP-UMI thresholds the block assignment is compared at.

The first two leave every merged gene interval standing alone on this
instance; the last two force the second-level loop to grow a run and then to
merge it backwards, which is the branch that decides where a genome segment
ends. A comparison run only at the shipped threshold would exercise neither.
"""


@pytest.fixture(scope="module")
def staged(
    planted_instance: tuple[Any, Any, Any, Any],  # noqa: F811
    gate_config: Any,  # noqa: F811
) -> tuple[Any, Any, Any]:
    """The loaded instance and its gene-SNP table, once for the module."""
    from cnaster.io import load_input_data
    from cnaster.omics import form_gene_snp_table

    _, _, written, _ = planted_instance
    loaded = load_input_data(gate_config)
    table = form_gene_snp_table(
        loaded.unique_snp_ids, str(written.hgtable), loaded.adata
    )

    return loaded, table, written


@pytest.mark.patch
@pytest.mark.parametrize("min_umi", MIN_UMIS)
def test_the_blocks_are_cnasters_blocks(
    staged: tuple[Any, Any, Any], min_umi: int
) -> None:
    """**Every block id, bitwise, at four thresholds.**

    A fresh copy per call, because `cnaster` writes both of its block columns
    by position and a second call on one frame writes into the wrong one
    (#189). That is also why this cannot be a fixture returning one result.
    """
    from cnaster.omics import assign_initial_blocks as upstream
    from port.patch.omics.blocks import assign_initial_blocks as patched

    loaded, table, _ = staged
    alleles = (loaded.cell_snp_Aallele, loaded.cell_snp_Ballele)

    reference = upstream(
        table.copy(), loaded.adata, *alleles, loaded.unique_snp_ids, min_umi
    )
    realized = patched(
        table.copy(), loaded.adata, *alleles, loaded.unique_snp_ids, min_umi
    )

    pd.testing.assert_frame_equal(realized, reference)


@pytest.mark.analytic
def test_the_merge_sweep_restarts_at_every_chromosome() -> None:
    """**Positions restart per chromosome, and the running reach must too.**

    The defect this pins was real: a single `np.maximum.accumulate` over the
    gene rows carries the last chromosome's largest end onto the next one,
    where every start is below it, so the whole chromosome merges into one
    interval. On the dev instance that turned 400 merged intervals into 223
    and was invisible in any single-chromosome check.

    Two chromosomes below, each with two disjoint genes. The second
    chromosome's genes sit **inside** the first chromosome's span by position,
    so an unreset sweep yields two intervals where the answer is four.
    """
    from port.patch.omics.blocks import merged_gene_intervals

    chromosome = np.array([1, 1, 2, 2])
    start = np.array([100, 5_000, 100, 5_000])
    end = np.array([200, 9_000, 200, 9_000])

    np.testing.assert_array_equal(
        merged_gene_intervals(chromosome, start, end), [0, 1, 2, 3]
    )


@pytest.mark.analytic
def test_the_merge_sweep_joins_a_chain_of_overlaps() -> None:
    """A gene overlapping only the one before it still joins the same run.

    The run's reach is the largest end seen in it, not the last one, so
    `(0, 100), (50, 60), (70, 200)` is one interval: the third overlaps the
    first even though it misses the second. A sweep comparing against the
    previous row's end alone would split it.
    """
    from port.patch.omics.blocks import merged_gene_intervals

    chromosome = np.ones(4, dtype=int)
    start = np.array([0, 50, 70, 300])
    end = np.array([100, 60, 200, 400])

    np.testing.assert_array_equal(merged_gene_intervals(chromosome, start, end), [0, 3])
