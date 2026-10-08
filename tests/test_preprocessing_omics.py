"""`port.patch.omics.blocks.form_gene_snp_table` against cnaster's, bitwise (#190).

The window search -- nearest preceding gene within `num_preceeding_rows` rows, same
chromosome, containing the SNP -- is also checked against the intervals.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from port.sim.run_config import PlantedInstance

pytestmark = pytest.mark.preprocessing


@pytest.fixture(scope="module")
def both_tables(
    planted_instance: PlantedInstance,
    gate_config: Any,
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
    """Every column, row and index entry equals cnaster's frame."""
    reference, realized = both_tables

    pd.testing.assert_frame_equal(realized, reference)


@pytest.mark.patch
def test_every_snp_kept_got_the_gene_that_contains_it(
    both_tables: tuple[Any, Any],
) -> None:
    """Each surviving SNP's gene satisfies the containment condition, recomputed from
    the intervals.
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
    """The nearest preceding containing gene is chosen and the search stops at the
    chromosome.

    Row 2 (the SNP) lies in `outer` and `inner`; row 4, on chromosome 2, lies in `later`
    numerically only.
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
    """A gene further back than `num_preceeding_rows` rows is not found, as in cnaster."""
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
"""SNP-UMI thresholds; the last two exercise the second-level grow-and-merge branch."""


@pytest.fixture(scope="module")
def staged(
    planted_instance: PlantedInstance,
    gate_config: Any,
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
    """Every block id equals cnaster's at four thresholds, bitwise.

    A fresh copy per call: cnaster writes its block columns by position (#189).
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
    """The running reach resets per chromosome, so the second chromosome yields two
    intervals.
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
    """The run's reach is its largest end, so `(0, 100), (50, 60), (70, 200)` is one
    interval.
    """
    from port.patch.omics.blocks import merged_gene_intervals

    chromosome = np.ones(4, dtype=int)
    start = np.array([0, 50, 70, 300])
    end = np.array([100, 60, 200, 400])

    np.testing.assert_array_equal(merged_gene_intervals(chromosome, start, end), [0, 3])


@pytest.mark.patch
def test_a_known_segmentation_is_cnasters_under_the_swaps(
    staged: tuple[Any, Any, Any],
) -> None:
    """With `known_id`, the swapped name returns what cnaster does, without recursing
    (#466).
    """
    from cnaster.omics import assign_initial_blocks as upstream
    from port.pipeline import SWAPS, patched

    loaded, table, _ = staged
    alleles = (loaded.cell_snp_Aallele, loaded.cell_snp_Ballele)
    known = table.copy()
    known["known_id"] = (np.arange(len(known)) >= len(known) // 2).astype(int)

    reference = upstream(known.copy(), loaded.adata, *alleles, loaded.unique_snp_ids, 1)

    import cnaster.omics

    with patched(SWAPS):
        realized = cnaster.omics.assign_initial_blocks(
            known.copy(), loaded.adata, *alleles, loaded.unique_snp_ids, 1
        )

    pd.testing.assert_frame_equal(realized, reference)
