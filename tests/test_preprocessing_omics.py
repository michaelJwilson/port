"""`port.patch.omics.blocks.form_gene_snp_table` against cnaster's, bitwise (#190).

The window search -- nearest preceding gene within `num_preceeding_rows` rows, same
chromosome, containing the SNP -- is also checked against the intervals.
"""

from typing import Any

import cnaster.omics
import numpy as np
import pandas as pd
import pytest
from cnaster.io import load_input_data
from cnaster.omics import assign_initial_blocks as upstream
from cnaster.omics import form_gene_snp_table as cnaster_form_gene_snp_table
from port.patch.omics.blocks import assign_initial_blocks as port_assign_initial_blocks
from port.patch.omics.blocks import form_gene_snp_table as patched
from port.patch.omics.blocks import merged_gene_intervals, preceding_gene
from port.pipeline import SWAPS
from port.pipeline import patched as pipeline_patched
from port.sim.run_config import PlantedInstance

pytestmark = pytest.mark.preprocessing


@pytest.fixture(scope="module")
def both_tables(
    planted_instance: PlantedInstance,
    gate_config: Any,
) -> tuple[Any, Any]:
    """Both implementations run once on the planted instance."""

    _, _, written, _ = planted_instance
    loaded = load_input_data(gate_config)
    arguments = (loaded.unique_snp_ids, str(written.hgtable), loaded.adata)

    return cnaster_form_gene_snp_table(*arguments), patched(*arguments)


@pytest.mark.patch
def test_the_table_is_cnasters_table(both_tables: tuple[Any, Any]) -> None:
    """Every column, row and index entry equals cnaster's frame."""
    reference, realized = both_tables

    pd.testing.assert_frame_equal(realized, reference)


@pytest.mark.patch
def test_every_snp_kept_got_the_gene_that_contains_it(
    both_tables: tuple[Any, Any],
) -> None:
    """Each surviving SNP's gene satisfies the containment condition, recomputed from the intervals."""
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
    planted_instance: PlantedInstance, gate_table: tuple[Any, Any]
) -> tuple[Any, Any, Any]:
    """The loaded instance and its gene-SNP table, once for the module."""
    return *gate_table, planted_instance[2]


def _under_the_swaps(*arguments: Any) -> Any:
    """The swapped name, called as cnaster's pipeline calls it."""
    with pipeline_patched(SWAPS):
        return cnaster.omics.assign_initial_blocks(*arguments)


@pytest.mark.patch
@pytest.mark.parametrize(
    ("min_umi", "known", "assign"),
    [
        *((min_umi, False, port_assign_initial_blocks) for min_umi in MIN_UMIS),
        (1, True, _under_the_swaps),
    ],
    ids=[*(f"min-umi-{min_umi}" for min_umi in MIN_UMIS), "known-under-the-swaps"],
)
def test_the_blocks_are_cnasters_blocks(
    staged: tuple[Any, Any, Any], min_umi: int, known: bool, assign: Any
) -> None:
    """Every block id equals cnaster's at four thresholds, bitwise, and with `known_id` the swapped name returns what cnaster does, without recursing (#466); a fresh copy per call: cnaster writes its block columns by position (#189)."""

    loaded, table, _ = staged
    alleles = (loaded.cell_snp_Aallele, loaded.cell_snp_Ballele)
    if known:
        table = table.copy()
        table["known_id"] = (np.arange(len(table)) >= len(table) // 2).astype(int)

    reference = upstream(
        table.copy(), loaded.adata, *alleles, loaded.unique_snp_ids, min_umi
    )
    realized = assign(
        table.copy(), loaded.adata, *alleles, loaded.unique_snp_ids, min_umi
    )

    pd.testing.assert_frame_equal(realized, reference)


@pytest.mark.analytic
def test_the_merge_sweep_restarts_at_every_chromosome() -> None:
    """The running reach resets per chromosome, so the second chromosome yields two intervals."""

    chromosome = np.array([1, 1, 2, 2])
    start = np.array([100, 5_000, 100, 5_000])
    end = np.array([200, 9_000, 200, 9_000])

    np.testing.assert_array_equal(
        merged_gene_intervals(chromosome, start, end), [0, 1, 2, 3]
    )


@pytest.mark.analytic
def test_the_merge_sweep_joins_a_chain_of_overlaps() -> None:
    """The run's reach is its largest end, so `(0, 100), (50, 60), (70, 200)` is one interval."""

    chromosome = np.ones(4, dtype=int)
    start = np.array([0, 50, 70, 300])
    end = np.array([100, 60, 200, 400])

    np.testing.assert_array_equal(merged_gene_intervals(chromosome, start, end), [0, 3])
