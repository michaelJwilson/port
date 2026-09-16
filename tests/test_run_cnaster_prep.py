"""`cnaster` derives its own bins from the written files, and they are the planted ones (#92).

`tests/test_tmp_input_round_trip.py` hands `summarize_counts_for_bins` a table
that already carries the bin assignment. `run_cnaster` has no such column: it
reads gene and SNP **coordinates** and decides the partition itself, through
`form_gene_snp_table`, `assign_initial_blocks` and `create_bin_ranges`.

So this is the stronger claim. What is supplied is where the genes and the
SNPs are; what is recovered is the segmentation **and** the counts in it.

The partition is asserted before any count is compared, because a count
comparison over a partition nobody checked could be measuring a coincidence.
And it is asserted as the whole `lengths` vector rather than a bin total: a
different split of the same total is exactly the error this arrangement could
make, and a scalar would not see it.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.fixtures import CoreInferenceTruth, core_inference_truth
from tests.tmp_inputs import write_tmp_inputs, written_config
from tests.unsegment import unsegment

pytestmark = [pytest.mark.preprocessing, pytest.mark.cnaster]

INITIAL_MIN_UMI = 1
SECONDARY_MIN_UMI = 1
"""The floors `assign_initial_blocks` and `create_bin_ranges` apply.

At one, because the fixture's job here is the **partition**: a floor that
merged blocks would be measuring the floor, and what it does at a realistic
setting is a separate question with its own instance.
"""


def _prepared(tmp_path: Path, n_obs: int = 20) -> tuple[CoreInferenceTruth, Any]:
    """Drive the prep chain `run_cnaster` runs, in its order, and bin at the end.

    One call, because every function in the chain reads the global config and
    the chain has to hold it open across all of them.
    """
    from cnaster.omics import (
        assign_initial_blocks,
        create_bin_ranges,
        form_gene_snp_table,
        summarize_counts_for_bins,
        summarize_counts_for_blocks,
    )

    truth = core_inference_truth(
        n_clones=2, n_states=3, lattice=(6, 5), n_obs=n_obs, n_segments=2
    )
    # No flipped haplotype: the files carry allele counts, and the phase is
    # `cnaster`'s to infer. The flipped form is exercised at the binner.
    written = write_tmp_inputs(truth, unsegment(truth, flip_every=0), tmp_path)

    with written_config(written):
        from cnaster.config import get_global_config
        from cnaster.io import load_input_data

        loaded = load_input_data(get_global_config())
        alleles = (loaded.cell_snp_Aallele, loaded.cell_snp_Ballele)

        table = form_gene_snp_table(
            loaded.unique_snp_ids, str(written.hgtable), loaded.adata
        )
        table = assign_initial_blocks(
            table,
            loaded.adata,
            *alleles,
            loaded.unique_snp_ids,
            initial_min_umi=INITIAL_MIN_UMI,
        )
        blocks = summarize_counts_for_blocks(
            table, loaded.adata, *alleles, loaded.unique_snp_ids
        )
        table = create_bin_ranges(
            table,
            loaded.adata,
            *alleles,
            loaded.unique_snp_ids,
            blocks.X,
            blocks.total_bb_RD,
            blocks.lengths,
            secondary_min_umi=SECONDARY_MIN_UMI,
            secondary_min_snp_umi=SECONDARY_MIN_UMI,
            secondary_min_normal_umi=0,
        )
        binned = summarize_counts_for_bins(
            table,
            loaded.adata,
            blocks.X,
            blocks.total_bb_RD,
            np.ones(int(table.block_id.dropna().nunique()), dtype=bool),
            nu=1.0,
            logphase_shift=0.0,
            geneticmap_file=None,
        )

    return truth, Prepared(table=table, blocks=blocks, binned=binned)


@dataclass(frozen=True)
class Prepared:
    """What the chain produced: the table it built, the blocks, the bins."""

    table: Any
    blocks: Any
    binned: Any


def test_the_blocks_are_one_per_planted_bin(tmp_path: Path) -> None:
    """`assign_initial_blocks` merges overlapping intervals, and these do not.

    The first place the arrangement could go wrong: intervals spaced closer
    than a gene's length would merge, and the partition would be coarser than
    the planted one before `create_bin_ranges` ever ran.
    """
    truth, prepared = _prepared(tmp_path)

    assert int(prepared.table.block_id.dropna().nunique()) == truth.n_obs
    np.testing.assert_array_equal(
        prepared.blocks.X[:, 0, :], truth.counts_nb.astype(np.int64)
    )
    np.testing.assert_array_equal(
        prepared.blocks.total_bb_RD, truth.total_bb_RD.astype(np.int64)
    )


def test_the_derived_segmentation_is_the_planted_one(tmp_path: Path) -> None:
    """`lengths`, as a vector, from coordinates alone.

    `create_bin_ranges` groups blocks into bins under three UMI floors and a
    5 Mb cap, and `summarize_counts_for_bins` counts distinct bins per
    chromosome. Nothing here was told the segmentation; it comes back because
    the coordinates encode it.
    """
    truth, prepared = _prepared(tmp_path, n_obs=20)

    assert int(prepared.table.bin_id.dropna().nunique()) == truth.n_obs
    np.testing.assert_array_equal(prepared.binned.lengths, truth.lengths)


def test_the_counts_in_the_derived_bins_are_the_planted_ones(tmp_path: Path) -> None:
    """All three channels, bitwise, over a partition `cnaster` chose.

    The claim #92 asks for. The expression channel travels as genes through an
    `.h5ad`, the alleles as two sparse matrices, and both are summed back into
    bins the fixture never named.
    """
    truth, prepared = _prepared(tmp_path, n_obs=20)

    np.testing.assert_array_equal(
        prepared.binned.X[:, 0, :], truth.counts_nb.astype(np.int64)
    )
    np.testing.assert_array_equal(
        prepared.binned.X[:, 1, :], truth.counts_bb.astype(np.int64)
    )
    np.testing.assert_array_equal(
        prepared.binned.total_bb_RD, truth.total_bb_RD.astype(np.int64)
    )
