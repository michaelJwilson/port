"""A binned fixture, unsegmented and binned back (issue #68; the rows #392 stage 1 folds).

`run_cnamaste` starts at genes and SNPs; the fixture plants at bins. `unsegment`
builds a **pre-image** -- gene-level and block-level counts that `cnamaste`'s
own `summarize_counts_for_bins` carries back to exactly the bins they came
from. The referee is the fixture, and the bar is bitwise: the aggregation sums
integers, so nothing may move.

That makes the segmentation code testable without inverting it, and it is what
lets a temporary-file fixture for `run_cnamaste` carry planted truth all the way
down: the truth is stated at the bins, written out at the genes and SNPs, and
the code under test is what puts it back.
"""

from typing import Any

import numpy as np
import pytest
from sim.truth import planted
from sim.unsegment import Unsegmented, unsegment


def _rebin(pre_image: Unsegmented) -> Any:
    """`cnamaste`'s own aggregation, on the pre-image."""
    from cnamaste.omics import summarize_counts_for_bins

    return summarize_counts_for_bins(
        pre_image.df_gene_snp,
        pre_image.adata,
        pre_image.block_single_X,
        pre_image.block_single_total_bb_RD,
        pre_image.phase_indicator,
        nu=1.0,
        logphase_shift=0.0,
        geneticmap_file=None,
    )


@pytest.mark.end2end
@pytest.mark.parametrize("n_obs", [60, 240])
def test_the_round_trip_returns_the_binned_fixture(n_obs: int) -> None:
    """Both channels come back bitwise, at two bin counts.

    Swept because the partition is drawn per bin: a single size could land on
    a run where every bin happened to take the same number of genes, and the
    variable grouping is half of what this checks.
    """
    truth = planted(n_clones=2, n_states=3, lattice=(6, 5), n_obs=n_obs, n_segments=2)
    rebinned = _rebin(unsegment(truth))

    np.testing.assert_array_equal(rebinned.X[:, 0, :], truth.counts_nb.astype(np.int64))
    np.testing.assert_array_equal(rebinned.X[:, 1, :], truth.counts_bb.astype(np.int64))
    np.testing.assert_array_equal(
        rebinned.total_bb_RD, truth.total_bb_RD.astype(np.int64)
    )


@pytest.mark.end2end
def test_the_round_trip_returns_the_segmentation() -> None:
    """`lengths` comes back, which is what #67's decision is about.

    The binner counts distinct bins per chromosome in order of first
    appearance, so a pre-image that grouped the rows differently would return
    the right counts against the wrong segmentation.
    """
    truth = planted(n_clones=2, n_states=3, lattice=(6, 5), n_obs=240, n_segments=4)
    rebinned = _rebin(unsegment(truth))

    np.testing.assert_array_equal(rebinned.lengths, truth.lengths)


@pytest.mark.end2end
def test_the_unassigned_genes_never_reach_a_bin() -> None:
    """Counts outside the table's assignment are dropped, not summed.

    `unsegment` plants 25 genes carrying counts and no `bin_id`. A binner that
    summed `adata` by position rather than by the grouping the table declares
    would pass every other assertion here and fail this one, because those
    counts would land somewhere.

    Asserted by construction and by consequence: the pre-image's gene matrix
    carries strictly more than the fixture, and the round trip still returns
    the fixture exactly.
    """
    truth = planted(n_clones=2, n_states=3, lattice=(6, 5), n_obs=60, n_segments=2)
    pre_image = unsegment(truth)

    unassigned = pre_image.df_gene_snp.bin_id.isnull().sum()
    assert unassigned == 25

    total_in_matrix = pre_image.adata.layers["count"].sum()
    assert total_in_matrix > truth.counts_nb.sum()

    rebinned = _rebin(pre_image)
    np.testing.assert_array_equal(rebinned.X[:, 0, :].sum(), truth.counts_nb.sum())


@pytest.mark.end2end
def test_the_flipped_blocks_are_unflipped_by_the_binner() -> None:
    """`phase_indicator` is read, not assumed true.

    Every third block is stored on the opposite haplotype, so the binner has
    to apply `total - B` to recover it. Driven rather than asserted: the same
    pre-image with every indicator forced true returns a **different** answer,
    and that difference is the branch.
    """
    from dataclasses import replace

    truth = planted(n_clones=2, n_states=3, lattice=(6, 5), n_obs=60, n_segments=2)
    pre_image = unsegment(truth)
    assert not pre_image.phase_indicator.all(), "no block is flipped"

    honest = _rebin(pre_image)
    lying = _rebin(
        replace(pre_image, phase_indicator=np.ones_like(pre_image.phase_indicator))
    )

    np.testing.assert_array_equal(honest.X[:, 1, :], truth.counts_bb.astype(np.int64))
    assert not np.array_equal(lying.X[:, 1, :], honest.X[:, 1, :])


@pytest.mark.sim
def test_the_pre_image_is_a_partition_and_not_a_copy() -> None:
    """Each bin is split across several genes and blocks, and the counts vary.

    One gene per bin would make the aggregation a copy, and a copy round-trips
    under an implementation that picks rather than sums. Pinned so a change to
    `unsegment` that flattened the split would fail here rather than quietly
    weaken every test above.
    """
    truth = planted(n_clones=2, n_states=3, lattice=(6, 5), n_obs=240, n_segments=4)
    table = unsegment(truth).df_gene_snp
    assigned = table[table.bin_id.notnull()]

    genes = assigned.groupby("bin_id")["gene"].count()
    blocks = assigned.groupby("bin_id")["snp_id"].count()

    assert genes.max() > genes.min() > 0, f"genes per bin: {genes.min()}-{genes.max()}"
    assert blocks.max() > blocks.min() > 0, (
        f"blocks per bin: {blocks.min()}-{blocks.max()}"
    )
    assert genes.mean() > 1.5, "the expression split is close to a copy"
