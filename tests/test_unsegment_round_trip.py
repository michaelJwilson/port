"""A binned fixture, unsegmented and re-binned by cnaster's `summarize_counts_for_bins`
(#68).

Referee: the fixture, bitwise, since the aggregation sums integers.
"""

from typing import Any

import numpy as np
import pytest
from port.sim.truth import core_inference_truth
from port.sim.unsegment import Unsegmented, unsegment


def _rebin(pre_image: Unsegmented) -> Any:
    """`cnaster`'s own aggregation, on the pre-image."""
    from cnaster.omics import summarize_counts_for_bins

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
@pytest.mark.preprocessing
@pytest.mark.critical
@pytest.mark.parametrize("n_obs", [60, 240])
def test_the_round_trip_returns_the_binned_fixture(n_obs: int) -> None:
    """Both channels come back bitwise, at two bin counts."""
    truth = core_inference_truth(
        n_clones=2, n_states=3, lattice=(6, 6), n_obs=n_obs, n_segments=2
    )
    rebinned = _rebin(unsegment(truth))

    np.testing.assert_array_equal(rebinned.X[:, 0, :], truth.counts_nb.astype(np.int64))
    np.testing.assert_array_equal(rebinned.X[:, 1, :], truth.counts_bb.astype(np.int64))
    np.testing.assert_array_equal(
        rebinned.total_bb_RD, truth.total_bb_RD.astype(np.int64)
    )


@pytest.mark.snapshot
@pytest.mark.preprocessing
def test_the_round_trip_returns_the_segmentation() -> None:
    """`lengths` comes back (#67)."""
    truth = core_inference_truth(
        n_clones=2, n_states=3, lattice=(6, 6), n_obs=240, n_segments=4
    )
    rebinned = _rebin(unsegment(truth))

    np.testing.assert_array_equal(rebinned.lengths, truth.lengths)


@pytest.mark.snapshot
@pytest.mark.preprocessing
def test_the_unassigned_genes_never_reach_a_bin() -> None:
    """Counts outside the table's assignment are dropped, not summed."""
    truth = core_inference_truth(
        n_clones=2, n_states=3, lattice=(6, 6), n_obs=60, n_segments=2
    )
    pre_image = unsegment(truth)

    unassigned = pre_image.df_gene_snp.bin_id.isnull().sum()
    assert unassigned == 25

    total_in_matrix = pre_image.adata.layers["count"].sum()
    assert total_in_matrix > truth.counts_nb.sum()

    rebinned = _rebin(pre_image)
    np.testing.assert_array_equal(rebinned.X[:, 0, :].sum(), truth.counts_nb.sum())


@pytest.mark.snapshot
@pytest.mark.preprocessing
def test_the_flipped_blocks_are_unflipped_by_the_binner() -> None:
    """`phase_indicator` is read: forcing it true changes the result."""
    from dataclasses import replace

    truth = core_inference_truth(
        n_clones=2, n_states=3, lattice=(6, 6), n_obs=60, n_segments=2
    )
    pre_image = unsegment(truth)
    assert not pre_image.phase_indicator.all(), "no block is flipped"

    honest = _rebin(pre_image)
    lying = _rebin(
        replace(pre_image, phase_indicator=np.ones_like(pre_image.phase_indicator))
    )

    np.testing.assert_array_equal(honest.X[:, 1, :], truth.counts_bb.astype(np.int64))
    assert not np.array_equal(lying.X[:, 1, :], honest.X[:, 1, :])


@pytest.mark.smoke
@pytest.mark.preprocessing
def test_the_pre_image_is_a_partition_and_not_a_copy() -> None:
    """Each bin is split across several genes and blocks, with varying counts."""
    truth = core_inference_truth(
        n_clones=2, n_states=3, lattice=(6, 6), n_obs=240, n_segments=4
    )
    table = unsegment(truth).df_gene_snp
    assigned = table[table.bin_id.notnull()]

    genes = assigned.groupby("bin_id")["gene"].count()
    blocks = assigned.groupby("bin_id")["snp_id"].count()

    assert genes.max() > genes.min() > 0, f"genes per bin: {genes.min()}-{genes.max()}"
    assert blocks.max() > blocks.min() > 0, (
        f"blocks per bin: {blocks.min()}-{blocks.max()}"
    )
    assert genes.mean() > 1.5, "the expression split is close to a copy"
