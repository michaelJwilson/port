"""`cnaster` derives its bins from written coordinates, and they are the planted ones
(#92).

The `lengths` vector is asserted before the counts.
"""

from pathlib import Path

import numpy as np
import pytest
from port.sim.inputs import Binned, read_to_bins, write_tmp_inputs, written_config
from port.sim.truth import CoreInferenceTruth, core_inference_truth
from port.sim.unsegment import unsegment

pytestmark = [pytest.mark.preprocessing]

INITIAL_MIN_UMI = 1
SECONDARY_MIN_UMI = 1
"""The floors `assign_initial_blocks` and `create_bin_ranges` apply, at one so no block
merges.
"""


def _prepared(tmp_path: Path, n_obs: int = 20) -> tuple[CoreInferenceTruth, Binned]:
    """Drive `run_cnaster`'s prep chain in order under one global config, and bin at the
    end.
    """
    truth = core_inference_truth(
        n_clones=2, n_states=3, lattice=(6, 6), n_obs=n_obs, n_segments=2
    )
    # No flipped haplotype: phase is `cnaster`'s to infer.
    written = write_tmp_inputs(truth, unsegment(truth, flip_every=0), tmp_path)

    with written_config(written):
        chain = read_to_bins(
            written,
            initial_min_umi=INITIAL_MIN_UMI,
            secondary_min_umi=SECONDARY_MIN_UMI,
        )

    return truth, chain


@pytest.mark.snapshot
def test_the_blocks_are_one_per_planted_bin(tmp_path: Path) -> None:
    """`assign_initial_blocks` gives one block per planted bin."""
    truth, prepared = _prepared(tmp_path)

    assert int(prepared.table.block_id.dropna().nunique()) == truth.n_obs
    np.testing.assert_array_equal(
        prepared.blocks.X[:, 0, :], truth.counts_nb.astype(np.int64)
    )
    np.testing.assert_array_equal(
        prepared.blocks.total_bb_RD, truth.total_bb_RD.astype(np.int64)
    )


@pytest.mark.end2end
@pytest.mark.critical
def test_the_derived_segmentation_is_the_planted_one(tmp_path: Path) -> None:
    """`lengths`, derived from coordinates alone, equals the planted segmentation."""
    truth, prepared = _prepared(tmp_path, n_obs=20)

    assert int(prepared.table.bin_id.dropna().nunique()) == truth.n_obs
    np.testing.assert_array_equal(prepared.bins.lengths, truth.lengths)


@pytest.mark.snapshot
def test_the_counts_in_the_derived_bins_are_the_planted_ones(tmp_path: Path) -> None:
    """All three channels in `cnaster`'s derived bins equal the planted counts bitwise."""
    truth, prepared = _prepared(tmp_path, n_obs=20)

    np.testing.assert_array_equal(
        prepared.bins.X[:, 0, :], truth.counts_nb.astype(np.int64)
    )
    np.testing.assert_array_equal(
        prepared.bins.X[:, 1, :], truth.counts_bb.astype(np.int64)
    )
    np.testing.assert_array_equal(
        prepared.bins.total_bb_RD, truth.total_bb_RD.astype(np.int64)
    )
