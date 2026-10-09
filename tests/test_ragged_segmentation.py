"""Ragged chromosome lengths and per-chromosome segmentation, checked against `Ragged`
(#667).
"""

import numpy as np
import pytest
from cnaster.hmrf_utils import clone_stack_obs
from port.sim.truth import (
    MINIMUM_SEGMENT,
    core_inference_truth,
    dev_instance,
    ragged_lengths,
)
from sal.ragged import Ragged

BOUNDARY_SEGMENTS = 200
"""Segments in the restart instance: 200 over three clones gives 597 boundaries."""


@pytest.mark.end2end
@pytest.mark.critical
def test_the_dev_instance_plants_unequal_chromosomes() -> None:
    """Ten chromosomes, 50 to 182 bins, summing to the genome, pinned as realized."""
    truth = dev_instance()

    np.testing.assert_array_equal(
        truth.lengths, [69, 150, 53, 100, 112, 182, 107, 50, 118, 59]
    )
    assert truth.lengths.sum() == truth.n_obs
    assert truth.lengths.max() / truth.lengths.min() == pytest.approx(3.64, abs=0.01)


@pytest.mark.smoke
def test_upstream_accepts_the_planted_genome() -> None:
    """The partition constructs as upstream's `Ragged`."""
    truth = dev_instance()
    batch = truth.ragged

    assert batch.n_segments == truth.lengths.size
    assert not batch.rectangular
    assert batch.offsets[-1] == truth.n_obs
    assert [len(segment) for segment in batch.segments()] == list(truth.lengths)


@pytest.mark.smoke
def test_a_chromosome_below_the_floor_is_refused() -> None:
    """Upstream admits one bin since sal #1233; `ragged_lengths` keeps its floor of two."""
    assert Ragged(values=np.zeros(4), lengths=(1, 3)).n_segments == 2
    assert MINIMUM_SEGMENT == 2

    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="do not fit"):
        ragged_lengths(3, 4, rng=rng)

    # NB a genome barely wider than the floor: no draw goes below it.
    for seed in range(1000):
        drawn = ragged_lengths(30, 10, rng=np.random.default_rng(seed))
        assert drawn.min() >= MINIMUM_SEGMENT
        assert drawn.sum() == 30


@pytest.mark.end2end
def test_the_partition_is_exact_and_reproducible() -> None:
    """It sums to the genome, and the same seed gives the same chromosomes."""
    for seed in (0, 7, 11):
        first = ragged_lengths(1000, 10, rng=np.random.default_rng(seed))
        again = ragged_lengths(1000, 10, rng=np.random.default_rng(seed))

        np.testing.assert_array_equal(first, again)
        assert first.sum() == 1000


@pytest.mark.end2end
def test_no_event_crosses_a_chromosome_boundary() -> None:
    """No placed event spans a chromosome boundary (#120)."""
    truth = core_inference_truth(
        n_clones=3,
        n_states=4,
        lattice=(6, 10),
        n_obs=2_000,
        n_segments=BOUNDARY_SEGMENTS,
        seed=3,
    )

    edges = np.concatenate(([0], np.cumsum(truth.lengths)))
    placed = 0

    for clone, events in enumerate(truth.events):
        for chromosome, offset, extent, state in events:
            start, stop = int(edges[chromosome]), int(edges[chromosome + 1])
            assert start <= offset, f"clone {clone}: event starts before {chromosome}"
            assert offset + extent <= stop, (
                f"clone {clone}: an event of {extent} bins at {offset} runs "
                f"past chromosome {chromosome}, which ends at {stop}"
            )
            assert state != 0, "an event that plants the neutral state is not an event"
            placed += 1

    assert placed > 0, "no events were placed, so nothing was checked"


@pytest.mark.snapshot
def test_the_clone_stacked_lengths_are_cnasters_own() -> None:
    """`stacked_lengths` tiles the ragged segmentation per clone, as `clone_stack_obs` does (#97)."""

    truth = dev_instance()
    n_clones, n_obs = truth.n_clones, truth.n_obs

    counts = np.zeros((n_obs, 2, n_clones))
    exposure = np.ones((n_obs, n_clones))
    trials = np.ones((n_obs, n_clones))
    sitewise = np.zeros((n_obs, 2))

    _, _, _, stacked, _, _ = clone_stack_obs(
        counts, exposure, trials, truth.lengths, sitewise, None
    )

    np.testing.assert_array_equal(stacked, truth.stacked_lengths())
    assert stacked.size == n_clones * truth.lengths.size
    assert int(stacked.sum()) == n_clones * n_obs


@pytest.mark.end2end
def test_the_equal_mode_is_still_reachable_and_still_refuses() -> None:
    """Rectangular lengths remain a named mode and keep the divisibility check (#97)."""
    truth = core_inference_truth(n_obs=240, n_segments=4, segmentation="equal", seed=5)

    np.testing.assert_array_equal(truth.lengths, [60, 60, 60, 60])
    assert truth.ragged.rectangular

    with pytest.raises(ValueError, match="do not partition"):
        core_inference_truth(n_obs=250, n_segments=4, segmentation="equal")

    with pytest.raises(ValueError, match="unknown segmentation"):
        core_inference_truth(segmentation="chromosomal")
