"""Unequal chromosomes, and a chain that restarts at each one.

Before #667 upstream had no shape that could carry a batch of unequal
chains, so the fixture planted `np.full(n_segments, n_obs // n_segments)` and
required `n_segments` to divide `n_obs`. Worse, the segmentation was a label
rather than the truth: one Markov chain ran the length of the genome and
`lengths` was handed to `cnaster` beside it, so the planted model and the
fitted one disagreed at every boundary.

`Ragged` is what both halves are checked against here -- the shape by
constructing it, the floor by driving its refusal.
"""

import numpy as np
import pytest
from snakes_and_ladders.ragged import MINIMUM_LENGTH, Ragged

from tests.fixtures import core_inference_truth, dev_instance, ragged_lengths

BOUNDARY_SEGMENTS = 200
"""Segments in the instance that measures the restart.

Each boundary is one observation of "does the chain carry across", so the
count is the sample size, and 200 segments over three clones gives 597.
"""


@pytest.mark.end2end
@pytest.mark.planted
@pytest.mark.critical
def test_the_dev_instance_plants_unequal_chromosomes() -> None:
    """Ten chromosomes, 50 to 182 bins, summing to the genome.

    Pinned as the realized partition rather than as "they differ": a draw
    that collapsed towards equal would still satisfy the weaker claim, and
    equal chromosomes are what this fixture exists to stop planting.
    """
    truth = dev_instance()

    np.testing.assert_array_equal(
        truth.lengths, [69, 150, 53, 100, 112, 182, 107, 50, 118, 59]
    )
    assert truth.lengths.sum() == truth.n_obs
    assert truth.lengths.max() / truth.lengths.min() == pytest.approx(3.64, abs=0.01)


@pytest.mark.infra
@pytest.mark.upstream
def test_upstream_accepts_the_planted_genome() -> None:
    """The partition is a `Ragged`, and constructing it is the check.

    `Ragged` refuses a `lengths` that does not tile its values and a segment
    below the floor, so this is upstream validating the shape `port` plants
    rather than `port` asserting it about itself.

    Not in the early gate, though it is fast and it does catch a wrong answer:
    `upstream` is not an external referee under the rule the gate runs on, and
    the two tests around this one cover the same partition against the planted
    truth and against `cnaster`. The guard refused this marker before review
    did.
    """
    truth = dev_instance()
    batch = truth.ragged

    assert batch.n_segments == truth.lengths.size
    assert not batch.rectangular
    assert batch.offsets[-1] == truth.n_obs
    assert [len(segment) for segment in batch.segments()] == list(truth.lengths)


@pytest.mark.infra
@pytest.mark.upstream
def test_a_chromosome_below_the_floor_is_refused() -> None:
    """One bin is an initial distribution and no transition, so upstream says no.

    The floor is upstream's, and `ragged_lengths` takes it from there rather
    than choosing its own: `MINIMUM_SEGMENT` is an alias, so a change upstream
    is a change here.
    """
    with pytest.raises(ValueError, match="at least 2 positions"):
        Ragged(values=np.zeros(4), lengths=(1, 3))

    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="do not fit"):
        ragged_lengths(3, 4, rng=rng)

    # NB the draw itself never reaches the floor from above: a thousand
    #    partitions of a genome barely wider than the floor, all legal.
    for seed in range(1000):
        drawn = ragged_lengths(30, 10, rng=np.random.default_rng(seed))
        assert drawn.min() >= MINIMUM_LENGTH
        assert drawn.sum() == 30


@pytest.mark.end2end
@pytest.mark.planted
def test_the_partition_is_exact_and_reproducible() -> None:
    """It sums to the genome, and the same seed gives the same chromosomes."""
    for seed in (0, 7, 11):
        first = ragged_lengths(1000, 10, rng=np.random.default_rng(seed))
        again = ragged_lengths(1000, 10, rng=np.random.default_rng(seed))

        np.testing.assert_array_equal(first, again)
        assert first.sum() == 1000


@pytest.mark.end2end
@pytest.mark.planted
def test_no_event_crosses_a_chromosome_boundary() -> None:
    """What `lengths` means once the path is events on a neutral backbone.

    This replaces a measurement of chain restart, and the replacement is
    forced: #120 stopped drawing the path from a Markov chain, so there is no
    transition rate for a boundary to disagree with. What survives is the
    claim the restart was evidence for -- **a chromosome boundary is a real
    boundary** -- and for a piecewise-constant path that means no event spans
    one.

    Asserted against the placed events rather than the path, because the path
    cannot be read back into them: two events on adjacent chromosomes that
    draw the same state abut, and an abutment looks exactly like a crossing.
    That is why `place_events` returns what it placed.

    Events here are long enough to cross if nothing stopped them -- up to 40
    bins against chromosomes of 10 and up -- so a placement ignoring `lengths`
    fails within a few draws.
    """
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


@pytest.mark.cnaster
@pytest.mark.subject
@pytest.mark.critical
def test_the_clone_stacked_lengths_are_cnasters_own() -> None:
    """`stacked_lengths` is what `clone_stack_obs` builds, for a ragged genome.

    `cnaster` tiles the segmentation with the clones, so the fit runs over
    `n_clones * n_segments` chains. With unequal chromosomes that stacked
    vector is ragged too, and it is the shape #97's rung has to hand upstream.
    """
    from cnaster.hmrf_utils import clone_stack_obs

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
@pytest.mark.planted
def test_the_equal_mode_is_still_reachable_and_still_refuses() -> None:
    """A rectangular genome is a mode, not the default.

    #97's rung retreats to equal lengths where the correspondence needs it, so
    the old behaviour stays available and stays named -- and it keeps the
    divisibility check, which is a property of that mode rather than of the
    fixture.
    """
    truth = core_inference_truth(n_obs=240, n_segments=4, segmentation="equal", seed=5)

    np.testing.assert_array_equal(truth.lengths, [60, 60, 60, 60])
    assert truth.ragged.rectangular

    with pytest.raises(ValueError, match="do not partition"):
        core_inference_truth(n_obs=250, n_segments=4, segmentation="equal")

    with pytest.raises(ValueError, match="unknown segmentation"):
        core_inference_truth(segmentation="chromosomal")
