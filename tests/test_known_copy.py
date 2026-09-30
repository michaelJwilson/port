"""#540: the known-clone copy-state problem's floor and its Missed count, each against an independent answer."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from port.sandbox import known_copy as kc


@pytest.mark.analytic
def test_the_floor_merges_within_a_chromosome_until_each_bin_holds_the_floor() -> None:
    """Every merged bin reaches the floor but a chromosome's only bin, never crosses a chromosome, and keeps order."""
    rng = np.random.default_rng(0)
    chrom = np.repeat([1, 2, 3], [40, 25, 3])
    weight = rng.gamma(0.5, 200.0, chrom.size)
    merged = kc.floored_bins(chrom, weight, 300.0)
    held = np.bincount(merged, weights=weight)

    assert (np.diff(merged) >= 0).all()
    assert (np.diff(merged) <= 1).all()
    for bin_id in np.unique(merged):
        chromosomes = np.unique(chrom[merged == bin_id])
        assert chromosomes.size == 1
        only = np.unique(merged[chrom == chromosomes[0]]).size == 1
        assert held[bin_id] >= 300.0 or only


@pytest.mark.oracle
def test_missed_is_the_fewest_misses_over_every_matching_of_states() -> None:
    """Against brute force over all permutations of 4 states."""
    rng = np.random.default_rng(1)
    truth = rng.integers(0, 4, 200)
    label = np.where(rng.random(200) < 0.7, (truth + 1) % 4, rng.integers(0, 4, 200))
    brute = min(
        int((np.asarray(perm)[label] != truth).sum())
        for perm in itertools.permutations(range(4))
    )

    assert kc.missed(label, truth) == brute
