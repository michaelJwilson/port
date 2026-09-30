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


@pytest.mark.bug
@pytest.mark.parametrize("name", ["cnaster-gmm", "distinct", "calicost-gmm", "prior"])
def test_every_start_returns_one_state_per_planted_state(name: str) -> None:
    """On two clones stacked along the genome, `n_states` states with positive rates and p in (0, 1).

    Clones handed to `cnaster`'s `gmm_init` as columns returned a state per
    clone per state, and `prior` wrote its B coordinate over totals, a rate
    above 1 on dev_tree_1s_hard.
    """
    from types import SimpleNamespace

    from tests.studies.copy_state_stream import seed_states

    rng = np.random.default_rng(0)
    n = 300
    clone = np.repeat([0, 1], n)
    state = np.where((np.arange(2 * n) % n < n // 2) & (clone == 1), 1, 0)
    exposure = np.full(2 * n, 300.0)
    trials = rng.integers(20, 40, 2 * n).astype(float)
    problem = SimpleNamespace(
        realization=0, total=rng.poisson(exposure * np.where(state == 1, 0.5, 1.0)).astype(float),
        b=rng.binomial(trials.astype(int), np.where(state == 1, 0.1, 0.5)).astype(float), exposure=exposure,
        trials=trials, clone=clone, contig=np.full(2 * n, "1"), start=np.tile(np.arange(n) * 1e6, 2),
        length=np.full(2 * n, 1e6), lengths=np.array([n, n]), planted=np.column_stack([1 - state, np.ones(2 * n, int)]),
        n_states=2, truth_label=state,
    )  # fmt: skip
    log_mu, p = seed_states(name, problem, np.random.default_rng(1))

    assert log_mu.size == p.size == 2
    assert np.isfinite(log_mu).all()
    assert ((p > 0) & (p < 1)).all()
