"""The generator against what it planted: a fixture that drew the wrong model
would referee every other test against the wrong truth."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import scipy.stats
from sim.truth import COPY_LATTICE, LOH_P, Truth, planted

pytestmark = pytest.mark.sim

LARGE: dict[str, Any] = {
    "n_clones": 3,
    "n_states": 4,
    "lattice": (40, 30),
    "n_obs": 200,
    "n_segments": 5,
}
"""240,000 draws per channel: enough that a moment off by 1% is 4 standard
errors from the planted value."""


@pytest.fixture(scope="module")
def large() -> Truth:
    return planted(**LARGE)


def test_segments_tile_the_genome_in_runs_of_two_or_more(large: Truth) -> None:
    assert int(large.lengths.sum()) == large.n_obs
    assert int(large.lengths.min()) >= 2
    assert len(large.lengths) == LARGE["n_segments"]


def test_clones_are_contiguous_row_bands(large: Truth) -> None:
    rows = large.labels.reshape(large.lattice)
    assert np.all(rows == rows[:, :1]), "a row holds more than one clone"
    assert np.all(np.diff(rows[:, 0]) >= 0), "bands are not in order"
    assert set(np.unique(large.labels)) == set(range(large.n_clones))


def test_the_normal_clone_is_neutral_and_every_tumour_clone_is_not(
    large: Truth,
) -> None:
    assert np.all(large.states[0] == 0)
    assert np.all(np.any(large.states[1:] != 0, axis=1))
    assert len({row.tobytes() for row in large.states}) == large.n_clones


def test_read_depth_has_the_planted_mean_and_variance(large: Truth) -> None:
    """`E[X] = m` and `Var[X] = m + alpha m^2`, pooled as standardized residuals."""
    mean = large.expected_nb()
    alpha = large.alphas[large.spot_states()]
    z = (large.counts_nb - mean) / np.sqrt(mean + alpha * mean**2)

    n = z.size
    assert abs(z.mean()) < 4.0 / np.sqrt(n)
    # NB `Var[z^2]` is under 5 for these moments; 4 standard errors of the variance.
    assert abs(z.var() - 1.0) < 4.0 * np.sqrt(5.0 / n)


def test_b_allele_has_the_planted_mean_once_the_phase_is_undone(large: Truth) -> None:
    """`E[B] = n p` and `Var[B] = n p (1 - p) (n + tau) / (1 + tau)`."""
    b = large.unphased_bb()
    n = large.total_bb_RD
    p = large.p_binom[large.spot_states()]
    tau = large.taus[large.spot_states()]
    z = (b - n * p) / np.sqrt(n * p * (1 - p) * (n + tau) / (1 + tau))

    assert abs(z.mean()) < 4.0 / np.sqrt(z.size)
    assert abs(z.var() - 1.0) < 4.0 * np.sqrt(5.0 / z.size)


def test_phase_flips_at_the_planted_rate_and_never_across_a_segment(
    large: Truth,
) -> None:
    starts = np.concatenate([[0], np.cumsum(large.lengths)[:-1]])
    flips = np.diff(large.phase, prepend=large.phase[0]) != 0
    assert not np.any(flips[starts])

    inside = np.setdiff1d(np.arange(large.n_obs), starts)
    expected = large.switch_prob[inside].sum()
    observed = flips[inside].sum()
    # NB a sum of independent Bernoullis: 4 standard deviations.
    sd = np.sqrt((large.switch_prob[inside] * (1 - large.switch_prob[inside])).sum())
    assert abs(observed - expected) < 4 * sd


def test_a_draw_is_a_function_of_the_seed_and_the_spot() -> None:
    first = planted(seed=5)
    again = planted(seed=5)
    other = planted(seed=6)

    assert np.array_equal(first.counts_nb, again.counts_nb)
    assert np.array_equal(first.counts_bb, again.counts_bb)
    assert not np.array_equal(first.counts_nb, other.counts_nb)


def test_the_copy_lattice_plants_integer_rates_and_allele_fractions() -> None:
    truth = planted(n_states=8, copy_lattice=True)
    assert truth.copies is not None

    total = truth.copies.sum(axis=1)
    np.testing.assert_array_equal(truth.copies, np.array(COPY_LATTICE))
    np.testing.assert_allclose(np.exp(truth.log_mu), total / 2.0, rtol=0, atol=1e-15)

    loh = truth.copies[:, 1] == 0
    np.testing.assert_array_equal(truth.p_binom[loh], LOH_P)
    np.testing.assert_allclose(
        truth.p_binom[~loh], truth.copies[~loh, 0] / total[~loh], rtol=0, atol=1e-15
    )


def test_counts_follow_the_planted_negative_binomial_by_goodness_of_fit() -> None:
    """One state, one exposure: the draws against `scipy.stats.nbinom` by chi-square."""
    truth = planted(
        n_clones=1,
        n_states=1,
        lattice=(50, 40),
        n_obs=40,
        n_segments=2,
        normal_clone=False,
        depth=(1.0, 1.0),
        exposure_scale=20.0,
    )
    # NB one state and a unit library factor leave the per-bin profile as the
    #    only exposure; one bin is 2,000 draws at a single mean.
    counts = truth.counts_nb[0]
    mean = truth.base_nb_mean[0, 0] * np.exp(truth.log_mu[0])
    r = 1.0 / truth.alphas[0]

    edges = np.arange(0, int(counts.max()) + 2)
    observed = np.bincount(counts, minlength=len(edges) - 1)
    expected = scipy.stats.nbinom.pmf(edges[:-1], r, r / (r + mean)) * counts.size
    keep = expected >= 5
    statistic = ((observed[keep] - expected[keep]) ** 2 / expected[keep]).sum()
    assert scipy.stats.chi2.sf(statistic, keep.sum() - 1) > 1e-3
