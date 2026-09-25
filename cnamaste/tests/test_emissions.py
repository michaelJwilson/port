"""The emission kernels against `scipy.stats`, an independent implementation of
the same two families, and against normalization, which holds of any pmf."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.stats
from sim.truth import Truth, planted

from cnamaste.hmm_nophasing import (
    _bb_logpmf_1d,
    _dense_bb_logpmf,
    _dense_nb_logpmf,
    _nb_logpmf_1d,
)

TOLERANCE = 1e-10
"""Absolute, in nats: both sides are float64 `lgamma` sums of a few terms."""


@pytest.fixture(scope="module")
def small() -> Truth:
    return planted(n_clones=2, n_states=4, lattice=(6, 5), n_obs=40, n_segments=2)


@pytest.mark.oracle
@pytest.mark.parametrize(("mu", "alpha"), [(0.5, 0.01), (1.0, 0.2), (3.0, 1e-4)])
def test_negative_binomial_kernel_is_scipys(
    mu: float, alpha: float, small: Truth
) -> None:
    obs = small.counts_nb[:, 0].astype(np.float64)
    exposure = small.base_nb_mean[:, 0]
    out = np.empty_like(obs)
    _nb_logpmf_1d(obs, exposure, mu, alpha, out)

    r = 1.0 / alpha
    expected = scipy.stats.nbinom.logpmf(obs, r, 1.0 / (1.0 + alpha * exposure * mu))
    np.testing.assert_allclose(out, expected, rtol=0, atol=TOLERANCE)


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("p", "tau"), [(0.5, 100.0), (0.88, 30.0), (1.0 - 1e-5, 100.0)]
)
def test_beta_binomial_kernel_is_scipys(p: float, tau: float, small: Truth) -> None:
    obs = small.counts_bb[:, 0].astype(np.float64)
    total = small.total_bb_RD[:, 0].astype(np.float64)
    out = np.empty_like(obs)
    _bb_logpmf_1d(obs, total, p, tau, out)

    expected = scipy.stats.betabinom.logpmf(obs, total, p * tau, (1.0 - p) * tau)
    np.testing.assert_allclose(out, expected, rtol=0, atol=TOLERANCE)


@pytest.mark.analytic
@pytest.mark.parametrize(("mean", "alpha"), [(2.0, 0.01), (40.0, 0.5)])
def test_negative_binomial_kernel_sums_to_one(mean: float, alpha: float) -> None:
    support = np.arange(0.0, 4_000.0)
    out = np.empty_like(support)
    _nb_logpmf_1d(support, np.full_like(support, mean), 1.0, alpha, out)
    assert abs(np.exp(out).sum() - 1.0) < 1e-12


@pytest.mark.analytic
@pytest.mark.parametrize("total", [1, 17, 60])
def test_beta_binomial_kernel_sums_to_one(total: int) -> None:
    support = np.arange(0.0, total + 1.0)
    out = np.empty_like(support)
    _bb_logpmf_1d(support, np.full_like(support, total), 0.66, 100.0, out)
    assert abs(np.exp(out).sum() - 1.0) < 1e-12


@pytest.mark.analytic
def test_zero_exposure_scores_zero() -> None:
    """A bin with no expected reads carries no read-depth evidence, whatever it holds."""
    obs = np.array([0.0, 3.0])
    out = np.full(2, np.nan)
    _nb_logpmf_1d(obs, np.zeros(2), 1.5, 0.01, out)
    np.testing.assert_array_equal(out, 0.0)


@pytest.mark.oracle
def test_dense_kernels_are_the_one_dimensional_kernels_per_state_and_spot(
    small: Truth,
) -> None:
    """`(n_states, n_obs, n_spots)`, each entry the 1-D kernel's."""
    nb = _dense_nb_logpmf(
        small.counts_nb.astype(np.float64),
        small.base_nb_mean,
        small.log_mu[:, None],
        small.alphas[:, None],
    )
    bb = _dense_bb_logpmf(
        small.counts_bb.astype(np.float64),
        small.total_bb_RD.astype(np.float64),
        small.p_binom[:, None],
        small.taus[:, None],
    )
    for state in range(small.n_states):
        mean = small.base_nb_mean * np.exp(small.log_mu[state])
        r = 1.0 / small.alphas[state]
        np.testing.assert_allclose(
            nb[state],
            scipy.stats.nbinom.logpmf(small.counts_nb, r, r / (r + mean)),
            rtol=0,
            atol=TOLERANCE,
        )
        p, tau = small.p_binom[state], small.taus[state]
        np.testing.assert_allclose(
            bb[state],
            scipy.stats.betabinom.logpmf(
                small.counts_bb, small.total_bb_RD, p * tau, (1 - p) * tau
            ),
            rtol=0,
            atol=TOLERANCE,
        )


@pytest.mark.analytic
def test_the_planted_state_is_the_likeliest_on_pooled_evidence(small: Truth) -> None:
    """Summed over a clone's spots, each bin's planted state scores highest.

    The fixture's states are separated in at least one channel by more than
    the pooled noise, so a kernel that swapped a parameter -- `p` for
    `1 - p`, `alpha` for `1 / alpha` -- moves some bin's maximum off it.
    """
    nb = _dense_nb_logpmf(
        small.counts_nb.astype(np.float64),
        small.base_nb_mean,
        small.log_mu[:, None],
        small.alphas[:, None],
    )
    bb = _dense_bb_logpmf(
        small.unphased_bb().astype(np.float64),
        small.total_bb_RD.astype(np.float64),
        small.p_binom[:, None],
        small.taus[:, None],
    )
    for clone in range(small.n_clones):
        spots = small.labels == clone
        pooled = (nb + bb)[:, :, spots].sum(axis=2)
        np.testing.assert_array_equal(pooled.argmax(axis=0), small.states[clone])
