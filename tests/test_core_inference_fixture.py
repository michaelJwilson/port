"""The planted instance `run_core_inference` is validated against (#4).

One negative binomial per `(segment, spot)` and one beta-binomial, through
`snakes_and_ladders`'
families (#66): pinned as that draw, scored as upstream does, and recoverable.
"""

import numpy as np
import pytest
from port.qa.statistics import chi_square_pvalue
from port.sim.truth import CoreInferenceTruth, core_inference_truth
from scipy import stats

from tests.adapters import from_core_inference_truth

CHI_SQUARE_ALPHA = 0.001
"""Rejection level for the goodness-of-fit tests; loose because they run at a fixed seed."""


def _recover(truth: CoreInferenceTruth) -> dict[str, np.ndarray]:
    """Moment estimates of the planted parameters per state, by inverting the NB and beta-
    binomial variances.
    """
    per_spot = truth.states[truth.labels].T  # (n_obs, n_spots)
    mu = np.empty(truth.n_states)
    alpha = np.empty(truth.n_states)
    p_binom = np.empty(truth.n_states)
    rho = np.empty(truth.n_states)

    for state in range(truth.n_states):
        mask = per_spot == state
        x, lam = truth.counts_nb[mask], truth.base_nb_mean[mask]
        y, trials = truth.counts_bb[mask], truth.total_bb_RD[mask]

        mu[state] = (x / lam).mean()
        rate = lam * np.exp(truth.log_mu[state])
        alpha[state] = (((x - rate) ** 2).mean() - rate.mean()) / (rate**2).mean()

        share = truth.p_binom[state]
        p_binom[state] = (y / trials).mean()
        spread = ((y - trials * share) ** 2).mean() - (
            trials * share * (1 - share)
        ).mean()
        rho[state] = spread / (share * (1 - share) * (trials * (trials - 1)).mean())

    return {"mu": mu, "alpha": alpha, "p_binom": p_binom, "rho": rho}


@pytest.mark.end2end
@pytest.mark.critical
def test_the_draw_recovers_the_planted_parameters() -> None:
    """Every planted parameter is recovered from the counts by moments."""
    truth = core_inference_truth(n_obs=3000, lattice=(20, 20), n_segments=6)
    got = _recover(truth)

    np.testing.assert_allclose(got["mu"], np.exp(truth.log_mu), rtol=0.05)
    np.testing.assert_allclose(got["alpha"], truth.alphas, rtol=0.15)
    np.testing.assert_allclose(got["p_binom"], truth.p_binom, rtol=0.05)
    np.testing.assert_allclose(got["rho"], 1.0 / (truth.taus + 1.0), rtol=0.15)


@pytest.mark.smoke
def test_each_entry_is_a_negative_binomial_draw() -> None:
    """Pooled counts per state against `nbinom(r, r / (r + lambda mu))` by goodness of fit
    (#78).
    """
    truth = core_inference_truth(
        n_obs=3000,
        lattice=(20, 20),
        n_segments=6,
        exposure="constant",
        depth=(1.0, 1.0),
        reads=(40, 41),
    )
    per_spot = truth.states[truth.labels].T
    state = 1
    counts = truth.counts_nb[per_spot == state]

    r = 1.0 / truth.alphas[state]
    rate = np.exp(truth.log_mu[state])
    edges = np.arange(0, int(np.quantile(counts, 0.999)) + 2)

    observed = np.bincount(
        np.clip(counts.astype(int), 0, edges[-1]), minlength=edges.size
    )[: edges.size]
    expected = stats.nbinom.pmf(edges, r, r / (r + rate)) * counts.size
    expected[-1] += counts.size - expected.sum()

    chi, pvalue = chi_square_pvalue(observed, expected)

    assert pvalue > CHI_SQUARE_ALPHA, f"chi2 = {chi:.1f}, p = {pvalue:.2e}"


@pytest.mark.smoke
def test_each_entry_is_a_beta_binomial_draw() -> None:
    """Pooled successes per state against `betabinom(n, a, b)` by goodness of fit."""
    truth = core_inference_truth(
        n_obs=3000,
        lattice=(20, 20),
        n_segments=6,
        exposure="constant",
        depth=(1.0, 1.0),
        reads=(40, 41),
    )
    per_spot = truth.states[truth.labels].T
    state = 2
    successes = truth.counts_bb[per_spot == state].astype(int)

    trials = 40
    a = truth.p_binom[state] * truth.taus[state]
    b = (1.0 - truth.p_binom[state]) * truth.taus[state]

    observed = np.bincount(successes, minlength=trials + 1)[: trials + 1]
    expected = stats.betabinom.pmf(np.arange(trials + 1), trials, a, b) * successes.size

    chi, pvalue = chi_square_pvalue(observed, expected)

    assert pvalue > CHI_SQUARE_ALPHA, f"chi2 = {chi:.1f}, p = {pvalue:.2e}"


@pytest.mark.oracle
@pytest.mark.critical
def test_cnaster_scores_the_fixture_as_upstream_does() -> None:
    """`cnaster` and upstream score the fixture at the planted parameters to the realized
    agreement.
    """
    import torch
    from cnaster.hmm_nophasing import _bb_logpmf_1d, _nb_logpmf_1d
    from sal.emissions import (
        BetaBinomialEmission,
        NegativeBinomialEmission,
    )

    truth = core_inference_truth()
    spot, state = 7, 2
    counts = truth.counts_nb[:, spot]
    exposure = truth.base_nb_mean[:, spot]

    theirs = np.zeros(truth.n_obs)
    _nb_logpmf_1d(
        counts,
        exposure,
        float(np.exp(truth.log_mu[state])),
        float(truth.alphas[state]),
        theirs,
    )
    ours = (
        NegativeBinomialEmission(
            torch.as_tensor(1.0 / truth.alphas), torch.as_tensor(np.exp(truth.log_mu))
        )
        .log_density(
            torch.as_tensor(counts), covariate=torch.as_tensor(exposure)[:, None]
        )[:, state]
        .numpy()
    )
    nb_diff = np.abs(theirs - ours).max()

    successes, trials = truth.counts_bb[:, spot], truth.total_bb_RD[:, spot]
    theirs_bb = np.zeros(truth.n_obs)
    _bb_logpmf_1d(
        successes,
        trials,
        float(truth.p_binom[state]),
        float(truth.taus[state]),
        theirs_bb,
    )
    ours_bb = (
        BetaBinomialEmission(
            torch.ones(truth.n_states),
            torch.as_tensor(truth.p_binom * truth.taus),
            torch.as_tensor((1.0 - truth.p_binom) * truth.taus),
        )
        .log_density(
            torch.as_tensor(successes), covariate=torch.as_tensor(trials)[:, None]
        )[:, state]
        .numpy()
    )
    bb_diff = np.abs(theirs_bb - ours_bb).max()

    assert nb_diff < 1e-10, f"negative binomial channel: {nb_diff:.3e}"
    assert bb_diff < 1e-10, f"beta-binomial channel: {bb_diff:.3e}"


@pytest.mark.end2end
def test_the_field_recovers_the_planted_clone_assignment() -> None:
    """At the planted states every spot's own clone wins its
    `compute_loglike_spot_assignment` row.
    """
    from cnaster.hmm_nophasing import _dense_bb_logpmf, _dense_nb_logpmf
    from cnaster.hmrf import compute_loglike_spot_assignment

    truth = core_inference_truth(n_obs=600, lattice=(12, 10), n_segments=6)

    log_mu = np.repeat(truth.log_mu[:, None], 1, axis=1)
    alphas = np.repeat(truth.alphas[:, None], 1, axis=1)
    rdr = _dense_nb_logpmf(truth.counts_nb, truth.base_nb_mean, log_mu, alphas)
    baf = _dense_bb_logpmf(
        truth.counts_bb,
        truth.total_bb_RD,
        np.repeat(truth.p_binom[:, None], 1, axis=1),
        np.repeat(truth.taus[:, None], 1, axis=1),
    )

    field = compute_loglike_spot_assignment(
        truth.n_spots,
        np.ones(truth.n_spots),
        np.ones(truth.n_spots),
        np.empty(0),
        False,
        rdr,
        baf,
        np.ascontiguousarray(truth.states.T),
        truth.n_obs,
        truth.n_clones,
    )

    recovered = field.argmax(axis=1)
    accuracy = float((recovered == truth.labels).mean())

    assert accuracy == 1.0, f"the planted labelling is not the argmax: {accuracy:.3f}"


@pytest.mark.end2end
def test_the_pseudobulk_recovers_the_mean_and_not_the_dispersion() -> None:
    """The pseudobulk fit recovers the planted `mu` but not the per-spot dispersion (#78)."""
    from cnaster.pseudobulk import merge_pseudobulk_by_index_mix

    truth = core_inference_truth(n_obs=3000, lattice=(20, 20), n_segments=6)
    inputs = from_core_inference_truth(truth)

    pooled_X, pooled_base, _, _ = merge_pseudobulk_by_index_mix(
        inputs.single_X,
        inputs.single_base_nb_mean,
        inputs.single_total_bb_RD,
        inputs.initial_clone_index,
    )

    # NB clone 1: clone 0 is the normal clone (#298) and plants no state 1.
    clone = 1
    state_of = truth.states[clone]
    state = 1
    rows = state_of == state

    counts = pooled_X[rows, 0, clone]
    exposure = pooled_base[rows, clone]
    rate = exposure * np.exp(truth.log_mu[state])

    mu_hat = (counts / exposure).mean()
    alpha_hat = (((counts - rate) ** 2).mean() - rate.mean()) / (rate**2).mean()
    spots = truth.clone_index[clone].size

    np.testing.assert_allclose(mu_hat, np.exp(truth.log_mu[state]), rtol=0.05)
    assert alpha_hat < truth.alphas[state] / 4, (
        f"the aggregate's dispersion {alpha_hat:.5f} is not below the per-spot "
        f"{truth.alphas[state]:.5f} over {spots} spots, which #78 measures"
    )


@pytest.mark.smoke
def test_a_spot_s_counts_come_from_its_own_stream() -> None:
    """Column `s` equals a direct draw from `default_rng([seed, s])`."""
    import torch
    from port.sim.truth import emission_family

    truth = core_inference_truth()
    spot = 13

    family = emission_family(truth.log_mu, truth.alphas, truth.p_binom, truth.taus)
    covariate = np.stack(
        [truth.base_nb_mean[:, spot], truth.total_bb_RD[:, spot]], axis=-1
    )
    drawn = family.sample(
        truth.states[truth.labels[spot]],
        np.random.default_rng([truth.seed, spot]),
        covariate=torch.as_tensor(covariate),
    )

    np.testing.assert_array_equal(drawn[..., 0], truth.counts_nb[:, spot])
    np.testing.assert_array_equal(drawn[..., 1], truth.counts_bb[:, spot])


@pytest.mark.smoke
def test_the_fixture_is_bitwise_reproducible() -> None:
    """The same arguments give the same instance, every array of it."""
    first, second = core_inference_truth(), core_inference_truth()

    for name in ("counts_nb", "counts_bb", "base_nb_mean", "total_bb_RD", "states"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("lattice", "n_clones"), [((12, 10), 3), ((40, 40), 4), ((20, 5), 10)]
)
def test_every_size_plants_a_normal_clone_of_at_least_thirty_per_cent(
    lattice: tuple[int, int], n_clones: int
) -> None:
    """Clone 0 is all state 0 and holds at least 30% of the spots, at three sizes (#298)."""
    from port.sim.truth import NORMAL_SHARE

    truth = core_inference_truth(
        n_clones=n_clones, n_states=4, lattice=lattice, n_obs=60, n_segments=2
    )

    assert np.all(truth.states[0] == 0)
    assert np.mean(truth.labels == 0) >= NORMAL_SHARE
    assert np.unique(truth.labels).size == n_clones
