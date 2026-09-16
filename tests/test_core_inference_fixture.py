"""The planted instance `run_core_inference` is validated against (issue #4).

#66 decides the draw: **one negative binomial per `(segment, spot)` for the
total and one beta-binomial for the successes**, independently, through
`snakes_and_ladders`' own families. These tests pin that it is that draw and
not something with the same marginals, that `cnaster` scores it as upstream
does, and that the planted labels and states are recoverable from it.

Component-wise, in the order #14's ladder puts them: the draw, the score, the
field. The end-to-end run and its size are
`tests/test_core_inference_at_scale.py`.
"""

import numpy as np
import pytest
from scipy import stats

from tests.adapters import from_core_inference_truth
from tests.fixtures import CoreInferenceTruth, core_inference_truth

CHI_SQUARE_ALPHA = 0.001
"""Rejection level for the goodness-of-fit tests.

Loose on purpose: these run at a fixed seed, so a level that rejects one draw
in twenty would be a test that fails for being run. At 0.001 a false rejection
is a fixture that changed, and `test_the_draw_is_a_function_of_the_seed_and_the_spot`
is what says whether it did.
"""


def _recover(truth: CoreInferenceTruth) -> dict[str, np.ndarray]:
    """Moment estimates of the planted parameters, per state.

    `Var(X_i) = lambda_i mu + alpha (lambda_i mu)^2` and
    `Var(y_i) = n_i p (1-p) [1 + (n_i - 1) rho]` are the two relations that
    make these count models, so inverting them is a referee independent of the
    sampler that drew them.
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


@pytest.mark.planted
def test_the_draw_recovers_the_planted_parameters() -> None:
    """Every planted parameter is recovered from the counts it generated.

    The weakest of these tests and the one that has to pass first: a fixture
    that does not recover its own parameters is not truth, and every referee
    above it would be scoring against a mislabelled instance.
    """
    truth = core_inference_truth(n_obs=3000, lattice=(20, 20), n_segments=6)
    got = _recover(truth)

    np.testing.assert_allclose(got["mu"], np.exp(truth.log_mu), rtol=0.05)
    np.testing.assert_allclose(got["alpha"], truth.alphas, rtol=0.15)
    np.testing.assert_allclose(got["p_binom"], truth.p_binom, rtol=0.05)
    np.testing.assert_allclose(got["rho"], 1.0 / (truth.taus + 1.0), rtol=0.15)


@pytest.mark.analytic
def test_each_entry_is_a_negative_binomial_draw() -> None:
    """Not a mixture whose marginals are negative binomial -- the draw itself.

    At a constant exposure every entry in a state shares one distribution, so
    the pooled counts are testable against `nbinom(r, r / (r + lambda mu))`
    directly. This is the assertion that separates the decided draw from a
    hierarchical one: those agree on this test and disagree on the clone sums,
    which is #78.
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

    keep = expected >= 5
    chi = (((observed[keep] - expected[keep]) ** 2) / expected[keep]).sum()
    pvalue = stats.chi2.sf(chi, keep.sum() - 1)

    assert pvalue > CHI_SQUARE_ALPHA, f"chi2 = {chi:.1f}, p = {pvalue:.2e}"


@pytest.mark.analytic
def test_each_entry_is_a_beta_binomial_draw() -> None:
    """The same for the success channel, against `betabinom(n, a, b)`.

    A binomial draw at the planted `p` would pass a mean check and fail here,
    which is what makes the overdispersion part of the planted truth rather
    than an accident of the sampler.
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
    state = 2
    successes = truth.counts_bb[per_spot == state].astype(int)

    trials = 40
    a = truth.p_binom[state] * truth.taus[state]
    b = (1.0 - truth.p_binom[state]) * truth.taus[state]

    observed = np.bincount(successes, minlength=trials + 1)[: trials + 1]
    expected = stats.betabinom.pmf(np.arange(trials + 1), trials, a, b) * successes.size

    keep = expected >= 5
    chi = (((observed[keep] - expected[keep]) ** 2) / expected[keep]).sum()
    pvalue = stats.chi2.sf(chi, keep.sum() - 1)

    assert pvalue > CHI_SQUARE_ALPHA, f"chi2 = {chi:.1f}, p = {pvalue:.2e}"


@pytest.mark.upstream
def test_cnaster_scores_the_fixture_as_upstream_does() -> None:
    """Two implementations of the emission, on the instance one of them drew.

    Scored at the **planted** parameters, which removes the optimizer and the
    convergence criterion, so a disagreement is a defect in one of the two and
    can be nothing else. The realized agreement is asserted rather than a
    tolerance chosen in advance.
    """
    import torch
    from cnaster.hmm_nophasing import _bb_logpmf_1d, _nb_logpmf_1d
    from snakes_and_ladders.emissions import (
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


@pytest.mark.planted
def test_the_field_recovers_the_planted_clone_assignment() -> None:
    """At the planted states, every spot's own clone wins its field row.

    `compute_loglike_spot_assignment` is the seam the label solver optimizes,
    so if the planted labelling is not its argmax at the planted states, no
    solver above it can recover the labelling and a failure further up would
    be unattributable.
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


@pytest.mark.planted
def test_the_pseudobulk_recovers_the_mean_and_not_the_dispersion() -> None:
    """#78, on the fixture rather than on a standalone draw.

    The exposure is summed with the counts, so `mu` survives aggregation. The
    dispersion does not: a sum of independent negative binomials at unequal
    exposures is not negative binomial, and what `cnaster` fits to the
    aggregate is smaller than the per-spot value by roughly the clone's size.
    """
    from cnaster.pseudobulk import merge_pseudobulk_by_index_mix

    truth = core_inference_truth(n_obs=3000, lattice=(20, 20), n_segments=6)
    inputs = from_core_inference_truth(truth)

    pooled_X, pooled_base, _, _ = merge_pseudobulk_by_index_mix(
        inputs.single_X,
        inputs.single_base_nb_mean,
        inputs.single_total_bb_RD,
        inputs.initial_clone_index,
    )

    clone = 0
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


@pytest.mark.analytic
def test_a_spot_s_counts_come_from_its_own_stream() -> None:
    """Column `s` is drawn from `default_rng([seed, s])` and nothing else.

    So the order spots are visited in cannot move the data, and a rebuild is
    bitwise. Asserted by drawing one column directly from the family and
    matching it, which is the claim; a **prefix** property is not claimed and
    does not hold, because the bands and the shared draws both depend on the
    lattice, so widening it re-labels the spots that were already there.
    """
    import torch

    from tests.fixtures import _emission_families

    truth = core_inference_truth()
    spot = 13

    family = _emission_families(truth.log_mu, truth.alphas, truth.p_binom, truth.taus)
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


@pytest.mark.analytic
def test_the_fixture_is_bitwise_reproducible() -> None:
    """The same arguments give the same instance, every array of it."""
    first, second = core_inference_truth(), core_inference_truth()

    for name in ("counts_nb", "counts_bb", "base_nb_mean", "total_bb_RD", "states"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
