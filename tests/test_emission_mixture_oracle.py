"""`cnaster`'s emission and its mixture, refereed by `snakes_and_ladders`.

**#232, and the rung #229 stage 4 needs before it can have a backend.**
`cnaster` writes its own negative-binomial and beta-binomial densities and
its own mixture initializer on top of them. Upstream ships the same model as
`emissions.CountPairEmission` and fits a mixture of it in
`opt/emission_mixture`. Until now nothing here compared the two, so
`CLAUDE.md`'s "reach for what `snakes_and_ladders` already carries" had never
been tested at this seam.

`oracle` throughout: upstream decides the expected value, and `cnaster` is
judged against it. That is what moves `CountPairEmission` out of
`UNMATCHED_FAMILIES` -- it referees now, where before it only drew (#69).

**The regime is stated because it is not the whole problem.** The
correspondence is exact only where `base_nb_mean` and `total_bb_RD` are
constant across bins: upstream carries one mean and one trial count per
state, `cnaster` carries one of each per bin. That is #57 and #65 reaching
the initializer, and `port.extensions.emission_family.constant_covariate` refuses
rather than approximates outside it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from cnaster.hmm_nophasing import _bb_logpmf_1d, _nb_logpmf_1d
from port.extensions.emission_family import (
    CovariateNotConstant,
    constant_covariate,
    count_pair_family,
)
from snakes_and_ladders.emissions import CountPairEmission
from snakes_and_ladders.opt.emission_mixture import (
    CountPairSeeding,
    expectation_maximization,
    plus_plus_start,
)

EXPOSURE, TRIALS = 40.0, 60.0
"""The constant covariates the correspondence holds at."""

LOG_MU = np.array([-0.35, 0.0, 0.42])
ALPHAS = np.array([0.08, 0.12, 0.05])
P_BINOM = np.array([0.50, 0.32, 0.18])
TAUS = np.array([30.0, 22.0, 45.0])

ROUND_OFF = 1e-9
"""What a different summation order costs, not what the model is worth.

Realized 1.4e-13 over 600 values; the bound is four orders looser so a
`lgamma` implementation changing under a dependency bump is not a red test
about nothing. A real disagreement is not small.
"""


def _cnaster_log_density(observations: np.ndarray) -> np.ndarray:
    """`cnaster`'s own density, state by state, as its HMM evaluates it."""
    totals, successes = observations[:, 0], observations[:, 1]
    n_obs = totals.size
    out = np.zeros((LOG_MU.size, n_obs))

    for state in range(LOG_MU.size):
        rdr = np.zeros(n_obs)
        baf = np.zeros(n_obs)

        _nb_logpmf_1d(
            totals, np.full(n_obs, EXPOSURE), np.exp(LOG_MU[state]), ALPHAS[state], rdr
        )
        _bb_logpmf_1d(
            successes, np.full(n_obs, TRIALS), P_BINOM[state], TAUS[state], baf
        )

        out[state] = rdr + baf

    return out


@pytest.fixture
def observations() -> np.ndarray:
    rng = np.random.default_rng(7)
    totals = rng.poisson(EXPOSURE * np.exp(LOG_MU[1]), size=200).astype(float)
    successes = rng.binomial(int(TRIALS), P_BINOM[1], size=200).astype(float)

    return np.column_stack([totals, successes])


@pytest.mark.oracle
def test_cnasters_emission_is_upstreams_count_pair_family(
    observations: np.ndarray,
) -> None:
    """The same model, parameterized differently, agreeing to round-off.

    `r = 1/alpha`, `lambda = exposure * exp(log_mu)`, `a = p * tau`,
    `b = (1 - p) * tau`. If this fails, one of those four is wrong, and the
    per-state maxima below say which state.
    """
    theirs = _cnaster_log_density(observations)

    family = count_pair_family(
        LOG_MU, ALPHAS, P_BINOM, TAUS, exposure=EXPOSURE, trials=TRIALS
    )
    ours = (
        family.log_density(torch.as_tensor(observations, dtype=torch.float64)).numpy().T
    )

    assert ours.shape == theirs.shape

    difference = np.abs(theirs - ours)

    assert np.max(difference) < ROUND_OFF, (
        f"per-state maxima {np.max(difference, axis=1)}"
    )


@pytest.mark.oracle
def test_upstreams_mixture_recovers_the_planted_state(
    observations: np.ndarray,
) -> None:
    """`opt/emission_mixture` fits the family `cnaster` would have GMM'd.

    The data are drawn from one state, so a two-component fit has to put its
    mass on a component whose mean is that state's. This is the claim
    `cnaster`'s initializer makes and never checks: that the mixture finds
    where the data are.
    """
    rng = np.random.default_rng(11)
    seeding = CountPairSeeding(
        dispersion=1.0 / ALPHAS[1], concentration=TAUS[1], joint=False, trials=TRIALS
    )
    components = plus_plus_start(observations, 2, seeding, rng)

    weights = torch.full((2,), 0.5, dtype=torch.float64)
    fit = expectation_maximization(observations, weights, components)

    assert fit.iterations >= 1
    assert fit.log_likelihood <= 0.0, "a discrete likelihood is a probability"

    # NB `mean` is per state and per channel: column 0 the negative
    #    binomial's, column 1 the beta-binomial's. Both are checked, because
    #    a fit that finds the depth and misses the allele fraction has found
    #    half the state.
    components = fit.components

    assert isinstance(components, CountPairEmission), (
        f"expected a CountPairEmission, got {type(components).__name__}"
    )

    fitted = np.asarray(components.mean)

    assert fitted.shape == (2, 2), (
        f"expected (n_states, n_channels), got {fitted.shape}"
    )

    dominant = int(torch.argmax(fit.weights))
    depth, allele = float(fitted[dominant, 0]), float(fitted[dominant, 1])

    assert depth == pytest.approx(EXPOSURE * np.exp(LOG_MU[1]), rel=0.25), (
        f"depth {depth:.3f} against planted {EXPOSURE * np.exp(LOG_MU[1]):.3f}"
    )

    assert allele == pytest.approx(TRIALS * P_BINOM[1], rel=0.25), (
        f"allele count {allele:.3f} against planted {TRIALS * P_BINOM[1]:.3f}"
    )


@pytest.mark.oracle
def test_the_fit_beats_a_wrong_state_on_its_own_referee(
    observations: np.ndarray,
) -> None:
    """The planted state scores higher than the others under `cnaster`'s density.

    Cheap, and it is the property an initializer exists to get right: the
    referee the selection in #229 uses has to rank the truth first, or
    ranking by it selects nothing.
    """
    totals = _cnaster_log_density(observations).sum(axis=1)

    assert int(np.argmax(totals)) == 1, f"per-state totals {totals}"


@pytest.mark.oracle
def test_a_varying_covariate_is_refused_rather_than_approximated(
    observations: np.ndarray,
) -> None:
    """Outside the regime there is no upstream form, and that is #57/#65.

    A family fitted to a mean that ignores a varying exposure answers a
    different question, so reporting it beside `cnaster`'s would compare two
    answers to two questions.
    """
    varying = np.linspace(EXPOSURE, 2.0 * EXPOSURE, 200)

    with pytest.raises(CovariateNotConstant, match="base_nb_mean varies"):
        count_pair_family(
            LOG_MU, ALPHAS, P_BINOM, TAUS, exposure=varying, trials=TRIALS
        )

    with pytest.raises(CovariateNotConstant, match="total_bb_RD varies"):
        count_pair_family(
            LOG_MU, ALPHAS, P_BINOM, TAUS, exposure=EXPOSURE, trials=varying
        )

    # NB float64 noise on a computed covariate is not variation.
    jittered = np.full(200, EXPOSURE) * (1.0 + 1e-15 * np.arange(200))

    assert constant_covariate(jittered, "base_nb_mean") == pytest.approx(EXPOSURE)
