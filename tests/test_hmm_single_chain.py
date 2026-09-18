"""Rung one of issue #14: a single chain, refereed against `snakes_and_ladders`.

No spatial layer and no factored state space, so this rung needs neither the
structured transfer matrix nor the exposure-aware count family, and the
correspondence with `cnaster` is exact today. Every measurement here is
taken in that regime, which `CLAUDE.md` requires each one to state.

The sharpest assertion is not recovery. Scoring fixed parameters removes the
optimizer, the convergence criterion and the local optima, so a disagreement
is a defect in one of the two implementations and cannot be anything else.
"""

import numpy as np
import pytest
import torch
from snakes_and_ladders.opt.hmm import forward_log_likelihood_from_density

from tests.adapters import (
    BAF_CHANNEL,
    CnasterChainInputs,
    cnaster_emission,
    cnaster_total_log_likelihood,
    from_negative_binomial_chains,
)
from tests.fixtures import (
    NegativeBinomialChains,
    circulant_transition,
    negative_binomial_chains,
)

# NB the two implementations sum in different orders -- one over a
#    concatenated genomic axis, one over sequences -- so they agree to the
#    accumulated rounding of a few hundred terms and not to the last bit.
TOLERANCE = 1e-9


def upstream_emission(
    fixture: NegativeBinomialChains, inputs: CnasterChainInputs
) -> np.ndarray:
    """The same scores from upstream, shaped as `cnaster` returns them."""
    density = fixture.family.log_density(
        torch.as_tensor(fixture.dataset.observations, dtype=torch.float64)
    )
    reshaped: np.ndarray = density.numpy().reshape(inputs.n_obs, inputs.n_states).T
    return reshaped


def upstream_total_log_likelihood(fixture: NegativeBinomialChains) -> float:
    """The summed forward log-likelihood, from upstream's recursion."""
    density = fixture.family.log_density(
        torch.as_tensor(fixture.dataset.observations, dtype=torch.float64)
    )
    return float(
        forward_log_likelihood_from_density(
            density,
            torch.log(torch.as_tensor(fixture.dataset.initial)),
            torch.log(torch.as_tensor(fixture.dataset.transition)),
        )
    )


@pytest.mark.oracle
@pytest.mark.upstream_oracle
@pytest.mark.critical
@pytest.mark.parametrize("n_states", [1, 2, 3, 5])
@pytest.mark.parametrize("separation", [1.5, 2.5])
def test_emission_matches_upstream(n_states: int, separation: float) -> None:
    """`cnaster` scores every observation under every state as upstream does."""
    fixture = negative_binomial_chains(n_states=n_states, separation=separation)
    inputs = from_negative_binomial_chains(fixture)

    np.testing.assert_allclose(
        cnaster_emission(inputs),
        upstream_emission(fixture, inputs),
        rtol=0.0,
        atol=TOLERANCE,
    )


@pytest.mark.oracle
@pytest.mark.upstream_oracle
@pytest.mark.critical
@pytest.mark.parametrize("n_sequences", [1, 4])
@pytest.mark.parametrize("sequence_length", [1, 60])
def test_total_log_likelihood_matches_upstream(
    n_sequences: int, sequence_length: int
) -> None:
    """The forward recursions agree on the total, including at length one.

    A single position exercises the initialization alone, with no transition
    applied; a single sequence removes the concatenation `lengths` restarts.
    """
    fixture = negative_binomial_chains(
        n_sequences=n_sequences, sequence_length=sequence_length
    )
    inputs = from_negative_binomial_chains(fixture)

    assert cnaster_total_log_likelihood(inputs) == pytest.approx(
        upstream_total_log_likelihood(fixture), abs=TOLERANCE
    )


@pytest.mark.oracle
@pytest.mark.upstream_oracle
@pytest.mark.critical
@pytest.mark.parametrize("drift", [0.2, 0.8])
def test_total_log_likelihood_matches_upstream_asymmetric(drift: float) -> None:
    """The totals agree under a transition that is not symmetric.

    The default fixture's transition is circulant and therefore symmetric,
    so it scores identically under either row convention and cannot detect a
    transpose between the two implementations. This case can: a mutation
    test transposing `log_transmat` passes against the symmetric fixture and
    fails against this one.
    """
    fixture = negative_binomial_chains(n_states=4, drift=drift)
    inputs = from_negative_binomial_chains(fixture)

    assert cnaster_total_log_likelihood(inputs) == pytest.approx(
        upstream_total_log_likelihood(fixture), abs=TOLERANCE
    )


@pytest.mark.infra
@pytest.mark.analytic
def test_drift_outside_the_unit_interval_is_refused() -> None:
    """The fixture refuses a drift that is not a share."""
    with pytest.raises(ValueError, match="drift"):
        negative_binomial_chains(drift=0.0)


@pytest.mark.infra
@pytest.mark.analytic
def test_beta_binomial_channel_is_inert_at_zero_depth() -> None:
    """The adapter isolates the count channel, rather than hoping to.

    Every comparison above is against a count-only family, so a beta-binomial
    term leaking into the score would shift both sides of nothing and show up
    as an unexplained offset. This pins the assumption instead.
    """
    fixture = negative_binomial_chains()
    inputs = from_negative_binomial_chains(fixture)
    assert np.all(inputs.total_bb_RD == 0.0)

    successes = inputs.single_X[:, BAF_CHANNEL, :]
    assert np.all(successes == 0.0)

    perturbed = from_negative_binomial_chains(fixture)
    perturbed.p_binom[:] = 0.9
    perturbed.taus[:] = 3.0

    np.testing.assert_array_equal(cnaster_emission(inputs), cnaster_emission(perturbed))


@pytest.mark.infra
@pytest.mark.analytic
@pytest.mark.parametrize("exposure", [0.5, 1.0, 7.0])
def test_constant_exposure_is_absorbed(exposure: float) -> None:
    """A constant `base_nb_mean` leaves the scored model unchanged.

    This is the claim the flat-exposure regime rests on: `cnaster` forms
    `lam = exposure * mu` per observation, so a constant factors into
    `log_mu` and the correspondence with a per-state upstream mean is exact.
    Exposure that varies along a chain does not, which is why the adapter
    takes a scalar.
    """
    fixture = negative_binomial_chains()
    reference = cnaster_emission(from_negative_binomial_chains(fixture))
    scored = cnaster_emission(from_negative_binomial_chains(fixture, exposure=exposure))

    np.testing.assert_allclose(scored, reference, rtol=0.0, atol=TOLERANCE)


@pytest.mark.cnaster
@pytest.mark.analytic
@pytest.mark.parametrize("n_states", [1, 2, 5])
def test_transition_matches_cnaster_construction(n_states: int) -> None:
    """The fixture's transition is the one `cnaster` builds for itself.

    Two constructions of one matrix, pinned so a later change to either is a
    failing test rather than a silent divergence in what is being compared.
    """
    from cnaster.hmm_nophasing import get_log_transmat

    self_transition = 0.8
    mine = circulant_transition(n_states, self_transition)
    theirs = np.exp(get_log_transmat(n_states, self_transition))

    np.testing.assert_allclose(mine, theirs, rtol=0.0, atol=1e-15)


@pytest.mark.infra
@pytest.mark.analytic
def test_separation_below_one_is_refused() -> None:
    """The fixture refuses a separation that does not order the state means."""
    with pytest.raises(ValueError, match="separation"):
        negative_binomial_chains(separation=0.5)
