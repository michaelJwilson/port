"""Lattice properties checked against analytic identities rather than a second implementation."""

import numpy as np
import pytest
from cnaster.hmm_nophasing import hmm_nophasing
from scipy.special import logsumexp

from tests.adapters import (
    CnasterChainInputs,
    cnaster_lattice_arguments,
    cnaster_log_emission,
    from_negative_binomial_chains,
)
from tests.fixtures import negative_binomial_chains


def emission_and_inputs(**kwargs: object) -> tuple[np.ndarray, CnasterChainInputs]:
    """The scores and the arguments they were computed from."""
    fixture = negative_binomial_chains(**kwargs)  # type: ignore[arg-type]
    inputs = from_negative_binomial_chains(fixture)
    return cnaster_log_emission(inputs), inputs


@pytest.mark.analytic
@pytest.mark.parametrize("n_states", [1, 2, 3, 5])
@pytest.mark.parametrize("n_sequences", [1, 3])
def test_forward_and_backward_agree_at_every_position(
    n_states: int, n_sequences: int
) -> None:
    """`logsumexp(alpha + beta)` equals the forward total at every position."""

    log_emission, inputs = emission_and_inputs(
        n_states=n_states, n_sequences=n_sequences, sequence_length=40
    )
    args = cnaster_lattice_arguments(inputs, log_emission)
    log_alpha = hmm_nophasing.forward_lattice(*args)
    log_beta = hmm_nophasing.backward_lattice(*args)

    marginal = logsumexp(log_alpha + log_beta, axis=0)

    # NB chains are independent, so the identity holds per chain, not across them.
    start = 0
    for length in inputs.lengths:
        within = marginal[start : start + length]
        np.testing.assert_allclose(within, within[0], rtol=0.0, atol=1e-9)
        start += length


@pytest.mark.analytic
@pytest.mark.parametrize("n_states", [2, 4])
def test_state_posteriors_normalise(n_states: int) -> None:
    """The posteriors are a distribution over states at every position."""

    log_emission, inputs = emission_and_inputs(n_states=n_states, sequence_length=30)
    log_gamma = hmm_nophasing().get_state_posteriors(
        *cnaster_lattice_arguments(inputs, log_emission)
    )

    np.testing.assert_allclose(np.exp(log_gamma).sum(axis=0), 1.0, rtol=0.0, atol=1e-9)


@pytest.mark.analytic
def test_copy_states_fold_the_phase_only_when_asked() -> None:
    """`includes_phased` folds the phase index only when set (#9)."""

    n_copy_states = 3
    log_gamma = np.full((2 * n_copy_states, 4), -np.inf)
    # NB one of each phase, so folding is observable.
    for position, state in enumerate([0, 1 + n_copy_states, 2, 0 + n_copy_states]):
        log_gamma[state, position] = 0.0

    unfolded = hmm_nophasing.get_copy_states(log_gamma, includes_phased=False)
    folded = hmm_nophasing.get_copy_states(log_gamma, includes_phased=True)

    np.testing.assert_array_equal(unfolded, [0, 4, 2, 3])
    np.testing.assert_array_equal(folded, [0, 1, 2, 0])
