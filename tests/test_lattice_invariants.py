"""Properties of the lattice that hold without a second implementation.

`CLAUDE.md` asks every test to name what it is checked against. These are
checked against analytic identities rather than against upstream, which is
what makes them the cheapest rung available: they need no adapter, no
correspondence and no regime statement.

They also cover what the upstream comparisons cannot. Only the forward
recursion is exercised by a likelihood, so a sign or an index error in
`backward_lattice` is invisible to every test in `test_hmm_single_chain`;
the first identity below is the one that would catch it.
"""

import numpy as np
import pytest
from scipy.special import logsumexp

from tests.adapters import CnasterChainInputs, from_negative_binomial_chains
from tests.fixtures import negative_binomial_chains


def emission_and_inputs(**kwargs: object) -> tuple[np.ndarray, CnasterChainInputs]:
    """The scores and the arguments they were computed from."""
    from cnaster.hmm_nophasing import hmm_nophasing

    fixture = negative_binomial_chains(**kwargs)  # type: ignore[arg-type]
    inputs = from_negative_binomial_chains(fixture)
    log_emit_rdr, log_emit_baf = (
        hmm_nophasing.compute_emission_probability_nb_betabinom(
            inputs.single_X,
            inputs.base_nb_mean,
            inputs.log_mu,
            inputs.alphas,
            inputs.total_bb_RD,
            inputs.p_binom,
            inputs.taus,
        )
    )
    return log_emit_rdr + log_emit_baf, inputs


@pytest.mark.infra
@pytest.mark.analytic
@pytest.mark.parametrize("n_states", [1, 2, 3, 5])
@pytest.mark.parametrize("n_sequences", [1, 3])
def test_forward_and_backward_agree_at_every_position(
    n_states: int, n_sequences: int
) -> None:
    """`logsumexp(alpha + beta)` is the total likelihood, at every position.

    The identity that ties the two recursions together: marginalizing the
    state at any one position gives the same number, and that number is what
    the forward recursion reports at the end. It holds by construction and
    fails under a sign error, a transposed transition or an off-by-one in
    either sweep — none of which any likelihood comparison can see, because
    a likelihood only uses the forward pass.
    """
    from cnaster.hmm_nophasing import hmm_nophasing

    log_emission, inputs = emission_and_inputs(
        n_states=n_states, n_sequences=n_sequences, sequence_length=40
    )
    args = (
        inputs.lengths,
        inputs.log_transmat,
        inputs.log_startprob,
        log_emission,
        inputs.log_sitewise_transmat,
    )
    log_alpha = hmm_nophasing.forward_lattice(*args)
    log_beta = hmm_nophasing.backward_lattice(*args)

    marginal = logsumexp(log_alpha + log_beta, axis=0)

    # NB the chains are independent and each carries its own total, so the
    #    identity holds within a chain rather than across the concatenation.
    start = 0
    for length in inputs.lengths:
        within = marginal[start : start + length]
        np.testing.assert_allclose(within, within[0], rtol=0.0, atol=1e-9)
        start += length


@pytest.mark.infra
@pytest.mark.analytic
@pytest.mark.parametrize("n_states", [2, 4])
def test_state_posteriors_normalise(n_states: int) -> None:
    """The posteriors are a distribution over states at every position."""
    from cnaster.hmm_nophasing import hmm_nophasing

    log_emission, inputs = emission_and_inputs(n_states=n_states, sequence_length=30)
    log_gamma = hmm_nophasing().get_state_posteriors(
        inputs.lengths,
        inputs.log_transmat,
        inputs.log_startprob,
        log_emission,
        inputs.log_sitewise_transmat,
    )

    np.testing.assert_allclose(np.exp(log_gamma).sum(axis=0), 1.0, rtol=0.0, atol=1e-9)


@pytest.mark.infra
@pytest.mark.analytic
def test_copy_states_fold_the_phase_only_when_asked() -> None:
    """`includes_phased` decides whether a phase index is folded away.

    The default is not to fold, over a space that may be paired, so a caller
    that forgets it receives a phase index where a copy state is meant. #9
    records that nothing pins which callers pass it; this pins what the two
    settings do.
    """
    from cnaster.hmm_nophasing import hmm_nophasing

    n_copy_states = 3
    log_gamma = np.full((2 * n_copy_states, 4), -np.inf)
    # NB argmax per position: copy state 0 phase 0, then 1 phase 1, 2 phase 0,
    #    0 phase 1 -- one of each phase, so folding is observable.
    for position, state in enumerate([0, 1 + n_copy_states, 2, 0 + n_copy_states]):
        log_gamma[state, position] = 0.0

    unfolded = hmm_nophasing.get_copy_states(log_gamma, includes_phased=False)
    folded = hmm_nophasing.get_copy_states(log_gamma, includes_phased=True)

    np.testing.assert_array_equal(unfolded, [0, 4, 2, 3])
    np.testing.assert_array_equal(folded, [0, 1, 2, 0])
