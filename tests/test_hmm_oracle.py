"""`cnaster.hmm` and the dense recursion against upstream `sal` (#140).

Posterior and evidence against `forward_backward`; Baum-Welch ascent scored by
upstream's evidence.
"""

import itertools
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from tests.adapters import CnasterChainInputs, from_negative_binomial_chains
from tests.fixtures import NegativeBinomialChains, negative_binomial_chains

if TYPE_CHECKING:
    from sal.likelihood.forward_backward import ForwardBackward

TOLERANCE = 1e-9
"""Two `logsumexp` recursions over the same floats, in the same order."""

POSTERIOR_TOLERANCE = 1e-10
"""Tighter: a normalized posterior loses the evidence's magnitude."""


def _cnaster_emission(inputs: CnasterChainInputs) -> np.ndarray:
    from cnaster.hmm_nophasing import hmm_nophasing

    rdr, baf = hmm_nophasing.compute_emission_probability_nb_betabinom(
        inputs.single_X,
        inputs.base_nb_mean,
        inputs.log_mu,
        inputs.alphas,
        inputs.total_bb_RD,
        inputs.p_binom,
        inputs.taus,
    )
    emission: np.ndarray = rdr + baf
    return emission


def _cnaster_posterior(inputs: CnasterChainInputs) -> np.ndarray:
    """Return `log gamma` over the concatenated axis via `compute_copy_state_posterior`."""
    from cnaster.hmm import compute_copy_state_posterior
    from cnaster.hmm_nophasing import hmm_nophasing

    emission = _cnaster_emission(inputs)
    log_alpha = hmm_nophasing.forward_lattice(
        inputs.lengths,
        inputs.log_transmat,
        inputs.log_startprob,
        emission,
        inputs.log_sitewise_transmat,
    )
    log_beta = hmm_nophasing.backward_lattice(
        inputs.lengths,
        inputs.log_transmat,
        inputs.log_startprob,
        emission,
        inputs.log_sitewise_transmat,
    )
    posterior: np.ndarray = compute_copy_state_posterior(log_alpha, log_beta)
    return posterior


def _upstream_forward_backward(
    fixture: NegativeBinomialChains, chain: int
) -> "ForwardBackward":
    """Upstream's two passes on one chain of the fixture."""
    from sal.likelihood.forward_backward import forward_backward

    # NB upstream takes one row per chain; `cnaster` concatenates them
    observations = np.asarray(fixture.dataset.observations)[chain]

    density = fixture.family.log_density(
        torch.as_tensor(observations, dtype=torch.float64)
    )
    return forward_backward(
        np.asarray(density, dtype=float),
        np.log(np.asarray(fixture.dataset.initial, dtype=float)),
        np.log(np.asarray(fixture.dataset.transition, dtype=float)),
    )


@pytest.mark.oracle
@pytest.mark.parametrize("n_sequences", [1, 3])
def test_the_state_posterior_agrees_with_upstream(n_sequences: int) -> None:
    """The state posterior equals upstream's, to `POSTERIOR_TOLERANCE`, per chain."""
    fixture = negative_binomial_chains(
        n_states=3, sequence_length=40, n_sequences=n_sequences
    )
    inputs = from_negative_binomial_chains(fixture)

    posterior = np.exp(_cnaster_posterior(inputs))
    length = posterior.shape[1] // n_sequences

    for chain in range(n_sequences):
        expected = _upstream_forward_backward(fixture, chain)
        window = posterior[:, chain * length : (chain + 1) * length].T
        np.testing.assert_allclose(
            window,
            np.asarray(expected.posterior),
            rtol=0.0,
            atol=POSTERIOR_TOLERANCE,
        )


@pytest.mark.oracle
def test_every_posterior_column_is_a_distribution() -> None:
    """Each posterior column sums to one, to `POSTERIOR_TOLERANCE`."""
    fixture = negative_binomial_chains(n_states=4, sequence_length=25, n_sequences=2)
    inputs = from_negative_binomial_chains(fixture)

    posterior = np.exp(_cnaster_posterior(inputs))

    np.testing.assert_allclose(
        posterior.sum(axis=0), np.ones(posterior.shape[1]), atol=POSTERIOR_TOLERANCE
    )


@pytest.mark.oracle
@pytest.mark.parametrize("n_sequences", [2, 4])
def test_the_evidence_agrees_over_a_rectangular_batch(n_sequences: int) -> None:
    """Rectangular-batch evidence equals upstream's per-chain sum, to `TOLERANCE` (#97)."""
    fixture = negative_binomial_chains(
        n_states=3, sequence_length=30, n_sequences=n_sequences
    )
    inputs = from_negative_binomial_chains(fixture)

    from cnaster.hmm_nophasing import hmm_nophasing
    from scipy.special import logsumexp

    log_alpha = hmm_nophasing.forward_lattice(
        inputs.lengths,
        inputs.log_transmat,
        inputs.log_startprob,
        _cnaster_emission(inputs),
        inputs.log_sitewise_transmat,
    )
    ends = np.cumsum(inputs.lengths) - 1
    theirs = sum(
        float(_upstream_forward_backward(fixture, chain).log_evidence)
        for chain in range(n_sequences)
    )
    ours = float(sum(logsumexp(log_alpha[:, end]) for end in ends))

    assert ours == pytest.approx(theirs, abs=TOLERANCE)


def _upstream_evidence_at(
    fixture: NegativeBinomialChains,
    log_mu: np.ndarray,
    alphas: np.ndarray,
) -> float:
    """Return upstream's evidence at `cnaster`'s fitted parameters, summed over chains."""
    from sal.emissions import NegativeBinomialEmission
    from sal.likelihood.forward_backward import forward_backward

    family = NegativeBinomialEmission(
        torch.as_tensor(1.0 / np.asarray(alphas).ravel(), dtype=torch.float64),
        torch.as_tensor(np.exp(np.asarray(log_mu).ravel()), dtype=torch.float64),
    )
    observations = np.asarray(fixture.dataset.observations)
    log_initial = np.log(np.asarray(fixture.dataset.initial, dtype=float))
    log_transition = np.log(np.asarray(fixture.dataset.transition, dtype=float))

    total = 0.0
    for chain in range(observations.shape[0]):
        density = family.log_density(
            torch.as_tensor(observations[chain], dtype=torch.float64)
        )
        total += float(
            forward_backward(
                np.asarray(density, dtype=float), log_initial, log_transition
            ).log_evidence
        )
    return total


def _fit(fixture: NegativeBinomialChains, inputs: CnasterChainInputs, max_iter: int):  # type: ignore[no-untyped-def]
    """Run `pipeline_baum_welch` at a budget with the fixture's transition (#80)."""
    import warnings

    from cnaster.hmm import pipeline_baum_welch
    from cnaster.hmm_nophasing import hmm_nophasing

    # NB the start is supplied: the driver's own initializer raises (#143)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pipeline_baum_welch(
            None,
            inputs.single_X,
            inputs.lengths,
            inputs.n_states,
            inputs.base_nb_mean,
            inputs.total_bb_RD,
            inputs.log_sitewise_transmat,
            hmmclass=hmm_nophasing,
            params="sp",
            t=float(np.exp(inputs.log_transmat[0, 0])),
            init_log_mu=np.asarray(inputs.log_mu, dtype=float).reshape(-1, 1),
            init_p_binom=np.asarray(inputs.p_binom, dtype=float).reshape(-1, 1),
            init_alphas=np.asarray(inputs.alphas, dtype=float).reshape(-1, 1),
            init_taus=np.asarray(inputs.taus, dtype=float).reshape(-1, 1),
            max_iter=max_iter,
            tol=1e-12,
        )


@pytest.mark.oracle
def test_baum_welch_does_not_lower_the_upstream_evidence(cnaster_config: None) -> None:
    """Upstream's evidence does not fall across Baum-Welch iterations, to `TOLERANCE`."""
    fixture = negative_binomial_chains(n_states=3, sequence_length=60, n_sequences=2)
    inputs = from_negative_binomial_chains(fixture)

    evidence = [
        _upstream_evidence_at(
            fixture, result.params.new_log_mu, result.params.new_alphas
        )
        for result in (_fit(fixture, inputs, budget) for budget in (1, 2, 3, 5, 8))
    ]

    for before, after in itertools.pairwise(evidence):
        assert after >= before - TOLERANCE, (
            f"the evidence fell from {before:.9f} to {after:.9f}"
        )


@pytest.mark.oracle
def test_baum_welch_reaches_a_fixed_point(cnaster_config: None) -> None:
    """20 and 40 iterations reach the same upstream evidence, to 1e-6."""
    fixture = negative_binomial_chains(n_states=3, sequence_length=60, n_sequences=2)
    inputs = from_negative_binomial_chains(fixture)

    settled = _fit(fixture, inputs, 20)
    further = _fit(fixture, inputs, 40)

    assert _upstream_evidence_at(
        fixture, further.params.new_log_mu, further.params.new_alphas
    ) == pytest.approx(
        _upstream_evidence_at(
            fixture, settled.params.new_log_mu, settled.params.new_alphas
        ),
        abs=1e-6,
    )


@pytest.mark.bug
def test_the_driver_cannot_initialize_itself(cnaster_config: None) -> None:
    """`pipeline_baum_welch`'s default init calls `gmm_init` with too few args (#143)."""
    import warnings

    from cnaster.hmm import pipeline_baum_welch
    from cnaster.hmm_nophasing import hmm_nophasing

    fixture = negative_binomial_chains(n_states=2, sequence_length=20, n_sequences=1)
    inputs = from_negative_binomial_chains(fixture)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(TypeError, match="gmm_init"):
            pipeline_baum_welch(
                None,
                inputs.single_X,
                inputs.lengths,
                inputs.n_states,
                inputs.base_nb_mean,
                inputs.total_bb_RD,
                inputs.log_sitewise_transmat,
                hmmclass=hmm_nophasing,
                params="sp",
                max_iter=1,
            )


IMPOSED_SELF_TRANSITIONS = [0.99999, 0.999, 0.99, 0.95, 0.9, 0.5]
"""Wrong self-transitions on both sides of the fixture's 0.8 (#142)."""


def _uniform_transition(self_transition: float, n_states: int) -> np.ndarray:
    """`cnaster`'s own shape: `t` on the diagonal, the rest spread evenly."""
    transition = np.full(
        (n_states, n_states), (1.0 - self_transition) / (n_states - 1), dtype=float
    )
    np.fill_diagonal(transition, self_transition)
    return transition


def _paper_transition_update(
    fixture: NegativeBinomialChains, log_transition: np.ndarray
) -> np.ndarray:
    """Return the paper's transition M step from upstream's pairwise posterior, row-normalized."""
    from sal.likelihood.forward_backward import forward_backward

    observations = np.asarray(fixture.dataset.observations)
    log_initial = np.log(np.asarray(fixture.dataset.initial, dtype=float))

    counts = np.zeros(log_transition.shape, dtype=float)
    for chain in range(observations.shape[0]):
        density = fixture.family.log_density(
            torch.as_tensor(observations[chain], dtype=torch.float64)
        )
        counts += np.asarray(
            forward_backward(
                np.asarray(density, dtype=float), log_initial, log_transition
            ).pairwise
        ).sum(axis=0)
    updated: np.ndarray = counts / counts.sum(axis=1, keepdims=True)
    return updated


def _fit_at(  # type: ignore[no-untyped-def]
    fixture: NegativeBinomialChains,
    inputs: CnasterChainInputs,
    self_transition: float,
    max_iter: int,
):
    """`_fit` with the self-transition imposed and `params="stp"`."""
    import warnings

    from cnaster.hmm import pipeline_baum_welch
    from cnaster.hmm_nophasing import hmm_nophasing

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pipeline_baum_welch(
            None,
            inputs.single_X,
            inputs.lengths,
            inputs.n_states,
            inputs.base_nb_mean,
            inputs.total_bb_RD,
            inputs.log_sitewise_transmat,
            hmmclass=hmm_nophasing,
            params="stp",
            t=self_transition,
            init_log_mu=np.asarray(inputs.log_mu, dtype=float).reshape(-1, 1),
            init_p_binom=np.asarray(inputs.p_binom, dtype=float).reshape(-1, 1),
            init_alphas=np.asarray(inputs.alphas, dtype=float).reshape(-1, 1),
            init_taus=np.asarray(inputs.taus, dtype=float).reshape(-1, 1),
            max_iter=max_iter,
            tol=1e-12,
        )


@pytest.mark.bug
@pytest.mark.parametrize("imposed", IMPOSED_SELF_TRANSITIONS)
def test_the_m_step_returns_the_transition_it_was_given(
    cnaster_config: None, imposed: float
) -> None:
    """The M step returns the imposed transition unchanged, bitwise in log space (#142)."""
    fixture = negative_binomial_chains(n_states=3, sequence_length=60, n_sequences=2)
    inputs = from_negative_binomial_chains(fixture)

    fitted = _fit_at(fixture, inputs, imposed, 8)

    # NB compared in log space, where the driver built it and the identity is exact
    np.testing.assert_array_equal(
        np.asarray(fitted.params.new_log_transmat),
        np.log(_uniform_transition(imposed, inputs.n_states)),
    )


@pytest.mark.oracle
@pytest.mark.parametrize("imposed", IMPOSED_SELF_TRANSITIONS)
def test_the_paper_would_move_the_transition_toward_the_truth(imposed: float) -> None:
    """One paper M step from upstream's pairwise posterior moves `t` toward truth (#142)."""
    fixture = negative_binomial_chains(n_states=3, sequence_length=60, n_sequences=2)

    truth = float(np.diag(np.asarray(fixture.dataset.transition, dtype=float)).mean())
    n_states = np.asarray(fixture.dataset.transition).shape[0]
    updated = _paper_transition_update(
        fixture, np.log(_uniform_transition(imposed, n_states))
    )
    paper = float(np.diag(updated).mean())

    assert abs(paper - truth) < abs(imposed - truth), (
        f"the paper's update did not improve on {imposed}: {paper} against {truth}"
    )
