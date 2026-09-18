"""`cnaster.hmm` and the dense recursion, refereed by upstream (#140).

**Two gaps, one module.** `cnaster/hmm.py` is 59 statements at zero: every
rung so far has gone *around* the Baum-Welch driver into `hmm_nophasing`, so
the loop itself has never had a referee. And `hmm_nophasing`'s backward pass
and posterior have none either -- `test_hmm_single_chain.py` referees the
emission and the **forward** total, and `test_oracle_rung.py` the clone-stacked
**ragged** recursion, which leaves the rectangular batch `cnaster` actually
runs, and the backward half of it, unchecked.

Upstream's `likelihood.forward_backward` returns the evidence, the posterior
and the pairwise posterior from one call, so it referees all three at once.
The posterior is the strongest of them: it is a function of **both** passes,
so a backward recursion that were wrong could not produce the right gamma.

`opt.hmm.forward_log_likelihood` referees the driver. Baum-Welch's defining
guarantee is that the evidence does not decrease, and the referee has to be
an independent evaluation of that evidence -- `cnaster`'s own `llf` grading
its own ascent is what `CLAUDE.md` calls measuring one implementation against
itself.
"""

import itertools
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from tests.adapters import CnasterChainInputs, from_negative_binomial_chains
from tests.fixtures import NegativeBinomialChains, negative_binomial_chains

if TYPE_CHECKING:
    from snakes_and_ladders.likelihood.forward_backward import ForwardBackward

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
    """`log gamma` over the concatenated axis, through `cnaster`'s own pieces.

    `hmm.compute_copy_state_posterior` is the function the driver uses, so
    this exercises `hmm.py` as well as both recursions in `hmm_nophasing`.
    """
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
    from snakes_and_ladders.likelihood.forward_backward import forward_backward

    # NB `observations` is `(n_sequences, sequence_length)`, one row per chain,
    #    where `cnaster` takes them concatenated along the observation axis.
    #    The row is what upstream's single-chain recursion wants.
    observations = np.asarray(fixture.dataset.observations)[chain]

    density = fixture.family.log_density(
        torch.as_tensor(observations, dtype=torch.float64)
    )
    return forward_backward(
        np.asarray(density, dtype=float),
        np.log(np.asarray(fixture.dataset.initial, dtype=float)),
        np.log(np.asarray(fixture.dataset.transition, dtype=float)),
    )


@pytest.mark.upstream_oracle
@pytest.mark.parametrize("n_sequences", [1, 3])
def test_the_state_posterior_agrees_with_upstream(n_sequences: int) -> None:
    """**The backward pass's first referee, and the driver's posterior.**

    `log gamma` is a function of both recursions and of
    `hmm.compute_copy_state_posterior`, so this is the one comparison that a
    wrong backward pass cannot survive: a forward-only error would already
    show in the evidence, and a backward-only error shows only here.

    Swept over the sequence count because the recursion restarts at every
    boundary `lengths` declares, and a restart the backward pass got wrong
    would be invisible on one chain.
    """
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


@pytest.mark.upstream_oracle
def test_every_posterior_column_is_a_distribution() -> None:
    """What the comparison above would miss if both sides were unnormalized.

    `compute_copy_state_posterior` subtracts a `logsumexp` across states, so
    each column sums to one. Upstream's rows do too, and an agreement between
    two arrays that were both wrong by the same constant would pass the
    previous test on a single chain.
    """
    fixture = negative_binomial_chains(n_states=4, sequence_length=25, n_sequences=2)
    inputs = from_negative_binomial_chains(fixture)

    posterior = np.exp(_cnaster_posterior(inputs))

    np.testing.assert_allclose(
        posterior.sum(axis=0), np.ones(posterior.shape[1]), atol=POSTERIOR_TOLERANCE
    )


@pytest.mark.upstream_oracle
@pytest.mark.parametrize("n_sequences", [2, 4])
def test_the_evidence_agrees_over_a_rectangular_batch(n_sequences: int) -> None:
    """The shape `cnaster` runs, which the ragged rung does not cover.

    #97 refereed the clone-stacked **ragged** recursion against
    `baum_welch_family`. This is the rectangular batch -- equal-length chains
    concatenated, which is what `pipeline_baum_welch` receives from the
    pseudobulk -- scored per chain against upstream and summed.
    """
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
    """Upstream's evidence at **`cnaster`'s** fitted parameters.

    The referee for the driver. `cnaster`'s own `llf` grading its own ascent
    is one implementation measured against itself, so the family is rebuilt
    upstream from the parameters the fit returned and the evidence is summed
    over the fixture's chains.
    """
    from snakes_and_ladders.emissions import NegativeBinomialEmission
    from snakes_and_ladders.likelihood.forward_backward import forward_backward

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
    """`pipeline_baum_welch` at a budget, on the single-spot chain fixture.

    `params="sp"` so the transition stays at the fixture's, which is what lets
    the referee score a likelihood under a transition both sides agree on.
    `hmmclass=hmm_nophasing` because the default raises on any instance with
    more than one spot (#80).
    """
    import warnings

    from cnaster.hmm import pipeline_baum_welch
    from cnaster.hmm_nophasing import hmm_nophasing

    # NB the starting point is supplied rather than left to the driver's own
    #    initializer, which cannot run (#143): `hmm.py:77` calls `gmm_init`
    #    with five positional arguments where it takes eight. Supplying it
    #    also fixes the ascent's start, which a monotonicity claim needs.
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


@pytest.mark.upstream_oracle
def test_baum_welch_does_not_lower_the_upstream_evidence(cnaster_config: None) -> None:
    """**Baum-Welch's defining guarantee, refereed rather than self-reported.**

    The evidence is non-decreasing across iterations. `cnaster` reports an
    `llf` of its own, so the check has to come from elsewhere: the family is
    rebuilt upstream at each iteration's fitted parameters and scored by
    upstream's recursion.

    This is **#4's scope item 4**, which has had no test anywhere: the driver
    is 59 statements and every rung so far went around it into
    `hmm_nophasing`.

    A failure means the M step is not maximising what the E step computed --
    the classic symptom of a mismatched sufficient statistic, and the one
    defect a per-function comparison cannot find.
    """
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


@pytest.mark.upstream_oracle
def test_baum_welch_reaches_a_fixed_point(cnaster_config: None) -> None:
    """More iterations stop moving the parameters, scored by upstream.

    Monotonicity alone is satisfied by a fit that never converges, so the
    other half of the guarantee is that the ascent terminates. Compared at
    the evidence rather than at the parameters, because two parameter sets
    that differ below the tolerance are the same fixed point and the evidence
    is the quantity the ascent is on.
    """
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


@pytest.mark.cnaster
def test_the_driver_cannot_initialize_itself(cnaster_config: None) -> None:
    """**`pipeline_baum_welch`'s own default raises (#143).**

    Left to initialize, `hmm.py:77` calls `gmm_init` with five positional
    arguments -- `n_states, X, base_nb_mean, total_bb_RD, params` -- where the
    function takes eight: `lengths`, `log_transmat` and
    `log_sitewise_transmat` follow `params` and have no defaults.

    So the branch that fires whenever a caller omits `init_log_mu` or
    `init_p_binom`, which is the signature's own default, cannot execute. The
    same class of defect as #9's `cna_mixture_init`, on a second path.

    Pinned so a fix upstream turns this red rather than passing unnoticed,
    and because the tests above have to work around it.
    """
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
