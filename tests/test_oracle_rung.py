"""`cnaster`'s recursion against upstream's, on the shape the pipeline fits.

#97, half of it. `cnaster` stacks clones along the genomic axis and tiles the
segmentation with them -- `clone_stack_lengths = np.tile(lengths, n_clones)`
(`hmrf_utils.py:51`) -- so the fit runs over `n_clones * n_chromosomes`
chains of unequal length. #667 gave upstream a shape that can hold that, and
this is the first rung that hands it one.

**What is refereed here is the E step, not the fit.** Both sides are given the
same parameters and the same data, so the comparison is between two
implementations of one recursion rather than between two optimizers' local
optima. The fit-level rung needs `fit_spatio_sequential` under a per-channel
covariate, which upstream still refuses; the last test pins that refusal so it
fails the day it lands.
"""

from dataclasses import dataclass

import numpy as np
import pytest
from snakes_and_ladders.ragged import Ragged

from tests.fixtures import (
    CoreInferenceTruth,
    circulant_transition,
    core_inference_truth,
)

LATTICE_SIDE = 6
"""A 36-node lattice: enough for the spatial prior to bind, small enough to fit."""

FIT_CLASSES = 2
FIT_STATES = 2
FIT_POSITIONS = 12
FIT_SEGMENTS = (5, 7)
"""Unequal segments, so the fit takes the ragged path rather than converting."""

TRIALS = 40
"""The pair's trial count, the covariate's second channel."""

FIT_ACCURACY = 0.9
"""What the label solver recovers of the planted labelling."""

TOLERANCE = 1e-9
"""Absolute agreement required between the two recursions.

Both work in log space over the same doubles, so the difference is summation
order rather than algorithm; the realized figure is in the test's message.
"""


@pytest.fixture(scope="module")
def planted() -> CoreInferenceTruth:
    """Unequal chromosomes, so the stacked batch is genuinely ragged."""
    return core_inference_truth(
        n_clones=3,
        n_states=4,
        lattice=(12, 20),
        n_obs=120,
        n_segments=5,
        self_transition=0.9,
        seed=13,
    )


@dataclass(frozen=True)
class Stacked:
    """The pseudobulk, clone-stacked exactly as `cnaster` stacks it.

    `clone_stack_obs` returns six values positionally; naming them here is what
    keeps the rung readable, since the batch shape is the whole point of it.
    """

    X: np.ndarray
    base_nb_mean: np.ndarray
    total_bb_RD: np.ndarray
    lengths: np.ndarray
    sitewise: np.ndarray


def _stacked(truth: CoreInferenceTruth) -> Stacked:
    """Aggregate to pseudobulk and stack the clones along the genomic axis."""
    from cnaster.hmrf_utils import clone_stack_obs
    from cnaster.pseudobulk import merge_pseudobulk_by_index_mix

    counts = np.stack([truth.counts_nb, truth.counts_bb], axis=1)
    X, base, total, _ = merge_pseudobulk_by_index_mix(
        counts, truth.base_nb_mean, truth.total_bb_RD, truth.clone_index
    )
    stack_X, stack_base, stack_total, lengths, sitewise, _ = clone_stack_obs(
        X, base, total, truth.lengths, np.zeros((truth.n_obs, 2)), None
    )
    return Stacked(stack_X, stack_base, stack_total, lengths, sitewise)


def _emission(truth: CoreInferenceTruth, stacked: Stacked) -> np.ndarray:
    """`cnaster`'s per-state score over the stacked batch, `(n_obs, n_states)`."""
    from cnaster.hmm_nophasing import hmm_nophasing

    n_states = truth.n_states

    rdr, baf = hmm_nophasing.compute_emission_probability_nb_betabinom(
        stacked.X,
        stacked.base_nb_mean,
        truth.log_mu.reshape(n_states, 1),
        truth.alphas.reshape(n_states, 1),
        stacked.total_bb_RD,
        truth.p_binom.reshape(n_states, 1),
        truth.taus.reshape(n_states, 1),
    )
    summed: np.ndarray = (rdr + baf)[:, :, 0].T
    return summed


@pytest.mark.upstream_oracle
@pytest.mark.critical
def test_the_two_recursions_agree_on_the_clone_stacked_batch(
    planted: CoreInferenceTruth,
) -> None:
    """The total log-likelihood, `cnaster`'s forward against upstream's ragged one.

    `cnaster` walks the concatenated axis and restarts at each boundary
    `lengths` declares; upstream walks the segments of a `Ragged` and returns
    one evidence per segment. Two implementations of the same recursion over
    the same fifteen chains of unequal length, and the agreement is the claim.
    """
    from cnaster.hmm_nophasing import hmm_nophasing
    from scipy.special import logsumexp
    from snakes_and_ladders.likelihood.ragged_rust import posteriors

    stacked = _stacked(planted)
    lengths = stacked.lengths
    density = _emission(planted, stacked)

    initial = np.full(planted.n_states, 1.0 / planted.n_states)
    transition = circulant_transition(planted.n_states, planted.self_transition)

    log_alpha = hmm_nophasing.forward_lattice(
        lengths,
        np.log(transition),
        np.log(initial),
        density.T[:, :, None],
        stacked.sitewise,
    )
    ends = np.cumsum(lengths) - 1
    theirs = float(sum(logsumexp(log_alpha[:, end]) for end in ends))

    _, _, evidence = posteriors(
        Ragged(values=density, lengths=tuple(int(x) for x in lengths)),
        np.log(initial),
        np.log(transition),
    )
    ours = float(evidence.sum())

    assert lengths.size == planted.n_clones * planted.lengths.size
    assert len(set(lengths.tolist())) > 1, "the stacked batch is not ragged"
    assert abs(theirs - ours) < TOLERANCE, (
        f"cnaster {theirs:.10f} against upstream {ours:.10f}, "
        f"difference {abs(theirs - ours):.3e}"
    )


@pytest.mark.upstream
def test_a_covariate_carrying_its_own_channel_axis_is_refused_with_a_singleton() -> (
    None
):
    """The shape `m_step` used to build, and why the fit-level rung is not here.

    #77 item 1: `search/spatio_sequential.py`'s `m_step` appended a trailing
    singleton unconditionally, so an `(S, V, 2)` covariate -- one that already
    names the family's two channels -- arrived as `(S, V, 2, 1)`, whose last
    axis names nothing. This pins that the family refuses it.

    **The fix has landed and this still passes, which is correct.** Upstream's
    #674 changed the *caller*: `m_step` now calls `covariate_block` and moves
    the axis, with a comment naming the defect, and the pin carries it.
    Refusing a covariate whose last axis names no channel is right either way,
    so what this guards now is the regression rather than the defect.

    The rung #674 unblocks is #109 and is not here yet: a first covaried
    ragged instance built from scratch reaches the negative binomial's M step
    with `mean and weight must be positive, got nan and 0.0`, which is a
    fixture-construction problem on this side rather than an upstream one.
    """
    import torch
    from snakes_and_ladders.emissions import (
        BetaBinomialEmission,
        CovariateNotSupportedError,
        NegativeBinomialEmission,
    )
    from snakes_and_ladders.sim.count_pairs import IndependentCountPair

    family = IndependentCountPair(
        NegativeBinomialEmission(dispersion=np.full(2, 6.0), mean=np.array([1.0, 3.0])),
        BetaBinomialEmission(
            trials=np.ones(2), alpha=np.array([15.0, 25.0]), beta=np.array([15.0, 5.0])
        ),
    )

    with pytest.raises(CovariateNotSupportedError):
        family.reestimate(
            torch.zeros((4, 6, 2), dtype=torch.float64),
            torch.full((4, 6, 2), 0.5, dtype=torch.float64),
            covariate=torch.ones((4, 6, 2, 1), dtype=torch.float64),
        )
