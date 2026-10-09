"""`cnaster`'s forward recursion against upstream's ragged one, clone-stacked (#97, #667).

E step only; the fit-level rung waits on upstream (last test).
"""

import numpy as np
import pytest
import torch
from cnaster.hmm_nophasing import hmm_nophasing
from port.sim.truth import CoreInferenceTruth, core_inference_truth
from sal.emissions import (
    BetaBinomialEmission,
    CovariateNotSupportedError,
    NegativeBinomialEmission,
)
from sal.likelihood.ragged import posteriors
from sal.ragged import Ragged
from sal.sim.count_pairs import IndependentCountPair
from scipy.special import logsumexp

from tests.adapters import Stacked, stacked_clones
from tests.fixtures import circulant_transition

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
"""Absolute agreement required between the two recursions (summation order only)."""


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


def _emission(truth: CoreInferenceTruth, stacked: Stacked) -> np.ndarray:
    """`cnaster`'s per-state score over the stacked batch, `(n_obs, n_states)`."""

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


@pytest.mark.oracle
@pytest.mark.critical
def test_the_two_recursions_agree_on_the_clone_stacked_batch(
    planted: CoreInferenceTruth,
) -> None:
    """Total log-likelihood: `cnaster`'s forward against upstream's ragged one, within `ATOL`."""

    stacked = stacked_clones(planted)
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


@pytest.mark.warning
def test_a_covariate_carrying_its_own_channel_axis_is_refused_with_a_singleton() -> (
    None
):
    """Upstream refuses an `(S, V, 2, 1)` covariate, guarding #674 (#77 item 1, #109)."""

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
