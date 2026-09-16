"""`run_core_inference` on a planted instance (issue #4).

The top of #14's ladder. Every rung below it is refereed alone, so a failure
here is attributable: the draw is pinned in `test_core_inference_fixture.py`,
the emission against upstream, and the field at the planted states.

**Two things the fixture established about the entry point itself**, both
recorded as tests rather than as prose:

*   At its own default `hmmclass=hmm_phased` it cannot complete one outer
    iteration on any instance with more than one spot.
*   `icm_sweep_deque`'s `min_clone_spots` defaults to 200 and
    `pipeline_clone_assignment` does not pass one, so every clone below that
    size is merged away and no labelling can be recovered under it.

The declared scale -- `M = K = 10`, `G = 10,000`, `S = 5,000` -- carries the
`release` marker and the reason is in `test_the_declared_scale_is_out_of
_reach_here`: the emission alone is 8.00 GB per outer iteration.
"""

import warnings
from typing import Any

import numpy as np
import pytest

from tests.adapters import from_core_inference_truth
from tests.fixtures import CoreInferenceTruth, core_inference_truth

MIN_CLONE_SPOTS = 200
"""`icm_sweep_deque`'s default, which `run_core_inference` does not expose.

A clone smaller than this is emptied into another during the label solve, so a
fixture below it measures the merge and not the solver.
"""


def _run(truth: CoreInferenceTruth, **kwargs: object) -> Any:
    from cnaster.hmm_nophasing import hmm_nophasing
    from cnaster.hmrf import run_core_inference

    with warnings.catch_warnings():
        # `scipy` rejects the `ftol` the shipped solver options pass (#46); the
        # warning is that defect firing on the live path, not this test's.
        warnings.simplefilter("ignore")
        return run_core_inference(
            **from_core_inference_truth(truth).as_kwargs(),
            hmmclass=hmm_nophasing,
            **kwargs,
        )


def _best_permutation_accuracy(fitted: np.ndarray, planted: np.ndarray) -> float:
    """Labelling accuracy up to a permutation of clone names.

    Clone indices are arbitrary -- the model is invariant to relabelling them
    -- so a comparison that fixes them measures the ordering and not the
    partition.
    """
    from itertools import permutations

    classes = int(planted.max()) + 1
    return max(
        float((np.array(order)[fitted] == planted).mean())
        for order in permutations(range(classes))
    )


@pytest.mark.cnaster
def test_the_default_hmm_class_cannot_complete_an_outer_iteration(
    cnaster_config: None,
) -> None:
    """`hmm_phased`, the default, indexes `log_mu` past its own shape.

    `run_core_inference` fits the **clone-stacked** pseudobulk, which has one
    column, so `new_log_mu` is `(n_states, 1)`.
    `compute_emission_probability_nb_betabinom_coded` then reads
    `n_states, n_spots = log_mu.shape` and immediately overwrites `n_spots`
    with the encoder's, while still indexing `log_mu[i, s]` over that larger
    range (`hmm_phased.py:118-145`).

    So the shipped default raises for any instance with more than one spot,
    and it raises in `pipeline_clone_assignment` after the whole HMM fit has
    run. Pinned so that a fix upstream turns this red rather than passing
    unnoticed.
    """
    from cnaster.hmrf import run_core_inference

    truth = core_inference_truth(
        n_clones=2, n_states=3, lattice=(6, 5), n_obs=60, n_segments=2
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(IndexError, match="out of bounds"):
            run_core_inference(
                **from_core_inference_truth(truth).as_kwargs(),
                max_iter_outer=1,
                max_iter=5,
            )


@pytest.mark.cnaster
def test_a_clone_below_the_solver_s_floor_is_merged_away(cnaster_config: None) -> None:
    """Under 200 spots a clone cannot survive, whatever the data says.

    `pipeline_clone_assignment` calls `icm_sweep_deque` without
    `min_clone_spots`, so the default of 200 applies and the enforcement
    reassigns every spot of any smaller clone. The planted labelling here is
    separable -- the field recovers it exactly in
    `test_core_inference_fixture.py` -- and the run still returns one clone.
    """
    truth = core_inference_truth(
        n_clones=2, n_states=3, lattice=(6, 5), n_obs=60, n_segments=2
    )

    result = _run(truth, max_iter_outer=1, max_iter=5)
    fitted = np.asarray(result.assignment.new_assignment)

    assert truth.clone_index[0].size < MIN_CLONE_SPOTS
    assert np.unique(fitted).size == 1, "a clone survived below the floor"


@pytest.mark.planted
@pytest.mark.release
def test_the_run_recovers_the_planted_labelling(cnaster_config: None) -> None:
    """Above the floor, the labelling comes back up to a permutation.

    `release` because it is over the per-pull-request budget: the clone floor
    forces at least `200 * n_clones` spots before the label solve is a solve
    at all, and the emission grows with the product of every extent.
    """
    truth = core_inference_truth(
        n_clones=2, n_states=3, lattice=(30, 20), n_obs=300, n_segments=4
    )
    assert min(index.size for index in truth.clone_index) >= MIN_CLONE_SPOTS

    result = _run(truth, max_iter_outer=2, max_iter=10)
    fitted = np.asarray(result.assignment.new_assignment)

    accuracy = _best_permutation_accuracy(fitted, truth.labels)
    assert accuracy > 0.9, f"labelling accuracy {accuracy:.3f}"


@pytest.mark.planted
@pytest.mark.release
def test_the_run_recovers_the_extreme_states_and_not_the_middle_one(
    cnaster_config: None,
) -> None:
    """The allele extremes come back; the expression does not, at all.

    Measured on this instance -- planted `mu` `[0.50, 2.75, 5.00]`, `p_binom`
    `[0.52, 0.70, 0.88]`, under the Weierstrass exposure #4's fixture plants:

    | budget | fitted `mu` | fitted `p_binom` |
    | --- | --- | --- |
    | 2 outer, 10 EM | 0.498, 4.880, 10.657 | 0.521, 0.880, 0.884 |
    | 3 outer, 30 EM | 0.498, 4.880, 7.731 | 0.521, 0.880, 0.888 |

    Three things are established and each is asserted.

    The **allele** extremes are recovered to 0.02 and the middle state is not
    -- `0.70` lands on `0.880`, leaving two of three indistinguishable.

    The **expression** is not recovered beyond the lowest state: the worst
    relative error is 1.131, and at ten times the budget it is still 0.775.
    Under an exposure drawn i.i.d. over both axes the same fit reached 0.147,
    so what breaks it is not that the exposure varies but that it varies
    **along the bin axis**, where it aliases with the state path instead of
    averaging out.

    More iterations do not fix either channel. That is #82, and the numbers
    here are the sharper version of it.
    """
    truth = core_inference_truth(
        n_clones=2, n_states=3, lattice=(30, 20), n_obs=300, n_segments=4
    )

    result = _run(truth, max_iter_outer=2, max_iter=10)
    fitted_mu = np.sort(np.exp(np.asarray(result.params.new_log_mu).ravel()))
    fitted_p = np.sort(np.asarray(result.params.new_p_binom).ravel())
    planted_mu = np.sort(np.exp(truth.log_mu))
    planted_p = np.sort(truth.p_binom)

    np.testing.assert_allclose(fitted_p[[0, -1]], planted_p[[0, -1]], atol=0.02)

    middle_gap = abs(fitted_p[1] - planted_p[1])
    assert middle_gap > 0.1, (
        f"the middle allele state now recovers to {middle_gap:.3f}; if that is "
        "a fix upstream, this assertion is what should change"
    )

    np.testing.assert_allclose(fitted_mu[0], planted_mu[0], rtol=0.05)

    expression_error = float(np.abs(fitted_mu / planted_mu - 1.0).max())
    assert expression_error > 0.5, (
        f"the expression now recovers to {expression_error:.3f} under a "
        "bin-varying exposure; a fix upstream is what should change this"
    )


@pytest.mark.planted
@pytest.mark.release
def test_the_declared_scale_plants_and_recovers_its_parameters() -> None:
    """`M = K = 10`, `G = 10,000`, `S = 5,000`: the fixture, and its truth.

    The instance builds in about 17 s at 2.1 GB and every planted parameter is
    recovered from the counts it generated. What is **not** asserted here is
    `run_core_inference` on it, for the reason the next test measures.
    """
    truth = core_inference_truth(
        n_clones=10, n_states=10, lattice=(50, 100), n_obs=10_000, n_segments=20
    )

    assert (truth.n_clones, truth.n_states) == (10, 10)
    assert (truth.n_obs, truth.n_spots) == (10_000, 5_000)

    per_spot = truth.states[truth.labels].T
    for state in range(truth.n_states):
        mask = per_spot == state
        counts, exposure = truth.counts_nb[mask], truth.base_nb_mean[mask]
        successes, trials = truth.counts_bb[mask], truth.total_bb_RD[mask]

        np.testing.assert_allclose(
            (counts / exposure).mean(), np.exp(truth.log_mu[state]), rtol=0.02
        )
        np.testing.assert_allclose(
            (successes / trials).mean(), truth.p_binom[state], rtol=0.02
        )


@pytest.mark.analytic
def test_the_declared_scale_is_out_of_reach_of_a_single_run_here() -> None:
    """Why the scale above validates the fixture and not the inference.

    `cnaster` materializes `(n_states, n_obs, n_spots)` twice per outer
    iteration. At the declared extents that is 8.00 GB before the pooled
    copies, against a profile that peaked at 4.5x its emission array at
    `K = 7`, `G = 3,000`, `S = 2,500`. Recorded as a number so the decision to
    mark the run `release` is a measurement rather than a preference.
    """
    truth = core_inference_truth(
        n_clones=10, n_states=10, lattice=(10, 10), n_obs=100, n_segments=2
    )
    declared = 2.0 * 10 * 10_000 * 5_000 * 8 / 1e9

    assert truth.emission_gigabytes < 0.01
    assert declared == pytest.approx(8.0), f"{declared:.2f} GB"
