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

`dev_instance` is what these run against: 35.4 s, and it recovers its
labelling exactly, so a failure here is a failure of the code rather than of
the instance. `key_instance` is the declared scale, `M = K = 10`, `G = 10,000`,
`S = 5,000`. Its fixture is exercised here; **the inference on it is not**,
because it does not fit in memory (#90). That run is its own change and its
own pull request.
"""

import warnings
from typing import Any

import numpy as np
import pytest

from tests.adapters import from_core_inference_truth
from tests.fixtures import (
    CoreInferenceTruth,
    core_inference_truth,
    critical_instance,
    dev_instance,
    key_instance,
)

DECLARED_SPOTS = 5_000
"""`S` at the scale #87 names, and the one the inference does not fit in."""

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


def _adjusted_rand_index(planted: np.ndarray, fitted: np.ndarray) -> float:
    """Agreement between two partitions, invariant to how either is labelled.

    Written here rather than taken from `scikit-learn`: it is `cnaster`'s
    dependency and not this repository's, and a referee that arrives through
    the subject is not independent of it.

    Ten classes have 3.6 million permutations, so the exact-permutation
    accuracy below does not scale; this counts agreeing pairs instead and
    corrects for the agreement expected by chance.
    """
    from math import comb

    table = np.zeros((int(planted.max()) + 1, int(fitted.max()) + 1), dtype=np.int64)
    np.add.at(table, (planted, fitted), 1)

    pairs = sum(comb(int(n), 2) for n in table.ravel())
    by_planted = sum(comb(int(n), 2) for n in table.sum(axis=1))
    by_fitted = sum(comb(int(n), 2) for n in table.sum(axis=0))
    total = comb(int(planted.size), 2)

    expected = by_planted * by_fitted / total
    maximum = 0.5 * (by_planted + by_fitted)
    return float((pairs - expected) / (maximum - expected))


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


@pytest.mark.bug
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


@pytest.mark.warning
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


@pytest.mark.end2end
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


@pytest.mark.end2end
@pytest.mark.release
def test_the_run_recovers_every_planted_state_on_a_mostly_neutral_genome(
    cnaster_config: None,
) -> None:
    """**It recovers all of them, and that reverses #82 and #86.**

    Planted `p` of `[0.5, 0.58, 0.88]` comes back as `[0.500, 0.581, 0.881]`;
    planted `mu` of `[1, 1.5, 5]` comes back as `[0.997, 1.513, 4.871]`, a
    worst relative error of **0.026**.

    On the fixture this replaces -- a Markov chain visiting three states
    roughly equally -- the same call reached a worst `mu` error of **1.131**,
    and the middle allele state landed on the top one, leaving two of three
    indistinguishable. Ten times the budget left it at 0.775.

    So what #82 and #86 measured was the **fixture**, not `cnaster`. A genome
    whose states are visited uniformly under an exposure varying along the bin
    axis defeats the fit; a mostly-neutral genome carrying events -- the
    realistic one, #120 -- does not, at the same exposure and the same budget.
    Both tickets carry the correction.

    The occupancy is what changed: `[0.860, 0.063, 0.077]` here against three
    states near a third each before. Long neutral runs give the initializer a
    baseline to place the others against, which is what a real sample has.
    """
    truth = core_inference_truth(
        n_clones=2, n_states=3, lattice=(30, 20), n_obs=300, n_segments=4
    )

    result = _run(truth, max_iter_outer=2, max_iter=10)
    fitted_mu = np.sort(np.exp(np.asarray(result.params.new_log_mu).ravel()))
    fitted_p = np.sort(np.asarray(result.params.new_p_binom).ravel())
    planted_mu = np.sort(np.exp(truth.log_mu))
    planted_p = np.sort(truth.p_binom)

    np.testing.assert_allclose(fitted_p, planted_p, atol=0.005)

    worst = float(np.max(np.abs(fitted_mu - planted_mu) / planted_mu))
    assert worst < 0.05, f"worst relative error in mu {worst:.3f}"


@pytest.mark.end2end
@pytest.mark.release
def test_the_declared_scale_plants_and_recovers_its_parameters() -> None:
    """`M = K = 10`, `G = 10,000`, `S = 5,000`: the fixture, and its truth.

    The instance builds in about 17 s at 2.1 GB and every planted parameter is
    recovered from the counts it generated. What is **not** asserted here is
    `run_core_inference` on it, for the reason the next test measures.
    """
    truth = key_instance(n_segments=20)

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


@pytest.mark.smoke
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


@pytest.mark.end2end
@pytest.mark.release
def test_the_dev_instance_recovers_its_labelling(cnaster_config: None) -> None:
    """The dev instance, and it recovers the labelling exactly.

    `M = 4`, `K = 10`, `G = 1,000`, `S = 1,600`: 35.4 s against the key
    instance's 310 s, at an adjusted Rand index of **1.000**.

    `S` is 1,600 rather than 1,000 because the lattice is square (#137): four
    bands of 400 spots over `40 x 40`, at 120 boundary edges and a
    perimeter-to-area of **0.300**, against the old strip's 300 edges and
    1.200. The recovery is exact on both, so what squaring bought is not a
    better number -- it is a number measured where the spatial prior is not
    being asked to hold a ribbon. That combination is what makes it worth having -- an instance
    that failed to recover would give a developer nothing to work against, and
    one that took five minutes would stop them looking.

    Marked `release` with the key one because both drive the whole pipeline;
    the difference is that this is the one to run by hand while changing
    something.
    """
    truth = dev_instance()

    assert (truth.n_clones, truth.n_states) == (4, 10)
    assert (truth.n_obs, truth.n_spots) == (1_000, 1_600)
    assert min(index.size for index in truth.clone_index) >= MIN_CLONE_SPOTS

    result = _run(truth, max_iter_outer=1, max_iter=3)
    fitted = np.asarray(result.assignment.new_assignment)

    assert np.unique(fitted).size == truth.n_clones
    assert _adjusted_rand_index(truth.labels, fitted) == pytest.approx(1.0)


@pytest.mark.end2end
@pytest.mark.critical
def test_the_critical_instance_recovers_its_labelling(cnaster_config: None) -> None:
    """The early gate's end-to-end run: `M = K = 2`, `G = 1,000`, `S = 500`.

    The same claim as the dev instance's -- the planted labelling comes back
    exactly, adjusted Rand index **1.000** -- at the smallest instance that
    still clears the solver's clone floor. It is what `-m critical` runs so
    that a broken pipeline is found in seconds; the dev and key instances say
    whether it still holds at scale, and they are the tier for that.
    """
    truth = critical_instance()

    assert (truth.n_clones, truth.n_states) == (2, 2)
    assert (truth.n_obs, truth.n_spots) == (1_000, 500)
    assert min(index.size for index in truth.clone_index) >= MIN_CLONE_SPOTS

    result = _run(truth, max_iter_outer=1, max_iter=3)
    fitted = np.asarray(result.assignment.new_assignment)

    assert np.unique(fitted).size == truth.n_clones
    assert _adjusted_rand_index(truth.labels, fitted) == pytest.approx(1.0)
