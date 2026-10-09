"""`run_core_inference` on a planted instance (#4, top of #14's ladder).

Referee: the planted labelling and states. `key_instance`'s inference is not run: it
does
not fit in memory (#90).
"""

import warnings
from itertools import permutations

import numpy as np
import pytest
from cnaster.hmrf import run_core_inference
from port.sim.truth import (
    core_inference_truth,
    critical_instance,
    dev_instance,
    key_instance,
)

from tests.adapters import from_core_inference_truth
from tests.builders import cnaster_emission_pair, emission_inputs
from tests.fixtures import partition_ari, run_planted_core_inference

DECLARED_SPOTS = 5_000
"""`S` at the scale #87 names, which the inference does not fit in."""

MIN_CLONE_SPOTS = 200
"""`icm_sweep_deque`'s default `min_clone_spots`, which `run_core_inference` does not expose."""


def _best_permutation_accuracy(fitted: np.ndarray, planted: np.ndarray) -> float:
    """Labelling accuracy up to a permutation of clone names."""

    classes = int(planted.max()) + 1
    return max(
        float((np.array(order)[fitted] == planted).mean())
        for order in permutations(range(classes))
    )


@pytest.mark.bug
def test_the_default_hmm_class_cannot_complete_an_outer_iteration(
    cnaster_config: None,
) -> None:
    """The default `hmm_phased` indexes `log_mu` past its shape and raises (`hmm_phased.py:118-145`)."""

    truth = core_inference_truth(
        n_clones=2, n_states=3, lattice=(6, 6), n_obs=60, n_segments=2
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
@pytest.mark.merge
def test_a_clone_below_the_solver_s_floor_is_merged_away(cnaster_config: None) -> None:
    """A separable clone under 200 spots is merged away, on equal bands at 6 x 5 (#298, #417)."""
    truth = core_inference_truth(
        n_clones=2,
        n_states=3,
        lattice=(6, 5),
        n_obs=60,
        n_segments=2,
        normal_clone=False,
    )

    result = run_planted_core_inference(truth, max_iter_outer=1, max_iter=5)
    fitted = np.asarray(result.assignment.new_assignment)

    assert truth.clone_index[0].size < MIN_CLONE_SPOTS
    assert np.unique(fitted).size == 1, "a clone survived below the floor"


@pytest.mark.end2end
@pytest.mark.release
def test_the_run_recovers_the_planted_labelling(cnaster_config: None) -> None:
    """Above the clone floor the labelling is recovered up to a permutation."""
    truth = core_inference_truth(
        n_clones=2, n_states=3, lattice=(30, 20), n_obs=300, n_segments=4
    )
    assert min(index.size for index in truth.clone_index) >= MIN_CLONE_SPOTS

    result = run_planted_core_inference(truth, max_iter_outer=2, max_iter=10)
    fitted = np.asarray(result.assignment.new_assignment)

    accuracy = _best_permutation_accuracy(fitted, truth.labels)
    assert accuracy > 0.9, f"labelling accuracy {accuracy:.3f}"


@pytest.mark.end2end
@pytest.mark.release
def test_the_run_recovers_every_planted_state_on_a_mostly_neutral_genome(
    cnaster_config: None,
) -> None:
    """On a mostly neutral genome all planted `p` and `mu` are recovered, worst relative error 0.026 (#82, #86, #120)."""
    truth = core_inference_truth(
        n_clones=2, n_states=3, lattice=(30, 20), n_obs=300, n_segments=4
    )

    result = run_planted_core_inference(truth, max_iter_outer=2, max_iter=10)
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
    """At `M = K = 10`, `G = 10,000`, `S = 5,000` the fixture recovers its planted parameters by moments."""
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


@pytest.mark.patch
def test_the_declared_scale_is_out_of_reach_of_a_single_run_here() -> None:
    """At the declared scale `cnaster`'s two `(n_states, n_obs, n_spots)` arrays need 8.00 GB, by its returned pair."""
    truth = core_inference_truth(
        n_clones=10, n_states=10, lattice=(20, 5), n_obs=100, n_segments=2
    )
    declared = 2.0 * 10 * 10_000 * 5_000 * 8 / 1e9

    assert truth.emission_gigabytes < 0.01
    assert declared == pytest.approx(8.0), f"{declared:.2f} GB"

    # NB `emission_gigabytes` is what `cnaster` returns at the truth's shape
    rdr, baf = cnaster_emission_pair(
        emission_inputs(truth.n_states, truth.n_obs, truth.n_spots, seed=0).columns(
            truth.n_spots
        )
    )
    assert (rdr.nbytes + baf.nbytes) / 1e9 == truth.emission_gigabytes


@pytest.mark.end2end
@pytest.mark.release
def test_the_dev_instance_recovers_its_labelling(cnaster_config: None) -> None:
    """The dev instance (`M = 4`, `K = 10`, `G = 1,000`, `S = 1,600`) recovers its labelling at ARI 1.000 (#137)."""
    truth = dev_instance()

    assert (truth.n_clones, truth.n_states) == (4, 10)
    assert (truth.n_obs, truth.n_spots) == (1_000, 1_600)
    assert min(index.size for index in truth.clone_index) >= MIN_CLONE_SPOTS

    result = run_planted_core_inference(truth, max_iter_outer=1, max_iter=3)
    fitted = np.asarray(result.assignment.new_assignment)

    assert np.unique(fitted).size == truth.n_clones
    assert partition_ari(truth.labels, fitted) == pytest.approx(1.0)


@pytest.mark.end2end
@pytest.mark.critical
def test_the_critical_instance_recovers_its_labelling(cnaster_config: None) -> None:
    """The critical instance (`M = K = 2`, `G = 1,000`, `S = 500`) recovers its labelling at ARI 1.000."""
    truth = critical_instance()

    assert (truth.n_clones, truth.n_states) == (2, 2)
    assert (truth.n_obs, truth.n_spots) == (1_000, 500)
    assert min(index.size for index in truth.clone_index) >= MIN_CLONE_SPOTS

    result = run_planted_core_inference(truth, max_iter_outer=1, max_iter=3)
    fitted = np.asarray(result.assignment.new_assignment)

    assert np.unique(fitted).size == truth.n_clones
    assert partition_ari(truth.labels, fitted) == pytest.approx(1.0)
