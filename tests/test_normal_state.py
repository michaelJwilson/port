"""The planted normal state against `cnaster`'s `find_diploid_balanced_state` definition
(#106).

Balanced within `EPS_BAF`, occupying `min_prop_threshold`, `mu` closest to 1; the
paper's
normalization differs and is stated in the last test.
"""

import numpy as np
import pytest
from cnaster.integer_copy import find_diploid_balanced_state
from port.sim.truth import CoreInferenceTruth, dev_instance

NORMAL_STATE = 0
"""Which state the fixture plants as diploid and balanced."""

EPS_BAF = 0.05
"""`integer_copy.py:110`'s default: how far from balance still counts."""

MIN_PROPORTION = 0.1
"""`integer_copy.py:153`: how much of the genome a candidate must occupy."""

RDR_TOLERANCE = 0.1
"""`integer_copy.py:94`: outside `1 +/- this`, the candidate is warned about."""


@pytest.fixture(scope="module")
def planted() -> CoreInferenceTruth:
    """The instance the round trip runs on."""
    return dev_instance()


@pytest.mark.end2end
@pytest.mark.critical
def test_the_planted_normal_state_is_normal_by_cnasters_definition(
    planted: CoreInferenceTruth,
) -> None:
    """Every clause of `find_diploid_balanced_state`, on the planted parameters."""
    assert planted.p_binom[NORMAL_STATE] == pytest.approx(0.5, abs=EPS_BAF)
    assert np.exp(planted.log_mu[NORMAL_STATE]) == pytest.approx(1.0, abs=RDR_TOLERANCE)

    # Closest to one, since a state nearer unity would be chosen instead.
    distances = np.abs(1.0 - np.exp(planted.log_mu))
    assert int(np.argmin(distances)) == NORMAL_STATE
    assert distances[NORMAL_STATE] == 0.0


@pytest.mark.end2end
@pytest.mark.critical
def test_the_planted_normal_state_is_the_candidate_it_selects(
    planted: CoreInferenceTruth,
) -> None:
    """The planted normal state occupies 0.944 of bins, above `min_prop_threshold` (#120, #298)."""

    path = planted.states.reshape(-1)
    occupancy = np.bincount(path, minlength=planted.n_states) / path.size

    assert occupancy[NORMAL_STATE] == pytest.approx(0.944, abs=5e-3)
    assert occupancy[NORMAL_STATE] > MIN_PROPORTION

    chosen = find_diploid_balanced_state(
        planted.log_mu,
        planted.p_binom,
        path,
        min_prop_threshold=MIN_PROPORTION,
        EPS_BAF=EPS_BAF,
    )

    assert chosen == NORMAL_STATE


@pytest.mark.snapshot
def test_a_mostly_diploid_genome_is_the_candidate_it_wants(
    planted: CoreInferenceTruth,
) -> None:
    """A half-diploid path clears the threshold and selects the planted state."""

    rng = np.random.default_rng(101)
    path = np.where(
        rng.random(planted.n_obs) < 0.5,
        NORMAL_STATE,
        rng.integers(1, planted.n_states, planted.n_obs),
    )

    chosen = find_diploid_balanced_state(
        planted.log_mu,
        planted.p_binom,
        path,
        min_prop_threshold=MIN_PROPORTION,
        EPS_BAF=EPS_BAF,
    )

    assert chosen == NORMAL_STATE


@pytest.mark.warning
def test_the_planted_scale_is_cnasters_and_not_the_papers() -> None:
    """`mu = 1` is on `cnaster`'s scale, not the paper's: clone 1's exposure-weighted rate is 1.202 (#5, #298)."""
    truth = dev_instance()

    weights = truth.base_nb_mean.sum(axis=1)
    weights = weights / weights.sum()
    normalized = float(np.sum(weights * np.exp(truth.log_mu[truth.states[1]])))

    assert normalized != pytest.approx(1.0, abs=0.05), (
        f"the exposure-weighted rate is {normalized:.3f}; if it were one the "
        f"two scales would agree and this difference would not need stating"
    )
