"""The planted normal state, against the definition `cnaster` applies.

#106. The fixture plants state zero at `mu = 1`, `p = 0.5`, and two stages
require such a state to exist: `find_diploid_balanced_state` raises without
one, and the normal-spot path tests every bin against a beta-binomial with
`p` forced to 0.5.

**The definition is `cnaster`'s, read off the code rather than chosen.**
`integer_copy.py:63` takes the states whose BAF is within `EPS_BAF = 0.05` of
balance and which occupy at least `min_prop_threshold = 0.1` of the bins, and
among those picks `argmin |1 - exp(log_mu)|` -- closest to one. It then warns
if the winner's `exp(log_mu)` leaves `[0.9, 1.1]`. So normal is `mu = 1` and
`p = 0.5`, with a stated tolerance on each.

The paper defines the same quantity and normalizes it differently, which is a
stated difference and is recorded in the last test here.
"""

import numpy as np
import pytest

from tests.fixtures import CoreInferenceTruth, dev_instance

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


@pytest.mark.planted
@pytest.mark.critical
def test_the_planted_normal_state_is_normal_by_cnasters_definition(
    planted: CoreInferenceTruth,
) -> None:
    """Every clause of `find_diploid_balanced_state`, on the planted parameters."""
    assert planted.p_binom[NORMAL_STATE] == pytest.approx(0.5, abs=EPS_BAF)
    assert np.exp(planted.log_mu[NORMAL_STATE]) == pytest.approx(1.0, abs=RDR_TOLERANCE)

    # And it is the *closest* to one, which is what the selection turns on:
    # a second state nearer to unity would be chosen instead.
    distances = np.abs(1.0 - np.exp(planted.log_mu))
    assert int(np.argmin(distances)) == NORMAL_STATE
    assert distances[NORMAL_STATE] == 0.0


@pytest.mark.cnaster
@pytest.mark.critical
def test_the_planted_normal_state_is_too_rare_to_be_a_candidate(
    planted: CoreInferenceTruth,
) -> None:
    """Planting the state is not enough: it has to be common enough as well.

    `find_diploid_balanced_state` requires a candidate to occupy at least
    `min_prop_threshold = 0.1` of the genome, and the dev instance's normal
    state occupies **0.0858**. So on the planted truth the selection raises,
    with the same message #106 was opened for:

        ValueError: No candidate diploid balanced state found!

    The round trip reaches the end anyway because it fits **five** states
    against the ten planted (#90 is why), and a five-state fit concentrates
    enough mass on a balanced state to clear the threshold. That is the fit
    rescuing the fixture, not the fixture being right.

    The fix is realism rather than a constant: a genome is mostly copy
    neutral, and a chain that visits ten states uniformly puts each at about
    a tenth -- exactly the threshold, so which side it lands on is a
    coincidence of the seed. #106 carries the decision.
    """
    from cnaster.integer_copy import find_diploid_balanced_state

    path = planted.states.reshape(-1)
    occupancy = np.bincount(path, minlength=planted.n_states) / path.size

    assert occupancy[NORMAL_STATE] == pytest.approx(0.0858, abs=5e-4)
    assert occupancy[NORMAL_STATE] < MIN_PROPORTION

    with pytest.raises(ValueError, match="No candidate diploid balanced state"):
        find_diploid_balanced_state(
            planted.log_mu,
            planted.p_binom,
            path,
            min_prop_threshold=MIN_PROPORTION,
            EPS_BAF=EPS_BAF,
        )


@pytest.mark.cnaster
def test_a_mostly_diploid_genome_is_the_candidate_it_wants(
    planted: CoreInferenceTruth,
) -> None:
    """The same selection, on a path where the normal state is common.

    What the fixture would have to plant. Half the genome copy neutral is
    conservative for a tumour sample and clears the threshold by a factor of
    five, and then the selection picks the planted state rather than raising.
    """
    from cnaster.integer_copy import find_diploid_balanced_state

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


@pytest.mark.analytic
def test_the_planted_scale_is_cnasters_and_not_the_papers() -> None:
    """A stated difference: `mu = 1` means two things, and only one is tested.

    The paper defines de-biased rates `mu_bar_k` under the constraint
    `sum_g lambda_g mu_bar_k = 1` (`emission.tex:16`), so a state at one is
    one *on the exposure-weighted scale*, and the normalization is what makes
    the number meaningful.

    `cnaster` computes that normalization -- `compute_logmu_shifts` -- and
    never applies it (#5); the emission logs `logmu_shifts are not currently
    supported` on every call. Then `find_diploid_balanced_state` tests
    `|1 - exp(log_mu)|` against one anyway, which is the paper's criterion on
    an unnormalized quantity.

    So the fixture plants `mu = 1` on `cnaster`'s scale, because `cnaster` is
    the subject. Under the paper's constraint the same state would sit at a
    different value unless the exposure happened to be normalized, and this
    asserts that it does not: the planted exposure's weighted sum is not one,
    so the two scales are genuinely different here rather than coincidentally
    equal.
    """
    truth = dev_instance()

    weights = truth.base_nb_mean.sum(axis=1)
    weights = weights / weights.sum()
    normalized = float(np.sum(weights * np.exp(truth.log_mu[truth.states[0]])))

    assert normalized != pytest.approx(1.0, abs=0.05), (
        f"the exposure-weighted rate is {normalized:.3f}; if it were one the "
        f"two scales would agree and this difference would not need stating"
    )
