"""cnaster's four live emission entry points, against each other and scipy (#205 step
1).

Three claims are `patch` (implementations agree); the switch term as the allele swap is
the `oracle`, against scipy.
"""

import numpy as np
import pytest
import scipy.stats

from tests.builders import EmissionInputs, cnaster_emission_pair, emission_inputs

SWAP_TOLERANCE = 1.0e-12
"""Switch term vs scipy tolerance; realized 6.0e-14 (reassociation only)."""


def _inputs(n_states: int, *, n_obs: int = 60, n_spots: int = 1) -> EmissionInputs:
    """Counts and parameters at `cnaster`'s shapes, with repeats to deduplicate."""
    return emission_inputs(n_states, n_obs, n_spots, seed=17, exposure=(20, 60), trials=(5, 40), share=0.45,
                           ranges=((-0.4, 0.4), (0.1, 0.6), (0.2, 0.83), (7.5, 30.0))).columns(n_spots)  # fmt: skip


def _phased_emission(
    inputs: EmissionInputs, *, clone_stack: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    return cnaster_emission_pair(inputs, phased=True, clone_stack=clone_stack)


_unphased_emission = cnaster_emission_pair


@pytest.mark.oracle
@pytest.mark.usefixtures("cnaster_config")
@pytest.mark.parametrize("n_states", [1, 3, 5])
def test_the_switch_term_is_the_allele_swap(n_states: int) -> None:
    """`_switch_betabinom_1d` scores the B allele as scipy scores `n - k`."""
    from cnaster.hmm_nophasing import _bb_logpmf_1d
    from cnaster.hmm_phased import _switch_betabinom_1d

    generator = np.random.default_rng(11)
    successes = generator.integers(0, 40, 12).astype(np.float64)
    totals = successes + generator.integers(1, 40, 12).astype(np.float64)

    p_binom = np.linspace(0.2, 0.83, n_states)
    taus = np.linspace(7.5, 30.0, n_states)

    unswitched = np.zeros((n_states, successes.size))
    for state in range(n_states):
        _bb_logpmf_1d(successes, totals, p_binom[state], taus[state], unswitched[state])

    switched = _switch_betabinom_1d(unswitched, successes, totals, p_binom, taus)

    expected = np.vstack(
        [
            scipy.stats.betabinom.logpmf(
                totals - successes,
                totals,
                p_binom[state] * taus[state],
                (1.0 - p_binom[state]) * taus[state],
            )
            for state in range(n_states)
        ]
    )

    realized = float(np.abs(switched - expected).max())
    assert realized < SWAP_TOLERANCE, (
        f"switch term departs from scipy by {realized:.3g}"
    )


@pytest.mark.patch
@pytest.mark.usefixtures("cnaster_config")
@pytest.mark.parametrize("n_states", [1, 3])
def test_the_phased_wrapper_is_the_phased_coded_path(n_states: int) -> None:
    """`hmm_phased`'s dense entry point equals the deduplicated one."""
    from cnaster.count_encoder import CountEncoder
    from cnaster.hmm_phased import hmm_phased

    inputs = _inputs(n_states)

    wrapper_rdr, wrapper_baf = _phased_emission(inputs)
    coded_rdr, coded_baf = hmm_phased.compute_emission_probability_nb_betabinom_coded(
        CountEncoder(inputs.single_X[:, 0, :], inputs.base_nb_mean),
        CountEncoder(inputs.single_X[:, 1, :], inputs.total_bb_RD),
        inputs.log_mu,
        inputs.alphas,
        inputs.p_binom,
        inputs.taus,
        clone_stack=False,
    )

    np.testing.assert_array_equal(wrapper_rdr, coded_rdr)
    np.testing.assert_array_equal(wrapper_baf, coded_baf)


@pytest.mark.patch
@pytest.mark.usefixtures("cnaster_config")
@pytest.mark.parametrize("n_states", [1, 3, 5])
def test_the_phased_read_depth_is_the_unphased_one_twice(n_states: int) -> None:
    """Both halves of the phased RDR equal the unphased RDR."""
    inputs = _inputs(n_states)

    unphased_rdr, _ = _unphased_emission(inputs)
    phased_rdr, _ = _phased_emission(inputs)

    assert phased_rdr.shape[0] == 2 * n_states

    np.testing.assert_array_equal(phased_rdr[:n_states], unphased_rdr)
    np.testing.assert_array_equal(phased_rdr[n_states:], unphased_rdr)


@pytest.mark.patch
@pytest.mark.usefixtures("cnaster_config")
@pytest.mark.parametrize("n_states", [1, 3, 5])
def test_the_phased_allele_channel_opens_with_the_unphased_one(n_states: int) -> None:
    """The first half of the phased BAF equals the unphased BAF."""
    inputs = _inputs(n_states)

    _, unphased_baf = _unphased_emission(inputs)
    _, phased_baf = _phased_emission(inputs)

    np.testing.assert_array_equal(phased_baf[:n_states], unphased_baf)

    # The switched half differs, so the channel is not empty.
    assert np.abs(unphased_baf).max() > 0.0, "the allele channel carries no data"
    assert not np.array_equal(phased_baf[n_states:], unphased_baf)


EVIDENCE_TOLERANCE = 1.0e-12
"""Phased vs unphased recursion tolerance; realized 3.6e-15 (reassociation only)."""


@pytest.mark.analytic
@pytest.mark.parametrize("switch", [0.0, 1.0e-12, 1.0e-8])
@pytest.mark.parametrize("n_states", [2, 3, 5])
def test_phasing_a_phase_free_emission_changes_no_evidence(
    n_states: int, switch: float
) -> None:
    """With switching forbidden and equal phase emissions, the phased lattice matches
    the unphased one.

    Parametrized over a zero switch and two small ones, since `-inf` and a tiny finite
    log differ in arithmetic (#205 step 3).
    """
    from cnaster.hmm_nophasing import hmm_nophasing
    from cnaster.hmm_phased import hmm_phased
    from scipy.special import logsumexp

    generator = np.random.default_rng(3)
    lengths = np.array([25, 15])
    n_obs = int(lengths.sum())

    emission = generator.normal(size=(n_states, n_obs, 1))
    paired = np.vstack([emission, emission])

    log_startprob = np.log(np.full(n_states, 1.0 / n_states))
    self_transition = 0.9
    off = (1.0 - self_transition) / (n_states - 1)
    log_transmat = np.log(
        np.full((n_states, n_states), off) + np.eye(n_states) * (self_transition - off)
    )

    sitewise = np.full(n_obs, np.log(switch) if switch > 0.0 else -np.inf)

    def evidence(alpha: np.ndarray) -> float:
        return float(sum(logsumexp(alpha[:, end - 1]) for end in np.cumsum(lengths)))

    unphased = evidence(
        hmm_nophasing.forward_lattice(
            lengths, log_transmat, log_startprob, emission, sitewise
        )
    )
    phased = evidence(
        hmm_phased.forward_lattice(
            lengths, log_transmat, log_startprob, paired, sitewise
        )
    )

    assert abs(phased - unphased) < EVIDENCE_TOLERANCE, (
        f"phased evidence {phased:.12f} against unphased {unphased:.12f}"
    )
