"""The phased lattice, refereed against `snakes_and_ladders` at a constant phase kernel.

A position-varying kernel has no upstream form here. `cnaster`'s phased emission raises,
so scores are supplied as arrays and the defect is pinned separately.
"""

import numpy as np
import pytest
from cnaster.hmm_phased import hmm_phased, update_combined_transmat

from tests.adapters import (
    cnaster_phased_total_log_likelihood,
    from_phased_chains,
    upstream_phased_total_log_likelihood,
)
from tests.fixtures import phased_chains, phased_combined_transition

TOLERANCE = 1e-9

ASSEMBLIES = [
    pytest.param(False, id="kronecker"),
    pytest.param(True, id="conserved-copy-state-only"),
]


@pytest.mark.snapshot
@pytest.mark.parametrize("penalize", ASSEMBLIES)
@pytest.mark.parametrize("n_copy_states", [1, 2, 3, 4])
def test_combined_transition_matches_cnaster_construction(
    penalize: bool, n_copy_states: int
) -> None:
    """The fixture's transfer matrix equals `update_combined_transmat`'s."""

    fixture = phased_chains(
        n_copy_states=n_copy_states, penalize_phase_only_on_same_cnv=penalize
    )
    theirs = np.empty((fixture.n_paired_states, fixture.n_paired_states))
    update_combined_transmat(
        theirs,
        n_copy_states,
        np.log(fixture.base_transition),
        np.log(1.0 - fixture.switch),
        np.log(fixture.switch),
        penalize,
        np.log(0.5),
    )

    np.testing.assert_allclose(
        fixture.combined_transition, np.exp(theirs), rtol=0.0, atol=1e-15
    )


@pytest.mark.analytic
@pytest.mark.parametrize("penalize", ASSEMBLIES)
@pytest.mark.parametrize("switch", [0.01, 0.15, 0.5, 0.9])
@pytest.mark.parametrize("n_copy_states", [1, 2, 5])
def test_combined_transition_is_row_stochastic(
    penalize: bool, switch: float, n_copy_states: int
) -> None:
    """Both assemblies are row-stochastic at every kernel and size."""
    fixture = phased_chains(
        n_copy_states=n_copy_states, penalize_phase_only_on_same_cnv=penalize
    )
    combined = phased_combined_transition(
        fixture.base_transition, switch, penalize_phase_only_on_same_cnv=penalize
    )

    assert np.all(combined >= 0.0)
    np.testing.assert_allclose(combined.sum(axis=1), 1.0, rtol=0.0, atol=1e-14)


@pytest.mark.oracle
@pytest.mark.critical
@pytest.mark.parametrize("penalize", ASSEMBLIES)
@pytest.mark.parametrize("drift", [0.3, 0.5, 0.7])
@pytest.mark.parametrize("n_sequences", [1, 3])
@pytest.mark.parametrize("sequence_length", [1, 50])
def test_total_log_likelihood_matches_upstream(
    penalize: bool, drift: float, n_sequences: int, sequence_length: int
) -> None:
    """The phased total log-likelihood agrees with upstream at a constant kernel."""
    fixture = phased_chains(
        n_sequences=n_sequences,
        sequence_length=sequence_length,
        drift=drift,
        penalize_phase_only_on_same_cnv=penalize,
    )
    inputs = from_phased_chains(fixture)

    assert cnaster_phased_total_log_likelihood(inputs) == pytest.approx(
        upstream_phased_total_log_likelihood(fixture, inputs), abs=TOLERANCE
    )


@pytest.mark.smoke
def test_sitewise_kernel_changes_the_score() -> None:
    """The lattice reads the sitewise kernel: changing it changes the score."""
    fixture = phased_chains()
    scored = cnaster_phased_total_log_likelihood(from_phased_chains(fixture))
    perturbed = cnaster_phased_total_log_likelihood(
        from_phased_chains(fixture, switch=0.45)
    )

    assert abs(scored - perturbed) > 1.0


@pytest.mark.analytic
def test_zero_phase_switching_decouples_the_phases() -> None:
    """At a vanishing kernel the off-diagonal phase blocks vanish."""
    fixture = phased_chains(n_copy_states=3)
    combined = phased_combined_transition(
        fixture.base_transition, 1e-12, penalize_phase_only_on_same_cnv=False
    )
    n_copy = fixture.n_copy_states

    np.testing.assert_allclose(combined[:n_copy, n_copy:], 0.0, rtol=0.0, atol=1e-11)
    np.testing.assert_allclose(
        combined[:n_copy, :n_copy], fixture.base_transition, rtol=0.0, atol=1e-11
    )


@pytest.mark.smoke
def test_switch_outside_the_unit_interval_is_refused() -> None:
    """The assembly refuses a kernel that is not a probability."""
    fixture = phased_chains()
    with pytest.raises(ValueError, match="switch"):
        phased_combined_transition(
            fixture.base_transition, 1.0, penalize_phase_only_on_same_cnv=False
        )


@pytest.mark.bug
@pytest.mark.xfail(
    strict=True,
    reason=(
        "cnaster's phased emission is unreachable: "
        "hmm_phased.compute_emission_probability_nb_betabinom is a classmethod "
        "calling compute_emission_probability_nb_betabinom_coded, an instance "
        "method, through cls. See issue #9. Remove this test when it lands."
    ),
)
def test_phased_emission_is_reachable() -> None:
    """`cnaster`'s phased emission returns; strict xfail until `cnaster` fixes it."""

    fixture = phased_chains(n_copy_states=2)
    n_paired = fixture.n_paired_states
    n_obs = fixture.n_sequences * fixture.sequence_length

    single_X = np.zeros((n_obs, 2, 1))
    single_X[:, 0, 0] = fixture.dataset.observations.reshape(-1)

    hmm_phased.compute_emission_probability_nb_betabinom(
        single_X,
        np.ones((n_obs, 1)),
        np.zeros((n_paired, 1)),
        np.ones((n_paired, 1)),
        np.zeros((n_obs, 1)),
        np.full((n_paired, 1), 0.5),
        np.full((n_paired, 1), 100.0),
    )
