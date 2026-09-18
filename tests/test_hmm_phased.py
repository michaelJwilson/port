"""The phased lattice, refereed against `snakes_and_ladders`.

`cnaster` factors a state into a copy state and a phase and assembles its
transfer matrix per position from a `K x K` base and a two-element kernel.
Upstream's recursion takes a single transition, so the correspondence holds
**only where the kernel is constant along the chain** — then one matrix
stands for the whole of it. A kernel that varies by position has no upstream
form until the structured transfer matrix lands, and every measurement here
is taken in the constant regime, which `CLAUDE.md` requires each to state.

What this rung does not cover: `cnaster`'s phased *emission*, which raises
before it returns. The lattice takes its scores as an array, so supplying
them directly is what separates the transfer matrix from the defect below
it, and the defect is pinned by its own test rather than worked around
silently.
"""

import numpy as np
import pytest
import torch
from snakes_and_ladders.opt.hmm import forward_log_likelihood_from_density

from tests.adapters import (
    CnasterPhasedInputs,
    cnaster_phased_total_log_likelihood,
    from_phased_chains,
)
from tests.fixtures import PhasedChains, phased_chains, phased_combined_transition

TOLERANCE = 1e-9

ASSEMBLIES = [
    pytest.param(False, id="kronecker"),
    pytest.param(True, id="conserved-copy-state-only"),
]


def upstream_phased_total_log_likelihood(
    fixture: PhasedChains, inputs: CnasterPhasedInputs
) -> float:
    """The same total from upstream, at the assembled constant transition.

    The paired start is the copy-state initial halved across the phases,
    which is what `hmm_phased.forward_lattice` builds for itself.
    """
    n_sequences = inputs.lengths.size
    sequence_length = int(inputs.lengths[0])
    density = torch.as_tensor(
        inputs.log_emission[:, :, 0]
        .T.reshape(n_sequences, sequence_length, fixture.n_paired_states)
        .copy()
    )
    paired_initial = 0.5 * np.concatenate([fixture.initial, fixture.initial])

    return float(
        forward_log_likelihood_from_density(
            density,
            torch.log(torch.as_tensor(paired_initial)),
            torch.log(torch.as_tensor(fixture.combined_transition)),
        )
    )


@pytest.mark.cnaster
@pytest.mark.subject
@pytest.mark.critical
@pytest.mark.parametrize("penalize", ASSEMBLIES)
@pytest.mark.parametrize("n_copy_states", [1, 2, 3, 4])
def test_combined_transition_matches_cnaster_construction(
    penalize: bool, n_copy_states: int
) -> None:
    """The fixture assembles the matrix `update_combined_transmat` assembles.

    Two constructions of one object, pinned so a later change to either is a
    failing test rather than a silent divergence in what is compared.
    """
    from cnaster.hmm_phased import update_combined_transmat

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


@pytest.mark.oracle
@pytest.mark.property
@pytest.mark.parametrize("penalize", ASSEMBLIES)
@pytest.mark.parametrize("switch", [0.01, 0.15, 0.5, 0.9])
@pytest.mark.parametrize("n_copy_states", [1, 2, 5])
def test_combined_transition_is_row_stochastic(
    penalize: bool, switch: float, n_copy_states: int
) -> None:
    """Both assemblies conserve mass, at every kernel and every size.

    The second is not obviously stochastic — it overwrites four entries of
    an already-normalized matrix — so it is asserted rather than assumed.
    """
    fixture = phased_chains(
        n_copy_states=n_copy_states, penalize_phase_only_on_same_cnv=penalize
    )
    combined = phased_combined_transition(
        fixture.base_transition, switch, penalize_phase_only_on_same_cnv=penalize
    )

    assert np.all(combined >= 0.0)
    np.testing.assert_allclose(combined.sum(axis=1), 1.0, rtol=0.0, atol=1e-14)


@pytest.mark.oracle
@pytest.mark.upstream_oracle
@pytest.mark.critical
@pytest.mark.parametrize("penalize", ASSEMBLIES)
@pytest.mark.parametrize("drift", [0.3, 0.5, 0.7])
@pytest.mark.parametrize("n_sequences", [1, 3])
@pytest.mark.parametrize("sequence_length", [1, 50])
def test_total_log_likelihood_matches_upstream(
    penalize: bool, drift: float, n_sequences: int, sequence_length: int
) -> None:
    """The phased recursion agrees with upstream at a constant kernel.

    Length one exercises the paired start alone, with no transfer applied;
    a single sequence removes the concatenation `lengths` restarts.
    """
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


@pytest.mark.infra
@pytest.mark.analytic
def test_sitewise_kernel_changes_the_score() -> None:
    """The lattice reads the kernel it is handed.

    Without this the agreement above would hold just as well against a
    lattice that ignored the sitewise argument entirely, since the fixture
    and the reference would then differ by nothing observable.
    """
    fixture = phased_chains()
    scored = cnaster_phased_total_log_likelihood(from_phased_chains(fixture))
    perturbed = cnaster_phased_total_log_likelihood(
        from_phased_chains(fixture, switch=0.45)
    )

    assert abs(scored - perturbed) > 1.0


@pytest.mark.oracle
@pytest.mark.property
def test_zero_phase_switching_decouples_the_phases() -> None:
    """At a vanishing kernel the two phases stop exchanging mass.

    An analytic property of the assembly rather than a comparison: the
    off-diagonal blocks carry the switch, so as it goes to zero they do too
    and the matrix becomes block diagonal.
    """
    fixture = phased_chains(n_copy_states=3)
    combined = phased_combined_transition(
        fixture.base_transition, 1e-12, penalize_phase_only_on_same_cnv=False
    )
    n_copy = fixture.n_copy_states

    np.testing.assert_allclose(combined[:n_copy, n_copy:], 0.0, rtol=0.0, atol=1e-11)
    np.testing.assert_allclose(
        combined[:n_copy, :n_copy], fixture.base_transition, rtol=0.0, atol=1e-11
    )


@pytest.mark.infra
@pytest.mark.analytic
def test_switch_outside_the_unit_interval_is_refused() -> None:
    """The assembly refuses a kernel that is not a probability."""
    fixture = phased_chains()
    with pytest.raises(ValueError, match="switch"):
        phased_combined_transition(
            fixture.base_transition, 1.0, penalize_phase_only_on_same_cnv=False
        )


@pytest.mark.infra
@pytest.mark.analytic
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
    """The phased emission returns scores rather than raising.

    Written as the test that should pass, marked strict so it fails loudly
    the day `cnaster` fixes the defect rather than sitting green and unread.
    """
    from cnaster.hmm_phased import hmm_phased

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
