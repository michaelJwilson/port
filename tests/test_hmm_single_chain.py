"""A single NB chain scored by `cnaster` and `snakes_and_ladders` at fixed parameters (#14).

No spatial layer or factored state space, so the correspondence is exact.
"""

from typing import Any

import numpy as np
import pytest
import torch
from cnaster.hmm_nophasing import get_log_transmat

from tests.adapters import (
    BAF_CHANNEL,
    CnasterChainInputs,
    cnaster_emission,
    cnaster_total_log_likelihood,
    from_negative_binomial_chains,
    upstream_total_log_likelihood,
)
from tests.fixtures import (
    NegativeBinomialChains,
    circulant_transition,
    negative_binomial_chains,
)

# NB sums run in different orders, so agreement is to accumulated rounding
TOLERANCE = 1e-9


def upstream_emission(
    fixture: NegativeBinomialChains, inputs: CnasterChainInputs
) -> np.ndarray:
    """Return upstream's emission scores in `cnaster`'s `(n_states, n_obs)` layout."""
    density = fixture.family.log_density(
        torch.as_tensor(fixture.dataset.observations, dtype=torch.float64)
    )
    reshaped: np.ndarray = density.numpy().reshape(inputs.n_obs, inputs.n_states).T
    return reshaped


@pytest.mark.oracle
@pytest.mark.critical
@pytest.mark.parametrize("n_states", [1, 2, 3, 5])
@pytest.mark.parametrize("separation", [1.5, 2.5])
def test_emission_matches_upstream(n_states: int, separation: float) -> None:
    """`cnaster`'s emission equals upstream's, to `TOLERANCE`."""
    fixture = negative_binomial_chains(n_states=n_states, separation=separation)
    inputs = from_negative_binomial_chains(fixture)

    np.testing.assert_allclose(
        cnaster_emission(inputs),
        upstream_emission(fixture, inputs),
        rtol=0.0,
        atol=TOLERANCE,
    )


@pytest.mark.oracle
@pytest.mark.critical
@pytest.mark.parametrize(
    "chains",
    [
        *(
            {"n_sequences": n_sequences, "sequence_length": sequence_length}
            for sequence_length in (1, 60)
            for n_sequences in (1, 4)
        ),
        {"n_states": 4, "drift": 0.2},
        {"n_states": 4, "drift": 0.8},
    ],
    ids=lambda chains: "-".join(f"{k}={v}" for k, v in chains.items()),
)
def test_total_log_likelihood_matches_upstream(chains: dict[str, Any]) -> None:
    """Forward totals equal upstream's, to `TOLERANCE`, including at length one and under an asymmetric transition, catching a transpose."""
    fixture = negative_binomial_chains(**chains)
    inputs = from_negative_binomial_chains(fixture)

    assert cnaster_total_log_likelihood(inputs) == pytest.approx(
        upstream_total_log_likelihood(fixture), abs=TOLERANCE
    )


@pytest.mark.warning
def test_drift_outside_the_unit_interval_is_refused() -> None:
    """The fixture refuses a drift that is not a share."""
    with pytest.raises(ValueError, match="drift"):
        negative_binomial_chains(drift=0.0)


@pytest.mark.analytic
def test_beta_binomial_channel_is_inert_at_zero_depth() -> None:
    """At zero depth the beta-binomial channel does not change the emission."""
    fixture = negative_binomial_chains()
    inputs = from_negative_binomial_chains(fixture)
    assert np.all(inputs.total_bb_RD == 0.0)

    successes = inputs.single_X[:, BAF_CHANNEL, :]
    assert np.all(successes == 0.0)

    perturbed = from_negative_binomial_chains(fixture)
    perturbed.p_binom[:] = 0.9
    perturbed.taus[:] = 3.0

    np.testing.assert_array_equal(cnaster_emission(inputs), cnaster_emission(perturbed))


@pytest.mark.analytic
@pytest.mark.parametrize("exposure", [0.5, 1.0, 7.0])
def test_constant_exposure_is_absorbed(exposure: float) -> None:
    """A constant `base_nb_mean` leaves the emission unchanged, to `TOLERANCE`."""
    fixture = negative_binomial_chains()
    reference = cnaster_emission(from_negative_binomial_chains(fixture))
    scored = cnaster_emission(from_negative_binomial_chains(fixture, exposure=exposure))

    np.testing.assert_allclose(scored, reference, rtol=0.0, atol=TOLERANCE)


@pytest.mark.snapshot
@pytest.mark.parametrize("n_states", [1, 2, 5])
def test_transition_matches_cnaster_construction(n_states: int) -> None:
    """`circulant_transition` equals `cnaster`'s `get_log_transmat`, to 1e-15."""

    self_transition = 0.8
    mine = circulant_transition(n_states, self_transition)
    theirs = np.exp(get_log_transmat(n_states, self_transition))

    np.testing.assert_allclose(mine, theirs, rtol=0.0, atol=1e-15)


@pytest.mark.warning
def test_separation_below_one_is_refused() -> None:
    """The fixture refuses a separation that does not order the state means."""
    with pytest.raises(ValueError, match="separation"):
        negative_binomial_chains(separation=0.5)
