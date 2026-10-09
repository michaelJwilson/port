"""cnaster's dense and deduplicated emission implementations, against each other (#9).

Neither is designated the referee; a disagreement is a defect either way.
"""

import numpy as np
import pytest
from cnaster.count_encoder import CountEncoder
from cnaster.hmm_nophasing import hmm_nophasing

from tests.adapters import cnaster_emission, from_negative_binomial_chains
from tests.fixtures import negative_binomial_chains


@pytest.mark.backend
@pytest.mark.usefixtures("cnaster_config")
@pytest.mark.parametrize("n_states", [1, 3, 5])
@pytest.mark.parametrize("separation", [1.2, 2.5])
def test_deduplicated_emission_matches_dense(n_states: int, separation: float) -> None:
    """The `CountEncoder` path scores as the dense kernels do, bitwise."""

    fixture = negative_binomial_chains(n_states=n_states, separation=separation)
    inputs = from_negative_binomial_chains(fixture)

    dense = cnaster_emission(inputs)

    nb_encoder = CountEncoder(inputs.single_X[:, 0, :], inputs.base_nb_mean)
    bb_encoder = CountEncoder(inputs.single_X[:, 1, :], inputs.total_bb_RD)
    log_emit_rdr, log_emit_baf = (
        hmm_nophasing().compute_emission_probability_nb_betabinom_coded(
            nb_encoder,
            bb_encoder,
            inputs.log_mu,
            inputs.alphas,
            inputs.p_binom,
            inputs.taus,
            clone_stack=True,
        )
    )
    # NB `clone_stack` drops the spot axis; the adapter drops the dense path's.
    stacked = log_emit_rdr + log_emit_baf
    coded = stacked[:, :, 0] if stacked.ndim == 3 else stacked

    np.testing.assert_array_equal(dense, coded)


@pytest.mark.analytic
@pytest.mark.usefixtures("cnaster_config")
def test_encoder_round_trip_is_the_identity() -> None:
    """Decoding what was encoded returns the original, position by position."""

    fixture = negative_binomial_chains(n_states=3)
    inputs = from_negative_binomial_chains(fixture)
    encoder = CountEncoder(inputs.single_X[:, 0, :], inputs.base_nb_mean)

    counts = inputs.single_X[:, 0, 0]
    unique = encoder.get_unique_obs(0)
    decoded = encoder.decode_array(unique, 0)

    np.testing.assert_array_equal(np.asarray(decoded).reshape(-1), counts)


@pytest.mark.smoke
@pytest.mark.usefixtures("cnaster_config")
def test_deduplication_finds_fewer_uniques_than_positions() -> None:
    """Counts repeat, so the unique set is smaller than the observation set."""

    fixture = negative_binomial_chains(n_states=3, sequence_length=200)
    inputs = from_negative_binomial_chains(fixture)
    encoder = CountEncoder(inputs.single_X[:, 0, :], inputs.base_nb_mean)

    assert encoder.get_unique_obs(0).size < inputs.n_obs
