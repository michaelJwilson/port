"""`cnaster`'s two emission implementations, against each other.

The optimizer minimizes the deduplicated form and `pipeline_baum_welch`
reports the likelihood of the dense one, so the number optimized and the
number reported come from different code. Nothing in `cnaster` asserts they
agree. This is the half of issue #9 that needs no upstream correspondence:
one implementation is the referee for the other, and a disagreement is a
defect whichever way it falls.

The referee is not designated here. #9 asks `cnaster` to name which of its
implementations is the oracle, as upstream names `Backend.PYTHON`; until it
does, this pins that they agree without saying which would be wrong.
"""

import numpy as np
import pytest

from tests.adapters import cnaster_emission, from_negative_binomial_chains
from tests.fixtures import negative_binomial_chains


@pytest.mark.infra
@pytest.mark.usefixtures("cnaster_config")
@pytest.mark.analytic
@pytest.mark.parametrize("n_states", [1, 3, 5])
@pytest.mark.parametrize("separation", [1.2, 2.5])
def test_deduplicated_emission_matches_dense(n_states: int, separation: float) -> None:
    """The `CountEncoder` path scores as the dense kernels do.

    Bitwise: the deduplication changes which observations are evaluated, not
    how, so a difference here is a defect rather than a reordering.
    """
    from cnaster.count_encoder import CountEncoder
    from cnaster.hmm_nophasing import hmm_nophasing

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
    # NB `clone_stack` concatenates the clones along the genomic axis and
    #    drops the spot axis, so the result is already `(n_states, n_obs)`;
    #    the dense path keeps the trailing axis and the adapter drops it.
    stacked = log_emit_rdr + log_emit_baf
    coded = stacked[:, :, 0] if stacked.ndim == 3 else stacked

    np.testing.assert_array_equal(dense, coded)


@pytest.mark.infra
@pytest.mark.usefixtures("cnaster_config")
@pytest.mark.analytic
def test_encoder_round_trip_is_the_identity() -> None:
    """Decoding what was encoded returns the original, position by position.

    The deduplication is only sound if the mapping back is exact; a test of
    the scores alone would miss a mapping that is wrong on a value the
    fixture happens not to contain.
    """
    from cnaster.count_encoder import CountEncoder

    fixture = negative_binomial_chains(n_states=3)
    inputs = from_negative_binomial_chains(fixture)
    encoder = CountEncoder(inputs.single_X[:, 0, :], inputs.base_nb_mean)

    counts = inputs.single_X[:, 0, 0]
    unique = encoder.get_unique_obs(0)
    decoded = encoder.decode_array(unique, 0)

    np.testing.assert_array_equal(np.asarray(decoded).reshape(-1), counts)


@pytest.mark.infra
@pytest.mark.usefixtures("cnaster_config")
@pytest.mark.analytic
def test_deduplication_finds_fewer_uniques_than_positions() -> None:
    """The encoder earns its place on this fixture.

    Counts repeat, so the unique set is smaller than the observation set.
    Where it is not, the deduplicated path is doing the dense path's work
    with an indirection on top, and the comparison above would still pass
    while measuring nothing — so the premise is asserted, not assumed.
    """
    from cnaster.count_encoder import CountEncoder

    fixture = negative_binomial_chains(n_states=3, sequence_length=200)
    inputs = from_negative_binomial_chains(fixture)
    encoder = CountEncoder(inputs.single_X[:, 0, :], inputs.base_nb_mean)

    assert encoder.get_unique_obs(0).size < inputs.n_obs
