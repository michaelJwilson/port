"""`hmm_phased`'s coded emission raises on the `(n_states, 1)` parameter a fit returns (#269).

`bug` pins the defect; `patch` pins the replacement's broadcast over spots. Against
upstream where upstream runs, it is a `tests.correspondence` row.
"""

from __future__ import annotations

import numpy as np
import pytest
from cnaster.count_encoder import CountEncoder
from cnaster.hmm_phased import hmm_phased
from port.patch.hmm_phased import (
    compute_emission_probability_nb_betabinom_coded as replacement,
)

from tests.builders import coded_encoders


@pytest.mark.bug
def test_upstream_cannot_score_a_one_column_parameter_over_many_spots(
    cnaster_config: None,
) -> None:
    """`cnaster.hmm_phased` raises on many spots; written to fail when fixed."""

    nb, bb, parameters = coded_encoders(n_obs=20, n_spots=4, seed=0)

    with pytest.raises(IndexError, match="out of bounds"):
        hmm_phased.compute_emission_probability_nb_betabinom_coded(nb, bb, **parameters)


@pytest.mark.patch
def test_the_replacement_broadcasts_the_one_column_over_many_spots(
    cnaster_config: None,
) -> None:
    """Over many spots, matches scoring each spot alone (broadcast semantics)."""

    n_obs, n_spots = 20, 4
    nb, bb, parameters = coded_encoders(n_obs, n_spots, seed=2)

    rdr, baf = replacement(nb, bb, clone_stack=False, **parameters)

    assert rdr.shape[2] == n_spots
    assert baf.shape[2] == n_spots

    for spot in range(n_spots):
        one_nb = CountEncoder(
            nb.obs_count[:, spot : spot + 1], nb.total_count[:, spot : spot + 1]
        )
        one_bb = CountEncoder(
            bb.obs_count[:, spot : spot + 1], bb.total_count[:, spot : spot + 1]
        )

        alone_rdr, alone_baf = replacement(
            one_nb, one_bb, clone_stack=False, **parameters
        )

        np.testing.assert_array_equal(rdr[:, :, spot], alone_rdr[:, :, 0])
        np.testing.assert_array_equal(baf[:, :, spot], alone_baf[:, :, 0])


@pytest.mark.bug
def test_a_second_parameter_column_is_refused(cnaster_config: None) -> None:
    """The shape upstream indexes for, which the fit cannot produce (#267)."""

    nb, bb, parameters = coded_encoders(n_obs=10, n_spots=2, seed=3)
    parameters["log_mu"] = np.zeros((3, 2))

    with pytest.raises(ValueError, match="the fit produces no other"):
        replacement(nb, bb, **parameters)
