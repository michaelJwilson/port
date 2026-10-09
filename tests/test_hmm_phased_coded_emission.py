"""`hmm_phased`'s coded emission raises on the `(n_states, 1)` parameter a fit returns (#269).

`bug` pins the defect; `patch` pins the replacement against upstream where upstream
runs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from cnaster.count_encoder import CountEncoder
from cnaster.hmm_phased import hmm_phased
from port.patch.hmm_phased import (
    compute_emission_probability_nb_betabinom_coded as replacement,
)


def _encoders(n_obs: int, n_spots: int, seed: int) -> tuple[Any, Any, dict[str, Any]]:
    rng = np.random.default_rng(seed)

    counts = rng.poisson(60, size=(n_obs, n_spots)).astype(float)
    exposure = np.full((n_obs, n_spots), 60.0)
    alleles = rng.binomial(40, 0.4, size=(n_obs, n_spots)).astype(float)
    depth = np.full((n_obs, n_spots), 40.0)

    n_states = 3
    parameters = {
        "log_mu": rng.normal(0.0, 0.2, size=(n_states, 1)),
        "alphas": np.full((n_states, 1), 0.25),
        "p_binom": rng.uniform(0.2, 0.8, size=(n_states, 1)),
        "taus": np.full((n_states, 1), 30.0),
    }

    return CountEncoder(counts, exposure), CountEncoder(alleles, depth), parameters


@pytest.mark.bug
def test_upstream_cannot_score_a_one_column_parameter_over_many_spots(
    cnaster_config: None,
) -> None:
    """`cnaster.hmm_phased` raises on many spots; written to fail when fixed."""

    nb, bb, parameters = _encoders(n_obs=20, n_spots=4, seed=0)

    with pytest.raises(IndexError, match="out of bounds"):
        hmm_phased.compute_emission_probability_nb_betabinom_coded(nb, bb, **parameters)


@pytest.mark.patch
def test_the_replacement_matches_upstream_where_upstream_runs(
    cnaster_config: None,
) -> None:
    """Bitwise equal to upstream at one spot, the only width upstream runs."""

    nb, bb, parameters = _encoders(n_obs=25, n_spots=1, seed=1)

    theirs = hmm_phased.compute_emission_probability_nb_betabinom_coded(
        nb, bb, **parameters
    )
    ours = replacement(nb, bb, **parameters)

    for mine, upstream in zip(ours, theirs, strict=True):
        np.testing.assert_array_equal(np.asarray(mine), np.asarray(upstream))


@pytest.mark.patch
def test_the_replacement_broadcasts_the_one_column_over_many_spots(
    cnaster_config: None,
) -> None:
    """Over many spots, matches scoring each spot alone (broadcast semantics)."""

    n_obs, n_spots = 20, 4
    nb, bb, parameters = _encoders(n_obs, n_spots, seed=2)

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

    nb, bb, parameters = _encoders(n_obs=10, n_spots=2, seed=3)
    parameters["log_mu"] = np.zeros((3, 2))

    with pytest.raises(ValueError, match="the fit produces no other"):
        replacement(nb, bb, **parameters)
