"""Replaces `hmm_phased`'s coded emission, reading parameters by state (#269).

Upstream indexes the `(n_states, 1)` parameter's column by the data's spot,
raising `IndexError` for more than one spot. This reads `log_mu[i, 0]`
broadcast over spots, as `hmm_nophasing`'s dense kernels do (#267).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from cnaster.hmm_nophasing import _bb_logpmf_1d, _nb_logpmf_1d
from cnaster.hmm_phased import _switch_betabinom_1d
from cnaster.hmm_phased import hmm_phased as UPSTREAM

from port.patch._clone_paths import state_vector

__all__ = ["compute_emission_probability_nb_betabinom_coded", "hmm_phased"]


def compute_emission_probability_nb_betabinom_coded(
    nbEncoder: Any,
    bbEncoder: Any,
    log_mu: np.ndarray,
    alphas: np.ndarray,
    p_binom: np.ndarray,
    taus: np.ndarray,
    clone_stack: bool = True,
    scratch_rdr: Any = None,
    scratch_baf: Any = None,
    normal_log_lambda: Any = None,
    clone_lengths: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Upstream's, with the parameter read by state rather than by spot."""
    del normal_log_lambda, clone_lengths

    rates = np.exp(state_vector(log_mu))
    dispersions = state_vector(alphas)
    probabilities = state_vector(p_binom)
    concentrations = state_vector(taus)

    n_states = len(rates)
    n_spots = nbEncoder.n_spots

    if n_spots != bbEncoder.n_spots:  # invariant
        msg = "Encoders must have identical spot counts"
        raise AssertionError(msg)

    rdr_columns, baf_columns = [], []

    for spot in range(n_spots):
        nb_endog = nbEncoder.get_unique_obs(spot)
        nb_exposure = nbEncoder.get_unique_total(spot)

        bb_endog = bbEncoder.get_unique_obs(spot)
        bb_exposure = bbEncoder.get_unique_total(spot)

        rdr_uniq = (
            scratch_rdr[spot]
            if scratch_rdr is not None
            else np.zeros((n_states, len(nb_endog)))
        )
        baf_uniq = (
            scratch_baf[spot]
            if scratch_baf is not None
            else np.zeros((n_states, len(bb_endog)))
        )

        for state in range(n_states):
            _nb_logpmf_1d(
                nb_endog,
                nb_exposure,
                rates[state],
                dispersions[state],
                rdr_uniq[state, :],
            )
            _bb_logpmf_1d(
                bb_endog,
                bb_exposure,
                probabilities[state],
                concentrations[state],
                baf_uniq[state, :],
            )

        switched = _switch_betabinom_1d(
            baf_uniq, bb_endog, bb_exposure, probabilities, concentrations
        )

        rdr_columns.append(
            nbEncoder.decode_array(np.vstack((rdr_uniq, rdr_uniq)), spot)
        )
        baf_columns.append(
            bbEncoder.decode_array(np.vstack((baf_uniq, switched)), spot)
        )

    if clone_stack:
        return (
            np.concatenate(rdr_columns, axis=1),
            np.concatenate(baf_columns, axis=1),
        )

    return np.stack(rdr_columns, axis=2), np.stack(baf_columns, axis=2)


class hmm_phased(UPSTREAM):  # type: ignore[misc]
    """`cnaster.hmm_phased`, scoring the coded emission by state (#269, #517); equal on one spot."""

    compute_emission_probability_nb_betabinom_coded = staticmethod(
        compute_emission_probability_nb_betabinom_coded
    )
