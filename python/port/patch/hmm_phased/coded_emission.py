"""`hmm_phased`'s coded emission, reading the parameter column it has (#269).

**Upstream cannot score the shape the fit returns.** Its body opens:

    n_states, n_spots = log_mu.shape   # the parameter's column count
    n_spots = nbEncoder.n_spots        # discarded, replaced by the data's
    for s in range(n_spots):           # bound from the data
        ... log_mu[i, s] ...           # index into the parameter

Line two throws away what line one just bound. The loop then runs over the
*data's* spot axis and indexes the *parameter's* column axis with it, so any
instance with more than one spot raises::

    IndexError: index 1 is out of bounds for axis 1 with size 1

Every fit returns `(n_states, 1)`, so this fires wherever the emission is
scored against pooled spots -- `hmrf.py:248`, the clone-assignment step.
`run_core_inference` defaults to `hmm_phased`; the shipped script passes
`hmm_nophasing` at all four of its call sites, which is the only reason it
is not a production failure.

`hmm_nophasing`'s dense kernels take `log_mu[i, 0]` and broadcast over the
spots. That is the reading this takes, because it is the one the parameter's
shape admits: the column axis and the spot axis are different things, and
#267 is the audit of what follows from confusing them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from cnaster.hmm_nophasing import _bb_logpmf_1d, _nb_logpmf_1d
from cnaster.hmm_phased import _switch_betabinom_1d

from port.patch.plotting.clone_paths import state_vector

__all__ = ["compute_emission_probability_nb_betabinom_coded"]


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
    num_segments_clones: Any = None,
    copy_states: Any = None,
    clone_lengths: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Upstream's, with the parameter read by state rather than by spot."""
    del normal_log_lambda, num_segments_clones, copy_states, clone_lengths

    rates = np.exp(state_vector(log_mu))
    dispersions = state_vector(alphas)
    probabilities = state_vector(p_binom)
    concentrations = state_vector(taus)

    n_states = len(rates)
    n_spots = nbEncoder.n_spots

    assert n_spots == bbEncoder.n_spots, "Encoders must have identical spot counts"

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
