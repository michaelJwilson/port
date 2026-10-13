"""Replaces `cnaster.hmrf.compute_loglike_spot_assignment`, walked in stride order (#59).

Same signature, layout and output; only the loop order changes: `spot` is
innermost, so the C-contiguous `(n_states, n_obs, n_spots)` emission is swept
sequentially and accumulated as a vector. Referee: bitwise (same summation order).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit

if TYPE_CHECKING:  # pragma: no cover - `prange` is `range` to a type checker
    prange = range
else:
    from numba import prange

__all__ = ["compute_loglike_spot_assignment_strided"]


@njit(parallel=True, cache=True)
def compute_loglike_spot_assignment_strided(
    n_spots,
    num_valid_nb_spotwise,
    num_valid_bb_spotwise,
    single_tumor_prop,
    is_tumor_mixed,
    log_emission_rdr,
    log_emission_baf,
    pred,
    n_obs,
    n_clones,
    smooth_indices=None,
    smooth_indptr=None,
    non_zero_weight=True,
):
    """As `cnaster`'s, with the spot loop innermost; returns `(n_spots, n_clones)`.

    The relative-channel weight is carried unchanged (#58). `prange` is over
    clones: each writes its own column, so threads share no reduction.
    """
    loglike_spot_clone_assignment = np.zeros((n_spots, n_clones))
    rel_valid_emision_weight = np.ones(n_spots, dtype=np.float64)

    if non_zero_weight and smooth_indices is not None and smooth_indptr is not None:
        for i in prange(n_spots):
            start_idx, end_idx = smooth_indptr[i], smooth_indptr[i + 1]

            pooled_num_valid_nb_spotwise, pooled_num_valid_bb_spotwise = 0.0, 0.0

            for k in range(start_idx, end_idx):
                neighbor = smooth_indices[k]

                if is_tumor_mixed and np.isnan(single_tumor_prop[neighbor]):
                    continue

                pooled_num_valid_nb_spotwise += num_valid_nb_spotwise[neighbor]
                pooled_num_valid_bb_spotwise += num_valid_bb_spotwise[neighbor]

            if pooled_num_valid_nb_spotwise > 0 and pooled_num_valid_bb_spotwise > 0:
                rel_valid_emision_weight[i] = (
                    pooled_num_valid_bb_spotwise / pooled_num_valid_nb_spotwise
                )

    is_1d_pred = pred.ndim == 1

    for c in prange(n_clones):
        accumulated_rdr = np.zeros(n_spots)
        accumulated_baf = np.zeros(n_spots)

        for o in range(n_obs):
            copy_state = pred[c * n_obs + o] if is_1d_pred else pred[o, c]

            # NB contiguous axis: a vector accumulation, not a scalar reduction.
            for spot in range(n_spots):
                accumulated_rdr[spot] += log_emission_rdr[copy_state, o, spot]
                accumulated_baf[spot] += log_emission_baf[copy_state, o, spot]

        for spot in range(n_spots):
            loglike_spot_clone_assignment[spot, c] = (
                rel_valid_emision_weight[spot] * accumulated_rdr[spot]
                + accumulated_baf[spot]
            )

    return loglike_spot_clone_assignment
