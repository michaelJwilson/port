"""`cnaster.hmrf.compute_loglike_spot_assignment`, walked in stride order.

Issue #59 item 1. A drop-in replacement with `cnaster`'s signature, its
layout and its output. The only change is the **order of the loops**.

`cnaster` runs:

    for spot in prange(n_spots):
        for c in range(n_clones):
            for o in range(n_obs):
                rdr += log_emission_rdr[copy_state, o, spot]

On a C-contiguous `(n_states, n_obs, n_spots)` emission the contiguous axis is
`spot`, and it is the one held fixed by the outer loop. Each inner step
strides `n_spots` floats -- 39 KB at 5,000 spots -- so every access misses.

This runs the same three loops with `spot` innermost, accumulating into a
vector rather than a scalar. Two things follow, and the second is why the
ratio is what it is:

*   The inner walk `log_emission_rdr[k, o, :]` is contiguous, and consecutive
    `o` rows are adjacent, so the pass is one sequential sweep of the array
    rather than `n_obs` strided probes.
*   `acc[s] += rdr[k, o, s]` is a **vector accumulation**, not a reduction to
    a scalar. `CLAUDE.md` asks for inner loops "contiguous, unaliased,
    without early exit or data-dependent reduction, so NumPy and the Rust
    compiler vectorize"; `cnaster`'s form is contiguous in neither sense and
    is a reduction, and a form that fixes only the contiguity gets about half
    the gain (`docs/measurements.md`).

**Referee: bitwise.** The additions per `(spot, clone)` run over `o` in the
same order as `cnaster`'s, so the same floats are summed in the same
sequence. `np.array_equal` is the bar and a tolerance would be hiding
something.

**It needs nothing from the producer**, which is what makes it a
simplification rather than a port-wide change. The algorithmic cut is issue
#59 item 2 -- not materializing the `(n_states, n_obs, n_spots)` array at
all.

Measured, and the forms tried and rejected: `docs/measurements.md`,
`port.patch.hmrf.field`.
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
    """As `cnaster`'s, with the spot loop innermost.

    Every argument, every shape and the returned `(n_spots, n_clones)` array
    are `cnaster`'s. The relative-channel weight is carried unchanged rather
    than fixed: it is a separate finding (#58), and changing two things at
    once would make the bitwise comparison meaningless.

    `prange` moves to the clone loop because each clone writes its own column
    and its own accumulators, so there is no reduction across threads. The
    spot loop cannot carry it -- it is the vectorized one.
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

            # NB the contiguous axis, walked in stride order: a vector
            #    accumulation rather than a reduction to a scalar.
            for spot in range(n_spots):
                accumulated_rdr[spot] += log_emission_rdr[copy_state, o, spot]
                accumulated_baf[spot] += log_emission_baf[copy_state, o, spot]

        for spot in range(n_spots):
            loglike_spot_clone_assignment[spot, c] = (
                rel_valid_emision_weight[spot] * accumulated_rdr[spot]
                + accumulated_baf[spot]
            )

    return loglike_spot_clone_assignment
