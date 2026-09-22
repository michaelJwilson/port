"""The spot/clone field without the emission array, for `cnaster.hmrf`.

Issue #59 item 2, and the algorithmic cut item 1's profile pointed at.

`cnaster` builds the field in two steps at `hmrf.py:245-280`:
`compute_emission_probability_nb_betabinom` materializes
`(n_states, n_obs, n_spots)` per channel, then
`compute_loglike_spot_assignment` reduces it over each clone's decoded
profile. This does both in one pass and materializes nothing.

**Two consequences, and only one of them is a speedup.**

*The flops.* The two-step scores every one of `n_states` states at every
`(bin, spot)`; the field then reads `pred[o, c]`, so at most `n_clones` of
them are ever used. Fusing scores only what is read, a factor `n_states /
n_clones` fewer evaluations. Measured against item 1's reordered field:

| `n_obs` | `n_spots` | states | clones | two-step | fused | ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3,000 | 5,000 | 7 | 4 | 16,046 ms | 7,698 ms | 2.08 |
| 3,000 | 5,000 | 7 | 7 | 16,102 ms | 14,043 ms | 1.15 |
| 10,000 | 2,500 | 7 | 4 | 26,735 ms | 14,449 ms | 1.85 |

`CLAUDE.md` puts a speedup claim at 2x, and this clears it **only where
`n_clones < n_states`**. At `n_clones == n_states` there is no flop to save
and the 1.15x is the materialization alone. Stated rather than averaged:
the ratio is `n_states / n_clones` and a reader can compute their own.

*The memory, which is the claim that does not depend on the ratio.* The two
emission channels are `2 * n_states * n_obs * n_spots * 8` bytes -- 1.68 GB
at 3,000 x 5,000, **16.8 GB at 30,000 x 5,000**. The fused form holds its
four `(n_obs, n_spots)` inputs, 4.8 GB at that size, and allocates an
`(n_spots, n_clones)` output.

Measured at `n_states = 7`, `n_obs = 30,000`, `n_spots = 5,000` on a machine
with 13 GB free: the fused form completed in **67.9 s**; the two-step
allocated its first 8.4 GB channel and the process was **killed by the
kernel** on the second. That is a size the two-step cannot run and this can,
which is a capability rather than a ratio -- and the kill is worth naming,
because an OOM death reads as infrastructure breaking rather than as a stated
limit.

**Referee: bitwise.** The per-`(spot, clone)` additions run over `o` in the
same order as the two-step's, on the same values, so `np.array_equal` is the
bar.

**What is deliberately not changed.** `rel_valid_emision_weight` is carried
as `cnaster` computes it, for the reason item 1 gives: it is #58's finding,
and moving two things at once would make the bitwise comparison meaningless.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from cnaster.hmm_nophasing import _bb_logpmf_1d, _nb_logpmf_1d
from numba import njit

if TYPE_CHECKING:  # pragma: no cover - `prange` is `range` to a type checker
    prange = range
else:
    from numba import prange

__all__ = ["fused_spot_clone_field"]

# NB the per-bin kernels are `cnaster`'s own, imported rather than restated:
#    a patch that reimplemented them would be comparing two implementations
#    of the emission as well as two of the field, and the bitwise claim below
#    would then be about the wrong thing. Inside `cnaster` this is a local
#    import from the same package.


@njit(nogil=True, cache=True, parallel=True, error_model="numpy")
def fused_spot_clone_field(
    counts_nb,
    base_nb_mean,
    counts_bb,
    total_bb_RD,
    log_mu,
    alphas,
    p_binom,
    taus,
    pred,
    rel_valid_emision_weight,
    out=None,
):
    """The `(n_spots, n_clones)` field, without an emission array.

    Replaces the pair at `cnaster.hmrf`'s call site rather than either
    function alone, so it takes what the emission was built from:
    `(n_obs, n_spots)` counts and exposures per channel, the per-state
    parameters, the decoded profiles, and the relative channel weight
    `compute_loglike_spot_assignment` would have applied.

    `prange` runs over clones: each writes its own column and its own
    accumulators, so nothing reduces across threads. The spot loop is the
    vectorized one, as in item 1.

    `out` is an `(n_spots, n_clones)` buffer to write into, or `None` to
    allocate one. This is upstream's shape --
    `oxi_snakes_and_ladders.external_field(..., field)` writes in place --
    and it is carried here for that correspondence rather than for the
    bytes: **the buffer is 400 KB at the declared scale**, against the 8 GB
    the two-step materialized, so a caller that reuses it across outer
    iterations saves an allocation and not a footprint. Every entry is
    written before it is read, so a reused buffer needs no clearing.
    """
    n_obs, n_spots = counts_nb.shape
    n_clones = pred.shape[1]

    # NB SIM108's ternary would ask `numba` to unify `none` with an array
    #    rather than specialize on which was passed, which is the whole
    #    mechanism of the optional buffer.
    if out is None:  # noqa: SIM108
        field = np.zeros((n_spots, n_clones))
    else:
        field = out

    for c in prange(n_clones):
        accumulated_rdr = np.zeros(n_spots)
        accumulated_baf = np.zeros(n_spots)
        scratch = np.zeros(n_spots)

        for o in range(n_obs):
            copy_state = pred[o, c]

            # NB one row per bin, contiguous, and only the state this clone
            #    decoded to -- which is the cut: n_clones of n_states.
            _nb_logpmf_1d(
                counts_nb[o, :],
                base_nb_mean[o, :],
                np.exp(log_mu[copy_state]),
                alphas[copy_state],
                scratch,
            )
            accumulated_rdr += scratch

            _bb_logpmf_1d(
                counts_bb[o, :],
                total_bb_RD[o, :],
                p_binom[copy_state],
                taus[copy_state],
                scratch,
            )
            accumulated_baf += scratch

        for spot in range(n_spots):
            field[spot, c] = (
                rel_valid_emision_weight[spot] * accumulated_rdr[spot]
                + accumulated_baf[spot]
            )

    return field
