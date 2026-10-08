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
n_clones` fewer evaluations: a speedup only where `n_clones < n_states`
(`docs/measurements.md`, `port.patch.hmrf.fused_field`).

*The memory, which is the claim that does not depend on the ratio.* The two
emission channels are `2 * n_states * n_obs * n_spots * 8` bytes -- 1.68 GB
at 3,000 x 5,000, **16.8 GB at 30,000 x 5,000**. The fused form holds its
four `(n_obs, n_spots)` inputs, 4.8 GB at that size, and allocates an
`(n_spots, n_clones)` output.

At a size the two-step cannot allocate, the fused form completes
(`docs/measurements.md`, `port.patch.hmrf.fused_field`).

**Referee: bitwise.** The per-`(spot, clone)` additions run over `o` in the
same order as the two-step's, on the same values, so `np.array_equal` is the
bar. With `log_space=True` the per-bin kernels are the rows of
`port.pipeline.LOG_SPACE_SWAPS` (#560, #561), and the bar is the two-step
under that table, bitwise again.

**What is deliberately not changed.** `rel_valid_emision_weight` is carried
as `cnaster` computes it, for the reason item 1 gives: it is #58's finding,
and moving two things at once would make the bitwise comparison meaningless.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from cnaster.hmm_nophasing import _bb_logpmf_1d as cnaster_bb_logpmf_1d
from cnaster.hmm_nophasing import _nb_logpmf_1d as cnaster_nb_logpmf_1d
from numba import njit

from port.patch.emission import bb_log_pmf
from port.patch.hmm_nophasing.nb_logpmf import _nb_logpmf_1d as log_space_nb_logpmf_1d

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
# NB imported under other names, so `LOG_SPACE_SWAPS` does not rebind them
#    here: a compiled kernel resolves a global at its first compile, and one
#    first compiled under the table would be cached with the table's kernel.


def _log_space_baf(counts_bb, total_bb_RD, p_binom, taus, pred):
    """`(n_spots, n_clones)`: each clone's beta-binomial log pmf summed over the
    bins in order, `port.patch.emission.bb_log_pmf` (T- #776); `tau = inf`
    the binomial.

    Summed over `o` one bin at a time, as the compiled pass sums, so the
    field is bitwise the per-bin rows'.
    """
    n_obs, n_spots = counts_bb.shape
    p = np.asarray(p_binom, dtype=np.float64).reshape(-1)
    tau = np.asarray(taus, dtype=np.float64).reshape(-1)
    accumulated = np.zeros((n_spots, pred.shape[1]))
    for o in range(n_obs):
        states = pred[o]
        k, n = counts_bb[o][:, None], total_bb_RD[o][:, None]
        accumulated += bb_log_pmf(k, n, p[states][None, :], tau[states][None, :])
    return accumulated


def fused_spot_clone_field(
    counts_nb: np.ndarray,
    base_nb_mean: np.ndarray,
    counts_bb: np.ndarray,
    total_bb_RD: np.ndarray,
    log_mu: np.ndarray,
    alphas: np.ndarray,
    p_binom: np.ndarray,
    taus: np.ndarray,
    pred: np.ndarray,
    rel_valid_emision_weight: np.ndarray,
    out: np.ndarray | None = None,
    log_space: bool = False,
) -> np.ndarray:
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
    `sal.oxisal.external_field(..., field)` writes in place --
    and it is carried here for that correspondence rather than for the
    bytes: **the buffer is 400 KB at the declared scale**, against the 8 GB
    the two-step materialized, so a caller that reuses it across outer
    iterations saves an allocation and not a footprint. Every entry is
    written before it is read, so a reused buffer needs no clearing.

    `log_space` scores with `LOG_SPACE_SWAPS`' kernels (#560, #561) rather
    than `cnaster`'s; its beta-binomial is `sal`'s rising factorials, summed
    before the compiled pass (:func:`_log_space_baf`, T- #781).
    """
    baf = (
        _log_space_baf(counts_bb, total_bb_RD, p_binom, taus, pred)
        if log_space
        else np.empty((0, 0))
    )
    field: np.ndarray = _fused_kernel(
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
        out,
        log_space,
        baf,
    )
    return field


# NB not cached: the log-space row calls `sal`'s `gammaln` pointer, which
#    keeps a kernel that reaches it out of `numba`'s cache (T- #776).
@njit(nogil=True, parallel=True, error_model="numpy")
def _fused_kernel(
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
    out,
    log_space,
    baf,
):
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
            if log_space:
                log_space_nb_logpmf_1d(
                    counts_nb[o, :],
                    base_nb_mean[o, :],
                    np.exp(log_mu[copy_state]),
                    alphas[copy_state],
                    scratch,
                )
            else:
                cnaster_nb_logpmf_1d(
                    counts_nb[o, :],
                    base_nb_mean[o, :],
                    np.exp(log_mu[copy_state]),
                    alphas[copy_state],
                    scratch,
                )
            accumulated_rdr += scratch

            if not log_space:
                cnaster_bb_logpmf_1d(
                    counts_bb[o, :],
                    total_bb_RD[o, :],
                    p_binom[copy_state],
                    taus[copy_state],
                    scratch,
                )
                accumulated_baf += scratch

        if log_space:
            accumulated_baf[:] = baf[:, c]

        for spot in range(n_spots):
            field[spot, c] = (
                rel_valid_emision_weight[spot] * accumulated_rdr[spot]
                + accumulated_baf[spot]
            )

    return field
