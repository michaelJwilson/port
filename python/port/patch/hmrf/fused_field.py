"""The spot/clone field without the emission array, for `cnaster.hmrf` (#59 item 2).

Replaces `compute_emission_probability_nb_betabinom` then
`compute_loglike_spot_assignment` (`hmrf.py:245-280`) with one pass that scores
only each clone's decoded state and materializes no `(n_states, n_obs, n_spots)`
array. Referee: bitwise against the two-step (under `LOG_SPACE_SWAPS` when
`log_space`, #560, #561). `rel_valid_emision_weight` is kept as `cnaster` computes it (#58).
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

# NB `cnaster`'s own per-bin kernels, so the bitwise claim is about the field alone;
#    renamed so `LOG_SPACE_SWAPS` does not rebind them before the first compile.


def _log_space_baf(counts_bb, total_bb_RD, p_binom, taus, pred):
    """`(n_spots, n_clones)`: each clone's beta-binomial log pmf summed over bins in order (T- #776).

    Summed one bin at a time, as the compiled pass sums, so it is bitwise the per-bin rows'.
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

    Takes `(n_obs, n_spots)` counts and exposures per channel, per-state
    parameters, the decoded profiles `pred`, and the relative channel weight.
    `out` is an optional `(n_spots, n_clones)` buffer, fully overwritten.
    `log_space` scores with `LOG_SPACE_SWAPS`' kernels (#560, #561, T- #781).
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


# NB cached: the log-space row reaches `sal`'s `gammaln` by a registered
#    symbol, not a pointer, so `numba` can cache it (sal #1342).
@njit(nogil=True, parallel=True, cache=True, error_model="numpy")
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

    # NB a ternary would make `numba` unify `none` with an array rather than specialize.
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

            # NB only the state this clone decoded to: n_clones of n_states.
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
