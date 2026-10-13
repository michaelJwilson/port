"""Replaces `cnaster.hmm_nophasing._nb_logpmf_1d` (and `_dense_nb_logpmf`) in log space (#560).

Upstream's `p = 1 / (1 + alpha lambda)` rounds to 1 for a vanishing mean and
scores any count at probability 1. This scores `port.patch.emission`'s negative
binomial (T- #776) without forming `p`; `lambda <= 0` still scores 0. Agrees to
1e-9 relative, not bitwise, so installed by `LOG_SPACE_SWAPS`, not `SWAPS`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit
from sal.emissions import rising

from port.patch.emission import DISPERSION_FLOOR, nb_complete

if TYPE_CHECKING:  # pragma: no cover - `prange` is `range` to a type checker
    prange = range
else:
    from numba import prange

__all__ = ["_dense_nb_logpmf", "_nb_logpmf_1d"]

# NB `sal`'s compiled `S` and `gammaln`, bound as globals at compile time (sal #1342);
#    no public scalar compiled form, so the private names stay.
_scaled, _, _ = rising._kernels()
_gammaln = rising._gammaln
_SERIES_FROM = rising._SERIES_FROM
_SMALL_T = rising._SMALL_T
_PLAIN_ERROR = rising._PLAIN_ERROR
_LOG_PROMISE = rising._LOG_PROMISE
_TERM_FLOOR = rising._TERM_FLOOR


@njit(nogil=True, cache=True, error_model="numpy")
def _nb_logpmf_1d(obs, exposure, mu, alpha, out):
    r = np.inf if alpha <= 0.0 else 1.0 / max(alpha, DISPERSION_FLOOR)
    n = len(obs)
    shape = np.full(n, r)
    counts = np.empty(n)
    for i in range(n):
        counts[i] = obs[i]
    scaled = np.empty(n)
    _scaled(
        shape,
        counts,
        scaled,
        _SERIES_FROM,
        _SMALL_T,
        _PLAIN_ERROR,
        _LOG_PROMISE,
        _TERM_FLOOR,
        True,
    )
    for i in range(n):
        out[i] = nb_complete(
            scaled[i] - _gammaln(counts[i] + 1.0), counts[i], exposure[i] * mu, r
        )


@njit(nogil=True, parallel=True, cache=True, error_model="numpy")
def _dense_nb_logpmf(X_nb, base_nb_mean, log_mu, alphas):
    n_states = log_mu.shape[0]
    n_obs, n_spots = X_nb.shape
    out = np.zeros((n_states, n_obs, n_spots), dtype=np.float64)
    for i in prange(n_states):
        mu_val = np.exp(log_mu[i, 0])
        alpha_val = alphas[i, 0]
        for s in range(n_spots):
            _nb_logpmf_1d(
                X_nb[:, s], base_nb_mean[:, s], mu_val, alpha_val, out[i, :, s]
            )
    return out
