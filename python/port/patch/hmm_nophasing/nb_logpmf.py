"""`cnaster.hmm_nophasing._nb_logpmf_1d`, in log space so a vanishing mean cannot score a count at probability 1 (#560).

**A row (#560).** `port.pipeline.LOG_SPACE_SWAPS` rebinds `_nb_logpmf_1d`
and, since `cnaster`'s compiled `_dense_nb_logpmf` binds it as a global at
compile time, `_dense_nb_logpmf` beside it, wherever `cnaster` binds them.
`port.patch.hmrf`'s field, the M-step gradient, `copy_likelihood` and
`jax_hmm` score the same arithmetic. Retire when `cnaster` lands the fix.

**The defect.** Upstream forms `p = 1 / (1 + alpha * lambda)` and calls
`nbinom_logpmf_numba(k, r, p)`, which returns `0.0` -- probability 1, for
any count -- when `p >= 1.0`. In float64 `p` rounds to exactly 1.0 once
`alpha * lambda` is below about 1.1e-16, so a state whose mean falls far
enough scores every row it holds at probability 1. Baum-Welch finds it: on
`dev_tree_1s_hard` r0 one fit drove a state to `log mu = -43.22`, gave it
7,632 of 7,688 rows, and reported -23,359 nats against the planted states'
-76,306.

**The fix.** `port.patch.emission`'s negative binomial, the one every site
scores (T- #776): `((T(r, y) + y log(lambda / (1 + q))) - r log1p(q))`,
`T = S(r, y) - lgamma(y + 1)` with `S` `sal`'s scaled rising factorial and
`q = lambda / r`, so no probability is formed and none rounds, and no
`lgamma(r)`-sized term is cancelled at `r` up to 1e10 (1.3e-5 nats at
`alpha = 1e-10` before). `S` is `sal`'s compiled kernel and `lgamma` its
`gammaln`, called from this one, so the row is
:func:`~port.patch.emission.nb_log_pmf`'s arithmetic to the rounding of
`numba`'s `log` against NumPy's. `lambda <= 0` still scores 0, upstream's convention for an
unobserved bin. `alpha` is floored in `a` as upstream floors it in `r`;
upstream leaves `p` unfloored, which is the same defect reached through
`alpha < 1.1e-16 / lambda`.

**Referee.** `scipy.stats.nbinom.logpmf` where scipy is exact
(`alpha * lambda >= 1e-4`), `mpmath` at 50 digits below it, and upstream's
kernel where its `p < 1`, each to 1e-9 relative
(`tests/test_patch_nb_logpmf.py`, `tests/test_log_space_sites.py`). Not
bitwise: `log1p` and `log` of a sum differ from `log` of a quotient in the
last place, so this is its own table, not a `SWAPS` row.

**Ratio.** Not an optimization: `S`'s two `gammaln` per score where upstream
took three `lgamma`, and its series where those would cancel. Not cached by
`numba`: `sal`'s `gammaln` pointer keeps a kernel that calls it out of the
cache, so the row compiles once per process.
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

# NB `sal`'s compiled `S` and its `gammaln`, bound as globals at compile time;
#    `_kernels()` sets `rising._gammaln` to the `scipy` pointer.
_scaled, _ = rising._kernels()
_gammaln = rising._gammaln
_SERIES_FROM = rising._SERIES_FROM
_SMALL_T = rising._SMALL_T
_PLAIN_ERROR = rising._PLAIN_ERROR
_LOG_PROMISE = rising._LOG_PROMISE
_TERM_FLOOR = rising._TERM_FLOOR


@njit(nogil=True, error_model="numpy")
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


@njit(nogil=True, parallel=True, error_model="numpy")
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
