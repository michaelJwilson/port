"""`cnaster.hmm_nophasing._nb_logpmf_1d`, in log space so a vanishing mean cannot score a count at probability 1 (#560).

**`port.patch.hmm_nophasing.nb_logpmf` (#560), moved in by T- #670 PR4.**
`cnamaste.hmm_nophasing` imports `_nb_logpmf_1d` and `_dense_nb_logpmf` in
place of `cnaster`'s, so every compiled caller reads them under `cnaster`'s
names.

**The defect.** Upstream forms `p = 1 / (1 + alpha * lambda)` and calls
`nbinom_logpmf_numba(k, r, p)`, which returns `0.0` -- probability 1, for
any count -- when `p >= 1.0`. In float64 `p` rounds to exactly 1.0 once
`alpha * lambda` is below about 1.1e-16, so a state whose mean falls far
enough scores every row it holds at probability 1. Baum-Welch finds it: on
`dev_tree_1s_hard` r0 one fit drove a state to `log mu = -43.22`, gave it
7,632 of 7,688 rows, and reported -23,359 nats against the planted states'
-76,306.

**The fix.** With `a = max(alpha, 1e-10) * lambda`: `log p = -log1p(a)` and
`log(1 - p) = log(a) - log1p(a)`, so no probability is formed and none
rounds. `lambda <= 0` still scores 0, upstream's convention for an
unobserved bin. `alpha` is floored in `a` as upstream floors it in `r`;
upstream leaves `p` unfloored, which is the same defect reached through
`alpha < 1.1e-16 / lambda`.

**Referee.** `port`'s kernel, bitwise; `mpmath` at 50 digits, to 1e-11
absolute; upstream's kernel where its `p < 1`, to 1e-9 relative
(`tests/test_cnamaste_numerics.py`). Not bitwise against upstream: `log1p`
and `log` of a sum differ from `log` of a quotient in the last place.

**Ratio.** Not an optimization: one `log1p` and one `log` per score, where
upstream took one `log` of `p` and one of `1 - p`.
"""

from __future__ import annotations

from math import lgamma, log, log1p
from typing import TYPE_CHECKING

import numpy as np
from numba import njit

if TYPE_CHECKING:  # pragma: no cover - `prange` is `range` to a type checker
    prange = range
else:
    from numba import prange

__all__ = ["_dense_nb_logpmf", "_nb_logpmf_1d"]


@njit(nogil=True, cache=True, error_model="numpy")
def _nb_logpmf_1d(
    obs: np.ndarray, exposure: np.ndarray, mu: float, alpha: float, out: np.ndarray
) -> None:
    alpha = max(alpha, 1.0e-10)
    r = 1.0 / alpha
    for i in range(len(obs)):
        k = obs[i]
        lambda_i = exposure[i] * mu
        if lambda_i <= 0.0:
            out[i] = 0.0
            continue
        a = alpha * lambda_i
        out[i] = (
            lgamma(k + r)
            - lgamma(r)
            - lgamma(k + 1.0)
            - r * log1p(a)
            + k * (log(a) - log1p(a))
        )


@njit(nogil=True, cache=True, parallel=True, error_model="numpy")
def _dense_nb_logpmf(
    X_nb: np.ndarray, base_nb_mean: np.ndarray, log_mu: np.ndarray, alphas: np.ndarray
) -> np.ndarray:
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
