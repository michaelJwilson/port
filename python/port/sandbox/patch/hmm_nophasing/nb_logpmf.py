"""`cnaster.hmm_nophasing._nb_logpmf_1d`, in log space so a vanishing mean cannot score a count at probability 1 (#560).

Ticket: #560 -- cnaster's `nbinom_logpmf_numba` scores any count at
  probability 1 once `p` rounds to 1, and Baum-Welch reaches it.
Measurement: scipy to 1e-9 where scipy is exact (`alpha * lambda >= 1e-4`);
  cnaster to 1e-9 where its `p < 1`; dev_tree_1s_hard r0's degenerate fit,
  -23,359 nats unpatched, refits at -75,505 nats and 1.3% missed.
Exit: a swap row once port installs it by default; retire when cnaster
  lands the fix.

**Replaces** `_nb_logpmf_1d(obs, exposure, mu, alpha, out)` and, since
`cnaster`'s compiled `_dense_nb_logpmf` binds it as a global at compile time,
`_dense_nb_logpmf(X_nb, base_nb_mean, log_mu, alphas)` beside it.

**The defect.** Upstream forms `p = 1 / (1 + alpha * lambda)` and calls
`nbinom_logpmf_numba(k, r, p)`, which returns `0.0` -- probability 1, for
any count -- when `p >= 1.0`. In float64 `p` rounds to exactly 1.0 once
`alpha * lambda` is below about 1.1e-16, so a state whose mean falls far
enough scores every row it holds at probability 1. Baum-Welch finds it: on
`dev_tree_1s_hard` r0 one fit drove a state to `log mu = -43.22`, gave it
7,632 of 7,688 rows, and reported -23,359 nats against the planted states'
-76,306 (`port.sandbox.known_copy`).

**The fix.** With `a = alpha * lambda`: `log p = -log1p(a)` and
`log(1 - p) = log(a) - log1p(a)`, so no probability is formed and none
rounds. `lambda <= 0` still scores 0, upstream's convention for an
unobserved bin. `-lgamma(k + 1)` is kept, as upstream's
`parameter_terms_only=True` keeps it: the full log pmf.

**Referee.** `scipy.stats.nbinom.logpmf`, and upstream's
kernel where it is defined (`a >= 1e-8`), to 1e-9 relative
(`tests/test_patch_nb_logpmf.py`). Not bitwise: `log1p` and `log` of a sum
differ from `log` of a quotient in the last place, so this is installed by
:func:`patched` around a fit, not a `SWAPS` row.

**Ratio.** Not an optimization: one `log1p` and one `log` per score, where
upstream took one `log` of `p` and one of `1 - p`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from math import lgamma, log, log1p
from typing import TYPE_CHECKING

import numpy as np
from numba import njit

if TYPE_CHECKING:  # pragma: no cover - `prange` is `range` to a type checker
    prange = range
else:
    from numba import prange

__all__ = ["_dense_nb_logpmf", "_nb_logpmf_1d", "patched"]


@njit(nogil=True, cache=True, error_model="numpy")
def _nb_logpmf_1d(obs, exposure, mu, alpha, out):
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


@contextmanager
def patched() -> Iterator[None]:
    """`cnaster.hmm_nophasing`'s two NB kernels replaced by these, restored on the way out."""
    import cnaster.hmm_nophasing as upstream

    names = ("_nb_logpmf_1d", "_dense_nb_logpmf")
    originals = {name: getattr(upstream, name) for name in names}
    try:
        upstream._nb_logpmf_1d = _nb_logpmf_1d
        upstream._dense_nb_logpmf = _dense_nb_logpmf
        yield
    finally:
        for name, original in originals.items():
            setattr(upstream, name, original)
