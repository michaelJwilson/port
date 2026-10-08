"""`cnaster.hmm_nophasing._bb_logpmf_1d`, exact at a large concentration (#561).

**A row (#561).** `port.pipeline.LOG_SPACE_SWAPS` rebinds `_bb_logpmf_1d` and
`_dense_bb_logpmf` to these wherever `cnaster` binds them; `cnaster` calls
both from Python, never from compiled code. Each scores
`port.patch.emission.bb_log_pmf`, the one beta-binomial every site scores
(T- #776): `sal`'s scaled rising factorials, so no `lgamma(tau)`-sized term
is formed and cancelled. Retire when `cnaster` lands the fix.

**The defect.** Upstream evaluates
`lgamma(k + a) + lgamma(n - k + b) - lgamma(n + a + b) - (lgamma(a) +
lgamma(b) - lgamma(a + b))` with `a = p tau`, `b = (1 - p) tau`. Each
`lgamma` is near `tau log tau`, so the sum loses `eps tau log tau`: 1e-10
nats at `tau = 1e5`, 3e-9 at 1e6, 5e-3 at 1e12, and at 1e16 the pmf over
`n = 100` sums to `e^132`.

**Referee.** `mpmath.loggamma` at 50 digits, to 1e-11 absolute for `tau`
from 10 to 1e16 (`tests/test_bb_logpmf.py`); `cnaster`'s kernel to 1e-9
where its own loss is below that. `cnaster`'s conventions are kept: `a` and
`b` floored at `DISPERSION_FLOOR`, and `k > n` or a negative count scores
0; `tau = inf` is the binomial (T- #617).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from port.patch.emission import DISPERSION_FLOOR, bb_log_pmf

__all__ = ["DISPERSION_FLOOR", "_bb_logpmf_1d", "_dense_bb_logpmf"]


def _bb_logpmf_1d(
    obs: Any,
    total: Any,
    p_binom: float,
    tau: float,
    out: np.ndarray,
    EPS: float = DISPERSION_FLOOR,
) -> np.ndarray:
    """`cnaster`'s row, its signature: each bin's beta-binomial log pmf into `out`, and returned.

    `EPS` is `cnaster`'s, and only its default is supported: the floor is
    `DISPERSION_FLOOR` on every site.
    """
    if EPS != DISPERSION_FLOOR:
        msg = f"_bb_logpmf_1d: EPS {EPS:g} is not the floor every site reads ({DISPERSION_FLOOR:g})"
        raise ValueError(msg)
    out[:] = bb_log_pmf(obs, total, p_binom, tau)
    return out


def _dense_bb_logpmf(
    X_bb: np.ndarray,
    total_bb_RD: np.ndarray,
    p_binom: np.ndarray,
    taus: np.ndarray,
    EPS: float = DISPERSION_FLOOR,
) -> np.ndarray:
    """`(n_states, n_obs, n_spots)`: :func:`_bb_logpmf_1d` for every state and spot."""
    n_states = p_binom.shape[0]
    out = np.zeros((n_states, *X_bb.shape), dtype=np.float64)
    for i in range(n_states):
        _bb_logpmf_1d(X_bb, total_bb_RD, p_binom[i, 0], taus[i, 0], out[i], EPS)
    return out
