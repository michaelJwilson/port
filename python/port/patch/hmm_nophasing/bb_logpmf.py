"""`cnaster.hmm_nophasing._bb_logpmf_1d` and `_dense_bb_logpmf`, exact at large `tau` (#561).

Replaces upstream's `lgamma` sum, which cancels `tau log tau`-sized terms, with
`port.patch.emission.bb_log_pmf` (T- #776). `cnaster`'s conventions kept: `a`, `b`
floored at `DISPERSION_FLOOR`; `k > n` or negative counts score 0; `tau = inf` is
the binomial (T- #617). Referee: `mpmath.loggamma` (`tests/test_bb_logpmf.py`).
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
    """Each bin's beta-binomial log pmf into `out`, returned; only the default `EPS` is accepted."""
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
