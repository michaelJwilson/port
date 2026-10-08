"""`cnaster.hmm_nophasing._bb_logpmf_1d`, exact at a large concentration (#561).

**A row (#561).** `port.pipeline.LOG_SPACE_SWAPS` rebinds `_bb_logpmf_1d` and
`_dense_bb_logpmf` to these wherever `cnaster` binds them; `cnaster` calls
both from Python, never from compiled code. `port.patch.hmrf`'s field, the
copy likelihood (:func:`rises_on_distinct`) and the M-step gradient read the
same rising factorials. Every one is `sal`'s `log_rising` (`digamma_rising`
for the gradient): `sal` owns the arithmetic, `port` its arrangement into the
pmf (T- #781). Retire when `cnaster` lands the fix.

**The defect.** Upstream evaluates
`lgamma(k + a) + lgamma(n - k + b) - lgamma(n + a + b) - (lgamma(a) +
lgamma(b) - lgamma(a + b))` with `a = p tau`, `b = (1 - p) tau`. Each
`lgamma` is near `tau log tau`, so the sum loses `eps tau log tau`: 1e-10
nats at `tau = 1e5`, 3e-9 at 1e6, 5e-3 at 1e12, and at 1e16 the pmf over
`n = 100` sums to `e^132`.

**The fix.** Each pair is a rising factorial,
`R(x, m) = lgamma(x + m) - lgamma(x)`, so the log pmf is
`log C(n, k) + R(a, k) + R(b, n - k) - R(a + b, n)`, summed in that order.
`R` is `sal.emissions.rising.log_rising`: the plain `lgamma` difference
wherever its bound meets 1e-14, Stirling's series differenced term by term
elsewhere, within 1e-14 of `mpmath` over `max(|R|, 1)`, and chosen per
element, so a table of `R` and a direct evaluation agree bit for bit.
`log C(n, k)` is `port`'s: three `math.lgamma`, as `cnaster`'s kernel and
the field's tables form it.

**Referee.** `mpmath.loggamma` at 50 digits, to 1e-11 absolute for `tau`
from 10 to 1e16 (`tests/test_bb_logpmf.py`); `cnaster`'s kernel to 1e-9
where its own loss is below that. `cnaster`'s conventions are kept: `a` and
`b` floored at `EPS`, and `k > n` or a negative count scores 0.
"""

from __future__ import annotations

from math import lgamma
from typing import Any

import numpy as np
from numba import njit
from numpy.typing import ArrayLike
from sal.emissions.rising import log_rising

__all__ = [
    "DISPERSION_FLOOR",
    "_bb_logpmf_1d",
    "_dense_bb_logpmf",
    "bb_logpmf",
    "binomial_logpmf",
    "log_binomial",
    "rises_on_distinct",
]

DISPERSION_FLOOR = 1e-10
"""`cnaster`'s floor on `alpha` in `_nb_logpmf_1d` and on `a`, `b` in `_bb_logpmf_1d`.

The one statement of it (T- #617): `port.patch.hmm_nophasing.gradient`,
`port.patch.hmrf.tabulated_field` and the functions below read it.
"""


@njit(nogil=True, cache=True, error_model="numpy")
def _log_binomial_flat(n: np.ndarray, k: np.ndarray, out: np.ndarray) -> None:
    for i in range(n.size):
        out[i] = lgamma(n[i] + 1.0) - lgamma(k[i] + 1.0) - lgamma(n[i] - k[i] + 1.0)


def log_binomial(n: ArrayLike, k: ArrayLike) -> np.ndarray:
    """`log C(n, k)`, broadcast: `lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)`."""
    n_, k_ = np.broadcast_arrays(
        np.asarray(n, dtype=np.float64), np.asarray(k, dtype=np.float64)
    )
    out = np.empty(n_.shape)
    _log_binomial_flat(
        np.ascontiguousarray(n_).ravel(), np.ascontiguousarray(k_).ravel(), out.ravel()
    )
    return out


def bb_logpmf(k: ArrayLike, n: ArrayLike, a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """The beta-binomial log pmf at shapes `a`, `b`, broadcast; 0 where `cnaster`'s kernel scores 0."""
    k, n, a, b = np.broadcast_arrays(
        *(np.asarray(v, dtype=np.float64) for v in (k, n, a, b))
    )
    valid = (k >= 0.0) & (n >= 0.0) & (k <= n) & (a > 0.0) & (b > 0.0)
    # NB an invalid entry is scored at `k = n = 0`, `a = b = 1`, then zeroed:
    #    `log_rising` is never asked for a negative count.
    kk, nn = np.where(valid, k, 0.0), np.where(valid, n, 0.0)
    aa, bb = np.where(valid, a, 1.0), np.where(valid, b, 1.0)
    out = (
        log_binomial(nn, kk)
        + log_rising(aa, kk)
        + log_rising(bb, nn - kk)
        - log_rising(aa + bb, nn)
    )
    return np.where(valid, out, 0.0)


def binomial_logpmf(k: ArrayLike, n: ArrayLike, p: float) -> np.ndarray:
    """The binomial log pmf, the beta-binomial's `tau -> inf` limit, broadcast; 0 where it scores 0."""
    k, n = np.broadcast_arrays(np.asarray(k, np.float64), np.asarray(n, np.float64))
    valid = (k >= 0.0) & (n >= 0.0) & (k <= n)
    kk, nn = np.where(valid, k, 0.0), np.where(valid, n, 0.0)
    out = log_binomial(nn, kk)
    with np.errstate(divide="ignore", invalid="ignore"):
        hits = np.where(kk > 0.0, kk * np.log(p) if p > 0.0 else -np.inf, 0.0)
        misses = np.where(
            nn - kk > 0.0, (nn - kk) * np.log1p(-p) if p < 1.0 else -np.inf, 0.0
        )
    return np.where(valid, out + hits + misses, 0.0)


def _bb_logpmf_1d(
    obs: Any,
    total: Any,
    p_binom: float,
    tau: float,
    out: np.ndarray | None = None,
    EPS: float = DISPERSION_FLOOR,
) -> np.ndarray:
    """`cnaster`'s row: each bin's beta-binomial log pmf into `out`, and returned.

    `tau = inf` is the binomial (T- #617). `cnaster` passes `out` but for one
    caller that reads the return; both are served.
    """
    obs = np.asarray(obs, dtype=np.float64)
    if tau == np.inf:
        row = binomial_logpmf(obs, total, p_binom)
    else:
        alpha = max(p_binom * tau, EPS)
        beta = max((1.0 - p_binom) * tau, EPS)
        row = bb_logpmf(obs, total, alpha, beta)
    if out is None:
        return row
    out[:] = row
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


def rises_on_distinct(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """`sal`'s `log_rising`, evaluated on the distinct counts of `m` and gathered (#702).

    `m` is one count per bin along the last axis and `x` a shape constant
    along it -- a scalar, or one row per state as `(..., 1)`. Then a bin's
    value depends on its count alone, so `log_rising(x, distinct)` gathered
    by each bin's index is the per-bin evaluation: elementwise, so bitwise.
    Measured: `docs/measurements.md`,
    `port.patch.hmm_nophasing.bb_logpmf.rises_on_distinct`. `sal`'s own
    gather, `on_distinct` with `reusing_distinct`, takes tensors (T- #781).

    A shape that varies along the bins, or an `m` that is not one-dimensional,
    takes `log_rising` unchanged.
    """
    m = np.asarray(m, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if m.ndim != 1 or (x.ndim > 0 and x.shape[-1] != 1):
        return log_rising(x, m)
    distinct, inverse = np.unique(m, return_inverse=True)
    if distinct.size == m.size:
        return log_rising(x, m)
    gathered: np.ndarray = log_rising(x, distinct)[..., inverse]
    return gathered
