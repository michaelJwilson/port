"""`cnaster.hmm_nophasing._bb_logpmf_1d`, exact at a large concentration (#561).

**A row (#561).** `port.pipeline.LOG_SPACE_SWAPS` rebinds `_bb_logpmf_1d` and
`_dense_bb_logpmf` to these wherever `cnaster` binds them, and
`port.patch.hmrf`'s field, `dense_emission.bb_states` and the M-step
gradient read the same arithmetic. Retire when `cnaster` and `sal` land the
fix.

**The defect.** Upstream evaluates
`lgamma(k + a) + lgamma(n - k + b) - lgamma(n + a + b) - (lgamma(a) +
lgamma(b) - lgamma(a + b))` with `a = p tau`, `b = (1 - p) tau`. Each
`lgamma` is near `tau log tau`, so the sum loses `eps tau log tau`: 1e-10
nats at `tau = 1e5`, 3e-9 at 1e6, 5e-3 at 1e12, and at 1e16 the pmf over
`n = 100` sums to `e^132`. `sal`'s dense kernel fills its tables the same way.

**The fix.** Each pair is a rising factorial,
`R(x, m) = lgamma(x + m) - lgamma(x)`, so the log pmf is
`log C(n, k) + R(a, k) + R(b, n - k) - R(a + b, n)`. Below `x = 1e3`, `R` is
the two `lgamma`, whose loss there is under 2e-12. At and above, Stirling's
series is differenced term by term:
`R(x, m) = m log x + (x + m - 1/2) log1p(m / x) - m + c(x + m) - c(x)`, with
`c(y) = 1 / (12 y) - 1 / (360 y^3) + 1 / (1260 y^5)` and each difference of
`c` formed from `expm1`, so nothing near `tau log tau` is ever subtracted.
The truncation is below `1 / (1680 x^7)`, 6e-25 at 1e3.

**Referee.** `mpmath.loggamma` at 50 digits, to 1e-11 absolute for `tau`
from 10 to 1e16 (`tests/test_bb_logpmf.py`); `cnaster`'s kernel to 1e-9
where its own loss is below that. `cnaster`'s conventions are kept: `a` and
`b` floored at `EPS`, and `k > n` or a negative count scores 0.

**Ratio.** Not an optimization: a `log1p` and three `expm1` per term above
1e3, where upstream took six `lgamma`.
"""

from __future__ import annotations

from math import expm1, lgamma, log, log1p
from typing import TYPE_CHECKING

import numpy as np
from numba import njit
from scipy.special import digamma, gammaln

if TYPE_CHECKING:  # pragma: no cover - `prange` is `range` to a type checker
    prange = range
else:
    from numba import prange

__all__ = [
    "STIRLING",
    "_bb_logpmf_1d",
    "_dense_bb_logpmf",
    "bb_logpmf",
    "digamma_rise",
    "rise",
    "rises",
]

STIRLING = 1e3
"""The shape at and above which `R(x, m)` is Stirling's differenced series rather than two `lgamma`."""


@njit(nogil=True, cache=True, error_model="numpy")
def rise(x: float, m: float) -> float:
    """`lgamma(x + m) - lgamma(x)`, `x > 0`, `m >= 0`, without cancellation at large `x`."""
    if m == 0.0:
        return 0.0
    if x < STIRLING:
        return lgamma(x + m) - lgamma(x)

    step = log1p(m / x)
    # NB `c(x + m) - c(x)`, each power's difference as `-expm1(-j step)`, the
    #    relative shortfall `1 - (x / (x + m))^j`, so it is formed small.
    correction = (
        expm1(-step) / (12.0 * x)
        - expm1(-3.0 * step) / (360.0 * x**3)
        + expm1(-5.0 * step) / (1260.0 * x**5)
    )
    return m * log(x) + ((x + m - 0.5) * step - m) + correction


@njit(nogil=True, cache=True, error_model="numpy")
def bb_logpmf(k: float, n: float, a: float, b: float) -> float:
    """The beta-binomial log pmf at shapes `a`, `b`; 0 where `cnaster`'s kernel scores 0."""
    if k < 0.0 or n < 0.0 or k > n or a <= 0.0 or b <= 0.0:
        return 0.0
    binomial = lgamma(n + 1.0) - lgamma(k + 1.0) - lgamma(n - k + 1.0)
    return binomial + rise(a, k) + rise(b, n - k) - rise(a + b, n)


@njit(nogil=True, cache=True, error_model="numpy")
def _bb_logpmf_1d(obs, total, p_binom, tau, out, EPS=1e-10):
    alpha = max(p_binom * tau, EPS)
    beta = max((1.0 - p_binom) * tau, EPS)

    for i in range(len(obs)):
        out[i] = bb_logpmf(obs[i], total[i], alpha, beta)


@njit(nogil=True, cache=True, parallel=True, error_model="numpy")
def _dense_bb_logpmf(X_bb, total_bb_RD, p_binom, taus, EPS=1e-10):
    n_states = p_binom.shape[0]
    n_obs, n_spots = X_bb.shape
    out = np.zeros((n_states, n_obs, n_spots), dtype=np.float64)

    for i in prange(n_states):
        p_val = p_binom[i, 0]
        tau_val = taus[i, 0]

        for s in range(n_spots):
            _bb_logpmf_1d(
                X_bb[:, s], total_bb_RD[:, s], p_val, tau_val, out[i, :, s], EPS
            )
    return out


def rises(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """:func:`rise`, broadcast over arrays, for the NumPy callers."""
    x = np.asarray(x, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    large = x >= STIRLING
    safe = np.where(large, x, STIRLING)
    step = np.log1p(m / safe)
    correction = (
        np.expm1(-step) / (12.0 * safe)
        - np.expm1(-3.0 * step) / (360.0 * safe**3)
        + np.expm1(-5.0 * step) / (1260.0 * safe**5)
    )
    series = m * np.log(safe) + ((safe + m - 0.5) * step - m) + correction
    small = gammaln(x + m) - gammaln(x)
    return np.where(m == 0.0, 0.0, np.where(large, series, small))


def digamma_rise(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """`psi(x + m) - psi(x)`, broadcast, without cancellation at large `x`.

    The derivative of :func:`rise` in `x`. Below `STIRLING` the two
    `digamma`; at and above, the asymptotic series differenced as `rise`
    differences Stirling's: `log1p(m / x)` plus the `1 / (2 y)`,
    `1 / (12 y^2)` and `1 / (120 y^4)` terms, each from `expm1`. The
    truncation is below `1 / (252 x^6)`, 4e-21 at 1e3.
    """
    x = np.asarray(x, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    large = x >= STIRLING
    safe = np.where(large, x, STIRLING)
    step = np.log1p(m / safe)
    series = (
        step
        - np.expm1(-step) / (2.0 * safe)
        - np.expm1(-2.0 * step) / (12.0 * safe**2)
        + np.expm1(-4.0 * step) / (120.0 * safe**4)
    )
    return np.where(large, series, digamma(x + m) - digamma(x))
