"""`cnaster.hmm_nophasing._bb_logpmf_1d`, exact at a large concentration (#561).

**A row (#561).** `port.pipeline.LOG_SPACE_SWAPS` rebinds `_bb_logpmf_1d` and
`_dense_bb_logpmf` to these wherever `cnaster` binds them, and
`port.patch.hmrf`'s field and `dense_emission.bb_states` read the same
arithmetic. The NumPy callers -- the copy likelihood through
:func:`rises_on_distinct`, the M-step gradient -- take `sal`'s
`log_rising` and `digamma_rising` (T- #781). Retire when `cnaster` lands the
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

from math import expm1, inf, lgamma, log, log1p
from typing import TYPE_CHECKING

import numpy as np
from numba import njit
from sal.emissions.rising import log_rising

if TYPE_CHECKING:  # pragma: no cover - `prange` is `range` to a type checker
    prange = range
else:
    from numba import prange

__all__ = [
    "DISPERSION_FLOOR",
    "STIRLING",
    "_bb_logpmf_1d",
    "_dense_bb_logpmf",
    "bb_logpmf",
    "binomial_logpmf",
    "rise",
    "rises_on_distinct",
]

DISPERSION_FLOOR = 1e-10
"""`cnaster`'s floor on `alpha` in `_nb_logpmf_1d` and on `a`, `b` in `_bb_logpmf_1d`.

The one statement of it (T- #617): `port.patch.hmm_nophasing.gradient`,
`port.patch.hmrf.tabulated_field` and the kernels below read it.
"""

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
def binomial_logpmf(k: float, n: float, p: float) -> float:
    """The binomial log pmf, the beta-binomial's `tau -> inf` limit; 0 where it scores 0."""
    if k < 0.0 or n < 0.0 or k > n:
        return 0.0
    out = lgamma(n + 1.0) - lgamma(k + 1.0) - lgamma(n - k + 1.0)
    if k > 0.0:
        out += k * log(p) if p > 0.0 else -inf
    if n - k > 0.0:
        out += (n - k) * log1p(-p) if p < 1.0 else -inf
    return out


@njit(nogil=True, cache=True, error_model="numpy")
def _bb_logpmf_1d(obs, total, p_binom, tau, out, EPS=DISPERSION_FLOOR):
    # NB `tau = inf` is the binomial: the shapes are infinite and the series
    #    in `rise` is `inf * 0`, NaN (T- #617). Finite `tau` is unchanged.
    if tau == inf:
        for i in range(len(obs)):
            out[i] = binomial_logpmf(obs[i], total[i], p_binom)
        return

    alpha = max(p_binom * tau, EPS)
    beta = max((1.0 - p_binom) * tau, EPS)

    for i in range(len(obs)):
        out[i] = bb_logpmf(obs[i], total[i], alpha, beta)


@njit(nogil=True, cache=True, parallel=True, error_model="numpy")
def _dense_bb_logpmf(X_bb, total_bb_RD, p_binom, taus, EPS=DISPERSION_FLOOR):
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
