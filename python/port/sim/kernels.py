"""The SNP law's exact sampler, compiled (#445, #455).

`port.sim.entries.Mixture.draw_nonzero` draws each (SNP, spot) entry given
`k >= 1` from a mixture of NBs: one uniform picks the component, and the
other is inverted under that component's CDF above `P(0)`. The walk is in log
space and stops only at the quantile or at `MAX_COUNT`, so the NB's tail is
not cut. Its referee is the law's own pmf by `scipy.stats`
(`tests/test_sim_entries.py`).

The law, per component with mean `mu` and `dispersion`: NB with
`var = mu + dispersion mu^2`, Poisson at `dispersion = 0`. The quantile is
the smallest `k` with `CDF(k) > u`.
"""

from __future__ import annotations

import numpy as np
from numba import njit

MAX_COUNT = 10_000_000
"""A walk this long means the mean was not finite; the kernel stops there."""


@njit(cache=True)
def _walk(u: float, mu: float, dispersion: float) -> int:
    """The smallest `k >= 1` with `CDF(k) > u`, given `u >= P(0)`, in log space."""
    if dispersion > 0.0:
        log_p = -np.log1p(dispersion * mu) / dispersion
        r = 1.0 / dispersion
        log_q = np.log(dispersion * mu) - np.log1p(dispersion * mu)
    else:
        log_p = -mu
        r = 0.0
        log_q = np.log(mu)
    cdf = np.exp(log_p)
    k = 0

    while cdf <= u and k < MAX_COUNT:
        k += 1
        if dispersion > 0.0:
            log_p += np.log((k - 1 + r) / k) + log_q
        else:
            log_p += log_q - np.log(k)
        cdf += np.exp(log_p)

    return k


@njit(cache=True)
def zero_truncated(
    cumulative: np.ndarray,
    centres: np.ndarray,
    zeros: np.ndarray,
    dispersion: float,
    pick: np.ndarray,
    level: np.ndarray,
) -> np.ndarray:
    """`k >= 1` from a mixture of NBs, one `(pick, level)` uniform pair each.

    `pick` chooses the component by `cumulative`, its weights given `k >= 1`;
    `level` is mapped onto `[P(0), 1)` of that component and inverted, so the
    draw is the component's quantile given `k >= 1`, with no tail cut.
    """
    out = np.empty(pick.size, dtype=np.int64)
    last = centres.size - 1
    for i in range(pick.size):
        c = min(np.searchsorted(cumulative, pick[i], side="right"), last)
        u = zeros[c] + level[i] * (1.0 - zeros[c])
        out[i] = max(_walk(u, centres[c], dispersion), 1)
    return out


@njit(cache=True)
def seated(total: int, kappa: float, uniforms: np.ndarray) -> np.ndarray:
    """Table sizes of `total` customers seated by a Chinese restaurant (#549).

    Customer `i` (from 0) opens a table with probability `kappa / (kappa + i)`
    -- `uniforms[i, 0]` below it -- and otherwise joins the table of customer
    `floor(uniforms[i, 1] i)`, which is a table picked in proportion to its
    size. Labelling each table independently from `q` makes the counts
    Dirichlet-multinomial `DM(total, kappa q)`: the Pólya urn.
    """
    sizes = np.zeros(total, dtype=np.int64)
    table = np.zeros(total, dtype=np.int64)
    opened = 0
    for i in range(total):
        if uniforms[i, 0] * (kappa + i) < kappa:
            table[i] = opened
            opened += 1
        else:
            table[i] = table[int(uniforms[i, 1] * i)]
        sizes[table[i]] += 1
    return sizes[:opened]
