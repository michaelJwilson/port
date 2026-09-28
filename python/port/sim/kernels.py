"""Sparse count draws for `port.sim.draw` by inversion, compiled (#445).

A spot-by-gene count matrix at CalicoST's size is 3,000 x 35,291 entries, of
which 93 per cent are zero at the fitted baseline. Drawing each entry as a
Gamma times a Poisson visits and allocates every one of them twice. Here
each entry is the negative-binomial quantile of one uniform: it is zero
wherever the uniform falls under `P(0)`, and only the nonzero entries walk
the CDF. One uniform per entry, drawn in blocks from the seeded generator,
makes the draw deterministic whatever the thread count.

`counts_numpy` is the oracle (`CLAUDE.md`, **The Oracle**): the same
quantiles by `scipy.stats`' `ppf`, which `tests/test_sim_kernels.py` pins
the compiled kernel against.

The law, per entry with mean `mu` and `dispersion`: NB with
`var = mu + dispersion mu^2`, Poisson at `dispersion = 0`. The quantile is the
smallest `k` with `CDF(k) > u`; `scipy`'s is `CDF(k) >= u`, which differs
only on a set of measure zero.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numba import njit

if TYPE_CHECKING:  # pragma: no cover - `prange` is `range` to a type checker
    prange = range
else:
    from numba import prange

MAX_COUNT = 10_000_000
"""A walk this long means the mean was not finite; the kernel stops there."""


@njit(cache=True, inline="always")
def _zero(mu: float, dispersion: float) -> float:
    """`P(0)`."""
    if dispersion > 0.0:
        return float((1.0 + dispersion * mu) ** (-1.0 / dispersion))
    return float(np.exp(-mu))


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


@njit(cache=True, parallel=True)
def _block(
    uniforms: np.ndarray,
    depth: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    dispersion: float,
) -> np.ndarray:
    """`(n_rows, n_cols)` int32 quantiles at `depth[i] * weights[j, labels[i]]`."""
    n_rows, n_cols = uniforms.shape
    out = np.zeros((n_rows, n_cols), dtype=np.int32)

    for i in prange(n_rows):
        clone = labels[i]
        for j in range(n_cols):
            mu = depth[i] * weights[j, clone]
            if mu > 0.0 and uniforms[i, j] >= _zero(mu, dispersion):
                out[i, j] = _walk(uniforms[i, j], mu, dispersion)

    return out


@njit(cache=True, parallel=True)
def _compact(dense: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CSR `(indptr, indices, data)` of a dense int32 block."""
    n_rows, n_cols = dense.shape
    nonzero = np.zeros(n_rows, dtype=np.int64)

    for i in prange(n_rows):
        count = 0
        for j in range(n_cols):
            if dense[i, j] != 0:
                count += 1
        nonzero[i] = count

    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(nonzero)
    indices = np.empty(indptr[-1], dtype=np.int32)
    data = np.empty(indptr[-1], dtype=np.int64)

    for i in prange(n_rows):
        at = indptr[i]
        for j in range(n_cols):
            if dense[i, j] != 0:
                indices[at] = j
                data[at] = dense[i, j]
                at += 1

    return indptr, indices, data


def counts(
    uniforms: np.ndarray,
    depth: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    dispersion: float,
) -> Any:
    """`scipy.sparse.csr_matrix` of NB quantiles at `depth[i] * weights[:, labels[i]]`."""
    import scipy.sparse

    dense = _block(
        np.ascontiguousarray(uniforms, dtype=np.float64),
        np.ascontiguousarray(depth, dtype=np.float64),
        np.ascontiguousarray(weights, dtype=np.float64),
        np.ascontiguousarray(labels, dtype=np.int64),
        float(dispersion),
    )
    indptr, indices, data = _compact(dense)
    return scipy.sparse.csr_matrix((data, indices, indptr), shape=dense.shape)


def counts_numpy(
    uniforms: np.ndarray,
    depth: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    dispersion: float,
) -> Any:
    """The oracle: the same quantiles by `scipy.stats`' `ppf`, dense then sparse."""
    import scipy.sparse
    from scipy.stats import nbinom, poisson

    means = np.asarray(depth, dtype=np.float64)[:, None] * weights[:, labels].T
    positive = means > 0
    out = np.zeros(means.shape, dtype=np.int64)
    u, mu = uniforms[positive], means[positive]

    if dispersion > 0:
        out[positive] = nbinom.ppf(u, 1.0 / dispersion, 1.0 / (1.0 + dispersion * mu))
    else:
        out[positive] = poisson.ppf(u, mu)

    return scipy.sparse.csr_matrix(out)


def draw_rows(
    depth: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    dispersion: float,
    rng: np.random.Generator,
    block: int = 512,
) -> Any:
    """NB counts, row `s` at mean `depth[s] * weights[:, labels[s]]`, in row blocks.

    `weights` is `(n_cols, n_clones)`; one uniform per entry, drawn a block
    at a time, so memory is `block x n_cols` and the draw is the same bits
    at any thread count.
    """
    import scipy.sparse

    blocks = []
    for start in range(0, depth.size, block):
        rows = slice(start, start + block)
        shape = (depth[rows].size, weights.shape[0])
        blocks.append(
            counts(rng.random(shape), depth[rows], weights, labels[rows], dispersion)
        )
    return scipy.sparse.vstack(blocks, format="csr")
