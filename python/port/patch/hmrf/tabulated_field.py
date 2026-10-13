"""The spot/clone field with its `lgamma` tabulated by count, bitwise (#433).

Each `lgamma` in the fused field depends only on an integer count and a
per-state parameter, so it is computed once per `(state, count)`. Referee:
bitwise against `fused_spot_clone_field`. Under `log_space` the tables are
`port.patch.emission`'s (T- #776), equal to its pmfs to rounding. Non-integer
counts go to the fused kernel (:func:`spot_clone_field`).
"""

from __future__ import annotations

from functools import partial
from math import lgamma, log, log1p
from typing import TYPE_CHECKING, Any

import numpy as np
from numba import get_num_threads, njit

from port.patch.emission import (
    DISPERSION_FLOOR,
    bb_complete,
    bb_tables,
    nb_table,
)
from port.patch.hmrf.fused_field import fused_spot_clone_field

if TYPE_CHECKING:  # pragma: no cover - `prange` is `range` to a type checker
    prange = range
else:
    from numba import prange

__all__ = ["field_kernel", "spot_clone_field", "tabulated_spot_clone_field"]

EPS = DISPERSION_FLOOR
"""`cnaster`'s floor on `alpha`, `a` and `b` (T- #617)."""

LIMIT = 2**24
"""The largest count a table is built to; beyond it the fused kernel scores."""


def _extent(counts_bb: np.ndarray, total_bb_RD: np.ndarray) -> int:
    """The beta-binomial tables' length: the largest count or total, plus one."""
    if not counts_bb.size:
        return 1
    return int(max(counts_bb.max(), total_bb_RD.max())) + 1


def tabulated_spot_clone_field(
    counts_nb: np.ndarray,
    base_nb_mean: np.ndarray,
    counts_bb: np.ndarray,
    total_bb_RD: np.ndarray,
    log_mu: np.ndarray,
    alphas: np.ndarray,
    p_binom: np.ndarray,
    taus: np.ndarray,
    pred: np.ndarray,
    rel_valid_emision_weight: np.ndarray,
    out: np.ndarray,
    log_space: bool = False,
) -> np.ndarray:
    """`fused_spot_clone_field`'s `(n_spots, n_clones)` field, from tables.

    Counts must be non-negative integers (unchecked: see :func:`spot_clone_field`).
    """
    if log_space:
        nb_extent = int(counts_nb.max()) + 1 if counts_nb.size else 1
        counted, sizes = nb_table(alphas, nb_extent)
        tables = bb_tables(p_binom, taus, _extent(counts_bb, total_bb_RD))
        rises = np.stack([tables.success, tables.failure, tables.trial])
        rates = np.stack([tables.log_p, tables.log_q])
        log_factorial = tables.log_factorial
    else:
        counted, sizes = np.empty((0, 0)), np.empty(0)
        rises, rates, log_factorial = np.empty((3, 0, 0)), np.empty((2, 0)), np.empty(0)
    field: np.ndarray = _tabulated_kernel(
        counts_nb,
        base_nb_mean,
        counts_bb,
        total_bb_RD,
        log_mu,
        alphas,
        p_binom,
        taus,
        pred,
        rel_valid_emision_weight,
        out,
        counted,
        sizes,
        rises,
        rates,
        log_factorial,
        log_space,
    )
    return field


@njit(nogil=True, cache=True, parallel=True, error_model="numpy")
def _tabulated_kernel(
    counts_nb,
    base_nb_mean,
    counts_bb,
    total_bb_RD,
    log_mu,
    alphas,
    p_binom,
    taus,
    pred,
    rel_valid_emision_weight,
    out,
    counted,
    sizes_log_space,
    rises,
    rates,
    log_factorial_log_space,
    log_space,
):
    n_obs, n_spots = counts_nb.shape
    n_states = log_mu.shape[0]
    n_clones = pred.shape[1]

    nb_extent = int(counts_nb.max()) + 1 if counts_nb.size else 1
    bb_extent = (
        int(max(counts_bb.max(), total_bb_RD.max())) + 1 if counts_bb.size else 1
    )

    # NB `nbinom_logpmf_numba`'s coefficient, in its order: `(lgamma(k + r) -
    #    lgamma(r)) - lgamma(k + 1)`.
    nb_coefficient = np.empty((n_states, nb_extent))
    sizes = np.empty(n_states)
    inverse_sizes = np.zeros(n_states)

    for s in prange(n_states):
        if log_space:
            # NB `port.patch.emission.nb_table`, built before the pass.
            sizes[s] = sizes_log_space[s]
            inverse_sizes[s] = 1.0 / sizes_log_space[s]
            for k in range(nb_extent):
                nb_coefficient[s, k] = counted[s, k]
            continue

        r = 1.0 / max(alphas[s], DISPERSION_FLOOR)
        sizes[s] = r

        for k in range(nb_extent):
            value = lgamma(k + r) - lgamma(r)
            value -= lgamma(k + 1)
            nb_coefficient[s, k] = value

    # NB `betabinom_logpmf_numba`'s terms, each at the argument it computes.
    log_factorial = np.empty(bb_extent)

    for j in range(bb_extent):
        log_factorial[j] = log_factorial_log_space[j] if log_space else lgamma(j + 1)

    first = np.empty((n_states, bb_extent))
    second = np.empty((n_states, bb_extent))
    joint = np.empty((n_states, bb_extent))
    denominator = np.empty(n_states)

    for s in prange(n_states):
        a = max(p_binom[s] * taus[s], EPS)
        b = max((1.0 - p_binom[s]) * taus[s], EPS)

        if log_space:
            # NB `port.patch.emission.bb_tables`' scaled rising factorials.
            denominator[s] = 0.0

            for j in range(bb_extent):
                first[s, j] = rises[0, s, j]
                second[s, j] = rises[1, s, j]
                joint[s, j] = rises[2, s, j]
            continue

        denominator[s] = lgamma(a) + lgamma(b) - lgamma(a + b)

        for j in range(bb_extent):
            first[s, j] = lgamma(j + a)
            second[s, j] = lgamma(j + b)
            joint[s, j] = lgamma(j + a + b)

    field = out

    # NB each distinct state of a bin is scored once and added to every clone
    #    holding it, bins in order, so bitwise the per-clone loop's (T- #776).
    accumulated_rdr = np.zeros((n_spots, n_clones))
    accumulated_baf = np.zeros((n_spots, n_clones))
    n_blocks = max(1, min(n_spots, get_num_threads()))

    for block in prange(n_blocks):
        lo = block * n_spots // n_blocks
        hi = (block + 1) * n_spots // n_blocks

        held = np.empty(n_clones, dtype=np.int64)

        for o in range(n_obs):
            for c in range(n_clones):
                state = pred[o, c]
                seen = False
                for earlier in range(c):
                    if pred[o, earlier] == state:
                        seen = True
                        break
                if seen:
                    continue
                n_held = 0
                for holder in range(c, n_clones):
                    if pred[o, holder] == state:
                        held[n_held] = holder
                        n_held += 1

                mu = np.exp(log_mu[state])
                alpha = (
                    max(alphas[state], DISPERSION_FLOOR) if log_space else alphas[state]
                )
                r = sizes[state]
                denom = denominator[state]

                for spot in range(lo, hi):
                    # NB `_nb_logpmf_1d`: 0 for no baseline or `p` in {0, 1};
                    #    else summed in upstream's order.
                    rdr = 0.0
                    lambda_i = base_nb_mean[o, spot] * mu

                    if log_space:
                        # NB `nb_complete`'s order, `q = lambda * (1 / r)` (T- #776).
                        k = counts_nb[o, spot]
                        if lambda_i > 0.0:
                            q = lambda_i * inverse_sizes[state]
                            decay = lambda_i if q == 0.0 else r * log1p(q)
                            rated = 0.0 if k == 0.0 else k * log(lambda_i / (1.0 + q))
                            rdr = (nb_coefficient[state, int(k)] + rated) - decay
                    elif lambda_i > 0.0:
                        p = 1.0 / (1.0 + alpha * lambda_i)
                        k = counts_nb[o, spot]

                        if 0.0 < p < 1.0 and k >= 0.0:
                            rdr = (
                                nb_coefficient[state, int(k)]
                                + r * log(p)
                                + k * log(1.0 - p)
                            )

                    # NB `betabinom_logpmf_numba`: `(binomial + numerator) -
                    #    denominator`, each as upstream groups it.
                    baf = 0.0
                    k = counts_bb[o, spot]
                    n = total_bb_RD[o, spot]

                    if log_space and n == 0.0:
                        # NB no trials: every term of the score is 0.
                        baf = 0.0
                    elif n >= 0.0 and k >= 0.0 and k <= n:
                        kk = int(k)
                        nn = int(n)
                        binomial = (
                            log_factorial[nn]
                            - log_factorial[kk]
                            - log_factorial[nn - kk]
                        )
                        if log_space:
                            # NB `port.patch.emission.bb_complete`'s order.
                            baf = bb_complete(
                                binomial,
                                k,
                                n,
                                rates[0, state],
                                rates[1, state],
                                first[state, kk],
                                second[state, nn - kk],
                                joint[state, nn],
                            )
                        else:
                            numerator = (
                                first[state, kk]
                                + second[state, nn - kk]
                                - joint[state, nn]
                            )
                            baf = binomial + numerator - denom

                    for h in range(n_held):
                        accumulated_rdr[spot, held[h]] += rdr
                        accumulated_baf[spot, held[h]] += baf

    for spot in range(n_spots):
        for c in range(n_clones):
            field[spot, c] = (
                rel_valid_emision_weight[spot] * accumulated_rdr[spot, c]
                + accumulated_baf[spot, c]
            )

    return field


def _integral(values: np.ndarray) -> bool:
    """Every entry a non-negative integer a table can be built to."""
    return bool(
        values.size == 0
        or (
            np.all(values >= 0.0)
            and np.all(values == np.floor(values))
            and float(values.max()) < LIMIT
        )
    )


def field_kernel(
    counts_nb: np.ndarray,
    counts_bb: np.ndarray,
    total_bb_RD: np.ndarray,
    *,
    log_space: bool = False,
) -> Any:
    """The kernel :func:`spot_clone_field` would run on these counts, `log_space` bound (#488)."""
    if _integral(counts_nb) and _integral(counts_bb) and _integral(total_bb_RD):
        kernel = tabulated_spot_clone_field
    else:
        kernel = fused_spot_clone_field

    return partial(kernel, log_space=True) if log_space else kernel


def spot_clone_field(
    counts_nb: np.ndarray,
    base_nb_mean: np.ndarray,
    counts_bb: np.ndarray,
    total_bb_RD: np.ndarray,
    log_mu: np.ndarray,
    alphas: np.ndarray,
    p_binom: np.ndarray,
    taus: np.ndarray,
    pred: np.ndarray,
    rel_valid_emision_weight: np.ndarray,
    out: np.ndarray,
    *,
    log_space: bool = False,
) -> Any:
    """The field from tables where the counts are integers, else the fused kernel's; writes `out`."""
    arguments = (
        counts_nb,
        base_nb_mean,
        counts_bb,
        total_bb_RD,
        log_mu,
        alphas,
        p_binom,
        taus,
        pred,
        rel_valid_emision_weight,
        out,
    )

    return field_kernel(counts_nb, counts_bb, total_bb_RD, log_space=log_space)(
        *arguments
    )
