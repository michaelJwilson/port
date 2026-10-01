"""The spot/clone field with its `lgamma` tabulated by count, bitwise (#433).

`fused_spot_clone_field` scores every `(bin, spot, clone)` through
`cnaster`'s kernels: three `lgamma` a negative-binomial score and six a
beta-binomial one. Every one of the nine is a function of an integer count
and a per-state parameter -- `lgamma(k + r)`, `lgamma(k + a)`,
`lgamma(n - k + b)`, `lgamma(n + a + b)`, `lgamma(k + 1)` and their
constants -- and only the negative binomial's `r log p + k log(1 - p)`
reads the continuous exposure. So each is computed once per `(state, count)`
and read back, which is sal's approach to the same emission
(`sal.emissions.nb`, `sal.emissions.bb`, and `oxisal.external_field`, which
tabulates this field's own sum).

**Referee: bitwise.** A table entry is the same `lgamma` at the same
argument, built in the same order of operations as `cnaster`'s
`nbinom_logpmf_numba` and `betabinom_logpmf_numba` -- `lgamma(n + a + b)` is
`lgamma((n + a) + b)`, as upstream writes it -- and the per-bin sums run in
the fused kernel's order, so `np.array_equal` is the bar against it.

**Why not sal's kernels.** `sal.emissions.dense.log_emission` scores every
state at every observation, where the field reads one per `(bin, clone)`;
its negative binomial completes the exposure term in its own order (262.9 ulp
under `Order.TABULATED`); and `oxisal.external_field` sums both channels
with one weight, where `cnaster` weighs read depth per spot
(`rel_valid_emision_weight`). The measurement is in the pull request.

Counts that are not non-negative integers cannot index a table, and
:func:`spot_clone_field` hands those to the fused kernel unchanged.
"""

from __future__ import annotations

from math import lgamma, log
from typing import TYPE_CHECKING, Any

import numpy as np
from numba import njit

from port.patch.hmm_nophasing.gradient import DISPERSION_FLOOR
from port.patch.hmrf.fused_field import fused_spot_clone_field

if TYPE_CHECKING:  # pragma: no cover - `prange` is `range` to a type checker
    prange = range
else:
    from numba import prange

__all__ = ["field_kernel", "spot_clone_field", "tabulated_spot_clone_field"]

EPS = 1e-10
"""`cnaster`'s floors: `alpha` in `_nb_logpmf_1d`, `a` and `b` in `_bb_logpmf_1d`."""

LIMIT = 2**24
"""The largest count a table is built to; beyond it the fused kernel scores."""


@njit(nogil=True, cache=True, parallel=True, error_model="numpy")
def tabulated_spot_clone_field(
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
):
    """`fused_spot_clone_field`'s `(n_spots, n_clones)` field, from tables.

    Takes what the fused kernel takes, with integer-valued counts: the
    caller checks that (:func:`spot_clone_field`), because a count that is not
    an integer would index the wrong row rather than fail.
    """
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

    for s in prange(n_states):
        r = 1.0 / max(alphas[s], DISPERSION_FLOOR)
        sizes[s] = r

        for k in range(nb_extent):
            value = lgamma(k + r) - lgamma(r)
            value -= lgamma(k + 1)
            nb_coefficient[s, k] = value

    # NB `betabinom_logpmf_numba`'s terms, each at the argument it computes.
    log_factorial = np.empty(bb_extent)

    for j in range(bb_extent):
        log_factorial[j] = lgamma(j + 1)

    first = np.empty((n_states, bb_extent))
    second = np.empty((n_states, bb_extent))
    joint = np.empty((n_states, bb_extent))
    denominator = np.empty(n_states)

    for s in prange(n_states):
        a = max(p_binom[s] * taus[s], EPS)
        b = max((1.0 - p_binom[s]) * taus[s], EPS)
        denominator[s] = lgamma(a) + lgamma(b) - lgamma(a + b)

        for j in range(bb_extent):
            first[s, j] = lgamma(j + a)
            second[s, j] = lgamma(j + b)
            joint[s, j] = lgamma(j + a + b)

    field = out

    for c in prange(n_clones):
        accumulated_rdr = np.zeros(n_spots)
        accumulated_baf = np.zeros(n_spots)

        for o in range(n_obs):
            state = pred[o, c]
            mu = np.exp(log_mu[state])
            alpha = alphas[state]
            r = sizes[state]
            denom = denominator[state]

            for spot in range(n_spots):
                # NB `_nb_logpmf_1d`: no baseline, or `p` rounded to 0 or 1,
                #    scores 0; otherwise the coefficient, then `r log p`, then
                #    `k log(1 - p)`, summed in that order.
                rdr = 0.0
                lambda_i = base_nb_mean[o, spot] * mu

                if lambda_i > 0.0:
                    p = 1.0 / (1.0 + alpha * lambda_i)
                    k = counts_nb[o, spot]

                    if 0.0 < p < 1.0 and k >= 0.0:
                        rdr = (
                            nb_coefficient[state, int(k)]
                            + r * log(p)
                            + k * log(1.0 - p)
                        )

                accumulated_rdr[spot] += rdr

                # NB `betabinom_logpmf_numba`: `(binomial + numerator) -
                #    denominator`, each as upstream groups it.
                baf = 0.0
                k = counts_bb[o, spot]
                n = total_bb_RD[o, spot]

                if n >= 0.0 and k >= 0.0 and k <= n:
                    kk = int(k)
                    nn = int(n)
                    binomial = (
                        log_factorial[nn] - log_factorial[kk] - log_factorial[nn - kk]
                    )
                    numerator = (
                        first[state, kk] + second[state, nn - kk] - joint[state, nn]
                    )
                    baf = binomial + numerator - denom

                accumulated_baf[spot] += baf

        for spot in range(n_spots):
            field[spot, c] = (
                rel_valid_emision_weight[spot] * accumulated_rdr[spot]
                + accumulated_baf[spot]
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
    counts_nb: np.ndarray, counts_bb: np.ndarray, total_bb_RD: np.ndarray
) -> Any:
    """The kernel :func:`spot_clone_field` would run on these counts.

    The choice reads every count, so a caller scoring the same counts more
    than once -- per clone, per sweep -- makes it once (#488).
    """
    if _integral(counts_nb) and _integral(counts_bb) and _integral(total_bb_RD):
        return tabulated_spot_clone_field

    return fused_spot_clone_field


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
) -> Any:
    """The field from tables where the counts are integers, else the fused kernel's.

    Both write `out`, and the two are bitwise equal where both apply.
    """
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

    return field_kernel(counts_nb, counts_bb, total_bb_RD)(*arguments)
