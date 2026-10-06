"""The spot/clone field `pipeline_clone_assignment` reads, without the emission array (T- #670 PR7).

`port.patch.hmrf.fused_field`, `tabulated_field`, `invariants` and
`adjacency`, moved in by T- #670 PR7 for `cnamaste.hmrf`'s clone assignment
(`docs/port-forward.md` row 32). `cnaster` builds the field in two steps --
`compute_emission_probability_nb_betabinom` materializes `(n_states, n_obs,
n_spots)` per channel, `compute_loglike_spot_assignment` reduces it over each
clone's decoded profile -- and this does both in one pass (#59 item 2):

- :func:`fused_spot_clone_field` scores, per `(bin, clone)`, only the state
  that clone decoded to, with `cnamaste`'s log-space kernels (#560, #561);
- :func:`tabulated_spot_clone_field` reads each `lgamma` from a table by count
  (#433), bitwise the fused kernel where every count is an integer;
- :func:`boundary_invariants` holds the per-spot valid-segment counts the
  relative channel weight is built from (#59 item 4);
- :func:`adjacency_coo` is `cnaster`'s COO triple from the CSR arrays, three
  array expressions for two Python passes (#59 item 3).

**Log space only.** `port`'s kernels take `log_space`, `cnaster`'s kernels
or `LOG_SPACE_SWAPS`'; `cnamaste`'s kernels have been the latter since PR4, so
the field is the `log_space=True` path alone, bitwise `port`'s on it.

`rel_valid_emision_weight` is carried as `cnaster` computes it: #58's
finding, not changed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma, log, log1p
from typing import TYPE_CHECKING, Any

import numpy as np
from numba import njit

from cnamaste.bb_logpmf import DISPERSION_FLOOR, _bb_logpmf_1d, rise
from cnamaste.nb_logpmf import _nb_logpmf_1d

if TYPE_CHECKING:  # pragma: no cover - `prange` is `range` to a type checker
    from scipy.sparse import csr_matrix

    prange = range
else:
    from numba import prange

__all__ = [
    "LIMIT",
    "BoundaryInvariants",
    "adjacency_coo",
    "boundary_invariants",
    "field_kernel",
    "fused_spot_clone_field",
    "spot_clone_field",
    "tabulated_spot_clone_field",
]

LIMIT = 2**24
"""The largest count a table is built to; beyond it the fused kernel scores."""


@njit(nogil=True, cache=True, parallel=True, error_model="numpy")
def fused_spot_clone_field(
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
) -> np.ndarray:
    """The `(n_spots, n_clones)` field into `out`, without an emission array.

    Takes what the emission was built from: `(n_obs, n_spots)` counts and
    exposures per channel, the per-state parameters, the decoded profiles
    `(n_obs, n_clones)` and the relative channel weight. `prange` runs over
    clones, each writing its own column; the spot loop is the contiguous one.
    """
    n_obs, n_spots = counts_nb.shape
    n_clones = pred.shape[1]
    field = out

    for c in prange(n_clones):
        accumulated_rdr = np.zeros(n_spots)
        accumulated_baf = np.zeros(n_spots)
        scratch = np.zeros(n_spots)

        for o in range(n_obs):
            copy_state = pred[o, c]

            # NB one row per bin, contiguous, and only the state this clone
            #    decoded to: n_clones of n_states.
            _nb_logpmf_1d(
                counts_nb[o, :],
                base_nb_mean[o, :],
                np.exp(log_mu[copy_state]),
                alphas[copy_state],
                scratch,
            )
            accumulated_rdr += scratch

            _bb_logpmf_1d(
                counts_bb[o, :],
                total_bb_RD[o, :],
                p_binom[copy_state],
                taus[copy_state],
                scratch,
            )
            accumulated_baf += scratch

        for spot in range(n_spots):
            field[spot, c] = (
                rel_valid_emision_weight[spot] * accumulated_rdr[spot]
                + accumulated_baf[spot]
            )

    return field


@njit(nogil=True, cache=True, parallel=True, error_model="numpy")
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
) -> np.ndarray:
    """:func:`fused_spot_clone_field`'s field, its `lgamma` read from tables by count.

    Integer-valued counts only: the caller checks that
    (:func:`spot_clone_field`), because a count that is not an integer would
    index the wrong row rather than fail. The negative binomial's coefficient
    `(lgamma(k + r) - lgamma(r)) - lgamma(k + 1)` is tabulated per state, and
    the beta-binomial's rising factorials `rise(x, j)` per state and count.
    """
    n_obs, n_spots = counts_nb.shape
    n_states = log_mu.shape[0]
    n_clones = pred.shape[1]
    nb_extent = int(counts_nb.max()) + 1 if counts_nb.size else 1
    bb_extent = (
        int(max(counts_bb.max(), total_bb_RD.max())) + 1 if counts_bb.size else 1
    )

    nb_coefficient = np.empty((n_states, nb_extent))
    sizes = np.empty(n_states)
    for s in prange(n_states):
        r = 1.0 / max(alphas[s], DISPERSION_FLOOR)
        sizes[s] = r
        for k in range(nb_extent):
            value = lgamma(k + r) - lgamma(r)
            value -= lgamma(k + 1)
            nb_coefficient[s, k] = value

    log_factorial = np.empty(bb_extent)
    for j in range(bb_extent):
        log_factorial[j] = lgamma(j + 1)

    first = np.empty((n_states, bb_extent))
    second = np.empty((n_states, bb_extent))
    joint = np.empty((n_states, bb_extent))
    for s in prange(n_states):
        a = max(p_binom[s] * taus[s], DISPERSION_FLOOR)
        b = max((1.0 - p_binom[s]) * taus[s], DISPERSION_FLOOR)
        for j in range(bb_extent):
            first[s, j] = rise(a, float(j))
            second[s, j] = rise(b, float(j))
            joint[s, j] = rise(a + b, float(j))

    field = out
    for c in prange(n_clones):
        accumulated_rdr = np.zeros(n_spots)
        accumulated_baf = np.zeros(n_spots)

        for o in range(n_obs):
            state = pred[o, c]
            mu = np.exp(log_mu[state])
            alpha = max(alphas[state], DISPERSION_FLOOR)
            r = sizes[state]

            for spot in range(n_spots):
                # NB `nb_logpmf._nb_logpmf_1d` (#560): the coefficient, then
                #    `- r log1p(a)`, then `k (log a - log1p(a))`; no baseline
                #    scores 0.
                rdr = 0.0
                lambda_i = base_nb_mean[o, spot] * mu
                if lambda_i > 0.0:
                    a = alpha * lambda_i
                    k = counts_nb[o, spot]
                    rdr = (
                        nb_coefficient[state, int(k)]
                        - r * log1p(a)
                        + k * (log(a) - log1p(a))
                    )
                accumulated_rdr[spot] += rdr

                # NB `bb_logpmf.bb_logpmf`'s order.
                baf = 0.0
                k = counts_bb[o, spot]
                n = total_bb_RD[o, spot]
                if n >= 0.0 and k >= 0.0 and k <= n:
                    kk = int(k)
                    nn = int(n)
                    binomial = (
                        log_factorial[nn] - log_factorial[kk] - log_factorial[nn - kk]
                    )
                    baf = (
                        binomial
                        + first[state, kk]
                        + second[state, nn - kk]
                        - joint[state, nn]
                    )
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
    """The kernel :func:`spot_clone_field` runs on these counts: tables where
    every count is an integer, else the fused kernel. A caller scoring the
    same counts more than once makes the choice once (#488)."""
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
) -> np.ndarray:
    """The field from tables where the counts are integers, else the fused
    kernel's; both write `out`, bitwise equal where both apply."""
    kernel = field_kernel(counts_nb, counts_bb, total_bb_RD)
    field: np.ndarray = kernel(
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
    return field


# --- the boundary's invariants and the COO triple (#59 items 3-4) ----------------


@dataclass(frozen=True)
class BoundaryInvariants:
    """Segments per spot with a positive baseline and with positive read
    depth, `(n_spots,)`: properties of the input data, which the outer loop
    never fits."""

    num_valid_nb_spotwise: np.ndarray
    num_valid_bb_spotwise: np.ndarray

    def relative_channel_weight(
        self,
        smooth_indptr: np.ndarray,
        smooth_indices: np.ndarray,
        single_tumor_prop: np.ndarray | None = None,
    ) -> np.ndarray:
        """`pooled_bb / pooled_nb` per spot, the weight the field applies;
        ones where a spot's pooled counts are not both positive, as `cnaster`.

        Pooled with `bincount`: the sums are of counts bounded by `n_obs`, so
        every partial sum is exact in `float64` and the reassociation is
        bitwise `cnaster`'s sequential one. With `single_tumor_prop`,
        neighbours with a `nan` proportion are skipped.
        """
        n_spots = self.num_valid_nb_spotwise.shape[0]
        owner = np.repeat(np.arange(n_spots), np.diff(smooth_indptr))
        neighbours = np.asarray(smooth_indices)

        if single_tumor_prop is not None:
            keep = ~np.isnan(single_tumor_prop[neighbours])
            owner, neighbours = owner[keep], neighbours[keep]

        pooled_nb = np.bincount(
            owner, weights=self.num_valid_nb_spotwise[neighbours], minlength=n_spots
        )
        pooled_bb = np.bincount(
            owner, weights=self.num_valid_bb_spotwise[neighbours], minlength=n_spots
        )
        weight = np.ones(n_spots, dtype=np.float64)
        both = (pooled_nb > 0.0) & (pooled_bb > 0.0)
        weight[both] = pooled_bb[both] / pooled_nb[both]

        return weight


def boundary_invariants(
    single_base_nb_mean: np.ndarray, single_total_bb_RD: np.ndarray
) -> BoundaryInvariants:
    """Both per-spot valid-segment counts; refused where the two shapes differ."""
    base = np.asarray(single_base_nb_mean)
    total = np.asarray(single_total_bb_RD)

    if base.shape != total.shape:
        msg = f"shapes must agree, got {base.shape} and {total.shape}"
        raise ValueError(msg)

    return BoundaryInvariants(
        num_valid_nb_spotwise=(base > 0).sum(axis=0),
        num_valid_bb_spotwise=(total > 0).sum(axis=0),
    )


def adjacency_coo(
    adjacency_mat: csr_matrix,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`unpack_adjacency(cast_csr(adjacency_mat))`, dtypes included: `int64`,
    `int64`, `float64`, from the CSR arrays."""
    counts = np.diff(adjacency_mat.indptr)
    return (
        np.repeat(np.arange(adjacency_mat.shape[0]), counts).astype(np.int64),
        adjacency_mat.indices.astype(np.int64),
        adjacency_mat.data.astype(np.float64),
    )
