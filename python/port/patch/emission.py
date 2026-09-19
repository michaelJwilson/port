"""One emission entry point, writing into a buffer (#205).

**#205's first and third steps, which turned out to be one module.** The
ticket counted "nine density routines for two distributions" and read that as
duplication. [#214](https://github.com/michaelJwilson/port/pull/214) corrected
it: six of the nine are a three-rung shape hierarchy -- scalar, vector,
tensor -- which is how a `numba` kernel is written for three call shapes, and
the genuine duplicate is the beta-binomial, written once for the HMM to score
with and once for the M step to optimize. Those two agree to 4.1e-13
(`tests/test_density_agreement.py`), so there is no defect to fix and nothing
to reconcile.

What is left of step 1 is the thing the count was standing in for: **four
entry points where one would do.** `hmm_nophasing` and `hmm_phased` carry
`compute_emission_probability_nb_betabinom` and its `_coded` form each, a
fifth is kept as a string literal inside `hmm_nophasing`'s class body, and
the phased pair differs from the unphased one by a `vstack` and a switched
allele. This is the one they collapse to, and `phased` is the argument that
was a subclass.

**And it writes into the caller's arrays**, which is step 3. `cnaster`
allocates `(n_states, n_obs, n_spots)` per channel on every call -- 8 GB at
the declared scale, twice per outer iteration (#90) -- and the phased form
doubles the state axis on top. Nothing about those arrays changes shape
between iterations, so the allocation is a decision nobody made:
`CLAUDE.md`'s "recompute or store is a decision, and unmade it defaults to
recompute", read for memory.

| | allocated per call |
| --- | ---: |
| `cnaster`, unphased | `2 * K * G * S * 8` bytes |
| `cnaster`, phased | `4 * K * G * S * 8` bytes |
| this, with buffers | **0** |

**The ratio is not the claim either.** At `K = 7`, `G = 3,000`, `S = 2,000`,
minimum over five rounds: `cnaster` 7,229 ms against 6,732 ms here, **1.07x**,
and 0.97x at the gate size. `CLAUDE.md` puts a speedup at 2x, so this is
reported rather than claimed -- the kernels are `cnaster`'s own and the loop
around them does the same work. What changes is the allocation.

**Referee: bitwise**, against `hmm_nophasing.compute_emission_probability_nb_betabinom`
and against `hmm_phased`'s, in `tests/test_buffered_emission.py`. That is
available because the kernels are `cnaster`'s own, imported rather than
restated -- a module that reimplemented the densities would be comparing two
implementations of the emission as well as two of the loop over it.

**Who would maintain it.** `snakes_and_ladders` carries this job:
`NegativeBinomialEmission`, `BetaBinomialEmission` and `CountPairEmission`
are one family with covariates, bounds, `Reestimate` and an oracle, and
`oxi_snakes_and_ladders.external_field` already writes into a caller's
buffer. So this is **functionality upstream could carry** rather than `port`
code -- what it needs first is `#32`'s covariate gap, since the exposure on
the count mean and the trials on the allele channel are per-observation
exogenous data and upstream cannot take them yet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from cnaster.hmm_nophasing import _bb_logpmf_1d, _nb_logpmf_1d
from cnaster.hmm_phased import _switch_betabinom_1d
from numba import njit

if TYPE_CHECKING:  # pragma: no cover - `prange` is `range` to a type checker
    prange = range
else:
    from numba import prange

__all__ = ["emission_buffers", "emission_into"]


def emission_buffers(
    n_states: int, n_obs: int, n_spots: int, *, phased: bool
) -> tuple[np.ndarray, np.ndarray]:
    """The pair of buffers :func:`emission_into` writes, allocated once.

    `np.empty` rather than `np.zeros`: every entry is written before it is
    read, so a caller reusing a buffer across outer iterations needs no
    clearing, and one that cleared it would be paying for the pass twice.

    Parameters
    ----------
    n_states : int
        Copy states. The phased buffers carry `2 * n_states` rows.
    phased : bool
        Whether the allele channel is paired with a switched copy.
    """
    rows = 2 * n_states if phased else n_states

    return (
        np.empty((rows, n_obs, n_spots), dtype=np.float64),
        np.empty((rows, n_obs, n_spots), dtype=np.float64),
    )


@njit(nogil=True, cache=True, parallel=True, error_model="numpy")
def emission_into(
    counts_nb,
    base_nb_mean,
    counts_bb,
    total_bb_RD,
    log_mu,
    alphas,
    p_binom,
    taus,
    out_rdr,
    out_baf,
    phased,
):
    """Score both channels into `out_rdr` and `out_baf`, for either class.

    Parameters
    ----------
    counts_nb, base_nb_mean : np.ndarray
        `(n_obs, n_spots)`. `cnaster` passes these as `X[:, 0, :]` and the
        baseline.
    counts_bb, total_bb_RD : np.ndarray
        `(n_obs, n_spots)`. `X[:, 1, :]` and the read depth.
    log_mu, alphas, p_binom, taus : np.ndarray
        `(n_states, n_columns)`. One column is read for every spot, which is
        `hmm_nophasing`'s indexing; `n_spots` columns are read one per spot,
        which is `hmm_phased`'s. Both live layouts, and the branch between
        them is the shape rather than a flag.
    out_rdr, out_baf : np.ndarray
        `(n_states, n_obs, n_spots)`, or `(2 * n_states, ...)` when `phased`.
        Written in full. See :func:`emission_buffers`.
    phased : bool
        When true the state axis is doubled: the read-depth channel is the
        unphased block repeated, and the allele channel is the unphased block
        above its switched copy. That is `hmm_phased`'s `vstack` pair, done
        in place.

    Notes
    -----
    `prange` runs over copy states: each writes its own slab of both buffers,
    so nothing reduces across threads. The spot loop is inside it, and the
    per-bin kernels are `cnaster`'s own -- `_nb_logpmf_1d`, `_bb_logpmf_1d`
    and `_switch_betabinom_1d` -- so what is being compared is the loop and
    the allocation, not the density.

    The switched allele is computed from the unswitched one, which is how
    `hmm_phased` computes it: `_switch_betabinom_1d` takes the unphased
    scores and returns the scores at `1 - p`. Recomputing it from the
    parameters would be a second implementation of the switch and would cost
    the bitwise claim.
    """
    n_obs, n_spots = counts_nb.shape
    n_states = log_mu.shape[0]

    # NB the two entry points read the parameter columns differently, and
    #    that is the only place they disagree about the parameters:
    #    `hmm_nophasing`'s dense kernels take column zero, `hmm_phased`'s
    #    encoder path takes `log_mu[i, s]` -- one column per spot, which on
    #    the clone-stacked layout is one per clone. Both are covered by
    #    reading column `spot` when there is one per spot and column zero
    #    otherwise, which is what each of them means by its own indexing.
    per_spot = log_mu.shape[1] == n_spots and n_spots > 1

    for state in prange(n_states):
        for spot in range(n_spots):
            column = spot if per_spot else 0

            _nb_logpmf_1d(
                counts_nb[:, spot],
                base_nb_mean[:, spot],
                np.exp(log_mu[state, column]),
                alphas[state, column],
                out_rdr[state, :, spot],
            )
            _bb_logpmf_1d(
                counts_bb[:, spot],
                total_bb_RD[:, spot],
                p_binom[state, column],
                taus[state, column],
                out_baf[state, :, spot],
            )

        if phased:
            # NB the read-depth channel is the unphased block repeated and
            #    the allele channel is it above its switched copy, which is
            #    `hmm_phased`'s `vstack` pair written in place.
            for spot in range(n_spots):
                column = spot if per_spot else 0

                switched = _switch_betabinom_1d(
                    out_baf[state : state + 1, :, spot].copy(),
                    counts_bb[:, spot],
                    total_bb_RD[:, spot],
                    p_binom[state : state + 1, column],
                    taus[state : state + 1, column],
                )

                for obs in range(n_obs):
                    out_rdr[state + n_states, obs, spot] = out_rdr[state, obs, spot]
                    out_baf[state + n_states, obs, spot] = switched[0, obs]
