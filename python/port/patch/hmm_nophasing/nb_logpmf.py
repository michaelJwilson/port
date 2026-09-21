"""`cnaster`'s negative-binomial log-pmf, vectorized, with the two hoists.

**Proposed for `cnaster`, written here.** #240. `hmm_nophasing._nb_logpmf_1d`
loops over observations calling `nbinom_logpmf_numba`, which makes **three
`lgamma` calls per element**:

| term | varies with | in `cnaster` |
| --- | --- | --- |
| `lgamma(r)` | state only | called per element, inside the loop |
| `lgamma(k + 1)` | **data only** | per element, per state, per EM iteration |
| `lgamma(k + r)` | data and state | irreducible here |

Two of the three are recomputation. This module removes both and evaluates
the third with `scipy.special.gammaln` over the whole array.

## Measured: 1.78x, which is below the bar, taken deliberately

`CLAUDE.md` puts a speedup claim at 2x, and this does not reach it. Three
forms were measured at 200,000 observations by 10 states, each agreeing with
`cnaster` to round-off:

| form | ratio |
| --- | ---: |
| hoist `lgamma(r)`, cache `lgamma(k + 1)`, numba loop | 1.94x |
| this module -- the same two cuts, vectorized | **1.78x** |
| the same, plus a prefix-sum table for `lgamma(k + r)` | 3.51x |

So vectorization is the *weakest* of the three: it allocates a temporary per
term and still performs one transcendental per element. The 3.51x form
replaces `lgamma(k + r) - lgamma(r)` with `sum_{j<k} log(r + j)`, a prefix sum
indexed by the count -- 78 logarithms for 200,000 elements at the measured
`k_max`. It is not here because it needs a `k_max` bound and a fallback, and
that branch is unmeasured (#240 item 2).

**This lands as the smallest correct diff, not as a speedup.** Its evidence is
the agreement below, not the ratio.

## The `lgamma(k + 1)` cache

Keyed by the observation array's identity, like `clone_assignment`'s boundary
cache: the same array is scored against every state and every iteration, and
`log(k!)` does not depend on either. The cache holds a reference, so it keeps
the array alive for as long as it is cached -- deliberate, because a run
evaluates the same observations repeatedly and the alternative is recomputing
a per-element transcendental for a quantity that also **cancels** in every
posterior the HMM takes (#241).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.special

__all__ = ["log_factorial", "nb_logpmf_1d"]

_LOG_FACTORIAL: dict[int, tuple[Any, np.ndarray]] = {}
"""`id(obs) -> (obs, lgamma(obs + 1))`, holding the array so the id stays valid."""


def log_factorial(obs: np.ndarray) -> np.ndarray:
    """`lgamma(obs + 1)`, computed once per observation array.

    Identity-keyed rather than value-keyed: hashing the values would cost an
    O(n) pass to save an O(n) pass, and `cnaster` scores the *same* array
    object against every state.
    """
    key = id(obs)
    cached = _LOG_FACTORIAL.get(key)

    if cached is not None and cached[0] is obs:
        return cached[1]

    values: np.ndarray = scipy.special.gammaln(np.asarray(obs, dtype=np.float64) + 1.0)
    _LOG_FACTORIAL[key] = (obs, values)

    return values


def nb_logpmf_1d(
    obs: np.ndarray,
    exposure: np.ndarray,
    mu: float,
    alpha: float,
    out: np.ndarray,
) -> None:
    """`hmm_nophasing._nb_logpmf_1d`, writing into the caller's buffer.

    Reproduces its two guards exactly, and they are the reason this is not a
    one-line expression:

    *   `lambda <= 0` writes **0.0**, not `-inf`. `cnaster` treats a
        zero-exposure bin as carrying no information rather than as
        impossible, which is a modelling choice and is reproduced.
    *   `nbinom_logpmf_numba` returns **0.0** when `p` leaves `(0, 1)`, `r` is
        non-positive or `k` is negative, for the same reason.

    Writing `-inf` for either would be the mathematically obvious thing and
    would change every downstream posterior.
    """
    counts = np.asarray(obs, dtype=np.float64)
    lam = np.asarray(exposure, dtype=np.float64) * mu

    r = 1.0 / max(alpha, 1.0e-10)

    # NB constant per state, and `cnaster` evaluates it per element.
    lgamma_r = float(scipy.special.gammaln(r))
    lgamma_k1 = log_factorial(obs)

    with np.errstate(divide="ignore", invalid="ignore"):
        p = 1.0 / (1.0 + alpha * lam)
        values = (
            scipy.special.gammaln(counts + r)
            - lgamma_r
            - lgamma_k1
            + r * np.log(p)
            + counts * np.log1p(-p)
        )

    # NB the guards, in the order `cnaster` applies them: the exposure test
    #    first, then the domain test inside the scalar kernel.
    usable = (lam > 0.0) & (p > 0.0) & (p < 1.0) & (counts >= 0.0) & (r > 0.0)

    out[:] = np.where(usable, values, 0.0)
