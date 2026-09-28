"""The phased HMM's state posteriors from sal's ragged kernel, Kronecker switch (#426).

`cnaster.hmm_phased` runs the forward and backward lattices over `2 K`
states -- `K` copy states times two phases -- with the step into position
`t` the slow chain `T` crossed with a phase switch `S_t`, then normalizes
`log_alpha + log_beta`. sal #1133's ragged kernel takes that step in its
factors (`2 K^2 + 4 K` terms against `4 K^2`) and returns the normalized
marginals directly.

Correspondence, stated because every index moves:

| `cnaster` | sal |
| --- | --- |
| state `a * K + i` (phase blocks) | state `2 i + a` (`np.kron`) |
| `log_sitewise_transmat[t - 1]` into `t` | `switch[t]`; a segment's first entry unread |
| `log(1/2) + [start, start]` | `log_initial` over `2 K` |
| `PEANLIZE_PHASE_ONLY_ON_SAME_CNV = False`: `S_t` on every block | `SwitchKind.KRONECKER` |
| `... = True`: `S_t` on the diagonal blocks, one half elsewhere | `SwitchKind.KRONECKER_DIAGONAL` |
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["kronecker_state_posteriors"]


def kronecker_state_posteriors(
    lengths: Any,
    log_transmat: np.ndarray,
    log_startprob: np.ndarray,
    log_emission: np.ndarray,
    log_sitewise_transmat: np.ndarray,
    *,
    diagonal: bool = False,
) -> np.ndarray:
    """`hmm_phased.get_state_posteriors`'s `(2 K, n_obs)` log marginals, from sal.

    `log_emission` is `cnaster`'s `(2 K, n_obs, n_spots)`, summed over spots
    as the lattices sum it. `diagonal` is `PEANLIZE_PHASE_ONLY_ON_SAME_CNV`.
    """
    from sal.likelihood.ragged import SwitchKind, posteriors
    from sal.ragged import Ragged
    from scipy.special import logsumexp

    n_paired, n_obs = log_emission.shape[0], log_emission.shape[1]
    n_states = n_paired // 2

    # NB `cnaster`'s state `a * K + i` is sal's `2 i + a`.
    ours = (2 * np.arange(n_states)[None, :] + np.arange(2)[:, None]).reshape(-1)
    density = np.empty((n_obs, n_paired))
    density[:, ours] = np.sum(log_emission, axis=2).T

    initial = np.empty(n_paired)
    initial[ours] = np.log(0.5) + np.concatenate([log_startprob, log_startprob])

    # NB the step into `t` reads `cnaster`'s entry `t - 1`; a segment's first
    #    position is never read, so its value only has to be a probability.
    sitewise = np.exp(np.asarray(log_sitewise_transmat, dtype=np.float64))
    switch = np.concatenate([[0.5], sitewise[:-1]])

    # NB sal refuses a one-position segment (sal #666): it has no transition.
    #    Its posterior is the start times the emission, normalized, which is
    #    what `cnaster`'s lattices give it, so it is computed here.
    lengths_array = np.asarray(lengths, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths_array)[:-1]])
    single = lengths_array == 1
    log_posterior = np.empty_like(density)

    for start in starts[single]:
        joint = initial + density[start]
        log_posterior[start] = joint - logsumexp(joint)

    if (~single).any():
        kept = np.concatenate(
            [
                np.arange(s, s + n)
                for s, n in zip(starts[~single], lengths_array[~single], strict=True)
            ]
        )
        result = posteriors(
            Ragged(density[kept], tuple(int(n) for n in lengths_array[~single])),
            initial,
            np.asarray(log_transmat, dtype=np.float64),
            switch=switch[kept],
            switch_kind=(
                SwitchKind.KRONECKER_DIAGONAL if diagonal else SwitchKind.KRONECKER
            ),
        )
        log_posterior[kept] = np.asarray(result.log_posterior)

    log_gamma: np.ndarray = log_posterior[:, ours].T
    return log_gamma
