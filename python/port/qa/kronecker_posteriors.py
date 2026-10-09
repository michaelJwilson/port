"""The phased HMM's state posteriors from sal's ragged kernel, Kronecker switch (#426).

Replaces `cnaster.hmm_phased`'s `2 K`-state lattices with sal #1133's factored
step. Index correspondence:

| `cnaster` | sal |
| --- | --- |
| state `a * K + i` (phase blocks) | state `2 i + a` (`np.kron`) |
| `log_sitewise_transmat[t - 1]` into `t` | `switch[t]`; a segment's first entry unread |
| `log(1/2) + [start, start]` | `log_initial` over `2 K` |
| `PEANLIZE_PHASE_ONLY_ON_SAME_CNV = False` | `SwitchKind.KRONECKER` |
| `... = True` | `SwitchKind.KRONECKER_DIAGONAL` |
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

    `log_emission` is `(2 K, n_obs, n_spots)`, summed over spots.
    `diagonal` is `PEANLIZE_PHASE_ONLY_ON_SAME_CNV`.
    """
    from sal.likelihood.ragged import SwitchKind, kronecker_order, posteriors
    from sal.ragged import Ragged

    n_paired = log_emission.shape[0]

    # NB `cnaster`'s state `a * K + i` is sal's `2 i + a` (sal #1144).
    order = kronecker_order(n_paired, layer_major=True)
    density = np.sum(log_emission, axis=2).T[:, order]
    initial = (np.log(0.5) + np.concatenate([log_startprob, log_startprob]))[order]

    # NB the step into `t` reads `cnaster`'s entry `t - 1`; a segment's first is unread.
    sitewise = np.exp(np.asarray(log_sitewise_transmat, dtype=np.float64))
    switch = np.concatenate([[0.5], sitewise[:-1]])

    # NB sal scores one-position segments too (sal #1233).
    lengths_array = np.asarray(lengths, dtype=np.int64)
    result = posteriors(
        Ragged(density, tuple(int(n) for n in lengths_array)),
        initial,
        np.asarray(log_transmat, dtype=np.float64),
        switch=switch,
        switch_kind=(
            SwitchKind.KRONECKER_DIAGONAL if diagonal else SwitchKind.KRONECKER
        ),
    )
    log_posterior = np.asarray(result.log_posterior)

    log_gamma: np.ndarray = log_posterior[:, np.argsort(order)].T
    return log_gamma
