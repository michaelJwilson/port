"""`cnaster.hmm_nophasing.compute_logmu_shifts`, one value per clone (#234).

**`port.patch.hmm_nophasing.logmu_shift` (#234), moved in by T- #670 PR5**,
for the shifted emission in `cnamaste.hmm_nophasing`. `port`'s
`clone_log_normalizers`, which the clone assignment and the genomic figure
read, stays behind until the stage that calls it moves.

`compute_logmu_shifts` computes a per-clone `logsumexp` of
`log_mu[state] + normal_log_lambda` and writes it across every one of the
clone's segments. :func:`shifts` returns one value per clone instead:
`np.repeat(shifts(...), clone_lengths)` is upstream's array exactly, and a
per-clone shape cannot be indexed by clone where a per-segment one hands
every clone the first clone's shift.

**Referee.** `port`'s `shifts`, bitwise (`tests/test_cnamaste_hmm.py`).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numba import njit

__all__ = ["shifts"]


@njit(nogil=True, cache=True, parallel=False, error_model="numpy")
def _per_clone(
    means: np.ndarray, states: np.ndarray, lambdas: np.ndarray, lengths: np.ndarray
) -> np.ndarray:
    """Upstream's two passes, writing one value per clone rather than per segment."""
    n_clones = lengths.size
    out = np.empty(n_clones, dtype=np.float64)

    start = 0

    for clone in range(n_clones):
        length = lengths[clone]
        largest = -np.inf

        for i in range(length):
            value = means[states[start + i]] + lambdas[start + i]
            largest = max(largest, value)

        if np.isinf(largest):
            out[clone] = largest
        else:
            total = 0.0

            for i in range(length):
                total += np.exp(means[states[start + i]] + lambdas[start + i] - largest)

            out[clone] = largest + np.log(total)

        start += length

    return out


def shifts(
    log_mus: np.ndarray,
    copy_states: np.ndarray,
    normal_log_lambda: np.ndarray,
    clone_lengths: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Per-clone `logsumexp` of `log_mus[state] + normal_log_lambda`, `(n_clones,)`.

    A clone whose every term is `-inf` returns `-inf`, as upstream's loop
    does. Lambdas or lengths that do not cover the states are refused.
    """
    states = np.asarray(copy_states, dtype=np.int64).reshape(-1)
    lambdas = np.asarray(normal_log_lambda, dtype=np.float64).reshape(-1)
    means = np.asarray(log_mus, dtype=np.float64).reshape(-1)
    lengths = np.asarray(clone_lengths, dtype=np.int64).reshape(-1)

    n_segments = states.size

    if lambdas.size != n_segments:
        msg = f"{lambdas.size} lambdas for {n_segments} segments"
        raise ValueError(msg)

    if int(lengths.sum()) != n_segments:
        msg = f"clone lengths sum to {int(lengths.sum())}, not {n_segments}"
        raise ValueError(msg)

    reduced: np.ndarray = _per_clone(means, states, lambdas, lengths)

    return reduced
