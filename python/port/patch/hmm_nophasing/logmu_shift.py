"""`cnaster.hmm_nophasing.compute_logmu_shifts` as a per-clone reduction (#234), and `clone_log_normalizers`.

`shifts` returns one value per clone where upstream returns one per segment;
`np.repeat(shifts(...), clone_lengths)` is upstream's array, bitwise. Not
installed: upstream's only call is commented out (`hmm_nophasing.py:279`).
Unequal clone lengths are admitted.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numba import njit

__all__ = ["clone_log_normalizers", "shifts"]


@njit(nogil=True, cache=True, parallel=False, error_model="numpy")
def _per_clone(means, states, lambdas, lengths):
    """Upstream's two-pass logsumexp, writing one value per clone rather than per segment."""
    n_clones = lengths.size
    out = np.empty(n_clones, dtype=np.float64)

    start = 0

    for clone in range(n_clones):
        length = lengths[clone]
        largest = -np.inf

        for i in range(length):
            value = means[states[start + i]] + lambdas[start + i]

            # NB `max` rather than the branch PLR1730 asks for, under `numba`.
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
    """Per-clone `logsumexp` of `log_mus[copy_states] + normal_log_lambda`, `(n_clones,)`.

    `copy_states` and `normal_log_lambda` have one entry per segment, clones
    concatenated; `clone_lengths` sums to the segment count. Upstream returns
    `(n_segments,)`. An all-`-inf` clone gives `-inf`, as upstream.
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


def clone_log_normalizers(
    log_mu: np.ndarray, paths: np.ndarray, single_base_nb_mean: np.ndarray
) -> np.ndarray | None:
    """`log Z_c = log sum_g lambda_g mu_{s_c(g)}`, one per column of `paths` (#517).

    `lambda` is the normalized spot-summed baseline (`hmrf.py:476`); `paths`
    is `(n_obs, n_clones)` state indices into `log_mu`. `None` where the
    baseline has no mass.
    """
    import scipy.special

    profile = np.asarray(single_base_nb_mean, dtype=np.float64).sum(axis=1)
    total = profile.sum()

    if total <= 0.0:
        return None

    with np.errstate(divide="ignore"):
        log_lambda = np.log(profile / total)

    terms = np.asarray(log_mu)[np.asarray(paths, dtype=np.int64)] + log_lambda[:, None]
    normalizers: np.ndarray = scipy.special.logsumexp(terms, axis=0)

    return normalizers
