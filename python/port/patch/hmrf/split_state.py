"""One unbalanced copy state split by depth, refitted on fixed clones (#471).

The read-depth + BAF HMM has `hmm.n_states` states. On `dev_tree` r0 it
spends them on the balanced ladder and fits a one-copy loss and a
copy-neutral LOH, which share a BAF, as one state at the LOH's depth. The
integer decode then cannot tell them apart, and each bin of the pair decodes
to the same `(A, B)`.

`split_init` reads that from the fit. For each unbalanced state
(`|p - 0.5| >= 0.1`) it takes the bins decoded to it, per clone, and their
log depth ratio against the baseline, pooled over `POOL` bins. A
two-means cut of those ratios farther apart than `MIN_GAP` makes the state a
candidate. It returns the rates and BAFs to refit from:

- the candidate with the largest `gap x min(cluster size)` keeps its BAF and
  takes the upper depth;
- the closest remaining pair of states, by `(dmu / 0.3, dp / 0.1)`, gives up
  its less occupied member, which takes the lower depth and the same BAF.

`run_core_inference` then refits once, with `max_iter_outer = 0`, on the
clones the first fit returned, and keeps that fit's assignment. **The clones
are the first fit's, spot for spot.** Letting the refit reassign them moved
CalicoST hard from clone ARI 0.982 to 0.380.

Off by default: `split_state()` installs it for a block. It needs the
shifted fit's `run_core_inference` row (`SHIFT_SWAPS`), which is the only one
that reaches it.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np

__all__ = [
    "MIN_GAP",
    "POOL",
    "installed",
    "pooled",
    "split_init",
    "split_state",
]

_INSTALLED: list[bool] = [False]

POOL = 5
"""Bins pooled per ratio, centred, within a chromosome arm segment."""

MIN_GAP = 0.3
"""Least log-ratio gap between the two depth clusters of one state."""

UNBALANCED = 0.1
"""`|p - 0.5|` at or over which a state is unbalanced; only those split."""

MIN_BINS = 20
"""Fewest decoded bins a state needs before its depths are clustered."""


def installed() -> bool:
    """Whether :func:`split_state` is active."""
    return _INSTALLED[0]


@contextmanager
def split_state() -> Iterator[None]:
    """Split one unbalanced state by depth after the RDR + BAF fit, for the block."""
    previous = _INSTALLED[0]
    _INSTALLED[0] = True

    try:
        yield
    finally:
        _INSTALLED[0] = previous


def pooled(
    values: np.ndarray, lengths: Sequence[int], window: int = POOL
) -> np.ndarray:
    """Centred sums of `window` bins, truncated at each segment's ends."""
    out = np.empty(values.shape, dtype=np.float64)
    start = 0

    for length in lengths:
        n = int(length)
        segment = np.asarray(values[start : start + n], dtype=np.float64)
        cumulative = np.concatenate([[0.0], np.cumsum(segment)])
        index = np.arange(n)
        low = np.clip(index - window // 2, 0, n)
        high = np.clip(index + window // 2 + 1, 0, n)
        out[start : start + n] = cumulative[high] - cumulative[low]
        start += n

    return out


def _two_means(values: np.ndarray) -> tuple[float, float, int, int]:
    """1-d two-means from the quartiles: centres and cluster sizes."""
    low, high = np.percentile(values, 25), np.percentile(values, 75)
    lower = values <= low

    for _ in range(50):
        lower = np.abs(values - low) <= np.abs(values - high)

        if lower.all() or not lower.any():
            break

        low, high = values[lower].mean(), values[~lower].mean()

    return float(low), float(high), int(lower.sum()), int((~lower).sum())


def split_init(
    res: Any,
    single_X: np.ndarray,
    lengths: Sequence[int],
    single_base_nb_mean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    """Rates and BAFs to refit from, with one state split; `None` if none qualifies.

    Returns `(log_mu, p_binom, split, freed)`, each array `(n_states, 1)`.
    """
    log_mu = np.asarray(res["new_log_mu"], dtype=np.float64).reshape(-1)
    p_binom = np.asarray(res["new_p_binom"], dtype=np.float64).reshape(-1)
    n_states = log_mu.size
    path = np.asarray(res["pred_cnv"]) % n_states
    assignment = np.asarray(res["new_assignment"])
    labels = np.unique(assignment)

    if path.ndim != 2 or labels.size != path.shape[1]:
        return None

    X = np.asarray(single_X)
    base = np.asarray(single_base_nb_mean)
    ratio = np.full(path.shape, np.nan)

    for column, label in enumerate(labels):
        spots = assignment == label
        depth = pooled(X[:, 0, spots].sum(axis=1), lengths)
        expected = pooled(base[:, spots].sum(axis=1), lengths)
        ok = (depth > 0) & (expected > 0)
        ratio[ok, column] = np.log(depth[ok] / expected[ok])

    ratio -= np.nanmedian(ratio)
    best: tuple[float, int, float, float, float] | None = None

    for state in range(n_states):
        if abs(p_binom[state] - 0.5) < UNBALANCED:
            continue

        values = ratio[path == state]
        values = values[np.isfinite(values)]

        if values.size < MIN_BINS:
            continue

        low, high, n_low, n_high = _two_means(values)
        score = (high - low) * min(n_low, n_high)

        if high - low > MIN_GAP and (best is None or score > best[0]):
            best = (score, state, low, high, float(np.median(values)))

    if best is None:
        return None

    _, state, low, high, median = best
    occupancy = np.bincount(path.ravel(), minlength=n_states)

    def distance(i: int, j: int) -> float:
        return float(
            np.hypot((log_mu[i] - log_mu[j]) / 0.3, (p_binom[i] - p_binom[j]) / 0.1)
        )

    pairs = [
        (distance(i, j), i, j)
        for i in range(n_states)
        for j in range(i + 1, n_states)
        if state not in (i, j)
    ]

    if not pairs:
        return None

    _, i, j = min(pairs)
    freed = i if occupancy[i] < occupancy[j] else j

    new_mu, new_p = log_mu.copy(), p_binom.copy()
    new_mu[state] = log_mu[state] + high - median
    new_mu[freed] = log_mu[state] + low - median
    new_p[freed] = p_binom[state]

    return (
        new_mu[:, None],
        np.clip(new_p, 1e-3, 1 - 1e-3)[:, None],
        state,
        freed,
    )
