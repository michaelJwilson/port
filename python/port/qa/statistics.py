"""Performance statistics: bars, ranks, bootstrap intervals, wall time and peak memory.

One home for what the audits, benchmarks and studies each computed for
themselves (T- #673 G1): the study figures' median-and-quantile bars and
ranks, the population study's cluster bootstrap, and the wall seconds and
peak resident memory every audit reports.
"""

# ruff: noqa: A005 -- T- #673 names the module; it imports the standard library's absolutely
from __future__ import annotations

import resource
import statistics
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import pandas as pd

__all__ = [
    "Measured",
    "bars",
    "bootstrap_interval",
    "measured",
    "median_wall",
    "peak_gb",
    "ranks",
    "resample_weights",
]

T = TypeVar("T")


def bars(values: pd.Series) -> tuple[float, list[list[float]]]:
    """The median and its distances to the 10% and 90% quantiles, as `errorbar` takes them."""
    m = float(values.median())
    return m, [[m - float(values.quantile(0.1))], [float(values.quantile(0.9)) - m]]


def ranks(values: dict[str, float]) -> dict[str, int]:
    """1 for the lowest value, ties sharing the lower rank; NaN unranked."""
    order = pd.Series(values, dtype=float).dropna().rank(method="min")
    return {str(k): int(v) for k, v in order.items()}


def resample_weights(
    n_members: int, draws: int, rng: np.random.Generator
) -> np.ndarray:
    """`(draws, n_members)`: how often each member is drawn in each resample."""
    picks = rng.integers(0, n_members, (draws, n_members))
    return np.stack([np.bincount(p, minlength=n_members) for p in picks])


def bootstrap_interval(
    sums: np.ndarray, counts: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The 2.5% and 97.5% percentiles of a ratio of sums over the resamples `weights` draws.

    `sums` and `counts` hold one row per member (`(n_members,)` or
    `(n_members, n_bins)`); a resample weights each member's row by how often
    it was drawn, so a member's items enter together: the cluster bootstrap,
    computed without copying rows. A resample with no count in a bin is NaN
    and is left out.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        rates = (weights @ sums) / (weights @ counts)
        low, high = np.nanpercentile(rates, [2.5, 97.5], axis=0)
    return low, high


def peak_gb(children: bool = False) -> float:
    """The peak resident set so far, in GiB: this process's, or its waited-for children's.

    `ru_maxrss` is a high-water mark over the process's life, in KiB on
    Linux, so a reading bounds every block before it rather than one block.
    """
    who = resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF
    return resource.getrusage(who).ru_maxrss / 1024**2


@dataclass
class Measured:
    """What `measured` read when its block ended: wall seconds and `peak_gb`."""

    wall_s: float = float("nan")
    peak_gb: float = float("nan")


@contextmanager
def measured(children: bool = False) -> Iterator[Measured]:
    """Time the block by `time.perf_counter`, then read `peak_gb(children)`.

    The result is filled when the block exits, an exception included.
    """
    found = Measured()
    started = time.perf_counter()
    try:
        yield found
    finally:
        found.wall_s = time.perf_counter() - started
        found.peak_gb = peak_gb(children)


def median_wall(run: Callable[[], T], repeats: int) -> tuple[T, float, list[float]]:
    """`run` called `repeats` times: the last result, the median wall seconds, every wall."""
    if repeats < 1:
        msg = f"repeats must be at least 1, got {repeats}"
        raise ValueError(msg)
    walls: list[float] = []
    for _ in range(repeats):
        with measured() as m:
            result = run()
        walls.append(m.wall_s)
    return result, statistics.median(walls), walls
