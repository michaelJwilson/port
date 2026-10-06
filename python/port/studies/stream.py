"""The harness the two solver streams share: pool, tuning rule, drain, redraw (T- #673 G5).

`potts_stream` (spatial solvers on Potts problems) and `copy_state_stream`
(copy-state starts on HMM problems) each draw realizations of one manifest,
tune their samplers on held-out realizations, run every job of a
realization in a pool, keep a pickled record current and redraw its figure
as each realization completes. What they solve differs; this is what they
did alike, written once:

- `pool`: a spawn-context `ProcessPoolExecutor` whose workers run the
  stream's initializer, which warms every solver up on a small problem so
  no compilation lands in a timed job;
- `cheapest`: the tuning rule, the cheapest setting within `tolerance` of the
  best median gap;
- `finished`: the jobs a wait returns, removed from the pending map;
- `redraw`: the stream's figure, redrawn by `run_study` in a child process.
"""

from __future__ import annotations

import multiprocessing as mp
import subprocess
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = ["cheapest", "finished", "pool", "redraw"]


def pool(workers: int, initializer: Callable[[], None]) -> ProcessPoolExecutor:
    """`workers` spawned processes, each running `initializer` once."""
    return ProcessPoolExecutor(
        workers, mp_context=mp.get_context("spawn"), initializer=initializer
    )


def cheapest(
    group: pd.DataFrame, tolerance: float
) -> tuple[Any, pd.Series, pd.DataFrame]:
    """The setting `key` whose median `gap` is within `tolerance` of the best, cheapest by median `seconds`.

    `group` holds one solver's runs with `key`, `gap` and `seconds` columns;
    returns the chosen key, its medians (`gap`, `seconds`), and every
    key's, so a caller can read an untuned default's.
    """
    by = group.groupby("key").agg(gap=("gap", "median"), seconds=("seconds", "median"))
    good = by[by.gap <= by.gap.min() + tolerance].sort_values("seconds")
    return good.index[0], good.iloc[0], by


def finished(
    futures: dict[Future[dict[str, Any]], int], block: bool
) -> list[tuple[int, dict[str, Any]]]:
    """The completed jobs as `(realization, row)`, popped from `futures`; waits for one if `block`."""
    done, _ = wait(futures, timeout=None if block else 0, return_when=FIRST_COMPLETED)
    return [(futures.pop(future), future.result()) for future in done]


def redraw(study: str, record: Path, merge: Sequence[Path] = ()) -> None:
    """`run_study --<study> RECORD [MERGE ...]` in a child; a failed figure does not stop the stream."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "port.scripts.run_study",
            f"--{study}",
            str(record),
            *map(str, merge),
        ],
        check=False,
    )
