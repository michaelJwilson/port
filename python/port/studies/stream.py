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
- `halve`: which settings a first, one-start-per-realization round sends on
  to the full count of starts (successive halving, #716);
- `finished`: the jobs a wait returns, removed from the pending map;
- `redraw`: the stream's figure, redrawn by `run_study` in a child process.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import subprocess
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = ["cheapest", "finished", "halve", "merge_settings", "pool", "redraw"]


def pool(
    workers: int,
    initializer: Callable[..., None],
    *initargs: Any,
    start_method: str = "spawn",
) -> ProcessPoolExecutor:
    """`workers` processes, started by `start_method`, each running `initializer(*initargs)` once.

    `fork` is for a study that hands its workers what it already holds rather
    than pickling it to each (`copy_start_arms`); every other study spawns.
    """
    return ProcessPoolExecutor(
        workers,
        mp_context=mp.get_context(start_method),
        initializer=initializer,
        initargs=initargs,
    )


def merge_settings(path: Path, provenance: str, chosen: dict[str, Any]) -> None:
    """Write `chosen` over `path`'s tuned settings, with how they were chosen (#749 WP6).

    Keys `chosen` does not name keep their earlier values and their places;
    `_provenance` is replaced. The one merge both streams' `retune` wrote out.
    """
    import json

    earlier = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(
        json.dumps({**earlier, "_provenance": provenance, **chosen}, indent=2) + "\n"
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


def halve(group: pd.DataFrame, tolerance: float, keep: float = 1 / 3) -> set[Any]:
    """The keys of one solver's first round that go on: every key within `tolerance` of the best median `gap`, and the best `keep` of all.

    A first round of one start per realization ranks the settings; only these
    get the rest of the starts, so `cheapest` chooses among settings measured
    at the full count and the rest cost one start each.
    """
    by = group.groupby("key").gap.median().sort_values()
    best = set(by.index[: max(1, math.ceil(len(by) * keep))])
    return best | set(by.index[by <= by.min() + tolerance])


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
