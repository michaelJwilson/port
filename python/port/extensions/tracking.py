"""Record what `run_cnaster`'s optimizations did, through the seam sal carries.

**#251.** A `run_cnaster` is a sequence of optimizations -- seven
`scipy.optimize` sites and six discrete stages -- and none of them reports a
series. A fit that stalls, a labelling that collapses, an outer loop that
converges on iteration 2 because `tol` was met early all look identical from
outside until the run ends. #247 and #249 were both found by instrumenting one
call by hand, which is the argument for this module in one line.

`snakes_and_ladders.track` is the store, and nothing here reimplements it:
`track.Run` is a `runtime_checkable` Protocol written with `aim.Run`'s own
signatures, so `aim.Run` satisfies it structurally. What this module adds is
the two things sal cannot know: which `cnaster` stage a number belongs to, and
how to open a store from `port`'s entry point without making `aim` mandatory.

**Nothing is recorded outside a `track()` block.** `current()` returns the null
optimization, `record` returns on its first line, and a run without `--track`
is the run that existed before this module -- which is what
`tests/test_patched_entry_point.py`'s bitwise comparison asserts and what the
`is_null` guard here is for: the diagnostics are not even assembled.

`aim` is an optional extra (`pyproject.toml`, the `track` extra) and is
imported **inside** `tracking`, never at module scope, so an install without
it imports this module and runs every path but the one that opens a store.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from snakes_and_ladders.track import Run, TrackedOptimization, current, track

__all__ = ["record_label_sweep", "tracking"]

MISSING_AIM = (
    "--track needs the `track` extra, which is not installed. "
    "`uv sync --locked --extra track` installs it; see README.md for what it "
    "carries."
)
"""What a run refuses with when `--track` is asked for without `aim`.

A refusal rather than a silent fall back to no tracking: a user who asked to
record a run and got an unrecorded one has a measurement they cannot tell from
a recorded one, which is worse than a run that did not start.
"""


@contextmanager
def tracking(repo: str) -> Iterator[Run]:
    """Open an Aim run at `repo` and bind it for the block.

    Parameters
    ----------
    repo : str
        Where Aim keeps its store, its own `Run(repo=...)` argument. A
        directory that does not exist is created by Aim.

    Yields
    ------
    Run
        The bound store, so a caller can stamp run-level parameters on it.

    Raises
    ------
    SystemExit
        If `aim` is not installed. `MISSING_AIM` says how to get it.
    """
    try:
        from aim import Run as AimRun
    except ModuleNotFoundError as missing:
        raise SystemExit(MISSING_AIM) from missing

    run: Run = AimRun(repo=repo)

    try:
        with track(run):
            yield run
    finally:
        run.close()


def record_label_sweep(
    step: int,
    *,
    solver: str,
    cost: float,
    niter: int,
    n_clones: int,
    n_spots: int,
    tracked: TrackedOptimization | None = None,
) -> bool:
    """Record one clone-assignment sweep, and say whether anything was recorded.

    The clone labelling is the stage `port` already replaces, so it is the one
    site that needs no new swap to instrument -- which is why #251 starts here
    rather than with the seven continuous fits.

    `cost` is `cnaster`'s own objective, which it **maximizes**, and it goes in
    under Aim's `objective` name unnegated. The name is sal's and says nothing
    about the sign; negating it here would make a series that reads as
    improving while the fit gets worse, which is the defect #249 cost a day to.

    Parameters
    ----------
    step : int
        Which visit to this stage the numbers belong to, counted by the caller.
        A run reaches clone assignment several times and Aim needs the series
        ordered.
    solver : str
        `icm` or `alpha`, recorded as Aim context rather than as a number: two
        solvers in one run are two series, not one series with a jump in it.
    cost, niter, n_clones, n_spots
        What the sweep returned and what it ran on.
    tracked : TrackedOptimization | None
        The bound optimization, or `None` to look it up. Passed in only by
        tests, which need a store they can read back.

    Returns
    -------
    bool
        Whether a store was bound. `False` is the ordinary case -- a run
        without `--track` -- and the caller assembles nothing for it.
    """
    optimization = current() if tracked is None else tracked

    if optimization.is_null:
        return False

    context: Mapping[str, Any] = {"stage": "clone_assignment", "solver": solver}

    optimization.record(
        step,
        objective=float(cost),
        context=context,
        niter=float(niter),
        n_clones=float(n_clones),
        n_spots=float(n_spots),
    )

    return True
