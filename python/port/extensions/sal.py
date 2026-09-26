"""`--sal`: `snakes_and_ladders` routines where `port` measured a gain (#312).

A row is admitted on `port`'s own measurement and on nothing else -- at
least 2x runtime at a stress size, a stated memory reduction, or a lower
Potts energy or better recovery against a planted truth -- with `cnaster`'s
path kept as the oracle. A row that stops clearing its bar is removed, and
the test that pinned it records the number that removed it.

**Off by default.** No row reproduces `cnaster`, which is what `SWAPS`
promises; the flag is how a run opts in, and `--list` prints what it adds.

The rows are settings rather than name rebinds: the clone labelling is
chosen through `port.extensions.label_solver`, which `port`'s
`pipeline_clone_assignment` (a `SWAPS` row) reads. So `--sal` needs that
swap installed, and `run_cnaster_port --no-patch --sal` installs it alone.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass

__all__ = ["SAL_ROWS", "SalRow", "sal"]


@dataclass(frozen=True)
class SalRow:
    """One `cnaster` stage, what replaces it, and the measurement behind it."""

    stage: str
    cnaster: str
    sal: str
    axis: str
    evidence: str
    solver: str
    honours_shift: bool = True
    ticket: int = 312


SAL_ROWS: tuple[SalRow, ...] = (
    SalRow(
        stage="clone labelling",
        cnaster="cnaster.icm.icm_sweep_deque",
        sal="search.alpha_expansion.alpha_expansion, Backend.RUST, then cnaster's ICM",
        axis="accuracy",
        evidence=(
            "dev instance, three runs each: clone ARI 0.919 -> 1.000 against "
            "the planted labels, wall 40.5 -> 28.8 s. Alpha expansion alone "
            "lowers each call's Potts energy by 12 to 7,954 nats but takes the "
            "ARI to 0.386: cnaster's 200-spot floor, which the ICM that "
            "follows applies, is load-bearing"
        ),
        solver="alpha-rust-icm",
    ),
)
"""The admitted rows. Measured and not admitted, selectable through
`PORT_LABEL_SOLVER`:

- `alpha-rust`: the lowest energy per call, and ARI 0.386 end to end;
- `icm-numba`: the same descent 39 to 61 times faster at stress, within
  31 nats either way per call, and ARI 0.827 end to end, having no floor.

Both wait on #312's R7: a label merge in `snakes_and_ladders`."""


@contextlib.contextmanager
def sal(rows: tuple[SalRow, ...] = SAL_ROWS, *, shift: bool = False) -> Iterator[None]:
    """Install the rows for the block, and restore the solver on the way out.

    Refuses, naming the row, when `shift` is on and a row cannot honour it,
    rather than fitting unshifted without saying so (#312, R8).
    """
    from port.extensions.label_solver import _SELECTED, set_label_solver

    refused = [row.stage for row in rows if shift and not row.honours_shift]

    if refused:
        msg = f"--sal cannot honour --shift in: {', '.join(refused)}"
        raise ValueError(msg)

    previous = _SELECTED.name

    try:
        for row in rows:
            set_label_solver(row.solver)

        yield
    finally:
        set_label_solver(previous)
