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
        sal=(
            "search.alpha_expansion.alpha_expansion, Backend.RUST, then "
            "search.icm.merge_small_labels at cnaster's floor"
        ),
        axis="accuracy",
        evidence=(
            "dev instance: clone ARI 0.9795 -> 1.000 against the planted "
            "labels, wall 28.4 -> 24.8 s; lattice instance 0.9946 -> 1.000, "
            "27.0 -> 25.7 s, against the row it replaced (expansion then "
            "cnaster's ICM, #312: 0.919 -> 1.000 over the default). The floor "
            "is load-bearing -- alpha expansion alone reaches ARI 0.386 -- "
            "and is now sal's (#1114), so no cnaster ICM runs under --sal"
        ),
        solver="alpha-rust-merge",
        ticket=410,
    ),
)
"""The admitted rows. Measured and not admitted, selectable through
`PORT_LABEL_SOLVER`:

- `alpha-rust`: the lowest energy per call, and ARI 0.386 end to end;
- `icm-numba`: the same descent 39 to 61 times faster at stress, within
  31 nats either way per call, and ARI 0.827 end to end, having no floor;
- `icm-numba-floor`: that descent with sal's floor (#1114), ARI 0.3385 and
  two clones for four on the dev instance -- the index-order descent from
  the RDR stage's start dissolves clones the expansion keeps;
- `alpha-rust-icm`: the row `alpha-rust-merge` replaced;
- `icm-argmax-floor` (#410, sal #1121): the `numba` descent from the field's
  argmax with the floor. Dev instance: clone ARI 1.000, copy ARI 0.9971,
  wall 22.5 s end to end. Per call at 10,000 spots and ten clones, 0.034 s
  against the admitted row's 0.124 s under #421 -- 3.7x -- but at a Potts
  energy 295 nats higher (-19,335 against -19,630); at 1,600 spots and four
  clones, 0.0049 against 0.0102 s and 54 nats higher. Not admitted: the
  speed is bought with a worse minimum, and the label step is 2.3 s of the
  run.

#312's R7, the label merge in sal, landed as `merge_small_labels`."""


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
