"""`--sal`: opt-in `snakes_and_ladders` routines where `port` measured a gain (#312).

Off by default, since no row reproduces `cnaster`. Rows are settings read by
`pipeline_clone_assignment` (a `SWAPS` row), not name rebinds.
"""

from __future__ import annotations

from typing import Any, NamedTuple

__all__ = ["SAL_ROWS", "SalRow", "sal_options"]


class SalRow(NamedTuple):
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
            "search.alpha_expansion.fuse of alpha_expansion (Backend.RUST) and "
            "the argmax ICM, then search.icm.merge_small_labels at cnaster's floor"
        ),
        axis="accuracy",
        evidence=(
            "dev instance: clone ARI 0.9795 -> 1.000 against the planted "
            "labels, wall 28.4 -> 24.8 s; lattice instance 0.9946 -> 1.000, "
            "27.0 -> 25.7 s, against the row it replaced (expansion then "
            "cnaster's ICM, #312: 0.919 -> 1.000 over the default). The floor "
            "is load-bearing -- alpha expansion alone reaches ARI 0.386 -- "
            "and is now sal's (#1114), so no cnaster ICM runs under --sal. "
            "The fusion (#1125) of the expansion with the argmax descent keeps "
            "ARI 1.000 on both and lowers the Potts energy at 10,000 spots and "
            "ten clones by 50 nats, 14.6 from TRW-S's lower bound against the "
            "expansion's 64.6, at 0.100 s against 0.084 s per call"
        ),
        solver="alpha-rust-fuse-merge",
        ticket=410,
    ),
)
"""The admitted rows; measured alternatives are in `port.sandbox.extensions.label_solvers` (#749)."""


def sal_options(
    rows: tuple[SalRow, ...] = SAL_ROWS, *, shift: bool = False
) -> dict[str, Any]:
    """The options `--sal` binds into `pipeline_clone_assignment` (#517).

    Refuses, naming the row, when `shift` is on and a row cannot honour it,
    rather than fitting unshifted without saying so (#312, R8).
    """
    refused = [row.stage for row in rows if shift and not row.honours_shift]

    if refused:
        msg = f"--sal cannot honour --shift in: {', '.join(refused)}"
        raise ValueError(msg)

    # NB every row selects the labelling; the last one's stands, as when
    #    each was set in turn.
    return {"label_solver": rows[-1].solver} if rows else {}
