"""Which solver `pipeline_clone_assignment` uses for the clone labelling.

**#246.** `cnaster` has one: `icm.icm_sweep_deque`, greedy single-site
descent. `snakes_and_ladders` has another that solves the same Potts MAP
problem with a proved bound. Both now sit behind one signature
(`port.patch.icm_interface.icm_sweep` and
`port.patch.alpha_expansion.alpha_expansion_sweep`), so the choice is a
setting rather than an edit.

**Process-wide rather than an argument**, because the call site is inside
`cnaster`'s pipeline and `port` reaches it by rebinding a name, not by
passing one. A parameter would have to be threaded through `cnaster` code
this repository does not own.

Default `icm`, so installing `port`'s patches does not silently change which
algorithm decides the clones -- that is a scientific choice and #246 is where
it is argued, not a side effect of a refactor.
"""

from __future__ import annotations

import os
from typing import Literal

__all__ = ["SOLVERS", "label_solver", "set_label_solver"]

Solver = Literal["icm", "alpha"]
SOLVERS: tuple[Solver, ...] = ("icm", "alpha")

ENVIRONMENT = "PORT_LABEL_SOLVER"
"""Read once per call, so a subprocess arm can select without a flag."""


class _Selection:
    """One mutable slot, so the module needs no `global` statement."""

    name: Solver = "icm"


_SELECTED = _Selection()


def _checked(name: str, source: str) -> Solver:
    if name not in SOLVERS:
        msg = (
            f"{source} is {name!r}, not one of {SOLVERS}; refusing rather "
            "than falling back, because a typo here would silently run the "
            "algorithm you were trying to replace"
        )
        raise ValueError(msg)

    return name


def set_label_solver(name: str) -> Solver:
    """Choose the solver, refusing a name that is not one."""
    _SELECTED.name = _checked(name, "the requested solver")

    return _SELECTED.name


def label_solver() -> Solver:
    """The selected solver, with the environment as an override.

    The environment is consulted on every call rather than at import, so a
    benchmark harness that sets it per subprocess does not depend on import
    order.
    """
    from_environment = os.environ.get(ENVIRONMENT)

    if from_environment:
        return _checked(from_environment, ENVIRONMENT)

    return _SELECTED.name
