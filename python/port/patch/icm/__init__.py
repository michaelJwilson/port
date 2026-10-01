"""`cnaster.icm`'s solver, its reduced interface, and the choice between them (#250).

`interface` is `icm_sweep_deque` reduced to the problem it solves (#206),
`alpha_expansion` is upstream's solver behind that same signature (#246), and
`label_solver` is which of the two the call site takes; it lives in
`port.extensions` and is re-exported here, because choosing between two
solvers replaces no `cnaster` function -- it is a setting, and #274's
four-job rule puts what has no counterpart under `extensions/` (#281).
Two modules here, one
`cnaster` module, one package name.

The submodules keep the split; this re-exports them so a swap row can name
`port.patch.icm` and a reader can open `cnaster.icm` and find it.
"""

from __future__ import annotations

from port.extensions.label_solver import SOLVERS, solver_for
from port.patch.icm.alpha_expansion import (
    alpha_expansion_sweep,
    potts_energy,
    potts_graph_from,
)
from port.patch.icm.interface import (
    CsrGraph,
    IcmResult,
    fold_unary,
    icm_sweep,
)

__all__ = [
    "SOLVERS",
    "CsrGraph",
    "IcmResult",
    "alpha_expansion_sweep",
    "fold_unary",
    "icm_sweep",
    "potts_energy",
    "potts_graph_from",
    "solver_for",
]
