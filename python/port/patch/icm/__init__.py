"""`cnaster.icm`'s solver, its reduced interface, and the choice between them (#250).

`interface` is `icm_sweep_deque` reduced to the problem it solves (#206),
`alpha_expansion` is upstream's solver behind that same signature (#246), and
`label_solver` is which of the two the call site takes. Three modules, one
`cnaster` module, one package name.

The submodules keep the split; this re-exports them so a swap row can name
`port.patch.icm` and a reader can open `cnaster.icm` and find it.
"""

from __future__ import annotations

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
from port.patch.icm.label_solver import (
    SOLVERS,
    label_solver,
    set_label_solver,
)

__all__ = [
    "SOLVERS",
    "CsrGraph",
    "IcmResult",
    "alpha_expansion_sweep",
    "fold_unary",
    "icm_sweep",
    "label_solver",
    "potts_energy",
    "potts_graph_from",
    "set_label_solver",
]
