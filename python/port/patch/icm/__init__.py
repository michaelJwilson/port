"""Re-exports `cnaster.icm`'s solver patches and the choice between them (#250).

`interface` is `icm_sweep_deque` reduced to its problem (#206); `alpha_expansion`
is the same signature's alternative solver (#246); `label_solver` (in
`port.extensions`, #281) chooses between them.
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
