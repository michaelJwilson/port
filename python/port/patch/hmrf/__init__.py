"""Five patches to one `cnaster` module, so the package carries its name (#250).

`cnaster.hmrf` is one file doing five separable jobs, and `port` replaced
them one ticket at a time: the spot-clone field (#59), the fused two-pass
build, the per-iteration invariants, the COO round trip, and the clone
assignment itself (#206). Splitting them is right; naming the split after the
tickets was not, because a reader holding `cnaster/hmrf.py` open had no way
to find them.

The submodules keep the split; this re-exports them so a swap row can name
`port.patch.hmrf` and a reader can open `cnaster.hmrf` and find it.
"""

from __future__ import annotations

from port.patch.hmrf.adjacency import (
    adjacency_coo,
)
from port.patch.hmrf.clone_assignment import (
    UPSTREAM,
    boundary,
    pipeline_clone_assignment,
)
from port.patch.hmrf.field import (
    compute_loglike_spot_assignment_strided,
)
from port.patch.hmrf.fused_field import (
    fused_spot_clone_field,
)
from port.patch.hmrf.invariants import (
    BoundaryInvariants,
    boundary_invariants,
)

__all__ = [
    "UPSTREAM",
    "BoundaryInvariants",
    "adjacency_coo",
    "boundary",
    "boundary_invariants",
    "compute_loglike_spot_assignment_strided",
    "fused_spot_clone_field",
    "pipeline_clone_assignment",
]
