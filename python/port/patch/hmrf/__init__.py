"""Patches to `cnaster.hmrf`, re-exported under its name (#250).

Submodules: spot-clone field (#59), fused two-pass build, per-iteration
invariants, COO adjacency, and clone assignment (#206).
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
from port.patch.hmrf.core_inference import (
    pin_neutral,
    reindex_clones,
    run_core_inference,
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
    "pin_neutral",
    "pipeline_clone_assignment",
    "reindex_clones",
    "run_core_inference",
]
