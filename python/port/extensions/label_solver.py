"""Which solver `pipeline_clone_assignment` uses for the clone labelling (#246).

All solvers share `icm_sweep`'s signature, so the choice is a setting. It is
process-wide because `port` reaches the call site by rebinding a name inside
`cnaster`. Default `icm` (`cnaster`'s), so installing patches changes no algorithm.
"""

from __future__ import annotations

import functools
import os
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from port.patch.icm.interface import IcmResult

__all__ = [
    "SOLVERS",
    "fusion_then_merge",
    "sal_icm_sweep",
    "solved_on_sal",
    "solver_for",
    "sweep_for",
]

Solver = Literal[
    "icm",
    "alpha",
    "alpha-rust",
    "icm-numba",
    "alpha-rust-fuse-merge",
]
SOLVERS: tuple[Solver, ...] = (
    "icm",
    "alpha",
    "alpha-rust",
    "icm-numba",
    "alpha-rust-fuse-merge",
)
"""`icm` is `cnaster`'s; `alpha`, `alpha-rust`, `icm-numba` are sal's (#246, #312);
`alpha-rust-fuse-merge` is `--sal`'s (#410). Others set aside (#749 WP5)."""

ENVIRONMENT = "PORT_LABEL_SOLVER"
"""Overrides the bound solver when set; read per call (departs from T- #617 rule 1)."""


def _checked(name: str, source: str) -> Solver:
    if name not in SOLVERS:
        msg = (
            f"{source} is {name!r}, not one of {SOLVERS}; refusing rather "
            "than falling back, because a typo here would silently run the "
            "algorithm you were trying to replace"
        )
        raise ValueError(msg)

    return name


def solver_for(requested: str) -> Solver:
    """The solver a call uses: `requested`, or the environment's override.

    Read per call, not at import; raises ValueError if either names no solver.
    """
    from_environment = os.environ.get(ENVIRONMENT)

    if from_environment:
        return _checked(from_environment, ENVIRONMENT)

    return _checked(requested, "the requested solver")


def sweep_for(name: Solver) -> Any:
    """The `icm_sweep`-signature solver a name selects."""
    from port.patch.icm import alpha_expansion as sal
    from port.patch.icm.interface import icm_sweep

    if name == "icm":
        return icm_sweep

    if name == "alpha":
        return sal.alpha_expansion_sweep

    if name == "alpha-rust":
        from sal.backend import Backend

        return functools.partial(sal.alpha_expansion_sweep, backend=Backend.RUST)

    if name == "alpha-rust-fuse-merge":
        return fusion_then_merge

    if name == "icm-numba":
        return sal_icm_sweep

    msg = f"{name!r} is not one of {SOLVERS}"
    raise ValueError(msg)


def _finite(field: Any, graph: Any, spatial_weight: float) -> Any:
    """`field` with each `-inf` replaced by a finite forbidding penalty (#373, #462, #466).

    sal's expansion makes no move on `-inf`; energies of allowed labellings are unchanged.
    """
    import numpy as np

    from port.patch.icm.alpha_expansion import forbidden_as_finite

    return forbidden_as_finite(
        np.asarray(field, dtype=np.float64), graph, spatial_weight
    )


def solved_on_sal(
    field: Any, graph: Any, assignment: Any, spatial_weight: float, search: Any
) -> IcmResult:
    """Run `search` on sal's finite field and Potts graph; shared by every sal row (#517).

    `search(potts, values, start)` returns `(labelling, sweeps, termination)`;
    `assignment` is written in place, dtype kept; cost is the Potts energy.
    """
    import numpy as np
    from sal.sim.potts import energy

    from port.patch.icm.alpha_expansion import potts_graph_from
    from port.patch.icm.interface import IcmResult

    values = _finite(field, graph, spatial_weight)
    potts = potts_graph_from(graph, spatial_weight)
    labelling, sweeps, termination = search(
        potts, values, np.asarray(assignment, dtype=np.int64).copy()
    )

    labelling = np.asarray(labelling, dtype=assignment.dtype)
    assignment[:] = labelling

    return IcmResult(
        niter=int(sweeps),
        cost=float(energy(potts, values, labelling)),
        termination=termination,
    )


def sal_icm_sweep(
    field: Any,
    graph: Any,
    assignment: Any,
    spatial_weight: float,
    *,
    tolerance: float = 0.0,
    epsilon: float = 0.0,
    min_clone_spots: int = 200,
    cost_zeropoint: float = 0.0,
    onehot_allowed_clones: Any = None,
) -> IcmResult:
    """`icm_sweep`'s signature over sal's Rust ICM, bitwise its `numba` one (sal #1368).

    Same move set as `icm_sweep_deque`, but visits in index order where
    `cnaster` shuffles from the unseeded global RNG (#45, #312). `assignment`
    is updated in place; the ICM knobs are accepted and ignored.
    """
    del tolerance, epsilon, min_clone_spots, cost_zeropoint, onehot_allowed_clones

    import numpy as np
    from sal.search.icm import iterated_conditional_modes

    def search(potts: Any, values: Any, start: Any) -> tuple[Any, int, Any]:
        result = iterated_conditional_modes(
            potts, values, np.random.default_rng(0), start=start
        )
        return result.labelling, 1, result.termination

    return solved_on_sal(field, graph, assignment, spatial_weight, search)


def fusion_then_merge(
    field: Any,
    graph: Any,
    assignment: Any,
    spatial_weight: float,
    *,
    tolerance: float = 0.0,
    epsilon: float = 0.0,
    min_clone_spots: int = 200,
    cost_zeropoint: float = 0.0,
    onehot_allowed_clones: Any = None,
) -> IcmResult:
    """Fuse the expansion's labelling with the argmax descent's, then sal's floor.

    sal #1125's fusion move over alpha expansion (from the caller's labelling)
    and sal's ICM (from the field's argmax), then `merge_small_labels`.
    """
    del tolerance, epsilon, cost_zeropoint, onehot_allowed_clones

    import numpy as np
    from sal.backend import Backend
    from sal.search.alpha_expansion import alpha_expansion, fuse
    from sal.search.icm import iterated_conditional_modes, merge_small_labels

    def search(potts: Any, values: Any, start: Any) -> tuple[Any, int, Any]:
        expanded = alpha_expansion(
            potts, values, start=start, backend=Backend.RUST
        ).labelling
        descended = iterated_conditional_modes(
            potts,
            values,
            np.random.default_rng(0),
            start=np.argmax(values, axis=1).astype(np.int64),
        ).labelling
        fused = fuse(
            potts,
            values,
            np.asarray(expanded, dtype=np.int64),
            np.asarray(descended, dtype=np.int64),
            backend=Backend.RUST,
        ).labelling
        result = merge_small_labels(
            potts,
            values,
            np.asarray(fused, dtype=np.int64),
            np.random.default_rng(0),
            min_sites=max(int(min_clone_spots), 1),
        )
        return result.labelling, result.sweeps, result.termination

    return solved_on_sal(field, graph, assignment, spatial_weight, search)
