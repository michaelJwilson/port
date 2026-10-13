"""The label-solver rows set aside from `port.extensions.label_solver` (#749 WP5).

Ticket: #749 -- kept live are ICM, alpha expansion and the rows the rendered
  study figure draws; these four were measured and not admitted.
Measurement: `port.extensions.sal`'s list (dev instance clone ARI and Potts
  energy per call) and `docs/study-potts-solvers.md`'s table.
Exit: retire with `port.studies.clone_label_arms`, their one runner; a row
  returns to `label_solver` only by beating `alpha-rust-fuse-merge` there.

`alpha-rust-icm` is alpha expansion then `cnaster`'s ICM at its 200-spot
floor (#312); `alpha-rust-merge` the expansion then sal's floor;
`icm-numba-floor` sal's descent with the floor built in; `icm-argmax-floor`
that descent from the field's argmax (#410, sal #1121).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from port.extensions.label_solver import solved_on_sal

if TYPE_CHECKING:
    from port.patch.icm.interface import IcmResult


def expansion_then_floor(
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
    """Alpha expansion (Rust cut), then `cnaster`'s ICM with its knobs.

    The expansion finds the lower-energy basin; the ICM keeps `cnaster`'s
    `min_clone_spots` floor, merging any clone the expansion left under it.
    `assignment` is updated in place by both, and the result is the ICM's,
    whose cost is the energy of the labelling returned.
    """
    from sal.backend import Backend

    from port.patch.icm import alpha_expansion as sal
    from port.patch.icm.interface import icm_sweep

    # NB a mask is already in `field`; the ICM's floor is what reads it.
    sal.alpha_expansion_sweep(
        field, graph, assignment, spatial_weight, backend=Backend.RUST
    )

    return icm_sweep(
        field,
        graph,
        assignment,
        spatial_weight,
        tolerance=tolerance,
        epsilon=epsilon,
        min_clone_spots=min_clone_spots,
        cost_zeropoint=cost_zeropoint,
        onehot_allowed_clones=onehot_allowed_clones,
    )


def _sal_floor(
    field: Any,
    graph: Any,
    assignment: Any,
    spatial_weight: float,
    min_clone_spots: int,
    *,
    expand: bool,
) -> IcmResult:
    """sal end to end: an optional Rust expansion, then its ICM at `cnaster`'s floor.

    `merge_small_labels` is `iterated_conditional_modes` from the given start
    with `min_sites` (sal #1114), so with `expand` off this is the descent
    with the floor built in, and with it on the floor follows the expansion
    -- #312's R7: the merge that `alpha-rust` and `icm-numba` waited on.
    """
    import numpy as np
    from sal.backend import Backend
    from sal.search.alpha_expansion import alpha_expansion
    from sal.search.icm import merge_small_labels

    def search(potts: Any, values: Any, start: Any) -> tuple[Any, int, Any]:
        if expand:
            start = np.asarray(
                alpha_expansion(
                    potts, values, start=start, backend=Backend.RUST
                ).labelling,
                dtype=np.int64,
            )

        result = merge_small_labels(
            potts,
            values,
            start,
            np.random.default_rng(0),
            min_sites=max(int(min_clone_spots), 1),
            backend=Backend.NUMBA,
        )
        return result.labelling, result.sweeps, result.termination

    return solved_on_sal(field, graph, assignment, spatial_weight, search)


def expansion_then_merge(
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
    """Alpha expansion (Rust cut), then sal's merge at `cnaster`'s floor."""
    del tolerance, epsilon, cost_zeropoint, onehot_allowed_clones

    return _sal_floor(
        field, graph, assignment, spatial_weight, min_clone_spots, expand=True
    )


def sal_icm_floor_sweep(
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
    """sal's `numba` descent with `cnaster`'s floor built in."""
    del tolerance, epsilon, cost_zeropoint, onehot_allowed_clones

    return _sal_floor(
        field, graph, assignment, spatial_weight, min_clone_spots, expand=False
    )


def sal_icm_argmax_sweep(
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
    """sal's `numba` descent from the field's argmax, with `cnaster`'s floor (#410).

    sal #1121: a cold start at each site's best clone rather than the
    labelling the caller holds, then index-order single-site descent with
    `min_sites` dissolving any clone under the floor. `assignment` is read
    only for its dtype and written in place.
    """
    del tolerance, epsilon, cost_zeropoint, onehot_allowed_clones

    import numpy as np
    from sal.backend import Backend
    from sal.search.icm import iterated_conditional_modes

    def search(potts: Any, values: Any, start: Any) -> tuple[Any, int, Any]:
        del start
        result = iterated_conditional_modes(
            potts,
            values,
            np.random.default_rng(0),
            start=np.argmax(values, axis=1).astype(np.int64),
            min_sites=max(int(min_clone_spots), 1),
            backend=Backend.NUMBA,
        )
        return result.labelling, result.sweeps, result.termination

    return solved_on_sal(field, graph, assignment, spatial_weight, search)


SWEEPS: dict[str, Callable[..., IcmResult]] = {
    "alpha-rust-icm": expansion_then_floor,
    "alpha-rust-merge": expansion_then_merge,
    "icm-numba-floor": sal_icm_floor_sweep,
    "icm-argmax-floor": sal_icm_argmax_sweep,
}
"""Each set-aside row by the name `label_solver` selected it under."""
