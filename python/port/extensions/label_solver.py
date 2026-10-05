"""Which solver `pipeline_clone_assignment` uses for the clone labelling.

**#246.** `cnaster` has one: `icm.icm_sweep_deque`, greedy single-site
descent. `snakes_and_ladders` has another that solves the same Potts MAP
problem with a proved bound. Both now sit behind one signature
(`port.patch.icm.interface.icm_sweep` and
`port.patch.icm.alpha_expansion.alpha_expansion_sweep`), so the choice is a
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

import functools
import os
from typing import Any, Literal

__all__ = [
    "SOLVERS",
    "expansion_then_floor",
    "expansion_then_merge",
    "fusion_then_merge",
    "sal_icm_argmax_sweep",
    "sal_icm_floor_sweep",
    "sal_icm_sweep",
    "solver_for",
    "sweep_for",
]

Solver = Literal[
    "icm",
    "alpha",
    "alpha-rust",
    "icm-numba",
    "alpha-rust-icm",
    "alpha-rust-merge",
    "icm-numba-floor",
    "icm-argmax-floor",
    "alpha-rust-fuse-merge",
]
SOLVERS: tuple[Solver, ...] = (
    "icm",
    "alpha",
    "alpha-rust",
    "icm-numba",
    "alpha-rust-icm",
    "alpha-rust-merge",
    "icm-numba-floor",
    "icm-argmax-floor",
    "alpha-rust-fuse-merge",
)
"""`icm` is `cnaster`'s. `alpha`, `alpha-rust` and `icm-numba` are
`snakes_and_ladders`' (#246, #312): alpha expansion with its Python or Rust
minimum cut, and single-site descent compiled with `numba`.

`alpha-rust-icm` is the two in sequence: alpha expansion to its local
minimum, then `cnaster`'s ICM from there, which applies the 200-spot clone
floor alpha expansion has no move for. The floor is load-bearing: on the dev
instance alpha expansion alone takes the clone ARI from 0.919 to 0.386, and
the sequence to 1.000 (#312)."""

ENVIRONMENT = "PORT_LABEL_SOLVER"
"""Read once per call, so a subprocess arm can select without a flag.

**Overrides the solver bound at install**: a stated departure from rule 1
of T- #617, held by `tests/test_environment_reads.py`. Unset, the bound
solver runs.
"""


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

    The environment is consulted on every call rather than at import, so a
    benchmark harness that sets it per subprocess does not depend on import
    order. Either is refused if it names no solver.
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

    if name == "alpha-rust-icm":
        return expansion_then_floor

    if name == "alpha-rust-merge":
        return expansion_then_merge

    if name == "icm-numba-floor":
        return sal_icm_floor_sweep

    if name == "icm-argmax-floor":
        return sal_icm_argmax_sweep

    if name == "alpha-rust-fuse-merge":
        return fusion_then_merge

    return sal_icm_sweep


def _finite(field: Any, graph: Any, beta: float) -> Any:
    """`field` with each `-inf` a penalty no labelling pays, before sal reads it.

    sal's expansion makes no move on a field holding `-inf` (#373 B0), and
    `alpha_expansion_sweep` applies `forbidden_as_finite` for that reason
    (#462). Every row here that hands sal the field goes through it too, so
    one fix covers every path to sal's solvers (#466). A labelling that takes
    no forbidden label has the same energy under either field.
    """
    import numpy as np

    from port.patch.icm.alpha_expansion import forbidden_as_finite

    return forbidden_as_finite(np.asarray(field, dtype=np.float64), graph, beta)


def expansion_then_floor(
    field: Any, graph: Any, assignment: Any, beta: float, **knobs: Any
) -> Any:
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
    sal.alpha_expansion_sweep(field, graph, assignment, beta, backend=Backend.RUST)

    return icm_sweep(field, graph, assignment, beta, **knobs)


def _solve(field: Any, graph: Any, assignment: Any, beta: float, search: Any) -> Any:
    """The part every sal row shares: the finite field, the Potts graph, the result.

    `search(potts, values, start)` returns `(labelling, sweeps, termination)`,
    the last stage's `Termination`; `assignment`
    is written in place with its dtype kept, and the cost is the Potts energy
    of the labelling returned. One implementation for the five rows (#517).
    """
    import numpy as np
    from sal.sim.potts import energy

    from port.patch.icm.alpha_expansion import potts_graph_from
    from port.patch.icm.interface import IcmResult

    values = _finite(field, graph, beta)
    potts = potts_graph_from(graph, beta)
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
    beta: float,
    *,
    tol: float = 0.0,
    epsilon: float = 0.0,
    min_clone_spots: int = 200,
    cost_zeropoint: float = 0.0,
    onehot_allowed_clones: Any = None,
) -> Any:
    """`icm_sweep`'s signature, upstream's single-site descent in `numba`.

    The same move set as `cnaster`'s `icm_sweep_deque`, so what differs is
    the implementation and the visit order: index order every sweep, where
    `cnaster` shuffles a two-queue worklist from the unseeded global RNG
    (#45). The two reach different local minima of the same energy; #312
    measured the difference within 31 nats either way on real fields, and
    39 to 61 times the speed at 5,000 to 10,000 spots.

    Deterministic, so `rng` is a fixed seed that the index order never
    draws from. `assignment` is updated in place, and the ICM knobs are
    accepted and ignored, as :func:`alpha_expansion_sweep` does and for the
    same reason -- including `min_clone_spots`: the merge below 200 spots is
    `cnaster`'s, not the descent's.
    """
    del tol, epsilon, min_clone_spots, cost_zeropoint, onehot_allowed_clones

    import numpy as np
    from sal.backend import Backend
    from sal.search.icm import iterated_conditional_modes

    def search(potts: Any, values: Any, start: Any) -> tuple[Any, int, Any]:
        result = iterated_conditional_modes(
            potts, values, np.random.default_rng(0), start=start, backend=Backend.NUMBA
        )
        return result.labelling, 1, result.termination

    return _solve(field, graph, assignment, beta, search)


def _sal_floor(
    field: Any,
    graph: Any,
    assignment: Any,
    beta: float,
    min_clone_spots: int,
    *,
    expand: bool,
) -> Any:
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

    return _solve(field, graph, assignment, beta, search)


def expansion_then_merge(
    field: Any,
    graph: Any,
    assignment: Any,
    beta: float,
    *,
    tol: float = 0.0,
    epsilon: float = 0.0,
    min_clone_spots: int = 200,
    cost_zeropoint: float = 0.0,
    onehot_allowed_clones: Any = None,
) -> Any:
    """Alpha expansion (Rust cut), then sal's merge at `cnaster`'s floor."""
    del tol, epsilon, cost_zeropoint, onehot_allowed_clones

    return _sal_floor(field, graph, assignment, beta, min_clone_spots, expand=True)


def sal_icm_floor_sweep(
    field: Any,
    graph: Any,
    assignment: Any,
    beta: float,
    *,
    tol: float = 0.0,
    epsilon: float = 0.0,
    min_clone_spots: int = 200,
    cost_zeropoint: float = 0.0,
    onehot_allowed_clones: Any = None,
) -> Any:
    """sal's `numba` descent with `cnaster`'s floor built in."""
    del tol, epsilon, cost_zeropoint, onehot_allowed_clones

    return _sal_floor(field, graph, assignment, beta, min_clone_spots, expand=False)


def sal_icm_argmax_sweep(
    field: Any,
    graph: Any,
    assignment: Any,
    beta: float,
    *,
    tol: float = 0.0,
    epsilon: float = 0.0,
    min_clone_spots: int = 200,
    cost_zeropoint: float = 0.0,
    onehot_allowed_clones: Any = None,
) -> Any:
    """sal's `numba` descent from the field's argmax, with `cnaster`'s floor (#410).

    sal #1121: a cold start at each site's best clone rather than the
    labelling the caller holds, then index-order single-site descent with
    `min_sites` dissolving any clone under the floor. `assignment` is read
    only for its dtype and written in place.
    """
    del tol, epsilon, cost_zeropoint, onehot_allowed_clones

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

    return _solve(field, graph, assignment, beta, search)


def fusion_then_merge(
    field: Any,
    graph: Any,
    assignment: Any,
    beta: float,
    *,
    tol: float = 0.0,
    epsilon: float = 0.0,
    min_clone_spots: int = 200,
    cost_zeropoint: float = 0.0,
    onehot_allowed_clones: Any = None,
) -> Any:
    """Fuse the expansion's labelling with the argmax descent's, then sal's floor.

    sal #1125's fusion move: per site, one proposal's label or the other's,
    chosen by the roof dual, and never worse than the better proposal. The
    proposals are alpha expansion (Rust cut) from the caller's labelling and
    sal's `numba` descent from the field's argmax, which reach different
    minima; the floor follows as in :func:`expansion_then_merge`.
    """
    del tol, epsilon, cost_zeropoint, onehot_allowed_clones

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
            backend=Backend.NUMBA,
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
            backend=Backend.NUMBA,
        )
        return result.labelling, result.sweeps, result.termination

    return _solve(field, graph, assignment, beta, search)
