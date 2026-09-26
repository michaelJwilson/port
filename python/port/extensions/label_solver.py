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
    "label_solver",
    "sal_icm_argmax_sweep",
    "sal_icm_sweep",
    "set_label_solver",
    "sweep_for",
]

Solver = Literal[
    "icm", "alpha", "alpha-rust", "icm-numba", "alpha-rust-icm", "icm-argmax-floor"
]
SOLVERS: tuple[Solver, ...] = (
    "icm",
    "alpha",
    "alpha-rust",
    "icm-numba",
    "alpha-rust-icm",
    "icm-argmax-floor",
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

    if name == "icm-argmax-floor":
        return sal_icm_argmax_sweep

    return sal_icm_sweep


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
    from sal.sim.potts import energy

    from port.patch.icm.alpha_expansion import potts_graph_from
    from port.patch.icm.interface import IcmResult

    values = np.asarray(field, dtype=np.float64)
    potts = potts_graph_from(graph, beta)
    result = iterated_conditional_modes(
        potts,
        values,
        np.random.default_rng(0),
        start=np.asarray(assignment, dtype=np.int64).copy(),
        backend=Backend.NUMBA,
    )

    labelling = np.asarray(result.labelling, dtype=assignment.dtype)
    assignment[:] = labelling

    return IcmResult(niter=1, cost=float(energy(potts, values, labelling)))


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
    from sal.sim.potts import energy

    from port.patch.icm.alpha_expansion import potts_graph_from
    from port.patch.icm.interface import IcmResult

    values = np.asarray(field, dtype=np.float64)
    potts = potts_graph_from(graph, beta)
    result = iterated_conditional_modes(
        potts,
        values,
        np.random.default_rng(0),
        start=np.argmax(values, axis=1).astype(np.int64),
        min_sites=max(int(min_clone_spots), 1),
        backend=Backend.NUMBA,
    )

    labelling = np.asarray(result.labelling, dtype=assignment.dtype)
    assignment[:] = labelling

    return IcmResult(
        niter=int(result.sweeps), cost=float(energy(potts, values, labelling))
    )
