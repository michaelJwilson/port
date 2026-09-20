"""`snakes_and_ladders`' alpha expansion behind `icm_sweep`'s signature.

**#246.** `cnaster` solves the clone labelling with iterated conditional
modes -- `icm.icm_sweep_deque`, a greedy single-site descent. Upstream carries
`search.alpha_expansion`, which solves the same Potts MAP problem as a
sequence of binary minimum cuts and **states a bound**: for a metric pairwise
term the local minimum is within `2 * c_max / c_min` of the global one, which
is exactly 2 for a uniform coupling (Boykov, Veksler & Zabih 2001).

`.coveragerc-oracle` already declares the correspondence
(`search.alpha_expansion` referees `cnaster.icm`), and #127 is where it was
owed. Nothing had built it.

## Why the two differ, and why that is the point

ICM changes one site at a time, so it stops at any labelling no single flip
improves. Alpha expansion changes **arbitrarily many sites at once**, so it
crosses barriers no sequence of single flips crosses. The two therefore reach
different labellings by construction, and the comparison is not "are they the
same" but **which reaches the lower Potts energy** -- which
`search.alpha_expansion.energy` computes for either.

That makes this one of the rare patches with an *absolute* referee. A lower
energy is a better MAP solution under the same model, whoever produced it.

## Sign convention, which is where this would go wrong

`cnaster`'s `field` is a **log-likelihood**: larger is better, and
`icm_sweep_deque` maximizes. Upstream's `field_values` is an **energy**:
smaller is better, and `alpha_expansion` minimizes. So the field is negated
on the way in. Getting this backwards produces a run that completes, reports
a cost, and returns the worst labelling available.

## What is not carried over

`min_clone_spots` -- `cnaster` merges any clone below 200 spots mid-sweep
(#81), using the unseeded global RNG. Alpha expansion has no such move, and
adding one would break the monotonicity its termination proof rests on. So a
run through this solver does **not** apply that floor, which is a behaviour
difference rather than an omission, and `IcmResult.niter` counts expansion
cycles rather than ICM iterations.
"""

from __future__ import annotations

import numpy as np
from snakes_and_ladders.search.alpha_expansion import alpha_expansion, energy
from snakes_and_ladders.sim.graph import PottsGraph

from port.patch.icm_interface import CsrGraph, IcmResult

__all__ = ["alpha_expansion_sweep", "potts_energy", "potts_graph_from"]


def potts_graph_from(graph: CsrGraph, beta: float) -> PottsGraph:
    """`port`'s CSR adjacency as upstream's `PottsGraph`.

    Each undirected edge is taken **once**, from the upper triangle, because
    `CsrGraph` stores both directions and `PottsGraph` counts an edge's
    coupling once per entry -- listing both would double every bond and
    halve the effective temperature without saying so.

    The coupling is `beta * weight`, non-negative by the metric condition the
    bound rests on; a negative weight is refused rather than clipped.
    """
    indptr = np.asarray(graph.indptr)
    indices = np.asarray(graph.indices)
    weights = np.asarray(graph.weights, dtype=np.float64)

    edges: list[tuple[int, int]] = []
    coupling: list[float] = []

    for site in range(indptr.size - 1):
        for slot in range(int(indptr[site]), int(indptr[site + 1])):
            neighbour = int(indices[slot])

            if neighbour <= site:
                continue

            value = float(beta) * float(weights[slot])

            if value < 0.0:
                msg = (
                    f"edge ({site}, {neighbour}) has coupling {value}; alpha "
                    "expansion's bound requires a metric, so a negative "
                    "coupling is refused rather than clipped"
                )
                raise ValueError(msg)

            edges.append((site, neighbour))
            coupling.append(value)

    return PottsGraph(
        n_nodes=int(indptr.size - 1),
        edges=tuple(edges),
        coupling=tuple(coupling),
    )


def potts_energy(
    field: np.ndarray, graph: CsrGraph, assignment: np.ndarray, beta: float
) -> float:
    """The Potts energy of a labelling, for comparing two solvers.

    Takes `cnaster`'s field and negates it, so the number returned is
    comparable across solvers and **lower is better**. This is the referee
    #246 uses: it says which labelling is the better MAP solution without
    needing either solver to be right.
    """
    return float(
        energy(
            potts_graph_from(graph, beta),
            -np.asarray(field, dtype=np.float64),
            np.asarray(assignment, dtype=np.int64),
        )
    )


def alpha_expansion_sweep(
    field: np.ndarray,
    graph: CsrGraph,
    assignment: np.ndarray,
    beta: float,
    *,
    tol: float = 0.0,
    epsilon: float = 0.0,
    min_clone_spots: int = 200,
    cost_zeropoint: float = 0.0,
) -> IcmResult:
    """`icm_sweep`'s signature, upstream's solver.

    `assignment` is **updated in place**, as `icm_sweep` does, because the
    call site reads the array rather than a return value.

    `tol`, `epsilon`, `min_clone_spots` and `cost_zeropoint` are accepted and
    **not used**: they are ICM's convergence and perturbation knobs and have
    no counterpart in an algorithm that terminates on monotonicity. Accepted
    rather than refused so the two solvers are interchangeable at the call
    site; ignored rather than approximated so nothing pretends to honour
    them.
    """
    del tol, epsilon, min_clone_spots, cost_zeropoint

    values = -np.asarray(field, dtype=np.float64)
    n_states = int(values.shape[1])

    result = alpha_expansion(
        potts_graph_from(graph, beta),
        values,
        n_states,
        start=np.asarray(assignment, dtype=np.int64).copy(),
    )

    labelling = np.asarray(result.labelling, dtype=assignment.dtype)
    assignment[:] = labelling

    return IcmResult(niter=int(result.cycles), cost=float(result.energy))
