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

## Sign convention, which is where this did go wrong

`cnaster`'s `field` is a **log-likelihood**: larger is better, and
`icm_sweep_deque` maximizes. Upstream minimizes
``E(s) = -sum_i h_i[s_i] - sum_ij J_ij [s_i == s_j]`` (`sim.potts.energies`),
which **already carries the negation**. So `field` is passed through
unchanged: `h = field` makes `-E` exactly `cnaster`'s objective up to the
constant `sum J`.

Negating it as well inverts the problem -- the run completes, reports a
cost, and returns the *worst* labelling available, which is what the first
version of this module did. `test_zero_coupling_recovers_the_field_argmax`
is the pin: with no coupling the minimizer is `field.argmax(axis=1)`, and
under the doubled negation it was `argmin`. The energy referee shared the
negation, so a solver-against-solver comparison could not see it -- an
oracle carrying the defect it refereed.

## What is not carried over

`min_clone_spots` -- `cnaster` merges any clone below 200 spots mid-sweep
(#81), using the unseeded global RNG. Alpha expansion has no such move, and
adding one would break the monotonicity its termination proof rests on. So a
run through this solver does **not** apply that floor, which is a behaviour
difference rather than an omission, and `IcmResult.niter` counts expansion
cycles rather than ICM iterations.

## Forbidden labels are passed finite (#366)

`cnaster` marks a label a spot may not take with `-inf` in the field.
`snakes_and_ladders` at 679d326 makes no expansion move at all on a field
holding `-inf` or entries of order `-1e6`, where 186bc59 did: on a
`--sal` run of CalicoST's pure hard sample every sweep with a forbidden
label returned its start, 464 to 670 nats above the old pin's labelling, and
the clone ARI fell from 0.994 to 0.666. So `forbidden_as_finite` replaces
each `-inf` by a penalty no move can pay: the site's least finite entry,
less the coupling of every edge at the site, less one. A site then gains
more by any allowed label than by a forbidden one whatever its neighbours
do, so the minimizer and its energy are the ones the `-inf` field states.
"""

from __future__ import annotations

import numpy as np
from sal.backend import Backend
from sal.search.alpha_expansion import alpha_expansion
from sal.sim.graph import PottsGraph
from sal.sim.potts import energy

from port.patch.icm.interface import CsrGraph, IcmResult

__all__ = [
    "alpha_expansion_sweep",
    "forbidden_as_finite",
    "potts_energy",
    "potts_graph_from",
]


def potts_graph_from(graph: CsrGraph, spatial_weight: float) -> PottsGraph:
    """`port`'s CSR adjacency as upstream's `PottsGraph`, one-way edges included.

    `cnaster`'s ICM sums each spot's own row, so on a directed graph its
    coupling is `spatial_weight * sum_i sum_{j in row i} A_ij [s_i = s_j]`. Over an
    unordered pair that is `spatial_weight * (A_ij + A_ji)`, which `PottsGraph` --
    counting each edge once -- carries as `spatial_weight * (A_ij + A_ji) / 2`: the
    same energy up to the factor of two every symmetric graph already had.
    A reciprocated pair keeps `spatial_weight * w`, bitwise the upper triangle this
    replaces; a one-way pair enters at half, where the upper-triangle read
    kept it whole when `i < j` and dropped it when `i > j` (#417).

    Built by sal's `PottsGraph.from_csr` (#1113) on the symmetrized matrix,
    with no Python loop over entries. Refused: a negative coupling -- alpha
    expansion's bound requires a metric -- and a graph whose reciprocated
    share of edges is under `port.extensions.adjacency.RECIPROCATED`, where
    one-way edges are no longer the boundary's.
    """
    import scipy.sparse as sp

    from port.extensions.adjacency import RECIPROCATED, AdjacencyError

    n_nodes = int(np.asarray(graph.indptr).size - 1)
    matrix = sp.csr_matrix(
        (
            np.asarray(graph.weights, dtype=np.float64),
            np.asarray(graph.indices),
            np.asarray(graph.indptr),
        ),
        shape=(n_nodes, n_nodes),
    )
    matrix.eliminate_zeros()

    if matrix.nnz and matrix.data.min() * float(spatial_weight) < 0.0:
        msg = (
            f"a coupling of {matrix.data.min() * float(spatial_weight)}; alpha expansion's "
            "bound requires a metric, so a negative coupling is refused rather "
            "than clipped"
        )
        raise ValueError(msg)

    # NB an empty graph -- every coupling zero -- is symmetric, not one-way.
    reciprocated = matrix.multiply(matrix.T).nnz / matrix.nnz if matrix.nnz else 1.0

    if reciprocated < RECIPROCATED:
        msg = (
            f"only {reciprocated:.3f} of edges are reciprocated, under "
            f"{RECIPROCATED}: too one-way to read as a Potts coupling"
        )
        raise AdjacencyError(msg)

    symmetric = ((matrix + matrix.T) * (0.5 * float(spatial_weight))).tocsr()
    symmetric.sort_indices()

    return PottsGraph.from_csr(symmetric.indptr, symmetric.indices, symmetric.data)


def forbidden_as_finite(
    values: np.ndarray, graph: CsrGraph, spatial_weight: float
) -> np.ndarray:
    """`values` with each `-inf` replaced by a penalty no labelling pays (#366).

    Upstream minimizes `-sum h[s] - sum J [s == s']`, so a label's field entry
    `h` is a gain, and moving a site from any allowed label to one with entry
    `p` changes the energy by at least `min_allowed(h) - p - sum_j J_ij`. With
    `p = min_allowed(h) - sum_j J_ij - 1` that change is at least one, so no
    minimum cut takes a forbidden label and no local minimum holds one. A site
    with no finite entry is refused: it has no label to take.
    """
    forbidden = np.isneginf(values)

    if not forbidden.any():
        return values

    finite = np.where(forbidden, np.inf, values)
    lowest = finite.min(axis=1)

    if not np.isfinite(lowest).all():
        msg = "a site forbids every label; there is no labelling to minimize over"
        raise ValueError(msg)

    # NB the coupling at a site as `potts_graph_from` builds it, from
    #    `(A + A^T) / 2`: its row and column sums, halved. On a symmetric graph
    #    that is the row sum; a one-way edge counts at half, as it enters.
    indptr = np.asarray(graph.indptr)
    n_sites = indptr.size - 1
    sites = np.repeat(np.arange(n_sites), np.diff(indptr))
    weights = spatial_weight * np.asarray(graph.weights, dtype=np.float64)
    rows = np.bincount(sites, weights=weights, minlength=n_sites)
    columns = np.bincount(np.asarray(graph.indices), weights=weights, minlength=n_sites)
    incident = 0.5 * (rows + columns)
    penalty = lowest - incident - 1.0

    return np.where(forbidden, penalty[:, None], values)


def potts_energy(
    field: np.ndarray, graph: CsrGraph, assignment: np.ndarray, spatial_weight: float
) -> float:
    """The Potts energy of a labelling, for comparing two solvers.

    `cnaster`'s field goes in unchanged: upstream's `energy` negates it
    itself, so `-potts_energy(...)` is `cnaster`'s own objective up to the
    constant `spatial_weight * sum(weights)`. **Lower is better.** This is the referee
    #246 uses -- it says which labelling is the better MAP solution without
    needing either solver to be right, which it can only do if it scores the
    objective `cnaster` maximizes rather than its negation.
    """
    return float(
        energy(
            potts_graph_from(graph, spatial_weight),
            np.asarray(field, dtype=np.float64),
            np.asarray(assignment, dtype=np.int64),
        )
    )


def alpha_expansion_sweep(
    field: np.ndarray,
    graph: CsrGraph,
    assignment: np.ndarray,
    spatial_weight: float,
    *,
    tolerance: float = 0.0,
    epsilon: float = 0.0,
    min_clone_spots: int = 200,
    cost_zeropoint: float = 0.0,
    backend: Backend = Backend.PYTHON,
    onehot_allowed_clones: np.ndarray | None = None,
) -> IcmResult:
    """`icm_sweep`'s signature, upstream's solver.

    `backend` picks the minimum-cut solver and nothing else. `Backend.RUST`
    returns the same labelling as the Python cut on every problem #312
    measured -- four from a dev run and six at stress -- at 7 to 38 times
    the speed; sal keeps it opt-in because a degenerate network can admit a
    second minimum cut of equal energy (search/alpha_expansion.py:430).

    `assignment` is **updated in place**, as `icm_sweep` does, because the
    call site reads the array rather than a return value.

    `tolerance`, `epsilon`, `min_clone_spots` and `cost_zeropoint` are accepted and
    **not used**: they are ICM's convergence and perturbation knobs and have
    no counterpart in an algorithm that terminates on monotonicity. Accepted
    rather than refused so the two solvers are interchangeable at the call
    site; ignored rather than approximated so nothing pretends to honour
    them.
    """
    del tolerance, epsilon, min_clone_spots, cost_zeropoint, onehot_allowed_clones

    # NB *not* negated: upstream's energy is `-sum h[s] - sum J [s == s]`,
    #    so `h = field` is already `cnaster`'s objective with the sign
    #    upstream's minimizer wants. See the module docstring.
    values = forbidden_as_finite(
        np.asarray(field, dtype=np.float64), graph, spatial_weight
    )

    # NB `n_states` is the field's column count, which sal infers (#410).
    result = alpha_expansion(
        potts_graph_from(graph, spatial_weight),
        values,
        start=np.asarray(assignment, dtype=np.int64).copy(),
        backend=backend,
    )

    labelling = np.asarray(result.labelling, dtype=assignment.dtype)
    assignment[:] = labelling

    return IcmResult(
        niter=int(result.cycles),
        cost=float(result.energy),
        termination=result.termination,
    )
