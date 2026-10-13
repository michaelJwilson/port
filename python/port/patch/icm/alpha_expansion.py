"""`snakes_and_ladders`' alpha expansion behind `icm_sweep`'s signature (#246).

Alternative to `cnaster.icm.icm_sweep_deque`: the same Potts MAP problem by
binary minimum cuts, within 2x of the global minimum for uniform coupling.
`field` is a log-likelihood passed unnegated: sal's energy already negates it,
so lower energy is better. Departures: no `min_clone_spots` merge (#81), and
`IcmResult.niter` counts expansion cycles; `-inf` labels are passed finite (#366).
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
    """`port`'s CSR adjacency as sal's `PottsGraph`, one-way edges at half weight (#417).

    Couples each unordered pair by `spatial_weight * (A_ij + A_ji) / 2`; sal
    refuses negative coupling (T- #777). Raises AdjacencyError if the
    reciprocated edge share is under `port.extensions.adjacency.RECIPROCATED`.
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

    # NB an empty graph -- every coupling zero -- is symmetric, not one-way.
    reciprocated = matrix.multiply(matrix.T).nnz / matrix.nnz if matrix.nnz else 1.0

    if reciprocated < RECIPROCATED:
        msg = (
            f"only {reciprocated:.3f} of edges are reciprocated, under "
            f"{RECIPROCATED}: too one-way to read as a Potts coupling"
        )
        raise AdjacencyError(msg)

    return PottsGraph.from_directed_csr(
        matrix.indptr, matrix.indices, matrix.data, scale=float(spatial_weight)
    )


def forbidden_as_finite(
    values: np.ndarray, graph: CsrGraph, spatial_weight: float
) -> np.ndarray:
    """`values` with each `-inf` replaced by a penalty no labelling pays (#366).

    Penalty: the site's least finite entry less its incident coupling, less 1.
    Raises ValueError if a site forbids every label.
    """
    forbidden = np.isneginf(values)

    if not forbidden.any():
        return values

    finite = np.where(forbidden, np.inf, values)
    lowest = finite.min(axis=1)

    if not np.isfinite(lowest).all():
        msg = "a site forbids every label; there is no labelling to minimize over"
        raise ValueError(msg)

    # NB incident coupling as `potts_graph_from` builds it, from `(A + A^T) / 2`.
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
    """The Potts energy of a labelling; lower is better (#246).

    `-potts_energy` is `cnaster`'s objective up to `spatial_weight * sum(weights)`.
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
    """`icm_sweep`'s signature, sal's alpha expansion; `assignment` updated in place.

    `backend` picks the minimum-cut solver only (#312). The ICM knobs are
    accepted for interchangeability and ignored.
    """
    del tolerance, epsilon, min_clone_spots, cost_zeropoint, onehot_allowed_clones

    # NB not negated: sal's energy already negates `h`.
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
