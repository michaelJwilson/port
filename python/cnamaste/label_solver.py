"""The clone labelling's solvers, its floor and its Potts terms (T- #670 PR7).

`port.patch.icm.interface`, `port.patch.icm.alpha_expansion`'s graph and
field helpers, `port.patch.icm.floor` and the live rows of
`port.extensions.label_solver`, moved in by T- #670 PR7 for
`cnamaste.hmrf.pipeline_clone_assignment` (`docs/port-forward.md` row 32):

- `icm`: `cnaster`'s `icm_sweep_deque` behind the reduced interface
  :func:`icm_sweep` (#59 item 5), the per-sample weights and the mask folded
  into the field (:func:`fold_unary`);
- `alpha-rust-fuse-merge`, `--sal`'s (#410): `sal`'s alpha expansion (Rust
  cut) fused with its argmax descent, then `sal`'s merge at the floor
  (:func:`fusion_then_merge`);
- :func:`enforce_floor`: the clone-size floor met smallest first, each spot to
  its best remaining clone (#348), at `hmrf.min_spots_per_clone` where the
  configuration states it (#468).

**The Potts energy is one rule everywhere** (#180, #483). `cnaster`'s kNN
adjacency is directed: on a 42 x 42 hex grid 260 of 5,292 pairs are one-way.
`sal`'s solvers read it as `(A + A^T) / 2` (:func:`potts_graph_from`), which
is `calc_assignment_cost`'s rule, each stored entry at `beta w / 2`. Two
places in `cnaster` used another, and both are fixed here:

- :func:`merge_assignment` counts each merge's boundary gain from both
  sides, `gain[u, v] + gain[v, u]`; `cnaster.icm.merge_assignment` counts
  `gain[u, v]` alone, half of it on a symmetric graph (#483 defect 1);
- :func:`spatial_log_prior`, the spatial term of the reported `total_llf`,
  is that rule; `cnaster.hmrf` counted the stored entries with `i < j`,
  ignoring their weights (#483 defect 2).

**Departures from `port`'s**, none in what a solver computes: two of
`port`'s nine solvers move, the live ones (`--sal`'s and `cnaster`'s); the
seven measured and not admitted stay in `port.extensions.label_solver`.
`port`'s `PORT_LABEL_SOLVER` environment override is `port`'s, not read here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import numpy as np
from sal.opt.termination import Termination

__all__ = [
    "CNASTER_FLOOR",
    "SOLVERS",
    "CsrGraph",
    "IcmResult",
    "configured_floor",
    "enforce_floor",
    "fold_unary",
    "forbidden_as_finite",
    "fusion_then_merge",
    "icm_sweep",
    "merge_assignment",
    "potts_graph_from",
    "solver_for",
    "spatial_log_prior",
    "sweep_for",
]

Solver = Literal["icm", "alpha-rust-fuse-merge"]

SOLVERS: tuple[Solver, ...] = ("icm", "alpha-rust-fuse-merge")
"""`icm` is `cnaster`'s; `alpha-rust-fuse-merge` is `--sal`'s (#410)."""


@dataclass(frozen=True)
class CsrGraph:
    """One spatial graph as the three CSR arrays, which are meaningless apart."""

    indptr: np.ndarray
    indices: np.ndarray
    weights: np.ndarray

    @classmethod
    def from_matrix(cls, adjacency_mat: Any) -> CsrGraph:
        """From the `scipy.sparse` CSR matrix the boundary holds; no copy."""
        return cls(
            indptr=adjacency_mat.indptr,
            indices=adjacency_mat.indices,
            weights=adjacency_mat.data,
        )


class IcmResult(NamedTuple):
    """What a sweep returns, named: `cnaster`'s bare `(niter, cost)` and why it stopped."""

    niter: int
    cost: float
    termination: Termination


def fold_unary(
    single_llf: np.ndarray,
    log_persample_weights: np.ndarray | None = None,
    sample_ids: np.ndarray | None = None,
) -> np.ndarray:
    """A copy of the field with the per-sample weights added, as the sweep
    adds them per visit, left to right: bitwise.

    `log_persample_weights` is `(n_clones, n_samples)`, indexed
    `[c, sample_ids[i]]`, and requires `sample_ids`.
    """
    field = np.array(single_llf, dtype=np.float64, copy=True)
    n_spots, n_clones = field.shape

    if log_persample_weights is not None:
        if sample_ids is None:
            msg = "log_persample_weights requires sample_ids"
            raise ValueError(msg)
        if log_persample_weights.shape[0] != n_clones:
            msg = (
                f"log_persample_weights has {log_persample_weights.shape[0]} clones, "
                f"the field has {n_clones}"
            )
            raise ValueError(msg)
        field += log_persample_weights[:, np.asarray(sample_ids)[:n_spots]].T

    return field


def icm_sweep(
    field: np.ndarray,
    graph: CsrGraph,
    assignment: np.ndarray,
    spatial_weight: float,
    *,
    min_clone_spots: int = 200,
    onehot_allowed_clones: np.ndarray | None = None,
) -> IcmResult:
    """`cnaster`'s live `icm_sweep_deque` on a folded field; `assignment` in place.

    A delegation: the solver is `cnaster`'s, with its unseeded global RNG
    (#45). `onehot_allowed_clones` reaches only its floor, which reads no
    field.
    """
    from cnamaste.icm import icm_sweep_deque

    niter, cost = icm_sweep_deque(
        single_llf=field,
        adj_indptr=graph.indptr,
        adj_indices=graph.indices,
        adj_weights=graph.weights,
        new_assignment=assignment,
        spatial_weight=spatial_weight,
        posterior=None,
        onehot_allowed_clones=onehot_allowed_clones,
        tol=0.0,
        log_persample_weights=None,
        sample_ids=None,
        cost_zeropoint=0.0,
        temp=1.0,
        min_clone_spots=min_clone_spots,
        epsilon=0.0,
    )

    # NB `icm_sweep_deque` has no iteration cap: every return is its criterion met.
    return IcmResult(
        niter=int(niter),
        cost=float(cost),
        termination=Termination.after(int(niter), converged=True),
    )


def potts_graph_from(graph: CsrGraph, spatial_weight: float) -> Any:
    """The adjacency as `sal`'s `PottsGraph`: `(A + A^T) / 2`, times `spatial_weight`.

    A reciprocated pair keeps `spatial_weight * w`; a one-way pair enters at
    half (#417, #483). Refused: a negative coupling, which alpha expansion's
    bound excludes, and a graph with under `RECIPROCATED` of its edges
    reciprocated.
    """
    import scipy.sparse as sp
    from sal.sim.graph import PottsGraph

    from cnamaste.spot_adjacency import RECIPROCATED, AdjacencyError

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
            f"a coupling of {matrix.data.min() * float(spatial_weight)}; alpha "
            "expansion's bound requires a metric, so a negative coupling is "
            "refused rather than clipped"
        )
        raise ValueError(msg)

    reciprocated = matrix.multiply(matrix.T).nnz / matrix.nnz if matrix.nnz else 1.0
    if reciprocated < RECIPROCATED:
        msg = (
            f"only {reciprocated:.3f} of edges are reciprocated, under "
            f"{RECIPROCATED}: too one-way to read as a Potts coupling"
        )
        raise AdjacencyError(msg)

    halved = PottsGraph.from_directed_csr(matrix.indptr, matrix.indices, matrix.data)
    return PottsGraph(
        halved.n_nodes,
        halved.edges,
        tuple((halved.edge_coupling * float(spatial_weight)).tolist()),
    )


def forbidden_as_finite(
    values: np.ndarray, graph: CsrGraph, spatial_weight: float
) -> np.ndarray:
    """`values` with each `-inf` a penalty no labelling pays (#366).

    `sal`'s expansion makes no move on a field holding `-inf`. The penalty is
    the site's least finite entry less its incident coupling, as
    :func:`potts_graph_from` builds it, less one. A site forbidding every
    label is refused.
    """
    forbidden = np.isneginf(values)
    if not forbidden.any():
        return values

    finite = np.where(forbidden, np.inf, values)
    lowest = finite.min(axis=1)
    if not np.isfinite(lowest).all():
        msg = "a site forbids every label; there is no labelling to minimize over"
        raise ValueError(msg)

    indptr = np.asarray(graph.indptr)
    n_sites = indptr.size - 1
    sites = np.repeat(np.arange(n_sites), np.diff(indptr))
    weights = spatial_weight * np.asarray(graph.weights, dtype=np.float64)
    rows = np.bincount(sites, weights=weights, minlength=n_sites)
    columns = np.bincount(np.asarray(graph.indices), weights=weights, minlength=n_sites)
    incident = 0.5 * (rows + columns)
    penalty = lowest - incident - 1.0

    finite_values: np.ndarray = np.where(forbidden, penalty[:, None], values)
    return finite_values


def fusion_then_merge(
    field: np.ndarray,
    graph: CsrGraph,
    assignment: np.ndarray,
    spatial_weight: float,
    *,
    min_clone_spots: int = 200,
    onehot_allowed_clones: np.ndarray | None = None,
) -> IcmResult:
    """`--sal`'s labelling (#410, sal #1125): `assignment` in place.

    Alpha expansion (Rust cut) from the caller's labelling and `sal`'s
    `numba` descent from the field's argmax, fused per site by the roof dual,
    then `sal`'s merge of every label under `min_clone_spots`. The cost is
    the Potts energy of the labelling returned.
    """
    del onehot_allowed_clones
    from sal.backend import Backend
    from sal.search.alpha_expansion import alpha_expansion, fuse
    from sal.search.icm import iterated_conditional_modes, merge_small_labels
    from sal.sim.potts import energy

    values = forbidden_as_finite(
        np.asarray(field, dtype=np.float64), graph, spatial_weight
    )
    potts = potts_graph_from(graph, spatial_weight)
    start = np.asarray(assignment, dtype=np.int64).copy()

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

    labelling = np.asarray(result.labelling, dtype=assignment.dtype)
    assignment[:] = labelling
    return IcmResult(
        niter=int(result.sweeps),
        cost=float(energy(potts, values, labelling)),
        termination=result.termination,
    )


def solver_for(requested: str) -> Solver:
    """`requested`, refused if it names no solver here."""
    for solver in SOLVERS:
        if solver == requested:
            return solver

    msg = (
        f"the requested solver is {requested!r}, not one of {SOLVERS}; refusing "
        "rather than falling back"
    )
    raise ValueError(msg)


def sweep_for(name: Solver) -> Any:
    """The `icm_sweep`-signature solver a name selects."""
    return icm_sweep if name == "icm" else fusion_then_merge


# --- the clone-size floor (#348, #468) ----------------------------------------------

CNASTER_FLOOR = 200
"""`icm_sweep_deque`'s default `min_clone_spots`, which no key reaches (#81)."""


def configured_floor() -> int:
    """`hmrf.min_spots_per_clone` from `cnamaste`'s global config, else 200."""
    from cnamaste.config import get_global_config

    section = getattr(get_global_config(), "hmrf", None)
    floor = getattr(section, "min_spots_per_clone", None)
    return CNASTER_FLOOR if floor is None else int(floor)


def enforce_floor(field: np.ndarray, assignment: np.ndarray, floor: int) -> int:
    """Merge clones under `floor`, smallest first, into their spots' best clones.

    `field` is the unary score per `(spot, clone)`, higher better, any mask
    already `-inf`; `assignment` is updated in place. Returns the clones
    emptied. A clone whose spots have no allowed alternative keeps them.
    """
    n_clones = field.shape[1]
    counts = np.bincount(assignment, minlength=n_clones)
    emptied = 0
    stuck: set[int] = set()

    while True:
        alive = np.flatnonzero(counts > 0)
        small = [c for c in alive if counts[c] < floor and c not in stuck]
        if not small or alive.size <= 1:
            return emptied

        smallest = min(small, key=lambda c: (counts[c], c))
        spots = np.flatnonzero(assignment == smallest)
        others = np.setdiff1d(alive, [smallest])
        scores = field[np.ix_(spots, others)]
        best = others[np.argmax(scores, axis=1)]
        movable = np.isfinite(scores.max(axis=1))

        if not movable.any():
            stuck.add(int(smallest))
            continue

        assignment[spots[movable]] = best[movable]
        counts = np.bincount(assignment, minlength=n_clones)

        if counts[smallest] == 0:
            emptied += 1
        else:
            stuck.add(int(smallest))


# --- the Potts terms `cnaster` counted by another rule (#483) -------------------------


def spatial_log_prior(
    assignment: np.ndarray, adjacency_mat: Any, spatial_weight: float
) -> float:
    """`spatial_weight * sum_ij A_ij / 2 [s_i == s_j]` over the stored entries.

    `calc_assignment_cost`'s rule and `sal`'s energy's coupling term under
    :func:`potts_graph_from`: a reciprocated pair counts `w`, a one-way one
    `w / 2`. `cnaster` reported `spatial_weight` times the stored entries with
    `i < j` and the same label, unweighted (#483 defect 2).
    """
    entries = adjacency_mat.tocoo()
    labels = np.asarray(assignment)
    aligned = labels[entries.row] == labels[entries.col]
    weights = np.asarray(entries.data, dtype=np.float64)
    return float(spatial_weight * np.sum(weights[aligned]) / 2.0)


def merge_assignment(
    single_llf: np.ndarray,
    adj_spots: np.ndarray,
    adj_neighbors: np.ndarray,
    adj_weights: np.ndarray,
    assignment: np.ndarray,
    spatial_weight: float,
    log_persample_weights: np.ndarray | None = None,
    sample_ids: np.ndarray | None = None,
) -> tuple[float, float, tuple[int, int]]:
    """`cnaster.icm.merge_assignment`, the merge's boundary gain from both sides.

    Returns the current cost, the best cost after merging one clone into
    another, and that pair; `(-inf, (-1, -1))` where no merge gains. A merge
    of `u` into `v` aligns every edge between them, stored from `u` or from
    `v`, so its spatial gain is `gain[u, v] + gain[v, u]`, each stored entry
    at `spatial_weight * w / 2`. `cnaster` adds `gain[u, v]` alone, so on a
    path 0-1-2-3 labelled `[0, 0, 1, 1]` with a zero field it predicts 2.5
    for a merge `calc_assignment_cost` and `sal` score 3.0 (#483 defect 1).
    Ties go to the first pair in `(u, v)` order, as `cnaster`'s loop.
    """
    n_spots, n_clones = single_llf.shape
    labels = np.asarray(assignment, dtype=np.int64)
    values = np.asarray(single_llf, dtype=np.float64)

    if log_persample_weights is not None:
        if sample_ids is None:
            msg = "log_persample_weights requires sample_ids"
            raise ValueError(msg)
        values = values + log_persample_weights[:, np.asarray(sample_ids)[:n_spots]].T

    # NB `unary_sum[u, k]`: the field at label `k` summed over the spots now `u`.
    unary_sum = np.zeros((n_clones, n_clones), dtype=np.float64)
    np.add.at(unary_sum, labels, values)

    source = labels[np.asarray(adj_spots, dtype=np.int64)]
    target = labels[np.asarray(adj_neighbors, dtype=np.int64)]
    halves = spatial_weight * np.asarray(adj_weights, dtype=np.float64) / 2.0
    aligned = source == target

    boundary = np.zeros((n_clones, n_clones), dtype=np.float64)
    np.add.at(boundary, (source[~aligned], target[~aligned]), halves[~aligned])
    gain = boundary + boundary.T

    current = float(np.trace(unary_sum) + np.sum(halves[aligned]))
    candidates = current + (unary_sum - np.diag(unary_sum)[:, None]) + gain
    eligible = (gain > 0.0) & ~np.eye(n_clones, dtype=bool)

    if not eligible.any():
        return current, -np.inf, (-1, -1)

    flat = int(np.argmax(np.where(eligible, candidates, -np.inf)))
    u, v = divmod(flat, n_clones)
    return current, float(candidates[u, v]), (int(u), int(v))
