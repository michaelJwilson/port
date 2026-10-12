"""The clone-labelling step's Potts energy, and the same problem in the pairwise form an external solver takes (S1).

`pipeline_clone_assignment` labels spots by maximizing `icm.calc_assignment_cost`: each spot's field
`single_llf[i, l_i]`, plus `spatial_weight * w / 2` for each stored edge whose ends agree. An edge stored both ways so
counts `w`, one stored one way `w / 2`: the accounting of the adjacency as assumed, not symmetrized (Ticket#917).
`potts_energy` is that quantity negated, so it is minimized. `cnaster`'s per-sample weights (`hmrf.inertia`) are
deprecated and have no term here.

`potts_problem` poses the same energy as unary costs and one coupling per unordered pair, `first < second`, with
`J = spatial_weight * (w_ij + w_ji) / 2` read as `-J [l_first == l_second]`: OpenGM's Potts factor (S2), so a labelling
has one energy whichever solver found it. A self-loop is refused, as the adjacency's definition refuses it: its ends
always agree, so it would only shift every labelling's energy by a constant.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import scipy.sparse as sp


class PottsProblem(NamedTuple):
    """Unary costs `(n_spots, n_clones)`, each pair's ends `first < second`, and its coupling `J`."""

    unary: np.ndarray
    first: np.ndarray
    second: np.ndarray
    coupling: np.ndarray


def refuse_self_loops(adjacency):
    """`adjacency` as a sparse matrix, refused if any stored edge joins a spot to itself."""
    a = sp.coo_matrix(adjacency)
    loops = int(np.count_nonzero((a.row == a.col) & (a.data != 0)))
    if loops:
        raise ValueError(f"the adjacency has {loops} self-loops: a spot is not its own neighbour")
    return a


def potts_energy(single_llf, adjacency, labels, spatial_weight):
    """`-icm.calc_assignment_cost` of `labels` on `adjacency` (a sparse matrix, as stored, no self-loop), vectorized."""
    labels = np.asarray(labels)
    a = refuse_self_loops(adjacency)
    aligned = np.sum(a.data * (labels[a.row] == labels[a.col]))
    field = np.asarray(single_llf, dtype=np.float64)
    return -float(field[np.arange(labels.size), labels].sum() + spatial_weight * aligned / 2.0)


def potts_problem(single_llf, adjacency, spatial_weight):
    """The energy as unary costs and pairwise couplings: `E(l) = sum unary[i, l_i] - sum J [l_first == l_second]`."""
    a = refuse_self_loops(adjacency)
    lo, hi = np.minimum(a.row, a.col), np.maximum(a.row, a.col)
    pairs = sp.coo_matrix((a.data, (lo, hi)), shape=a.shape).tocsr()
    pairs.sum_duplicates()
    pairs = pairs.tocoo()
    return PottsProblem(
        unary=np.ascontiguousarray(-np.asarray(single_llf, dtype=np.float64)),
        first=np.ascontiguousarray(pairs.row, dtype=np.int64),
        second=np.ascontiguousarray(pairs.col, dtype=np.int64),
        coupling=np.ascontiguousarray(spatial_weight * pairs.data / 2.0, dtype=np.float64),
    )


def problem_energy(problem, labels):
    """`problem`'s energy of `labels`: `potts_energy`'s, on the adjacency it was posed from."""
    labels = np.asarray(labels)
    unary = problem.unary[np.arange(labels.size), labels].sum()
    return float(unary - np.sum(problem.coupling * (labels[problem.first] == labels[problem.second])))
