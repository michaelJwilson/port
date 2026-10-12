"""The clone-labelling step's Potts energy, and the same problem in the pairwise form an external solver takes (S1).

`pipeline_clone_assignment` labels spots by maximizing `icm.calc_assignment_cost`: each spot's field
`single_llf[i, l_i]`, plus `log_persample_weights[l_i, sample_i]` where given, plus `spatial_weight * w / 2` for each
stored edge whose ends agree. An edge stored both ways so counts `w`, one stored one way `w / 2`: the accounting of the
adjacency as assumed, not symmetrized (Ticket#917). `potts_energy` is that quantity negated, so it is minimized.

`potts_problem` poses the same energy as unary costs and one coupling per unordered pair, `first < second`, with
`J = spatial_weight * (w_ij + w_ji) / 2` read as `-J [l_first == l_second]`: OpenGM's Potts factor (S2), so a labelling
has one energy whichever solver found it.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import scipy.sparse as sp


class PottsProblem(NamedTuple):
    """Unary costs `(n_spots, n_clones)`, each pair's ends `first < second`, its coupling `J`, and the constant a
    self-loop adds to every labelling."""

    unary: np.ndarray
    first: np.ndarray
    second: np.ndarray
    coupling: np.ndarray
    offset: float


def _field(single_llf, log_persample_weights=None, sample_ids=None):
    field = np.asarray(single_llf, dtype=np.float64)
    if log_persample_weights is not None:
        field = field + np.asarray(log_persample_weights, dtype=np.float64)[:, np.asarray(sample_ids)].T
    return field


def potts_energy(single_llf, adjacency, labels, spatial_weight, log_persample_weights=None, sample_ids=None):
    """`-icm.calc_assignment_cost` of `labels` on `adjacency` (a sparse matrix, as stored), vectorized."""
    labels = np.asarray(labels)
    field = _field(single_llf, log_persample_weights, sample_ids)
    a = sp.coo_matrix(adjacency)
    aligned = np.sum(a.data * (labels[a.row] == labels[a.col]))
    return -float(field[np.arange(labels.size), labels].sum() + spatial_weight * aligned / 2.0)


def potts_problem(single_llf, adjacency, spatial_weight, log_persample_weights=None, sample_ids=None):
    """The energy as unary costs and pairwise couplings: `E(l) = offset + sum unary[i, l_i] - sum J [l_first == l_second]`.

    A self-loop's ends always agree, so it is no coupling but the constant `offset`, `-spatial_weight * w / 2`."""
    a = sp.coo_matrix(adjacency)
    keep = a.row != a.col
    lo, hi = np.minimum(a.row[keep], a.col[keep]), np.maximum(a.row[keep], a.col[keep])
    pairs = sp.coo_matrix((a.data[keep], (lo, hi)), shape=a.shape).tocsr()
    pairs.sum_duplicates()
    pairs = pairs.tocoo()
    return PottsProblem(
        unary=np.ascontiguousarray(-_field(single_llf, log_persample_weights, sample_ids)),
        first=np.ascontiguousarray(pairs.row, dtype=np.int64),
        second=np.ascontiguousarray(pairs.col, dtype=np.int64),
        coupling=np.ascontiguousarray(spatial_weight * pairs.data / 2.0, dtype=np.float64),
        offset=-float(spatial_weight * np.sum(a.data[~keep]) / 2.0),
    )


def problem_energy(problem, labels):
    """`problem`'s energy of `labels`: `potts_energy`'s, on the adjacency it was posed from."""
    labels = np.asarray(labels)
    unary = problem.unary[np.arange(labels.size), labels].sum()
    return float(problem.offset + unary - np.sum(problem.coupling * (labels[problem.first] == labels[problem.second])))
