"""Wraps `cnaster.icm.icm_sweep_deque` (the live, last definition) behind a reduced interface (#59 item 5).

`posterior`, `log_persample_weights`, `sample_ids`, `onehot_allowed_clones`,
`temp` and the three CSR arrays fold into the field, the coupling and a
`CsrGraph`. A simplification, not a speedup; bitwise equal to the direct call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from sal.opt.termination import Termination

__all__ = ["CsrGraph", "IcmResult", "fold_unary", "icm_sweep"]


@dataclass(frozen=True)
class CsrGraph:
    """One spatial graph as CSR arrays, passed as one argument (#59 item 3)."""

    indptr: np.ndarray
    indices: np.ndarray
    weights: np.ndarray

    @classmethod
    def from_matrix(cls, adjacency_mat: object) -> CsrGraph:
        """From a `scipy.sparse` CSR matrix, without copying."""
        return cls(
            indptr=adjacency_mat.indptr,  # type: ignore[attr-defined]
            indices=adjacency_mat.indices,  # type: ignore[attr-defined]
            weights=adjacency_mat.data,  # type: ignore[attr-defined]
        )

    @property
    def n_spots(self) -> int:
        """Nodes, which is `len(indptr) - 1`."""
        return int(self.indptr.shape[0] - 1)


class IcmResult(NamedTuple):
    """The sweep's `(niter, cost)`, named, with its termination."""

    niter: int
    cost: float
    termination: Termination
    """Whether and why the sweep stopped (`snakes_and_ladders`' `Termination`, T- #617)."""


def fold_unary(
    single_llf: np.ndarray,
    log_persample_weights: np.ndarray | None = None,
    sample_ids: np.ndarray | None = None,
    onehot_allowed_clones: np.ndarray | None = None,
) -> np.ndarray:
    """Fold the per-`(spot, clone)` constants into a new `float64` field, bitwise as `cnaster` adds them.

    `single_llf` is `(n_spots, n_clones)`, not modified; `log_persample_weights`
    is `(n_clones, n_samples)` indexed `[c, sample_ids[i]]` and needs
    `sample_ids`; `onehot_allowed_clones` is boolean `(n_spots, n_clones)`,
    disallowed entries become `-inf`. Raises `ValueError` on missing
    `sample_ids` or mismatched shapes.
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

    if onehot_allowed_clones is not None:
        if onehot_allowed_clones.shape != field.shape:
            msg = (
                f"onehot_allowed_clones is {onehot_allowed_clones.shape}, "
                f"the field is {field.shape}"
            )
            raise ValueError(msg)

        field[~np.asarray(onehot_allowed_clones, dtype=bool)] = -np.inf

    return field


def icm_sweep(
    field: np.ndarray,
    graph: CsrGraph,
    assignment: np.ndarray,
    spatial_weight: float,
    *,
    tolerance: float = 0.0,
    epsilon: float = 0.0,
    min_clone_spots: int = 200,
    cost_zeropoint: float = 0.0,
    onehot_allowed_clones: np.ndarray | None = None,
) -> IcmResult:
    """Run `cnaster`'s live sweep through the reduced interface, by delegation.

    `field` has the constants folded in (:func:`fold_unary`); `spatial_weight`
    is `cnaster`'s `spatial_weight / temp`; `assignment` is updated in place.
    The sweep draws from the unseeded legacy global `numpy` RNG, as `cnaster`'s does.
    """
    from cnaster.icm import icm_sweep_deque

    niter, cost = icm_sweep_deque(
        single_llf=field,
        adj_indptr=graph.indptr,
        adj_indices=graph.indices,
        adj_weights=graph.weights,
        new_assignment=assignment,
        spatial_weight=spatial_weight,
        posterior=None,
        # NB `None` unless the refinement's mask applies (#348): only the floor's
        #    reassignment reads it; the field already carries it.
        onehot_allowed_clones=onehot_allowed_clones,
        tol=tolerance,
        log_persample_weights=None,
        sample_ids=None,
        cost_zeropoint=cost_zeropoint,
        temp=1.0,
        min_clone_spots=min_clone_spots,
        epsilon=epsilon,
    )

    # NB `icm_sweep_deque` has no iteration cap, so every return is convergence.
    return IcmResult(
        niter=int(niter),
        cost=float(cost),
        termination=Termination.after(int(niter), converged=True),
    )
