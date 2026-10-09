"""Per-spot valid-segment counts and channel weight, hoisted out of `cnaster.hmrf`'s outer loop (#59 item 4).

Both depend only on the input data, which the loop never fits; a
simplification, not a speedup.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["BoundaryInvariants", "boundary_invariants"]


@dataclass(frozen=True)
class BoundaryInvariants:
    """Per-spot segment counts with a positive baseline and positive read depth, `(n_spots,)`."""

    num_valid_nb_spotwise: np.ndarray
    num_valid_bb_spotwise: np.ndarray

    def relative_channel_weight(
        self,
        smooth_indptr: np.ndarray,
        smooth_indices: np.ndarray,
        single_tumor_prop: np.ndarray | None = None,
    ) -> np.ndarray:
        """`pooled_bb / pooled_nb` per spot, the weight the field applies (#58).

        Ones where either pooled count is zero, as `cnaster`. Bitwise
        `cnaster`'s, as the sums are exact in `float64`. With
        `single_tumor_prop`, `nan` neighbours are skipped (`is_tumor_mixed`).
        """
        n_spots = self.num_valid_nb_spotwise.shape[0]

        owner = np.repeat(np.arange(n_spots), np.diff(smooth_indptr))
        neighbours = np.asarray(smooth_indices)

        if single_tumor_prop is not None:
            keep = ~np.isnan(single_tumor_prop[neighbours])
            owner, neighbours = owner[keep], neighbours[keep]

        pooled_nb = np.bincount(
            owner, weights=self.num_valid_nb_spotwise[neighbours], minlength=n_spots
        )
        pooled_bb = np.bincount(
            owner, weights=self.num_valid_bb_spotwise[neighbours], minlength=n_spots
        )

        weight = np.ones(n_spots, dtype=np.float64)
        both = (pooled_nb > 0.0) & (pooled_bb > 0.0)
        weight[both] = pooled_bb[both] / pooled_nb[both]

        return weight


def boundary_invariants(
    single_base_nb_mean: np.ndarray, single_total_bb_RD: np.ndarray
) -> BoundaryInvariants:
    """Both per-spot valid-segment counts; `ValueError` if the shapes disagree."""
    base = np.asarray(single_base_nb_mean)
    total = np.asarray(single_total_bb_RD)

    if base.shape != total.shape:
        msg = f"shapes must agree, got {base.shape} and {total.shape}"
        raise ValueError(msg)

    return BoundaryInvariants(
        num_valid_nb_spotwise=(base > 0).sum(axis=0),
        num_valid_bb_spotwise=(total > 0).sum(axis=0),
    )
