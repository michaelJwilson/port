"""The boundary's loop invariants, computed once instead of per iteration.

Issue #59 item 4. `cnaster.hmrf` recomputes two full passes over
`(n_obs, n_spots)` on every outer iteration:

    num_valid_nb_spotwise = (single_base_nb_mean > 0).sum(axis=0)   # :261
    num_valid_bb_spotwise = (single_total_bb_RD > 0).sum(axis=0)    # :262

Both are properties of the **input data**. `single_base_nb_mean` and
`single_total_bb_RD` are read by `load_input_data` and conditioned on
throughout -- `cnaster` never fits them -- so neither count can change while
the loop runs. `CLAUDE.md`: recompute or store is a decision, and unmade it
defaults to recompute.

`compute_loglike_spot_assignment` then rebuilds `rel_valid_emision_weight`
from those counts, per call, inside its own `prange` -- also invariant, and
hoisted with them.

**Measured** by `pytest-benchmark`, both count passes together, minimum
over the rounds it took:

| `n_obs` | `n_spots` | per iteration |
| ---: | ---: | ---: |
| 240 | 160 | 0.041 ms |
| 3,000 | 5,000 | 25.2 ms |
| 10,000 | 2,500 | 48.5 ms |

Against a boundary costing about 16 s that is under two tenths of a per
cent, so this is a **simplification** rather than a speedup and is offered
as one: the value is that a quantity which cannot change stops being
recomputed, and the ratio is incidental. It multiplies by `max_iter_outer`,
which is the only reason the absolute number is worth writing down at all.

**What this module is for.** The patch is two expressions, so shipping them
is not the point -- the point is the pair of facts a caller needs before
hoisting: that the counts depend on nothing the loop changes, and that the
relative channel weight derived from them therefore does not either.
:func:`boundary_invariants` computes both together so the call site hoists
one call, and `tests/test_hmrf_invariants_patch.py` is where the invariance
is asserted rather than asserted about.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["BoundaryInvariants", "boundary_invariants"]


@dataclass(frozen=True)
class BoundaryInvariants:
    """What the HMM/spatial boundary recomputes and need not.

    Parameters
    ----------
    num_valid_nb_spotwise, num_valid_bb_spotwise : np.ndarray
        Segments per spot with a positive baseline and a positive read depth,
        shape `(n_spots,)`. `cnaster`'s own names, so the call site changes
        only where they are computed.
    """

    num_valid_nb_spotwise: np.ndarray
    num_valid_bb_spotwise: np.ndarray

    def relative_channel_weight(
        self,
        smooth_indptr: np.ndarray,
        smooth_indices: np.ndarray,
        single_tumor_prop: np.ndarray | None = None,
    ) -> np.ndarray:
        """`pooled_bb / pooled_nb` per spot, the weight the field applies.

        Also invariant, and for the same reason: the smoothing neighbourhood
        is built from the spatial layout and `single_tumor_prop` is read with
        the data, neither of which the loop changes.
        `compute_loglike_spot_assignment` recomputes this per call inside its
        own `prange`, once per clone-assignment update.

        Carried here so a caller hoisting the counts hoists the weight with
        them rather than leaving half the work in the loop. Ones where a
        spot's pooled counts are not both positive, as `cnaster` does.

        Pooled with `bincount` rather than a loop over spots. The sums are of
        counts bounded by `n_obs`, so every partial sum is exact in
        `float64` and reassociating them cannot move the result -- which is
        what lets the comparison against `cnaster`'s sequential accumulation
        be bitwise rather than approximate.

        Parameters
        ----------
        single_tumor_prop : np.ndarray or None
            When given, neighbours with a `nan` proportion are skipped --
            `cnaster`'s `is_tumor_mixed` branch. `None` pools every
            neighbour, which is its unmixed branch.

        **This is not an endorsement of the weight.** Down-weighting the read
        depth against the allele channel by a count ratio is issue #58's
        finding; this reproduces it so that hoisting changes nothing, and
        fixing it is a separate change.
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
    """Both per-spot valid-segment counts, in one call.

    Raises
    ------
    ValueError
        If the two arrays disagree in shape, where they cannot describe the
        same `(n_obs, n_spots)` grid and one of them is the wrong input.
    """
    base = np.asarray(single_base_nb_mean)
    total = np.asarray(single_total_bb_RD)

    if base.shape != total.shape:
        msg = f"shapes must agree, got {base.shape} and {total.shape}"
        raise ValueError(msg)

    return BoundaryInvariants(
        num_valid_nb_spotwise=(base > 0).sum(axis=0),
        num_valid_bb_spotwise=(total > 0).sum(axis=0),
    )
