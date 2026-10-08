"""The clone-size floor, merged smallest first rather than all at once (#348).

`cnaster.icm.icm_sweep_deque` enforces `min_clone_spots = 200`, which no
configuration key reaches (#81), inside the sweep. It works in two steps
(`icm.py:926-962`):

- every clone under the floor is emptied at once;
- each of its spots is moved to a clone drawn **uniformly at random** from
  those at or over the floor, whatever the field says about the spot.

When the floor cannot be met by every clone, the few clones that meet it
absorb the rest. Measured on `port.sim.truth.calicost_instance`: 16 read-depth
sub-clones of about 100 spots each, one of which reached 200, and 1,509 of
1,600 spots were moved into it; the run ended with one clone.

The floor is kept and the way it is met changed: one clone at a time, the
smallest first, so merging stops as soon as every clone left clears the
floor; each spot to its best remaining clone by the field, not at random;
and a spot with no allowed clone left (the mask is `-inf` in the field)
keeps its own, so the merge is not forced across a BAF clone.

**The rule is `sal`'s** (T- #777): `FloorPolicy.SMALLEST_FIRST_BEST_FIELD`,
whose floor alone is `sal.search.icm.numba.floor_smallest_first`, its Python
oracle `sal.search.icm._floor_smallest_first` (sal #1324). Port's
`enforce_floor`, which `sal` took up, is retired: on 3,000 random problems
(2-8 clones, 5-300 spots, `-inf` masks, floors to half the spots) the three
returned identical assignments (`tests/test_floor_merge.py`).
`merge_small_labels(policy=...)` is not used: it descends by ICM after the
floor, where the run sweeps with its own solver.

The floor is `hmrf.min_spots_per_clone` where the configuration sets it,
else `cnaster`'s 200. `floor_merge` installs it for a block: the ICM runs
with its own floor at 0, and this is applied after each sweep.
"""

from __future__ import annotations

import numpy as np

__all__ = ["configured_floor", "floor_clones"]

CNASTER_FLOOR = 200
"""`icm_sweep_deque`'s default `min_clone_spots`, which no key reaches (#81)."""


def configured_floor() -> int:
    """`hmrf.min_spots_per_clone` from `cnaster`'s global config, else 200."""
    from cnaster.config import get_global_config

    section = getattr(get_global_config(), "hmrf", None)
    floor = getattr(section, "min_spots_per_clone", None)

    return CNASTER_FLOOR if floor is None else int(floor)


def floor_clones(field: np.ndarray, assignment: np.ndarray, floor: int) -> int:
    """`sal`'s smallest-first, best-field floor on `assignment`, in place; the clones it emptied.

    `field` is the unary score per `(spot, clone)`, higher better, with any
    allowed-clone mask already `-inf`. A clone whose spots have no allowed
    alternative keeps them, and is left under the floor.
    """
    from sal.search.icm.numba import floor_smallest_first

    before = np.unique(assignment).size
    labels = np.ascontiguousarray(assignment, dtype=np.int64)
    floor_smallest_first(labels, np.ascontiguousarray(field, dtype=np.float64), floor)
    assignment[:] = labels
    return int(before - np.unique(assignment).size)
