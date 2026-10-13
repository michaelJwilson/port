"""The clone-size floor, merged smallest first rather than all at once (#348).

Replaces `cnaster.icm.icm_sweep_deque`'s floor, which empties every small
clone at once into random clones, with `sal`'s
`FloorPolicy.SMALLEST_FIRST_BEST_FIELD` (#777): smallest clone first, each spot
to its best allowed clone by the field. The floor is `hmrf.min_spots_per_clone`
where configured, else `cnaster`'s 200 (#81).
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

    `field` is `(spot, clone)` score, higher better, masked clones `-inf`; a
    spot with no allowed alternative stays.
    """
    from sal.search.icm.numba import floor_smallest_first

    before = np.unique(assignment).size
    labels = np.ascontiguousarray(assignment, dtype=np.int64)
    floor_smallest_first(labels, np.ascontiguousarray(field, dtype=np.float64), floor)
    assignment[:] = labels
    return int(before - np.unique(assignment).size)
