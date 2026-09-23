"""The clone-size floor, merged smallest first rather than all at once (#348).

`cnaster.icm.icm_sweep_deque` enforces `min_clone_spots = 200`, which no
configuration key reaches (#81), inside the sweep. It works in two steps
(`icm.py:926-962`):

- every clone under the floor is emptied at once;
- each of its spots is moved to a clone drawn **uniformly at random** from
  those at or over the floor, whatever the field says about the spot.

When the floor cannot be met by every clone, the few clones that meet it
absorb the rest. Measured on `tests.fixtures.calicost_instance`: 16 read-depth
sub-clones of about 100 spots each, one of which reached 200, and 1,509 of
1,600 spots were moved into it; the run ended with one clone.

`enforce_floor` keeps the floor and changes how it is met:

- **one clone at a time, the smallest first**, so merging stops as soon as
  every clone that is left clears the floor, rather than as soon as the few
  that already did have absorbed everything;
- **each spot to its best remaining clone** by the field, not at random. The
  field carries the allowed-clone mask as `-inf`, so a spot with no allowed
  clone left keeps its own and the merge is not forced across a BAF clone.

The floor is `hmrf.min_spots_per_clone` where the configuration sets it,
else `cnaster`'s 200. `floor_merge` installs it for a block: the ICM runs
with its own floor at 0, and this is applied after each sweep.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np

__all__ = ["configured_floor", "enforce_floor", "floor_merge", "installed"]

_INSTALLED: list[bool] = [False]

CNASTER_FLOOR = 200
"""`icm_sweep_deque`'s default `min_clone_spots`, which no key reaches (#81)."""


def installed() -> bool:
    """Whether :func:`floor_merge` is active."""
    return _INSTALLED[0]


@contextmanager
def floor_merge() -> Iterator[None]:
    """Replace the ICM's random floor with :func:`enforce_floor` for the block."""
    previous = _INSTALLED[0]
    _INSTALLED[0] = True

    try:
        yield
    finally:
        _INSTALLED[0] = previous


def configured_floor() -> int:
    """`hmrf.min_spots_per_clone` from `cnaster`'s global config, else 200."""
    from cnaster.config import get_global_config

    section = getattr(get_global_config(), "hmrf", None)
    floor = getattr(section, "min_spots_per_clone", None)

    return CNASTER_FLOOR if floor is None else int(floor)


def enforce_floor(field: np.ndarray, assignment: np.ndarray, floor: int) -> int:
    """Merge clones under `floor`, smallest first, into their spots' best clones.

    `field` is the unary score per `(spot, clone)`, higher better, with any
    allowed-clone mask already `-inf`. `assignment` is updated in place.
    Returns how many clones were emptied. A clone whose spots have no allowed
    alternative keeps them, and is left under the floor rather than forced
    across the mask.
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
