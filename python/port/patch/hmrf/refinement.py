"""Keep each read-depth sub-clone inside its BAF clone, as `cnaster` intends (#348).

`run_cnaster` refines every BAF clone into `n_clones_rdr` read-depth
sub-clones in one global HMRF. `cnaster.spatial.initialize_rdr_clone_refininement`
returns the start **and** a one-hot mask saying which sub-clones each spot may
take, its own BAF clone's, and the call to `run_core_inference` then
drops the mask (`run_cnaster.py:1105`, `# onehot_allowed_clones=None,`).

Without it the refinement is an unconstrained `n_baf * n_clones_rdr`-label
problem, and `icm_sweep_deque`'s floor finishes it: any clone under 200 spots
has its spots **randomly** reassigned to a clone that has 200 (`icm.py:940`),
regardless of BAF clone. Measured on `tests.fixtures.calicost_instance`, whose
BAF stage recovers the planted four clones exactly: 16 sub-clones of about 100
spots, one of which reached 200 after the first sweep, so 1,509 of 1,600
spots were moved into it and the run ended with one clone (ARI 0.000).

`initialize_rdr_clone_refininement` here is upstream's, keeping the mask it
returns; `mask_for` hands it to `port.patch.hmrf.clone_assignment` while the
problem is still the one it was built for, which folds it into the field
(less :data:`MASK_PENALTY`, so a sweep or merge crosses a BAF clone only on a
read-depth gain larger than that) and passes it to the ICM, which applies it
as `-inf`, and to the floor, whose reassignment reads it (`icm.py:939`). A problem of a different
shape, or an assignment that already breaks the mask, gets none.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from cnaster.spatial import (
    initialize_rdr_clone_refininement as UPSTREAM,
)

from port.patch._signature import as_upstream

__all__ = [
    "MASK_PENALTY",
    "UPSTREAM",
    "compact",
    "forget",
    "initialize_rdr_clone_refininement",
    "kept",
    "mask_for",
]

MASK_PENALTY = 100.0
"""Nats a spot's field loses on a sub-clone of another BAF clone (#467).

Finite, so a strong read-depth preference can still correct a spot the BAF
stage misplaced; `-inf` made the BAF stage's boundary final. Measured with
`--sal --refinement-mask --floor-merge` (clone ARI; `-inf` in brackets):
`dev` 1.000 (0.868), CalicoST hard 0.982 (0.982), easy 0.986, r0 0.9979,
#338's three-sample instance 1.000. At 10 hard falls to 0.554 with 3 clones
for 4; at 1,000 `dev` is `-inf`'s 0.868. The `--sal` rows read the mask
only through the field; `cnaster`'s ICM re-applies it as `-inf` from
`onehot_allowed_clones`, so on the default arm this changes nothing.
"""

_KEPT: list[np.ndarray] = []


@as_upstream(UPSTREAM)
def initialize_rdr_clone_refininement(arguments: dict[str, Any]) -> Any:
    """Upstream's refinement start, with its allowed-clone mask kept."""
    assignment, allowed, total = UPSTREAM(**arguments)

    _KEPT.clear()
    _KEPT.append(np.asarray(allowed, dtype=bool))

    return assignment, allowed, total


def mask_for(assignment: np.ndarray, n_clones: int) -> np.ndarray | None:
    """The kept mask, if it describes this problem and this labelling."""
    from cnaster.config import start_time
    from cnaster.logger import get_logger

    if not _KEPT:
        return None

    log = get_logger(__name__, start_time=start_time)
    mask = _KEPT[0]
    labels = np.asarray(assignment, dtype=np.int64)

    if mask.shape != (labels.size, n_clones):
        log.info(
            f"Refinement mask {mask.shape} not applied to {labels.size} x {n_clones}."
        )
        return None
    if (
        labels.max(initial=-1) >= n_clones
        or not mask[np.arange(labels.size), labels].all()
    ):
        log.info("Refinement mask not applied: the assignment already crosses it.")
        return None

    log.info(f"Applying the refinement mask over {n_clones} clones (#348).")
    return mask


def compact(assignment: np.ndarray) -> None:
    """Keep the mask's columns for the clones `assignment` still uses.

    `cnaster.hmrf.run_core_inference` relabels the survivors of an iteration
    by `np.unique(..., return_inverse=True)`, ascending (`hmrf.py:648`), so
    the mask's surviving columns in the same order describe the next
    iteration's problem.
    """
    if _KEPT:
        survivors = np.unique(np.asarray(assignment, dtype=np.int64))
        if survivors.size < _KEPT[0].shape[1]:
            _KEPT[0] = _KEPT[0][:, survivors]


def kept() -> bool:
    """Whether a mask is kept, whatever problem it describes."""
    return bool(_KEPT)


def forget() -> None:
    """Drop the kept mask; a run that ends should not leave it to the next."""
    _KEPT.clear()
