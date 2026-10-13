"""Keep each read-depth sub-clone inside its BAF clone, as `cnaster` intends (#348).

Wraps `cnaster.spatial.initialize_rdr_clone_refininement` to keep the allowed-clone
mask that `run_cnaster` drops; `mask_for` hands it to
`port.patch.hmrf.clone_assignment`, which folds it into the field (less
:data:`MASK_PENALTY`) and passes it to the ICM and the floor.
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
"""Nats a spot's field loses on a sub-clone of another BAF clone; finite, unlike `-inf` (#467)."""

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

    Matches `run_core_inference`'s ascending `np.unique` relabelling (`hmrf.py:648`).
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
