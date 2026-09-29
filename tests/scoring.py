"""One clone matcher for every recovery scorer (#517 step 7).

A fitted labelling is compared with a planted one after relabelling: each
planted label is paired with the fitted label it shares the most spots (or
bins) with, one to one, by `linear_sum_assignment` on the overlap counts.
`tests.recovery_audit`, `tests.sim_audit` and `tests.copy_audit` each built
the overlap and solved it themselves.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = ["matched", "overlap"]


def overlap(
    planted: np.ndarray, fitted: np.ndarray, n_planted: int, n_fitted: int
) -> np.ndarray:
    """`(n_planted, n_fitted)` counts of each planted label under each fitted one."""
    counts = np.zeros((n_planted, n_fitted), dtype=np.int64)
    np.add.at(counts, (planted, fitted), 1)
    return counts


def matched(counts: np.ndarray) -> dict[int, int]:
    """Each planted label's fitted label, one to one, maximizing total overlap."""
    rows, columns = linear_sum_assignment(-counts)
    return dict(zip(rows.tolist(), columns.tolist(), strict=True))
