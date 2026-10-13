"""Re-exports port's `cnaster.hmm_nophasing` patches (#250).

`shifts` is `compute_logmu_shifts` as an axis reduction (#234);
`hmm_nophasing` applies the shift upstream computes and discards (#276), off by default.
"""

from __future__ import annotations

from port.patch.hmm_nophasing.logmu_shift import (
    shifts,
)
from port.patch.hmm_nophasing.shifted_emission import hmm_nophasing

__all__ = [
    "hmm_nophasing",
    "shifts",
]
