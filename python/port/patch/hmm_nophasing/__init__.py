"""Two `cnaster.hmm_nophasing` kernels, rewritten (#250).

`nb_logpmf` is the vectorized negative-binomial log-pmf `NUMERIC_SWAPS`
installs (#240); `logmu_shift` is `compute_logmu_shifts` as an axis reduction,
not installed (#234).

The submodules keep the split; this re-exports them so a swap row can name
`port.patch.hmm_nophasing` and a reader can open `cnaster.hmm_nophasing` and find it.
"""

from __future__ import annotations

from port.patch.hmm_nophasing.logmu_shift import (
    shifts,
)
from port.patch.hmm_nophasing.nb_logpmf import (
    log_factorial,
    nb_logpmf_1d,
)

__all__ = [
    "log_factorial",
    "nb_logpmf_1d",
    "shifts",
]
