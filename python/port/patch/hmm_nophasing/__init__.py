"""`cnaster.hmm_nophasing`, rewritten where port applies the shift (#250).

`logmu_shift` is `compute_logmu_shifts` as an axis reduction,
not installed (#234); `shifted_emission` is the class that **applies** that
shift, which upstream computes and discards (#276), off by default;
`dense_emission` (the `emission_kernels="sal"` option) scores the coded emission with sal's dense log-emission,
which `--sal` enters (#425).

The submodules keep the split; this re-exports them so a swap row can name
`port.patch.hmm_nophasing` and a reader can open `cnaster.hmm_nophasing` and find it.
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
