"""`cnaster.hmrf.run_core_inference`, with the shifted rates pinned (#293).

**The pin sets a scale the shifted model does not have.** With the per-clone
`logmu_shift` folded in (`port.patch.hmm_nophasing`), the negative binomial's
mean is `lambda_g T_n mu / sum_g lambda_g mu`, which `mu -> c mu` leaves
unchanged. So the fitted rates carry an arbitrary common factor, and nothing
downstream -- integer copy, the plots -- can read them until it is fixed.

The balanced state (allele fraction within `NEUTRAL_BAF_TOLERANCE` of 0.5)
with the lowest `mu` is set to `mu = 1`, and every rate moves with it. That
changes no emission and no likelihood, so it is done **once, after the whole
optimization**: `run_core_inference` returns the HMM and HMRF's final fit,
and `run_cnaster` hands it straight to integer copy and plotting.

Unshifted fits are returned untouched: their scale is set by the baseline,
and a pin would move it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from cnaster.hmrf import run_core_inference as UPSTREAM

__all__ = ["UPSTREAM", "pin_neutral", "run_core_inference"]


def pin_neutral(result: Any) -> int:
    """Set the neutral state's `mu` to 1 in `result`, in place.

    Returns the state pinned. `result` is `cnaster`'s `CnaHMRFResult`, which
    may be locked; it is unlocked for the one assignment and locked again if
    it was.
    """
    from port.patch.hmm_nophasing.shifted_emission import neutral_state
    from port.patch.plotting.clone_paths import state_vector

    column = np.asarray(result["new_log_mu"])
    rates = state_vector(column)
    neutral = neutral_state(rates, state_vector(result["new_p_binom"]))

    locked = bool(getattr(result, "_locked", False))

    if locked:
        result.unlock()

    try:
        result["new_log_mu"] = (rates - rates[neutral]).reshape(column.shape)
    finally:
        if locked:
            result.lock()

    return neutral


def run_core_inference(*args: Any, **kwargs: Any) -> Any:
    """Upstream's inference, then the neutral pin when the fit was shifted."""
    result = UPSTREAM(*args, **kwargs)

    hmmclass = kwargs.get("hmmclass")
    shifted = bool(getattr(hmmclass, "apply_logmu_shift", False))

    if shifted and "m" in str(kwargs.get("params", "")):
        pin_neutral(result)

    return result
