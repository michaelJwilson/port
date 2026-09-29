"""`cnaster.hmrf.run_core_inference`, with the shifted rates pinned (#293).

**The pin sets a scale the shifted model does not have.** With the per-clone
`logmu_shift` folded in (`port.patch.hmm_nophasing`), the negative binomial's
mean is `lambda_g T_n mu / sum_g lambda_g mu`, which `mu -> c mu` leaves
unchanged. So the fitted rates carry an arbitrary common factor, and nothing
downstream -- integer copy, the plots -- can read them until it is fixed.

The normal clone's dominant balanced state (`neutral_state`, #299) is set
to `mu = 1`, and every rate moves with it. That
changes no emission and no likelihood, so it is done **once, after the whole
optimization**: `run_core_inference` returns the HMM and HMRF's final fit,
and `run_cnaster` hands it straight to integer copy and plotting.

Unshifted fits are returned untouched: their scale is set by the baseline,
and a pin would move it.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from cnaster.hmrf import run_core_inference as UPSTREAM

__all__ = ["UPSTREAM", "pin_neutral", "run_core_inference"]

logger = logging.getLogger(__name__)


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
    try:
        path = np.asarray(result["pred_cnv"])
    except KeyError:
        path = None

    neutral = neutral_state(
        rates,
        state_vector(result["new_p_binom"]),
        path if path is not None and path.ndim == 2 else None,
    )

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
    from port.patch.hmm_initialize import distinct

    # NB passed rather than rebound: upstream binds the initializer as a
    #    default argument (#348).
    if distinct.installed() and "hmm_initializer" not in kwargs:
        kwargs["hmm_initializer"] = distinct.gmm_init

    result = UPSTREAM(*args, **kwargs)

    hmmclass = kwargs.get("hmmclass")
    shifted = bool(getattr(hmmclass, "apply_logmu_shift", False))

    if shifted and "m" in str(kwargs.get("params", "")):
        pin_neutral(result)

        from port.patch.hmrf import split_state

        if split_state.installed():
            result = _split_and_refit(result, args, kwargs)

    return result


def _split_and_refit(result: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """`split_state`'s refit on the first fit's clones, under the RDR + BAF call only."""
    from port.patch.hmrf.split_state import split_init

    single_X, lengths, base = args[0], args[1], np.asarray(args[2])

    # NB the BAF-only call has no baseline, and no depth to split by.
    if not np.any(base > 0):
        return result

    init = split_init(result, single_X, lengths, base)

    if init is None:
        logger.info("split state: no unbalanced state splits by depth")
        return result

    log_mu, p_binom, state, freed = init
    logger.info(
        f"split state: state {state} at log mu {log_mu[state, 0]:.3f}, "
        f"state {freed} freed for {log_mu[freed, 0]:.3f}"
    )

    kept = np.asarray(result["new_assignment"])
    clones = [np.flatnonzero(kept == label) for label in np.unique(kept)]
    if len(args) > 5:
        args = (*args[:5], clones, *args[6:])
    else:
        kwargs = {**kwargs, "initial_clone_index": clones}

    refit = UPSTREAM(
        *args,
        **{
            **kwargs,
            "init_log_mu": log_mu,
            "init_p_binom": p_binom,
            "max_iter_outer": 0,
        },
    )
    pin_neutral(refit)

    locked = bool(getattr(refit, "_locked", False))

    if locked:
        refit.unlock()

    try:
        refit["new_assignment"] = kept
    finally:
        if locked:
            refit.lock()

    return refit
