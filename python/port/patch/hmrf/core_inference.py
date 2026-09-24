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

**Then each clone's own shift** (#362). The pin fixes the shared table's
amplitude, but the shifted likelihood reads a clone's rates only through
`mu / Z_c`, so a clone whose neutral bins decode to a state other than the
pinned one is still off-scale: on CalicoST's easy simulated sample the
tumour clones' neutral bins sat at `mu = 0.53`, and a planted `(2, 2)`
decoded as `(1, 1)`. So after the pin each clone's shift, `log Z_c =
logsumexp_g(log mu_{z(g, c)} + log lambda_g)` -- the one the emission
applies, with `lambda` the baseline `cnaster` sums over every spot
(`hmrf.py:476`) -- is recorded in `new_log_mu_shift`, `(n_clones,)`, the
normal clone's at zero while `ZERO_NORMAL_SHIFT` holds (the default).

**Per-clone rates are never stored.** The result carries the pinned table
and the shifts; `reindex_clones` permutes the shifts with the clones and
keeps them, with the decode and the **normal state** -- the pinned one,
defined by the normal clone and shared by every clone -- in memory; and
`port.patch.integer_copy` subtracts a clone's shift from the table it is
handed, found by matching the clone's decode, and hands the decoder that
normal state rather than letting `find_diploid_balanced_state` re-derive it,
per clone, as the balanced state whose raw `mu` is closest to 1
(`integer_copy.py:84`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from cnaster.hmrf import reindex_clones as UPSTREAM_REINDEX
from cnaster.hmrf import run_core_inference as UPSTREAM

__all__ = [
    "UPSTREAM",
    "ZERO_NORMAL_SHIFT",
    "clone_shifts",
    "pin_neutral",
    "reindex_clones",
    "run_core_inference",
    "shift_for",
]

ZERO_NORMAL_SHIFT: list[bool] = [True]
"""Whether the normal clone's shift is set to zero (the default)."""


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


_PROPAGATED: dict[str, np.ndarray] = {}
"""The reindexed decode, `(n_obs, n_clones)`, and its clones' shifts."""

_NORMAL: list[int] = []
"""The pinned normal state, shared by every clone (the normal clone's)."""


def clone_shifts(
    result: Any, base_nb_mean: np.ndarray, zero_normal: bool = True
) -> np.ndarray:
    """Record each clone's `log Z_c` over the pinned rates; return them.

    Sets `result["new_log_mu_shift"]` to `(n_clones,)` and leaves
    `new_log_mu` the pinned, shared table. The normal clone is the one with
    the largest share of bins in balanced states, as `neutral_state` chooses
    it; with `zero_normal` its shift is 0.
    """
    from scipy.special import logsumexp

    from port.patch.hmm_nophasing.shifted_emission import NEUTRAL_BAF_TOLERANCE
    from port.patch.plotting.clone_paths import state_vector

    rates = state_vector(np.asarray(result["new_log_mu"]))
    balanced = (
        np.abs(state_vector(np.asarray(result["new_p_binom"])) - 0.5)
        <= NEUTRAL_BAF_TOLERANCE
    )
    path = np.asarray(result["pred_cnv"], dtype=np.int64)
    path = path.reshape(path.shape[0], -1)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_lambda = np.log(np.sum(base_nb_mean, axis=1) / np.sum(base_nb_mean))

    kept = np.isfinite(log_lambda)
    shifts = np.array(
        [
            float(logsumexp(rates[path[kept, c]] + log_lambda[kept]))
            for c in range(path.shape[1])
        ]
    )

    if zero_normal:
        shifts[int(np.argmax(balanced[path].mean(axis=0)))] = 0.0

    locked = bool(getattr(result, "_locked", False))

    if locked:
        result.unlock()

    try:
        result["new_log_mu_shift"] = shifts
    finally:
        if locked:
            result.lock()

    return shifts


def reindex_clones(res_combine: Any, *args: Any, **kwargs: Any) -> Any:
    """Upstream's reindex, with the clones' shifts permuted alongside them.

    Upstream permutes the decode's columns and not `new_log_mu_shift`, so the
    permutation is recovered by matching columns, applied to the shifts, and
    the reindexed decode and shifts are kept for the integer decode.
    """
    # NB bound at import, as `UPSTREAM_REINDEX`: the swap rebinds the name in
    #    `cnaster.hmrf`, so reading it here at call time would call this back.
    before = np.asarray(res_combine["pred_cnv"])
    reindexed, posterior = UPSTREAM_REINDEX(res_combine, *args, **kwargs)
    shifts = (
        res_combine.get("new_log_mu_shift") if hasattr(res_combine, "get") else None
    )
    _PROPAGATED.clear()

    if shifts is None or np.ndim(shifts) != 1:
        return reindexed, posterior

    old = before.reshape(before.shape[0], -1)
    after = np.asarray(reindexed["pred_cnv"])
    new = after.reshape(after.shape[0], -1)
    order: list[int] = []

    for column in range(new.shape[1]):
        matches = [
            c
            for c in range(old.shape[1])
            if c not in order and np.array_equal(old[:, c], new[:, column])
        ]
        order.append(matches[0] if matches else column)

    permuted = np.asarray(shifts, dtype=np.float64)[order]
    reindexed["new_log_mu_shift"] = permuted
    _PROPAGATED.update(pred=new, shifts=permuted)

    if _NORMAL:
        _PROPAGATED["normal"] = np.asarray(_NORMAL[0], dtype=np.int64)

    return reindexed, posterior


def shift_for(pred_cnv: Any) -> tuple[float, int | None]:
    """The shift and normal state of the clone whose decode is `pred_cnv`.

    `(0.0, None)` where no shifted fit was reindexed or no clone matches.
    """
    if not _PROPAGATED:
        return 0.0, None

    path = np.asarray(pred_cnv).reshape(-1)
    paths = _PROPAGATED["pred"]

    for column in range(paths.shape[1]):
        if paths.shape[0] == path.size and np.array_equal(paths[:, column], path):
            normal = _PROPAGATED.get("normal")
            return float(_PROPAGATED["shifts"][column]), (
                None if normal is None else int(normal)
            )

    return 0.0, None


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
        _NORMAL[:] = [pin_neutral(result)]

        base = args[2] if len(args) > 2 else kwargs.get("single_base_nb_mean")

        try:
            decoded = result["pred_cnv"] is not None
        except KeyError:
            decoded = False

        if base is not None and decoded:
            clone_shifts(result, np.asarray(base), ZERO_NORMAL_SHIFT[0])

    return result
