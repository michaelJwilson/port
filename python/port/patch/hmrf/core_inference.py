"""Replaces `cnaster.hmrf.run_core_inference`, pinning the shifted rates (#293).

The shifted model's mean is invariant to `mu -> c mu`, so after the fit the
normal clone's neutral state is set to `mu = 1` (#299), and each clone's
`log Z_c` is recorded in `new_log_mu_shift`, normal clone at zero (#362).
Unshifted fits are returned untouched.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
from cnaster.hmrf import reindex_clones as UPSTREAM_REINDEX
from cnaster.hmrf import run_core_inference as UPSTREAM

from port.patch._signature import as_upstream
from port.patch.hmrf.reindex import reindex_clones as held_to_one_column

__all__ = [
    "UPSTREAM",
    "ZERO_NORMAL_SHIFT",
    "clone_shifts",
    "identity_remap",
    "pin_neutral",
    "reindex_clones",
    "release",
    "run_core_inference",
    "shift_for",
]

ZERO_NORMAL_SHIFT: Final = True
"""The normal clone's shift is set to zero; a constant (#617)."""


def pin_neutral(result: Any) -> int:
    """Set the neutral state's `mu` to 1 in `result`, in place; return that state.

    A locked `CnaHMRFResult` is unlocked for the write and relocked.
    """
    from port.patch._clone_paths import state_vector
    from port.patch.hmm_nophasing.shifted_emission import neutral_state

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


def release() -> None:
    """Drop what the run held; `port.pipeline.patched` calls this on exit (#517)."""
    _PROPAGATED.clear()
    _NORMAL.clear()


def clone_shifts(
    res: Any, base_nb_mean: np.ndarray, zero_normal: bool = True
) -> np.ndarray:
    """Set `res["new_log_mu_shift"]` to each clone's `log Z_c`, `(n_clones,)`; return them.

    `log Z_c` is `logmu_shift.clone_log_normalizers` over the pinned rates
    (#749). With `zero_normal`, the normal clone's (as `neutral_state` picks
    it) is 0.
    """
    from port.patch._clone_paths import state_vector
    from port.patch.hmm_nophasing.logmu_shift import clone_log_normalizers
    from port.patch.hmm_nophasing.shifted_emission import normal_clone

    rates = state_vector(np.asarray(res["new_log_mu"]))
    path = np.asarray(res["pred_cnv"], dtype=np.int64)
    path = path.reshape(path.shape[0], -1)

    normalizers = clone_log_normalizers(rates, path, base_nb_mean)
    shifts = np.full(path.shape[1], -np.inf) if normalizers is None else normalizers

    if zero_normal:
        shifts[normal_clone(state_vector(np.asarray(res["new_p_binom"])), path)[0]] = (
            0.0
        )

    locked = bool(getattr(res, "_locked", False))

    if locked:
        res.unlock()

    try:
        res["new_log_mu_shift"] = shifts
    finally:
        if locked:
            res.lock()

    return shifts


@as_upstream(UPSTREAM_REINDEX)
def reindex_clones(arguments: dict[str, Any]) -> Any:
    """Upstream's reindex via `port.patch.hmrf.reindex` (#278, #517), shifts permuted alongside.

    The permutation is recovered by matching decode columns; the reindexed
    decode and shifts are kept for the integer decode.
    """
    res_combine = arguments["res_combine"]
    before = np.asarray(res_combine["pred_cnv"])
    reindexed: Any
    reindexed, posterior = held_to_one_column(**arguments)
    # NB by `__getitem__`: `CnaHMRFResult` has no `get` (#501).
    try:
        shifts = res_combine["new_log_mu_shift"]
    except (KeyError, TypeError):
        shifts = None
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
    locked = bool(getattr(reindexed, "_locked", False))

    if locked:
        reindexed.unlock()

    try:
        reindexed["new_log_mu_shift"] = permuted
    finally:
        if locked:
            reindexed.lock()
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


def identity_remap(sample_ids: Any, sample_list: Any = None) -> None:
    """Refuse `sample_ids` that upstream's `np.unique` re-map would renumber (#418).

    Raises `ValueError` unless the ids are `0..n-1` and `sample_list` has `n` names.
    """
    if sample_ids is None:
        return

    unique = np.unique(np.asarray(sample_ids))

    if not np.array_equal(unique, np.arange(unique.size)):
        msg = (
            f"sample_ids {unique.tolist()} are not 0..{unique.size - 1}: "
            "run_core_inference would renumber them"
        )
        raise ValueError(msg)

    if sample_list is not None and len(sample_list) != unique.size:
        msg = f"{len(sample_list)} names in sample_list for {unique.size} sample ids"
        raise ValueError(msg)


@as_upstream(UPSTREAM, hmm_start=None, distinct_init=False)
def run_core_inference(arguments: dict[str, Any], options: dict[str, Any]) -> Any:
    """Upstream's inference, then the neutral pin when the fit was shifted.

    Options bound at install (#517): `hmm_start`, the read-depth copy-state
    start (#489, #547); `distinct_init`, distinct-component initializer (#348).
    """
    import functools

    from port.patch.hmm_initialize import distinct, sal_mixture

    # NB passed rather than rebound: upstream binds the initializer as a
    #    default argument (#348).
    if "hmm_initializer" not in arguments:
        if options["hmm_start"] is not None:
            # NB the BAF-only stage falls back to `distinct`'s inside it.
            arguments["hmm_initializer"] = functools.partial(
                sal_mixture.gmm_init,
                start=options["hmm_start"],
                distinct=options["distinct_init"],
            )
        elif options["distinct_init"]:
            arguments["hmm_initializer"] = distinct.gmm_init

    identity_remap(arguments.get("sample_ids"), arguments.get("sample_list"))

    result = UPSTREAM(**arguments)

    hmmclass = arguments.get("hmmclass")
    from port.patch.hmm_nophasing.shifted_emission import shifted

    is_shifted = shifted(hmmclass)

    if is_shifted and "m" in str(arguments.get("params", "")):
        _NORMAL[:] = [pin_neutral(result)]

        base = arguments.get("single_base_nb_mean")

        try:
            decoded = result["pred_cnv"] is not None
        except KeyError:
            decoded = False

        if base is not None and decoded:
            clone_shifts(result, np.asarray(base), ZERO_NORMAL_SHIFT)

    return result
