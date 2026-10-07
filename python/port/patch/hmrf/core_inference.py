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
normal clone's at zero (`ZERO_NORMAL_SHIFT`).

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
"""The normal clone's shift is set to zero.

A constant: it was a one-element list any caller could flip, a switch that
bypassed install (T- #617). Nothing in `port` set it otherwise.
"""


def pin_neutral(result: Any) -> int:
    """Set the neutral state's `mu` to 1 in `result`, in place.

    Returns the state pinned. `result` is `cnaster`'s `CnaHMRFResult`, which
    may be locked; it is unlocked for the one assignment and locked again if
    it was.
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
    """Record each clone's `log Z_c` over the pinned rates; return them.

    `log Z_c` is `logmu_shift.clone_log_normalizers`, the field's and the
    genomic figure's (#749 WP7). Its one axis-0 reduction keeps the bins of
    zero baseline as exact zeros where this once dropped them, which moves
    the sum by at most 4.4e-15 nats (200 random instances, up to 3,000 bins).

    Sets `res["new_log_mu_shift"]` to `(n_clones,)` and leaves
    `new_log_mu` the pinned, shared table. The normal clone is the one with
    the largest share of bins in balanced states, as `neutral_state` chooses
    it; with `zero_normal` its shift is 0.
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
    """Upstream's reindex, with the clones' shifts permuted alongside them.

    The reorder is `port.patch.hmrf.reindex`'s, which holds every fitted
    parameter to one column rather than reordering columns it cannot have
    (#278, #517). Upstream permutes the decode's columns and not
    `new_log_mu_shift`, so the permutation is recovered by matching columns,
    applied to the shifts, and the reindexed decode and shifts are kept for
    the integer decode.
    """
    res_combine = arguments["res_combine"]
    before = np.asarray(res_combine["pred_cnv"])
    reindexed: Any
    reindexed, posterior = held_to_one_column(**arguments)
    # NB by `__getitem__`: `cnaster`'s `CnaHMRFResult` has no `get`, and
    #    reading through `hasattr(res, "get")` left the shifts unpermuted on
    #    every real run (#501).
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

    `run_core_inference` maps each id to its rank among `np.unique(sample_ids)`
    (`hmrf.py:453`) and the plots read `sample_list` by position, so the two
    name one slice only where the ranks are `0..n-1` and `sample_list` has
    `n` entries. `port.patch.io.get_sample_list` builds them so; a caller
    that bypasses it is refused rather than silently renumbered.

    Raises
    ------
    ValueError
        If the re-map is not the identity, or `sample_list` is not one name
        per id.
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


@as_upstream(UPSTREAM, hmm_start=None, distinct_init=False, baf_start=None)
def run_core_inference(arguments: dict[str, Any], options: dict[str, Any]) -> Any:
    """Upstream's inference, then the neutral pin when the fit was shifted.

    Options, which `run_cnaster_port` binds at install (#517): `hmm_start`,
    the read-depth stage's copy-state start (#489, #547); `baf_start`, the
    BAF-only stage's (#540); `distinct_init`, the initializer choosing among
    distinct components (#348).
    """
    import functools

    from port.patch.hmm_initialize import distinct, sal_mixture

    # NB passed rather than rebound: upstream binds the initializer as a
    #    default argument (#348).
    if "hmm_initializer" not in arguments:
        if options["hmm_start"] is not None or options["baf_start"] is not None:
            # NB a stage without its start falls back to `distinct`'s inside it.
            arguments["hmm_initializer"] = functools.partial(
                sal_mixture.gmm_init,
                start=options["hmm_start"],
                distinct=options["distinct_init"],
                baf_start=options["baf_start"],
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
