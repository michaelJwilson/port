"""`cnaster.hmrf.pipeline_clone_assignment`, without the array or the round trip.

**The seam, rebound one level up (#206).** `port` has carried measured
patches for #59's five items since [#125](https://github.com/michaelJwilson/port/pull/125)
and could install none of them: each needs a call-site edit inside
`cnaster.hmrf`, which `CLAUDE.md` makes read only. They install here,
because `pipeline_clone_assignment` is itself a module-level name and
rebinding *it* rebinds its call sites with it.

**All five of #59's items are installed here**, and none of them moves a
number. #206's "done when" list, in its order:

*   **The field is written into a buffer rather than returned and reduced.**
    `cnaster` materializes `(n_states, n_obs, n_spots)` per channel and then
    reduces it to `(n_spots, n_clones)` by reading one decoded state per
    `(bin, clone)`. `port.patch.hmrf.fused_field` does both in one pass,
    materializes nothing, and writes into a caller's array -- upstream's
    `external_field(..., field)` shape. 8 GB at the declared scale, twice
    per outer iteration (#90), for an array whose only consumer is the
    reduction. Pinned **bitwise** in `tests/test_hmrf_fused_field.py`.
*   **The graph crosses the seam once, in the representation the solver
    reads.** `CsrGraph` carries the three arrays that are meaningless apart.
    The COO triple `merge_assignment` wants is built only where it is
    consumed, and by `port.patch.hmrf.adjacency.adjacency_coo` -- three array
    expressions against `cast_csr` plus `unpack_adjacency`, which walk every
    non-zero in pure Python. On a run with `merge=False` `cnaster` computes
    that round trip and discards it (#59 item 3).
*   **The invariants are computed where they are constant.** The two
    valid-segment counts and the channel weight derived from them are properties
    of the input data, which the outer loop never fits, and `cnaster`
    recomputes all three per iteration (#59 item 4). :func:`boundary` holds
    them.
*   **The solver takes the problem.** `fold_unary` folds the per-sample
    weights and the allowed-clone mask into the field, and `icm_sweep` takes
    a field, a graph, a labelling and a coupling -- against fifteen
    parameters of which seven are not information the solver reads (#59 item
    5).

**#45, #58 and #81 are pinned first**, which is the rest of #206's list:
`tests/test_external_field.py` for #58, `tests/test_seam_defects.py` for the
other two. A seam rewritten without pinning them carries them into the
rewrite, and then nothing can tell a defect that was always there from one
the rewrite introduced.

**Who would maintain each step**, which `CLAUDE.md` says decides whether an
optimization is upstream's, could be upstream's, or is `port`'s:

| step | class | why |
| --- | --- | --- |
| the fused field, written into a buffer | **could be upstream** | `sal.oxisal.external_field(totals, successes, ..., field)` is this function, in Rust, writing in place. What it cannot take is the per-observation exposure and trials, which is #32's covariate gap |
| one graph across the seam | **exists upstream** | `single_site_sweeps(state, field, offsets, neighbours, couplings, ...)` takes the CSR directly |
| the hoisted invariants | **`port`** | they are invariants of `cnaster`'s own loop, and upstream has no loop to hoist them out of |
| the COO triple, built where it is consumed | **`port`** | upstream has no COO form to build; this is `cnaster`'s own round trip removed |
| the reduced solver interface | **exists upstream** | upstream's solver already takes the problem rather than its call site, and `#141` refereed `cnaster`'s ICM against two of them |

So three of the five are things this repository should be asking `cnaster`
to take from upstream rather than from `port`, and the other two are reports
about `cnaster`'s own loop and its own round trip. None of them lands here:
`CLAUDE.md` makes both dependencies read only, and what `port` controls is
the pin and the measurement.

**What it does not do is reimplement the pipeline.** Everything else is
`cnaster`'s: the pooling, the solver call, the merge loop, the likelihood
and the log lines. Where an argument takes this function down a branch the
fused field does not cover, it **delegates to `cnaster`'s own** rather than
guessing -- a fallback says which regimes are measured and which are not,
where a silent approximation would not.

The uncovered branches, and why:

*   `single_tumor_prop is not None` -- the tumour-mixed field is a different
    quantity, and #135 says the E step accepts the proportion and never
    reads it, so what it should be is open.
*   more than one sample -- `log_persample_weights` and `sample_ids` are
    passed through to the solver unchanged, but the inertia they encode is
    what #28 deprecates and no fixture here carries two samples.

There is no third branch for a parameter with a clone axis. The fused kernel
reads `(n_states,)` and `state_vector` normalizes at the edge, so a shape the
fit cannot produce raises there rather than being delegated (#278).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from cnaster.config import get_global_config, start_time
from cnaster.hmrf import pipeline_clone_assignment as UPSTREAM
from cnaster.logger import get_logger

__all__ = ["UPSTREAM", "boundary", "pipeline_clone_assignment"]

logger = get_logger(__name__, start_time=start_time)
"""`cnaster`'s own function, captured at import.

**Not looked up through `cnaster.hmrf` when it is needed**, because by then
the name is this one: `port.pipeline.patched()` imports this module to
resolve the replacement and *then* rebinds the module attribute, so a
delegation that resolved late would call itself. It did, once, and the
symptom was a `RecursionError` two minutes into a whole run.
"""


def _channel_weight(
    valid_nb: np.ndarray,
    valid_bb: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
) -> np.ndarray:
    """`rel_valid_emision_weight`, as a segment sum rather than two loops.

    `cnaster` computes it inside `compute_loglike_spot_assignment`: pooled
    valid BAF segments over pooled valid RDR segments, across the spot's
    smoothed neighbourhood, and one where either is zero. The neighbourhood
    is a CSR row, so the pooling is `np.add.reduceat` over the non-zeros --
    the same arithmetic, without the Python loop over spots.

    What the factor *is* remains #58's: the paper's field carries no such
    term, and this reproduces it rather than endorsing it.
    """
    n_spots = len(indptr) - 1
    weight = np.ones(n_spots, dtype=np.float64)

    if indices.size == 0:
        return weight

    # NB `reduceat` needs the start of each row and misbehaves on an empty
    #    one, so rows are summed by segment id instead -- which also gives
    #    an empty row a pooled count of zero, as the loop does.
    rows = np.repeat(np.arange(n_spots), np.diff(indptr))

    pooled_nb = np.bincount(rows, weights=valid_nb[indices], minlength=n_spots)
    pooled_bb = np.bincount(rows, weights=valid_bb[indices], minlength=n_spots)

    live = (pooled_nb > 0) & (pooled_bb > 0)
    weight[live] = pooled_bb[live] / pooled_nb[live]

    return weight


@dataclass
class _Boundary:
    """What the seam recomputes per outer iteration and need not (#59 item 4).

    `num_valid_nb_spotwise`, `num_valid_bb_spotwise` and the relative channel
    weight derived from them are properties of the **input data**.
    `single_base_nb_mean` and `single_total_bb_RD` are read by
    `load_input_data` and conditioned on throughout -- `cnaster` never fits
    them -- so none of the three can change while the outer loop runs, and
    `cnaster` recomputes all three on every iteration anyway.

    `held` is the point of the dataclass rather than an afterthought. The
    cache is keyed on `id()`, and an `id()` is only unique while its object
    is alive, so the entry keeps a reference to every array it was keyed on.
    Those arrays are the pipeline's own inputs and outlive the loop regardless,
    so this costs nothing and closes the one way an identity cache goes
    wrong.
    """

    valid_nb: np.ndarray
    valid_bb: np.ndarray
    weight: np.ndarray
    held: tuple[Any, ...]


_BOUNDARY: dict[tuple[int, ...], _Boundary] = {}
"""One slot. A run conditions on one dataset, so a second entry is a bug."""


def boundary(
    single_base_nb_mean: np.ndarray,
    single_total_bb_RD: np.ndarray,
    smooth_mat: Any,
) -> _Boundary:
    """The seam's loop invariants, computed once per dataset.

    Measured as a **simplification** rather than a speedup: the two count
    passes are 25.2 ms at 3,000 x 5,000 and under two tenths of a per cent of
    the boundary (#59 item 4). What it buys is that a quantity which cannot
    change stops being recomputed, `max_iter_outer` times.
    """
    key = (
        id(single_base_nb_mean),
        id(single_total_bb_RD),
        id(smooth_mat) if smooth_mat is not None else 0,
    )

    cached = _BOUNDARY.get(key)

    if cached is not None:
        return cached

    valid_nb = (single_base_nb_mean > 0).sum(axis=0)
    valid_bb = (single_total_bb_RD > 0).sum(axis=0)

    weight = (
        _channel_weight(valid_nb, valid_bb, smooth_mat.indices, smooth_mat.indptr)
        if smooth_mat is not None
        else np.ones(single_base_nb_mean.shape[1], dtype=np.float64)
    )

    _BOUNDARY.clear()
    _BOUNDARY[key] = _Boundary(
        valid_nb=valid_nb,
        valid_bb=valid_bb,
        weight=weight,
        held=(single_base_nb_mean, single_total_bb_RD, smooth_mat),
    )

    return _BOUNDARY[key]


def _decoded(pred: np.ndarray, n_obs: int) -> np.ndarray:
    """`pred` as `(n_obs, n_clones)`, whichever form the caller passed.

    `cnaster` accepts both and reads `pred[c * n_obs + o]` from the flat one,
    so the clone-major reshape is the transpose of `(n_clones, n_obs)`. **The
    flat form is what the live pipeline passes** -- found by delegating on it
    and reading the log, not by reading the call sites -- so supporting it is
    the difference between a patch that runs and one that does not.
    """
    if pred.ndim == 2:
        return pred

    return np.ascontiguousarray(pred.reshape(-1, n_obs).T)


def _delegates(single_tumor_prop: Any) -> str | None:
    """Why this call goes to `cnaster`'s function, or `None` if it does not."""
    if single_tumor_prop is not None:
        return "the tumour-mixed field (#135)"

    return None


def _clone_shifts(
    hmmclass: Any, res: Any, decoded: np.ndarray, single_base_nb_mean: np.ndarray
) -> np.ndarray | None:
    """`log sum_g lambda_g mu_{s_c(g)}` per clone, or `None` when unshifted.

    Only when the fit was shifted (#276, #293): scoring spots with the shift
    off against rates fitted with it on is the inconsistency this repairs.
    `lambda` is built as `hmrf.py:476` builds `normal_lambda` -- the baseline
    summed over spots, normalized -- because that is what the fit's shift
    was taken against, and `decoded` is the `(n_obs, n_clones)` path the
    field reads.
    """
    import scipy.special

    from port.patch.plotting.clone_paths import state_vector

    if not getattr(hmmclass, "apply_logmu_shift", False):
        return None

    profile = np.asarray(single_base_nb_mean, dtype=np.float64).sum(axis=1)
    total = profile.sum()

    if total <= 0.0:
        return None

    with np.errstate(divide="ignore"):
        log_lambda = np.log(profile / total)

    rates = state_vector(res["new_log_mu"])
    terms = rates[np.asarray(decoded, dtype=np.int64)] + log_lambda[:, None]

    shifts: np.ndarray = scipy.special.logsumexp(terms, axis=0)

    return shifts


def pipeline_clone_assignment(
    single_X: np.ndarray,
    single_base_nb_mean: np.ndarray,
    single_total_bb_RD: np.ndarray,
    res: Any,
    pred: np.ndarray,
    adjacency_mat: Any,
    prev_assignment: np.ndarray,
    sample_ids: Any,
    spatial_weight: float,
    smooth_mat: Any = None,
    log_persample_weights: Any = None,
    single_tumor_prop: Any = None,
    hmmclass: Any = None,
    merge: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """What `cnaster.hmrf.pipeline_clone_assignment` returns, computed leaner."""
    import cnaster.hmrf as upstream

    from port.extensions.label_solver import label_solver, sweep_for
    from port.patch.hmrf.adjacency import adjacency_coo
    from port.patch.hmrf.fused_field import fused_spot_clone_field
    from port.patch.hmrf.refinement import compact, mask_for
    from port.patch.icm.floor import configured_floor, enforce_floor
    from port.patch.icm.floor import installed as floor_installed
    from port.patch.icm.interface import CsrGraph, fold_unary, icm_sweep
    from port.patch.plotting.clone_paths import state_vector

    reason = _delegates(single_tumor_prop)

    if reason is not None:
        logger.info_once(f"Delegating clone assignment to cnaster: {reason}.")

        return UPSTREAM(  # type: ignore[no-any-return]
            single_X,
            single_base_nb_mean,
            single_total_bb_RD,
            res,
            pred,
            adjacency_mat,
            prev_assignment,
            sample_ids,
            spatial_weight,
            smooth_mat=smooth_mat,
            log_persample_weights=log_persample_weights,
            single_tumor_prop=single_tumor_prop,
            hmmclass=hmmclass,
            merge=merge,
        )

    n_obs, _, n_spots = single_X.shape
    n_states = res["new_p_binom"].shape[0]

    decoded = _decoded(pred, n_obs)
    n_clones = decoded.shape[1]

    started = time.time()
    new_assignment = copy.copy(prev_assignment)

    logger.info(
        f"Solving (pooled) emission likelihood for X.shape={single_X.shape}, "
        f"n_states={n_states} and {n_clones} clones with {hmmclass.__name__}, "
        f"is_tumor_mixed=False and merge={merge}."
    )

    logger.info("Pooling hmrf data by smooth mat. (reduces necessary computation).")

    if smooth_mat is not None:
        pooled_X, pooled_base_nb_mean, pooled_total_bb_RD, _, _ = (
            upstream.pool_spatio_genomic_counts(
                single_X,
                single_base_nb_mean,
                single_total_bb_RD,
                smooth_mat.indices,
                smooth_mat.indptr,
                None,
                False,
            )
        )
    else:
        pooled_X = single_X.copy()
        pooled_base_nb_mean = single_base_nb_mean.copy()
        pooled_total_bb_RD = single_total_bb_RD.copy()

    # NB hoisted: all three are functions of the input data, which the outer
    #    loop never fits, and `cnaster` recomputes them per iteration (#59
    #    item 4).
    invariants = boundary(single_base_nb_mean, single_total_bb_RD, smooth_mat)

    # NB the two steps `cnaster` runs here -- build (n_states, n_obs, n_spots)
    #    per channel, then read one decoded state per (bin, clone) out of it --
    #    in one pass that materializes neither.
    shifts = _clone_shifts(hmmclass, res, decoded, single_base_nb_mean)

    if shifts is None:
        field = fused_spot_clone_field(
            pooled_X[:, 0, :],
            pooled_base_nb_mean,
            pooled_X[:, 1, :],
            pooled_total_bb_RD,
            # NB `(n_states,)`, normalized at the edge. The kernel indexes by
            #    state alone, because a state parameter has no second axis to
            #    index (#278).
            state_vector(res["new_log_mu"]),
            state_vector(res["new_alphas"]),
            state_vector(res["new_p_binom"]),
            state_vector(res["new_taus"]),
            decoded,
            invariants.weight,
            # NB the buffer is the caller's, which is upstream's shape --
            #    `external_field(..., field)` writes in place. It is allocated
            #    per call rather than reused, and that is deliberate: this
            #    function *returns* the field, so a reused buffer would
            #    rewrite an array its caller still holds, which is the defect
            #    `tests/test_seam_defects.py` pins on the solver. At 400 KB at
            #    the declared scale there is nothing to win by taking that
            #    risk.
            np.empty((n_spots, n_clones)),
        )
    else:
        # NB the shift is the **candidate** clone's, not the spot's current
        #    one: a spot scored against clone `c` is scored under `c`'s
        #    normalizer. It enters the mean as `base * exp(-shift_c)`, so each
        #    clone's column is the same kernel over a rescaled exposure --
        #    the same work as the one call, split by clone.
        #    Both factors are taken relative to the shifts' mean: the rates
        #    have no scale under the shift and drift along it (to near -7,024
        #    on #292's realization 3), so `exp(-shift)` and `exp(log_mu)` are
        #    each out of range while their product is not.
        field = np.empty((n_spots, n_clones))
        centre = float(np.mean(shifts))

        for clone in range(n_clones):
            column = fused_spot_clone_field(
                pooled_X[:, 0, :],
                pooled_base_nb_mean * np.exp(-(shifts[clone] - centre)),
                pooled_X[:, 1, :],
                pooled_total_bb_RD,
                state_vector(res["new_log_mu"]) - centre,
                state_vector(res["new_alphas"]),
                state_vector(res["new_p_binom"]),
                state_vector(res["new_taus"]),
                np.ascontiguousarray(decoded[:, clone : clone + 1]),
                invariants.weight,
                np.empty((n_spots, 1)),
            )
            field[:, clone] = column[:, 0]

    if get_global_config().hmrf.fixed_assignment:
        logger.warning("Assuming a fixed clone assignment")
    else:
        solver = label_solver()

        logger.info(f"Solving for updated clone assignment with {solver}.")

        # NB the solver takes the problem -- a unary field, one graph, one
        #    coupling -- rather than its call site (#59 item 5). The
        #    per-sample weights fold into the field, which is where they were
        #    added anyway, once per visit instead of once per sweep; the
        #    three adjacency arrays travel as the one graph they are.
        sweep = icm_sweep if solver == "icm" else sweep_for(solver)

        # NB the read-depth refinement's allowed-clone mask, which `cnaster`
        #    computes and drops (#348): into the field, so no move and no
        #    merge crosses a BAF clone, and into the floor, whose random
        #    reassignment reads nothing else. Absent, the call is as before.
        mask = mask_for(new_assignment, n_clones)
        knobs: dict[str, Any] = {} if mask is None else {"onehot_allowed_clones": mask}

        if mask is not None:
            unmasked = field
            field = np.where(mask, field, -np.inf)

        # NB the floor merged smallest first, into each spot's best clone,
        #    in place of the sweep's all-at-once random reassignment (#348):
        #    the sweep runs floorless, the floor is met after it, and a
        #    second sweep settles what the merge moved.
        folded = fold_unary(field, log_persample_weights, sample_ids)
        graph = CsrGraph.from_matrix(adjacency_mat)

        if floor_installed():
            knobs["min_clone_spots"] = 0

        result = sweep(folded, graph, new_assignment, spatial_weight, **knobs)

        if floor_installed():
            emptied = enforce_floor(folded, new_assignment, configured_floor())

            if emptied:
                logger.info(
                    f"Merged {emptied} clones under {configured_floor()} spots, "
                    "smallest first (#348)."
                )
                result = sweep(folded, graph, new_assignment, spatial_weight, **knobs)
                enforce_floor(folded, new_assignment, configured_floor())

        niter, new_cost = result.niter, result.cost

        logger.info(f"Ready for potential merging of clones?  {merge}.")

        # NB the COO triple is built here rather than above, and by three
        #    array expressions rather than two pure-Python passes over every
        #    non-zero (#59 item 3). `merge_assignment` is its only consumer,
        #    so on a run with `merge=False` `cnaster` computes the round trip
        #    and discards it.
        if merge:
            adj_spots, adj_neighbors, adj_weights = adjacency_coo(adjacency_mat)

        while merge:
            new_cost, best_merge_cost, best_merge_pair = upstream.merge_assignment(
                field,
                adj_spots,
                adj_neighbors,
                adj_weights,
                new_assignment,
                spatial_weight,
                log_persample_weights=log_persample_weights,
                sample_ids=sample_ids,
            )

            if best_merge_cost > new_cost:
                first, second = best_merge_pair
                merged = int((new_assignment == first).sum())

                new_assignment[new_assignment == first] = second

                logger.info(
                    f"Merged {merged} spots from clone {first} into clone {second} "
                    f"with new cost={best_merge_cost} given original "
                    f"cost={new_cost:.6e}."
                )

                new_cost = best_merge_cost
            else:
                logger.info(
                    f"No more beneficial merges available (best merge "
                    f"cost={best_merge_cost} given original cost={new_cost:.6e})."
                )
                break

        _, counts = np.unique(new_assignment, return_counts=True)

        logger.info(
            f"Found new clone assignment with new cost {new_cost:.6e} in {niter} "
            f"iterations ({time.time() - started:.2f}s with clone breakdown=\n"
            f"{[f'{share:.3f}' for share in counts / counts.sum()]})."
        )

        # NB `cnaster` compacts the surviving clones in ascending order
        #    (`hmrf.py:648`); the mask follows, so it still describes the
        #    next iteration's problem.
        if mask is not None:
            compact(new_assignment)
            field = unmasked

    logger.info("Computing ln likelihood for hmrf.")

    log_likelihood = float(
        np.sum(np.take_along_axis(field, new_assignment.astype(int)[:, None], axis=1))
    )

    rows, columns = adjacency_mat.nonzero()
    upper = rows < columns

    log_likelihood += spatial_weight * np.sum(
        new_assignment[rows[upper]] == new_assignment[columns[upper]]
    )

    return new_assignment, field, log_likelihood
