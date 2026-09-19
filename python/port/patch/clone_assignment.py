"""`cnaster.hmrf.pipeline_clone_assignment`, without the array or the round trip.

**The seam, rebound one level up (#206).** `port` has carried measured
patches for #59 items 1-4 since [#125](https://github.com/michaelJwilson/port/pull/125)
and could install none of them: each needs a call-site edit inside
`cnaster.hmrf`, which `CLAUDE.md` makes read only. They install here,
because `pipeline_clone_assignment` is itself a module-level name and
rebinding *it* rebinds its call sites with it.

Two changes, and neither moves a number:

*   **The field is fused.** `cnaster` materializes
    `(n_states, n_obs, n_spots)` per channel and then reduces it to
    `(n_spots, n_clones)` by reading one decoded state per `(bin, clone)`.
    `port.patch.hmrf_fused_field` does both in one pass and materializes
    nothing -- 8 GB at the declared scale, twice per outer iteration (#90),
    for an array whose only consumer is the reduction. Pinned **bitwise**
    against `cnaster`'s two-step in `tests/test_hmrf_fused_field.py`.
*   **The COO triple is built only when it is used.** `cast_csr` and
    `unpack_adjacency` walk every non-zero in pure Python to produce
    `adj_spots`, `adj_neighbors` and `adj_weights`, and the solver reads the
    CSR arrays instead. Their one consumer is `merge_assignment`, inside
    `while merge:` -- so on a run with `merge=False` the whole round trip is
    computed and discarded (#59 item 3).

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
*   more than one parameter column -- `log_mu[state, 0]` is what the fused
    kernel reads, which is the clone-stacked layout every live call uses.
"""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
from cnaster.config import get_global_config, start_time
from cnaster.hmrf import pipeline_clone_assignment as UPSTREAM
from cnaster.logger import get_logger

__all__ = ["UPSTREAM", "pipeline_clone_assignment"]

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


def _delegates(single_tumor_prop: Any, log_mu: np.ndarray) -> str | None:
    """Why this call goes to `cnaster`'s function, or `None` if it does not."""
    if single_tumor_prop is not None:
        return "the tumour-mixed field (#135)"

    if log_mu.shape[1] != 1:
        return f"{log_mu.shape[1]} parameter columns, against the clone-stacked one"

    return None


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

    from port.patch.hmrf_fused_field import fused_spot_clone_field

    reason = _delegates(single_tumor_prop, res["new_log_mu"])

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

    valid_nb = (single_base_nb_mean > 0).sum(axis=0)
    valid_bb = (single_total_bb_RD > 0).sum(axis=0)

    weight = (
        _channel_weight(valid_nb, valid_bb, smooth_mat.indices, smooth_mat.indptr)
        if smooth_mat is not None
        else np.ones(n_spots, dtype=np.float64)
    )

    # NB the two steps `cnaster` runs here -- build (n_states, n_obs, n_spots)
    #    per channel, then read one decoded state per (bin, clone) out of it --
    #    in one pass that materializes neither.
    field = fused_spot_clone_field(
        pooled_X[:, 0, :],
        pooled_base_nb_mean,
        pooled_X[:, 1, :],
        pooled_total_bb_RD,
        res["new_log_mu"],
        res["new_alphas"],
        res["new_p_binom"],
        res["new_taus"],
        decoded,
        weight,
    )

    if get_global_config().hmrf.fixed_assignment:
        logger.warning("Assuming a fixed clone assignment")
    else:
        logger.info("Solving for updated clone assignment with icm_sweep_dequeue.")

        niter, new_cost = upstream.icm_sweep_deque(
            single_llf=field,
            adj_indptr=adjacency_mat.indptr,
            adj_indices=adjacency_mat.indices,
            adj_weights=adjacency_mat.data,
            new_assignment=new_assignment,
            spatial_weight=spatial_weight,
            posterior=None,
            onehot_allowed_clones=None,
            log_persample_weights=log_persample_weights,
            sample_ids=sample_ids,
        )

        logger.info(f"Ready for potential merging of clones?  {merge}.")

        # NB the COO triple is built here rather than above: `merge_assignment`
        #    is its only consumer, and `cast_csr` plus `unpack_adjacency` are
        #    two pure-Python passes over every non-zero (#59 item 3).
        if merge:
            adj_spots, adj_neighbors, adj_weights = upstream.unpack_adjacency(
                upstream.cast_csr(adjacency_mat)
            )

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
