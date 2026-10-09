"""Replaces `cnaster.hmrf.pipeline_clone_assignment` with a leaner seam (#206).

Installs #59's items: a fused field written into a buffer, the graph passed
once as CSR, invariants hoisted per dataset, the COO triple built only for
merging, and a reduced solver interface. Delegates to `cnaster` for a
tumour-mixed field (#135). Clone-size floor departs from `cnaster` (#468).
"""

from __future__ import annotations

import copy
import time
from typing import Any, NamedTuple

import numpy as np
from cnaster.config import get_global_config, start_time
from cnaster.hmrf import pipeline_clone_assignment as UPSTREAM
from cnaster.logger import get_logger

from port.patch.hmrf.invariants import BoundaryInvariants, boundary_invariants

__all__ = [
    "UPSTREAM",
    "PooledSmoothing",
    "boundary",
    "pipeline_clone_assignment",
    "release",
    "require_unpooled",
]

logger = get_logger(__name__, start_time=start_time)
"""`cnaster`'s function (`UPSTREAM`) is captured at import, before `patched()` rebinds it."""


class _Boundary(NamedTuple):
    """Per-dataset loop invariants (#59 item 4); `held` pins the `id()`-keyed arrays alive."""

    counts: BoundaryInvariants
    weight: np.ndarray
    held: tuple[Any, ...]


_BOUNDARY: dict[tuple[int, ...], _Boundary] = {}
"""One slot. A run conditions on one dataset, so a second entry is a bug."""


def release() -> None:
    """Drop the run's `id()`-keyed boundary; `port.pipeline.patched` calls this on exit (#517)."""
    _BOUNDARY.clear()


def _self_only(smooth_mat: Any) -> bool:
    """Whether every spot's only pooling neighbour is itself, at weight one."""
    import scipy.sparse as sp

    matrix = sp.csr_matrix(smooth_mat)
    n_spots = matrix.shape[0]
    return bool(
        matrix.shape == (n_spots, n_spots)
        and np.array_equal(matrix.indptr, np.arange(n_spots + 1))
        and np.array_equal(matrix.indices, np.arange(n_spots))
        and np.all(matrix.data == 1)
    )


class PooledSmoothing(ValueError):
    """A `smooth_mat` that pools a spot with any spot but itself (#513)."""


def require_unpooled(smooth_mat: Any) -> None:
    """Refuse a `smooth_mat` other than `None` or the identity (#513): counts are read unpooled."""
    if smooth_mat is None or _self_only(smooth_mat):
        return

    msg = (
        "smooth_mat pools spots with their neighbours; port's clone assignment "
        "reads counts unpooled and supports only the identity (#513)"
    )
    raise PooledSmoothing(msg)


def boundary(
    single_base_nb_mean: np.ndarray,
    single_total_bb_RD: np.ndarray,
    smooth_mat: Any,
) -> _Boundary:
    """The seam's loop invariants, computed once per dataset."""
    key = (
        id(single_base_nb_mean),
        id(single_total_bb_RD),
        id(smooth_mat) if smooth_mat is not None else 0,
    )

    cached = _BOUNDARY.get(key)

    if cached is not None:
        return cached

    counts = boundary_invariants(single_base_nb_mean, single_total_bb_RD)
    n_spots = single_base_nb_mean.shape[1]

    require_unpooled(smooth_mat)
    # NB under the identity each spot's neighbourhood is itself, so the
    #    pooled ratio is the spot's own; `None` keeps upstream's unit weight.
    weight = (
        counts.relative_channel_weight(np.arange(n_spots + 1), np.arange(n_spots))
        if smooth_mat is not None
        else np.ones(n_spots, dtype=np.float64)
    )

    _BOUNDARY.clear()
    _BOUNDARY[key] = _Boundary(
        counts=counts,
        weight=weight,
        held=(single_base_nb_mean, single_total_bb_RD, smooth_mat),
    )

    return _BOUNDARY[key]


def _decoded(pred: np.ndarray, n_obs: int) -> np.ndarray:
    """`pred` as `(n_obs, n_clones)`, from that or its flat clone-major form."""
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
    """`log sum_g lambda_g mu_{s_c(g)}` per clone, or `None` when unshifted (#276, #293).

    `lambda` is the normalized spot-summed baseline, as `hmrf.py:476` builds it.
    """
    from port.patch._clone_paths import state_vector
    from port.patch.hmm_nophasing.logmu_shift import clone_log_normalizers
    from port.patch.hmm_nophasing.shifted_emission import shifted

    if not shifted(hmmclass):
        return None

    return clone_log_normalizers(
        state_vector(res["new_log_mu"]), decoded, single_base_nb_mean
    )


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
    *,
    label_solver: str = "icm",
    floor_merge: bool = False,
    log_space: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """`cnaster.hmrf.pipeline_clone_assignment`'s return, computed leaner.

    `label_solver` names a `port.extensions.label_solver.SOLVERS` entry
    (`"icm"` is `cnaster`'s); `floor_merge` uses `port.patch.icm.floor`;
    `log_space` selects the log-space field kernels (#560, #561).
    Departure (#468, #617): the clone floor is `hmrf.min_spots_per_clone`
    where configured, else `cnaster`'s 200.
    """
    import cnaster.hmrf as upstream

    from port.extensions.label_solver import solver_for, sweep_for
    from port.patch._clone_paths import state_vector
    from port.patch.hmm_nophasing.shifted_emission import shifted
    from port.patch.hmrf.adjacency import adjacency_coo
    from port.patch.hmrf.refinement import MASK_PENALTY, compact, mask_for
    from port.patch.hmrf.tabulated_field import field_kernel, spot_clone_field
    from port.patch.icm.floor import configured_floor, floor_clones
    from port.patch.icm.interface import CsrGraph, fold_unary, icm_sweep

    reason = _delegates(single_tumor_prop)

    if reason is not None:
        logger.info_once(f"Delegating clone assignment to cnaster: {reason}.")

        # NB `cnaster`'s function reads none of these, so a run that asked is
        #    warned (#466).
        from port.patch.hmrf.refinement import kept

        dropped = [
            flag
            for flag, on in (
                ("--refinement-mask", kept()),
                ("--floor-merge", floor_merge),
                ("--shift", shifted(hmmclass)),
            )
            if on
        ]
        if dropped:
            logger.warning_once(
                f"{' and '.join(dropped)} not applied: clone assignment "
                f"delegates to cnaster for {reason}."
            )

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

    # NB no pooling (#513): `cnaster` only builds the identity `smooth_mat`.
    require_unpooled(smooth_mat)
    pooled_X = single_X
    pooled_base_nb_mean = single_base_nb_mean
    pooled_total_bb_RD = single_total_bb_RD

    # NB hoisted invariants of the input data (#59 item 4).
    invariants = boundary(single_base_nb_mean, single_total_bb_RD, smooth_mat)

    # NB build and reduce the field in one pass.
    shifts = _clone_shifts(hmmclass, res, decoded, single_base_nb_mean)

    if shifts is None:
        field = spot_clone_field(
            pooled_X[:, 0, :],
            pooled_base_nb_mean,
            pooled_X[:, 1, :],
            pooled_total_bb_RD,
            # NB `(n_states,)`, normalized at the edge (#278).
            state_vector(res["new_log_mu"]),
            state_vector(res["new_alphas"]),
            state_vector(res["new_p_binom"]),
            state_vector(res["new_taus"]),
            decoded,
            invariants.weight,
            # NB a fresh buffer per call, since the field is returned.
            np.empty((n_spots, n_clones)),
            log_space=log_space,
        )
    else:
        # NB each column uses the candidate clone's shift, as
        #    `base * exp(-shift_c)`; shifts and `log_mu` are centred on the
        #    shifts' mean to stay in floating-point range.
        field = np.empty((n_spots, n_clones))
        centre = float(np.mean(shifts))
        # NB one kernel and one exposure buffer for every clone (#488).
        kernel = field_kernel(
            pooled_X[:, 0, :],
            pooled_X[:, 1, :],
            pooled_total_bb_RD,
            log_space=log_space,
        )
        scaled = np.empty_like(pooled_base_nb_mean)

        for clone in range(n_clones):
            np.multiply(
                pooled_base_nb_mean, np.exp(-(shifts[clone] - centre)), out=scaled
            )
            column = kernel(
                pooled_X[:, 0, :],
                scaled,
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
        solver = solver_for(label_solver)

        logger.info(f"Solving for updated clone assignment with {solver}.")

        # NB the solver takes the problem, not its call site (#59 item 5).
        sweep = icm_sweep if solver == "icm" else sweep_for(solver)

        # NB the refinement's allowed-clone mask, which `cnaster` drops
        #    (#348): penalized in the field by `MASK_PENALTY` (#467).
        mask = mask_for(new_assignment, n_clones)
        knobs: dict[str, Any] = {} if mask is None else {"onehot_allowed_clones": mask}

        if mask is not None:
            unmasked = field
            field = np.where(mask, field, field - MASK_PENALTY)

        # NB with `floor_merge`, sweep floorless, merge smallest first, then
        #    sweep again (#348).
        folded = fold_unary(field, log_persample_weights, sample_ids)
        graph = CsrGraph.from_matrix(adjacency_mat)

        # NB the floor is `hmrf.min_spots_per_clone` where configured (#468).
        knobs["min_clone_spots"] = 0 if floor_merge else configured_floor()

        result = sweep(folded, graph, new_assignment, spatial_weight, **knobs)

        if floor_merge:
            emptied = floor_clones(folded, new_assignment, configured_floor())

            if emptied:
                logger.info(
                    f"Merged {emptied} clones under {configured_floor()} spots, "
                    "smallest first (#348)."
                )
                result = sweep(folded, graph, new_assignment, spatial_weight, **knobs)
                floor_clones(folded, new_assignment, configured_floor())

        niter, new_cost = result.niter, result.cost

        logger.info(f"Ready for potential merging of clones?  {merge}.")

        # NB the COO triple only where `merge_assignment` consumes it (#59 item 3).
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

        # NB `cnaster` compacts surviving clones (`hmrf.py:648`).
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
