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

## The outer loop, restated (#433)

`hmrf.py:407` schedules its rounds by reassigning the counter: after a round
that converges -- the assignment's ARI to the last one at the configured
tolerance, one clone left, or the counter at `max_iter_outer - 2` -- it sets
`r = max_iter_outer - 1` and turns merging on, which may move the counter
*backwards*. Read through, the schedule is plain:

- **unmerged rounds**, up to `max_iter_outer + 1`, stopping after the first
  that converges;
- **then two merged rounds**, only if one converged. A run that never
  converges never merges;
- **then the final fit**, on the last assignment's pseudobulk.

:func:`inference` runs exactly that, and is pinned bitwise against
upstream's (`tests/test_core_inference_loop.py`). Three things change
without moving a number: `clone_lengths` is the current clones' rather than
the initial ones' (`hmm_nophasing`'s shift already re-tiled it, #293), the
empty-clone inertia weight is named (:data:`EMPTY_CLONE_LOG_WEIGHT`), and the
scratch `res` dict is gone.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
from cnaster.hmrf import run_core_inference as UPSTREAM

__all__ = [
    "EMPTY_CLONE_LOG_WEIGHT",
    "UPSTREAM",
    "inference",
    "pin_neutral",
    "run_core_inference",
]

EMPTY_CLONE_LOG_WEIGHT = -50.0
"""`hmrf.py:740`'s log inertia for a clone with no spots in a sample, before normalization."""


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
    """:func:`inference`, then the neutral pin when the fit was shifted."""
    from port.patch.hmm_initialize import distinct

    # NB passed rather than rebound: upstream binds the initializer as a
    #    default argument (#348).
    if distinct.installed() and "hmm_initializer" not in kwargs:
        kwargs["hmm_initializer"] = distinct.gmm_init

    result = inference(*args, **kwargs)

    hmmclass = kwargs.get("hmmclass")
    shifted = bool(getattr(hmmclass, "apply_logmu_shift", False))

    if shifted and "m" in str(kwargs.get("params", "")):
        pin_neutral(result)

    return result


def _persample_weights(
    assignment: np.ndarray, sample_ids: np.ndarray, n_clones: int, n_samples: int
) -> np.ndarray:
    """Each sample's clone frequencies as normalized log weights, `(n_clones, n_samples)`."""
    import scipy.special

    weights: np.ndarray = np.ones((n_clones, n_samples)) * (-np.log(n_clones))

    for sample in range(n_samples):
        index = np.where(sample_ids == sample)[0]
        share = np.bincount(assignment[index], minlength=n_clones) / len(index)
        weights[:, sample] = np.where(share > 0, np.log(share), EMPTY_CLONE_LOG_WEIGHT)
        weights[:, sample] = weights[:, sample] - scipy.special.logsumexp(
            weights[:, sample]
        )

    return weights


def inference(*args: Any, **kwargs: Any) -> Any:
    """`cnaster.hmrf.run_core_inference`, with its schedule stated (module docstring).

    Takes upstream's arguments with upstream's defaults, read from its
    signature. Every stage is looked up on `cnaster.hmrf` at call time, so a
    swapped stage is the one that runs, as it is under upstream's loop.
    """
    from cnaster import hmrf

    bound = inspect.signature(UPSTREAM).bind(*args, **kwargs)
    bound.apply_defaults()
    a = bound.arguments

    single_X = a["single_X"]
    single_base_nb_mean = a["single_base_nb_mean"]
    single_total_bb_RD = a["single_total_bb_RD"]
    single_tumor_prop = a["single_tumor_prop"]
    lengths = a["lengths"]
    log_sitewise_transmat = a["log_sitewise_transmat"]
    n_states = a["n_states"]
    params = a["params"]
    max_iter_outer = a["max_iter_outer"]
    threshold = a["tumorprop_threshold"]

    n_obs = single_X.shape[0]
    n_clones = len(a["initial_clone_index"])
    _, sample_ids = np.unique(a["sample_ids"], return_inverse=True)
    n_samples = int(sample_ids.max()) + 1 if sample_ids.size else 0

    hmrf.logger.info(
        f"Running hmrfmix_concatenate_pipeline for {n_clones} clones and "
        f"{n_samples} samples/slices."
    )

    # NB the normal baseline's profile, `lambda_g`: summed over spots, then
    #    normalized, in place, as upstream divides it.
    normal_lambda = None

    if np.count_nonzero(single_base_nb_mean > 0.0):
        with np.errstate(divide="ignore", invalid="ignore"):
            normal_lambda = np.sum(single_base_nb_mean, axis=1)
            normal_lambda /= np.sum(single_base_nb_mean)
    else:
        hmrf.logger.warning("Found ill-defined normal baseline; baf only run.")

    def stacked(clone_index: list[np.ndarray]) -> tuple[Any, ...]:
        """The pseudobulk by clone, stacked along the genome, and its clone count."""
        X, base_nb_mean, total_bb_RD, tumor_prop = hmrf.merge_pseudobulk_by_index_mix(
            single_X,
            single_base_nb_mean,
            single_total_bb_RD,
            clone_index,
            single_tumor_prop,
            threshold=threshold,
        )
        stack = hmrf.clone_stack_obs(
            X, base_nb_mean, total_bb_RD, lengths, log_sitewise_transmat, tumor_prop
        )
        return (*stack, X.shape[2])

    stack = stacked(a["initial_clone_index"])

    init_log_mu, init_p_binom = a["init_log_mu"], a["init_p_binom"]
    init_alphas, init_taus = a["init_alphas"], a["init_taus"]

    if init_log_mu is None or init_p_binom is None:
        new_log_mu, new_p_binom, _, _ = a["hmm_initializer"](
            n_states,
            stack[0],
            stack[1],
            stack[2],
            params,
            stack[3],
            hmrf.get_log_transmat(n_states, a["t"]),
            stack[4],
            random_state=a["random_state"],
            in_log_space=False,
            only_minor=False,
        )

        if init_log_mu is None:
            init_log_mu = new_log_mu

        if init_p_binom is None:
            init_p_binom = new_p_binom

    last = {
        "log_mu": init_log_mu if "m" in params else None,
        "p_binom": init_p_binom if "p" in params else None,
        "alphas": init_alphas,
        "taus": init_taus,
    }
    assignment = np.zeros(single_X.shape[2], dtype=int)

    for clone, index in enumerate(a["initial_clone_index"]):
        assignment[index] = clone

    inertia = bool(hmrf.get_global_config().hmrf.inertia)
    persample = (
        np.ones((n_clones, n_samples)) * (-np.log(n_clones)) if inertia else None
    )

    def fit(stack: tuple[Any, ...], **extra: Any) -> Any:
        """One Baum-Welch fit on the stacked pseudobulk, from the last parameters."""
        X, base_nb_mean, total_bb_RD, stacked_lengths, sitewise, tumor_prop = stack[:6]

        return hmrf.pipeline_baum_welch(
            None,
            X,
            stacked_lengths,
            n_states,
            base_nb_mean,
            total_bb_RD,
            sitewise,
            tumor_prop,
            hmmclass=a["hmmclass"],
            params=params,
            t=a["t"],
            random_state=a["random_state"],
            fix_NB_dispersion=a["fix_NB_dispersion"],
            shared_NB_dispersion=a["shared_NB_dispersion"],
            fix_BB_dispersion=a["fix_BB_dispersion"],
            shared_BB_dispersion=a["shared_BB_dispersion"],
            is_diag=a["is_diag"],
            init_log_mu=last["log_mu"],
            init_p_binom=last["p_binom"],
            init_alphas=last["alphas"],
            init_taus=last["taus"],
            max_iter=a["max_iter"],
            tol=a["tol"],
            normal_lambda=normal_lambda,
            clone_lengths=np.full(stack[6], n_obs, dtype=int),
            init_log_gamma=None,
            **extra,
        )

    def one_round(
        stack: tuple[Any, ...], assignment: np.ndarray, merge: bool
    ) -> tuple[Any, tuple[Any, ...], np.ndarray]:
        """Fit, assign, re-index a lost clone, and re-pool."""
        res = fit(stack)
        pred = np.argmax(res["log_gamma"], axis=0)

        new_assignment, _, total_llf = hmrf.pipeline_clone_assignment(
            single_X,
            single_base_nb_mean,
            single_total_bb_RD,
            res,
            pred,
            a["adjacency_mat"],
            assignment,
            sample_ids,
            smooth_mat=a["smooth_mat"],
            spatial_weight=a["spatial_weight"],
            log_persample_weights=persample,
            single_tumor_prop=single_tumor_prop,
            hmmclass=a["hmmclass"],
            merge=merge,
        )

        remaining, reindexed = np.unique(new_assignment, return_inverse=True)

        if len(remaining) < stack[6]:
            hmrf.logger.warning("Detected clone loss: re-indexing clones.")
            new_assignment = reindexed
            rows = (remaining[:, None] * n_obs + np.arange(n_obs)).ravel()
            res["log_gamma"] = res["log_gamma"][:, rows]
            res["pred_cnv"] = res["pred_cnv"][rows]

        res["prev_assignment"] = assignment
        res["new_assignment"] = new_assignment
        res["total_llf"] = total_llf

        clone_index = [
            np.where(new_assignment == clone)[0] for clone in np.unique(new_assignment)
        ]

        return res, stacked(clone_index), pred

    def converged(res: Any, r: int) -> bool:
        """Upstream's three triggers, `r` counted after the round."""
        return bool(
            hmrf.adjusted_rand_score(res["prev_assignment"], res["new_assignment"])
            >= hmrf.get_global_config().hmrf.ari_tolerance
            or len(np.unique(res["new_assignment"])) == 1
            or r == max_iter_outer - 2
        )

    def advance(res: Any, n_current: int) -> np.ndarray:
        """Carry the fit and the assignment forward, and the inertia with them."""
        nonlocal persample

        last.update(
            log_mu=res["new_log_mu"],
            p_binom=res["new_p_binom"],
            alphas=res["new_alphas"],
            taus=res["new_taus"],
        )

        if inertia:
            persample = _persample_weights(
                res["new_assignment"], sample_ids, n_current, n_samples
            )

        new_assignment: np.ndarray = res["new_assignment"]
        return new_assignment

    res: Any = {}
    merging = False

    for r in range(max_iter_outer + 1):
        res, stack, _ = one_round(stack, assignment, merge=False)
        assignment = advance(res, stack[6])

        if converged(res, r + 1):
            merging = True
            break

    if merging:
        for _ in range(2):
            res, stack, _ = one_round(stack, assignment, merge=True)
            assignment = advance(res, stack[6])

    final = fit(stack, propagate_errors=a["propagate_hmm_param_errors"])

    res.params = final.params
    res.param_errors = final.param_errors if a["propagate_hmm_param_errors"] else None
    res.profile = final.profile

    if a["deconcatenate_clones"]:
        res["log_gamma"] = np.stack(
            [
                res["log_gamma"][:, (c * n_obs) : (c * n_obs + n_obs)]
                for c in range(len(np.unique(res["new_assignment"])))
            ],
            axis=-1,
        )
        res["pred_cnv"] = np.argmax(res["log_gamma"], axis=0)

    return res
