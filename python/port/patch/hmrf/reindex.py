"""Replaces `cnaster.hmrf.reindex_clones`, enforcing the one-column contract (#278, #269).

Upstream asserts `new_p_binom.shape[1] == 1` yet reorders all four state
parameters under `shape[1] > 1`. This checks all four (raising, not asserting,
so `python -O` keeps it) and drops the reorder, which makes
`run_cnaster.py`'s `idx = s if ... shape[1] > 1 else 0` dead. `pred_cnv` keeps
both of upstream's layouts.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from cnaster.config import start_time
from cnaster.logger import get_logger

from port.patch._clone_paths import parameter_by_path, state_vector

logger = get_logger(__name__, start_time=start_time)

__all__ = ["PARAMETERS", "reindex_clones"]

PARAMETERS = ("new_log_mu", "new_alphas", "new_p_binom", "new_taus")
"""The four the M step fits, each of them one value per state."""

EPS_BAF = 0.05
"""Upstream's dead-band around a balanced BAF, kept at its value."""


def _state_parameters(res_combine: dict[str, Any]) -> None:
    """Raise unless every fitted parameter is one value per state (#267)."""
    for key in PARAMETERS:
        state_vector(res_combine[key], key)


def reindex_clones(
    res_combine: dict[str, Any],
    posterior: Any = None,
    single_tumor_prop: Any = None,
) -> tuple[dict[str, Any], Any]:
    """Upstream's, with the parameter contract enforced and the reorder gone."""
    if single_tumor_prop is not None:  # invariant
        msg = "single_tumor_prop must be None"
        raise AssertionError(msg)

    _state_parameters(res_combine)

    new_res_combine = copy.copy(res_combine)

    assignments = res_combine["new_assignment"]
    clone_labels = np.unique(assignments)
    n_clones = len(clone_labels)

    pred_cnv = np.asarray(res_combine["pred_cnv"])
    is_concatenated = pred_cnv.ndim == 1

    n_obs = len(pred_cnv) // n_clones if is_concatenated else pred_cnv.shape[0]

    # NB the path keeps both upstream layouts; only the parameter read narrows.
    probabilities = res_combine["new_p_binom"]

    baf_profiles = np.stack(
        [
            parameter_by_path(
                probabilities,
                pred_cnv[c * n_obs : (c + 1) * n_obs]
                if is_concatenated
                else pred_cnv[:, c],
            )
            for c in range(n_clones)
        ]
    )

    # NB the normal clone minimizes deviation from 0.5, outside the dead band.
    baf_penalty = np.maximum(np.abs(baf_profiles - 0.5) - EPS_BAF, 0)
    cid_normal = int(np.argmin(np.sum(baf_penalty, axis=1)))

    unique_clones, spot_counts = np.unique(assignments, return_counts=True)

    mask_rest = unique_clones != cid_normal
    cid_rest = unique_clones[mask_rest]
    counts_rest = spot_counts[mask_rest]

    cid_rest_sorted = cid_rest[np.argsort(counts_rest)]
    reidx = np.concatenate(([cid_normal], cid_rest_sorted)).astype(int)

    logger.info(
        f"Remapping clone index: {cid_normal} (normal) to 0, "
        f"otherwise sorted by spot count."
    )

    palette = np.zeros(int(np.max(unique_clones)) + 1, dtype=int)

    for new_idx, old_idx in enumerate(reidx):
        palette[old_idx] = new_idx

    new_res_combine["new_assignment"] = palette[assignments]

    # NB upstream's per-clone reorder of the four parameters is unreachable
    #    after `_state_parameters`, so it is gone.

    if is_concatenated:
        concat_idx = np.concatenate(
            [np.arange(c * n_obs, c * n_obs + n_obs) for c in reidx]
        )

        new_res_combine["pred_cnv"] = pred_cnv[concat_idx]

        # NB `.keys()`, as upstream: `CnaHMRFResult` defines no `__contains__`.
        if "log_gamma" in res_combine.keys():  # noqa: SIM118
            new_res_combine["log_gamma"] = res_combine["log_gamma"][:, concat_idx]

    else:
        if pred_cnv.shape[1] > 1:
            new_res_combine["pred_cnv"] = pred_cnv[:, reidx]

        if "log_gamma" in res_combine.keys():  # noqa: SIM118
            log_gamma = res_combine["log_gamma"]

            if log_gamma.ndim == 3 and log_gamma.shape[2] > 1:
                new_res_combine["log_gamma"] = log_gamma[:, :, reidx]

    if posterior is not None and posterior.shape[1] > 1:
        new_posterior = copy.copy(posterior)[:, reidx]
    else:
        new_posterior = posterior

    return new_res_combine, new_posterior
