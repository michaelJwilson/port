"""`reindex_clones`, making the one-column contract hold (#278, #269).

**This is how `scripts/run_cnaster.py:1366` gets patched without forking it.**

That line reads

    idx = s if res_combine["new_log_mu"].shape[1] > 1 else 0

and sits inside `run_cnaster`, which is 1,648 lines. A drop-in replacement
for that is a fork, not a patch. But the line is a *read* of a contract
established elsewhere: `reindex_clones` is called at `run_cnaster.py:1269`,
ninety-seven lines earlier, and it is 87 lines. Establish the contract there
and the branch below is provably dead -- `idx` is `0` because it cannot be
anything else, rather than because nothing has produced a second column yet.

## What changes

`cnaster` already asserts the contract for one of the four parameters::

    assert res_combine["new_p_binom"].shape[1] == 1

and then, forty lines later, reorders all four **as though it did not**::

    for key in ["new_log_mu", "new_alphas", "new_p_binom", "new_taus"]:
        if res_combine[key].shape[1] > 1:
            new_res_combine[key] = res_combine[key][:, reidx]

One function, both readings. The assert makes the branch unreachable for the
parameter it names and leaves it reachable in principle for the other three,
which is the shape of #267's finding in miniature.

This extends the check to all four and drops the reorder. The check raises
rather than asserts: `assert` vanishes under `python -O`, and a contract that
disappears when optimizations are on is not one.

**`pred_cnv` is untouched.** Reindexing a path by clone order is a different
job from reading a state parameter, and upstream's handling of both layouts
there is deliberate. The narrowing is the parameter axis only, which is what
the deprecation is about.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from cnaster.config import start_time
from cnaster.logger import get_logger

from port.patch.plot_genomic.clone_paths import parameter_by_path, state_vector

logger = get_logger(__name__, start_time=start_time)

__all__ = ["PARAMETERS", "reindex_clones"]

PARAMETERS = ("new_log_mu", "new_alphas", "new_p_binom", "new_taus")
"""The four the M step fits, each of them one value per state."""

EPS_BAF = 0.05
"""Upstream's dead-band around a balanced BAF, kept at its value."""


def _state_parameters(res_combine: dict[str, Any]) -> None:
    """Every fitted parameter is one value per state, or the run stops.

    `clone_stack_obs` reshapes observations to `(-1, n_comp, 1)` and
    `get_initial_params` refuses `n_spots != 1`, so a second column means the
    fit changed. Every consumer downstream reads one value per state -- #267
    found three incompatible readings of what another column would mean -- so
    stopping here is the only behaviour that cannot be silently wrong.

    `state_vector` is the one guard, and it carries the message. Restating
    the check here would make two places to keep in agreement.
    """
    for key in PARAMETERS:
        state_vector(res_combine[key], key)


def reindex_clones(
    res_combine: dict[str, Any],
    posterior: Any = None,
    single_tumor_prop: Any = None,
) -> tuple[dict[str, Any], Any]:
    """Upstream's, with the parameter contract enforced and the reorder gone."""
    assert single_tumor_prop is None, "single_tumor_prop must be None"

    _state_parameters(res_combine)

    new_res_combine = copy.copy(res_combine)

    assignments = res_combine["new_assignment"]
    clone_labels = np.unique(assignments)
    n_clones = len(clone_labels)

    pred_cnv = np.asarray(res_combine["pred_cnv"])
    is_concatenated = pred_cnv.ndim == 1

    n_obs = len(pred_cnv) // n_clones if is_concatenated else pred_cnv.shape[0]

    # NB the path keeps both of upstream's layouts, for the reason the module
    #    docstring gives; what narrows is the parameter read beside it, from
    #    `new_p_binom[path, 0]` to one value per state.
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

    # NB upstream reorders the four parameters by clone here, under
    #    `if shape[1] > 1`. `_state_parameters` above is what makes that branch
    #    unreachable, so it is gone rather than left as an unreachable
    #    reading of an axis that has one meaning.

    if is_concatenated:
        concat_idx = np.concatenate(
            [np.arange(c * n_obs, c * n_obs + n_obs) for c in reidx]
        )

        new_res_combine["pred_cnv"] = pred_cnv[concat_idx]

        if "log_gamma" in res_combine:
            new_res_combine["log_gamma"] = res_combine["log_gamma"][:, concat_idx]

    else:
        if pred_cnv.shape[1] > 1:
            new_res_combine["pred_cnv"] = pred_cnv[:, reidx]

        if "log_gamma" in res_combine:
            log_gamma = res_combine["log_gamma"]

            if log_gamma.ndim == 3 and log_gamma.shape[2] > 1:
                new_res_combine["log_gamma"] = log_gamma[:, :, reidx]

    if posterior is not None and posterior.shape[1] > 1:
        new_posterior = copy.copy(posterior)[:, reidx]
    else:
        new_posterior = posterior

    return new_res_combine, new_posterior
