"""`cnaster.hmrf.merge_by_minspots`, after CalicoST's Neyman-Pearson merge when installed (#497).

`cnaster` calls `merge_by_minspots` where CalicoST first merged similar clones
(`run_cnaster.py:743`, `:1172`, both commented out), so the minimum-size merge
is the one place the pipeline hands over a stage's clones. Under
`port.extensions.np_merge.np_merge()` this runs the Neyman-Pearson merge on the
stage's own pseudobulk -- the spot counts `run_core_inference` was fitted on,
summed per clone as `cnaster` sums them -- and then upstream's function on what
it leaves. Not installed, it is upstream's call unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from cnaster.config import start_time
from cnaster.hmrf import merge_by_minspots as UPSTREAM
from cnaster.logger import get_logger

from port.patch._signature import as_upstream

__all__ = ["UPSTREAM", "merge_by_minspots"]

logger = get_logger(__name__, start_time=start_time)


@as_upstream(UPSTREAM)
def merge_by_minspots(arguments: dict[str, Any]) -> Any:
    """Upstream's minimum-size merge, after the Neyman-Pearson merge where installed."""
    from port.extensions import np_merge

    assignment, res = arguments["assignment"], arguments["res"]
    held = np_merge._INPUTS
    source = res

    if held:
        import cnaster.pseudobulk

        labels = np.unique(np.asarray(assignment))
        pred = np.asarray(res["pred_cnv"])
        clone_index = [np.where(np.asarray(assignment) == c)[0] for c in labels]
        n_obs = np.asarray(held["single_X"]).shape[0]
        source = res

        if pred.size == n_obs * len(labels):
            X, base_nb_mean, total_bb_RD, _ = (
                cnaster.pseudobulk.merge_pseudobulk_by_index_mix(
                    held["single_X"],
                    held["single_base_nb_mean"],
                    held["single_total_bb_RD"],
                    clone_index,
                )
            )
            chosen = np_merge.groups(X, base_nb_mean, total_bb_RD, res, held["params"])

            # NB why each pair stays apart: the events that block its merge.
            for pair, events in np_merge.statistics(
                X, base_nb_mean, total_bb_RD, res, held["params"]
            ).items():
                blocking = [
                    (s1, s2, n, round(t, 3))
                    for s1, s2, n, t in events
                    if (n >= np_merge.MINLENGTH and t >= np_merge.THRESHOLD)
                    or (
                        n < np_merge.MINLENGTH
                        and n * t >= np_merge.THRESHOLD * np_merge.MINLENGTH
                    )
                ]
                logger.info(
                    f"Neyman-Pearson pair {pair}: blocking events (s1, s2, bins, t) "
                    f"{sorted(blocking, key=lambda e: -e[2] * e[3])[:3]}."
                )

            if len(chosen) < len(labels):
                logger.info(
                    f"Neyman-Pearson merge (#497): {len(labels)} clones into "
                    f"{len(chosen)}, groups {chosen}."
                )
                res = np_merge.merged(res, chosen)
                assignment = res["new_assignment"]
            else:
                logger.info(
                    f"Neyman-Pearson merge (#497): no pair of {len(labels)} clones merges."
                )
        else:
            logger.warning(
                "Neyman-Pearson merge (#497) skipped: the decode's clones are not the assignment's."
            )

    groups, result = UPSTREAM(**{**arguments, "assignment": assignment, "res": res})

    if held and "m" in held["params"]:
        # NB the read-depth stage: `cnaster` drops what this returns (#497).
        np_merge.hold(source, result)

    return groups, result
