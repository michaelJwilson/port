"""`cnaster.pseudobulk.merge_pseudobulk_by_index_mix`, summing a block of bins at a time.

**Proposed for `cnaster`, written here.** Upstream gathers every spot of a
clone, `single_X[:, :, idx]`, into a fresh `(n_obs, 2, len(idx))` array and
then sums it: at 6,000 spots the gather streams the whole count matrix through
memory once per clone and call, and the profile of #487's `--sal` run puts
24.4 s of self time over 31 calls here (#488). Gathering and summing
`BLOCK` bins at a time keeps the gathered block in cache. Every entry is the
same `np.sum` over the same spots in the same order, so the return is
**bitwise** upstream's, which `tests/test_pseudobulk_patch.py` pins.

The body is otherwise upstream's, line for line, logging through `cnaster`'s
own logger, so a run's log is unchanged.
"""

# ruff: noqa: G004, E501, N806, N803, PLR2004, PLW2901, B007, PLR1736
# NB upstream's body, kept line for line so the two diff.

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
from cnaster.pseudobulk import logger

__all__ = ["BLOCK", "merge_pseudobulk_by_index_mix"]

BLOCK = 256
"""Bins gathered and summed at once."""


def _blocks(n_obs: int) -> Iterator[slice]:
    """`BLOCK`-bin slices covering `range(n_obs)`, none of one bin unless all are.

    NB numpy sums a `(1, n)` block as one flat vector, in another order than
    the rows of a taller block, and the last bit differs (3.6e-15 at 257
    bins, `tests/test_pseudobulk_patch.py`); every height from 2 to 39 was
    measured equal. A trailing bin therefore joins the block before it.
    """
    starts = list(range(0, n_obs, BLOCK))

    if len(starts) > 1 and n_obs - starts[-1] == 1:
        starts.pop()

    for index, start in enumerate(starts):
        stop = starts[index + 1] if index + 1 < len(starts) else n_obs
        yield slice(start, stop)


def merge_pseudobulk_by_index_mix(
    single_X: np.ndarray,
    single_base_nb_mean: np.ndarray,
    single_total_bb_RD: np.ndarray,
    clone_index: Any,
    single_tumor_prop: np.ndarray | None = None,
    threshold: float = 0.5,
    normal_clone_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Upstream's pseudobulk per clone; see the module docstring."""
    n_obs = single_X.shape[0]

    # NB overloads 'spots' as clones.
    n_spots = len(clone_index)

    X = np.zeros((n_obs, 2, n_spots))

    base_nb_mean = np.zeros((n_obs, n_spots))
    total_bb_RD = np.zeros((n_obs, n_spots))

    tumor_prop = np.zeros(n_spots) if single_tumor_prop is not None else None

    for k, idx in enumerate(clone_index):
        if len(idx) == 0:
            logger.warning(f"Clone {k} has no cells, skipping")
            continue

        if single_tumor_prop is not None:
            logger.warning_once(
                f"Merging pseudobulk assigning threshold tumor proportion={threshold:.3f}"
            )

            # NB spots in this clone with a given proportion.
            tumor_mask = single_tumor_prop[idx] > threshold
            idx = idx[tumor_mask]

            # NB assumes mean tumor proportion for all spots assigned to this clone.
            tumor_prop[k] = np.mean(single_tumor_prop[idx]) if len(idx) > 0 else 0.0  # type: ignore[index]

        # NB upstream's `np.sum(x[..., idx], axis=-1)`, a block of rows at a
        #    time: each entry is the same sum over the same `idx` in the same
        #    order, and the gathered block stays in cache (#488).
        for rows in _blocks(n_obs):
            X[rows, :, k] = np.sum(single_X[rows, :, idx], axis=-1)
            total_bb_RD[rows, k] = np.sum(single_total_bb_RD[rows, idx], axis=1)
            base_nb_mean[rows, k] = np.sum(single_base_nb_mean[rows, idx], axis=1)

    for k, idx in enumerate(clone_index):
        percentiles = [50, 75, 90, 95, 99, 100]

        # TODO may be NAN if insufficient spots to aggregate at least one snp-covering umi for a segment.
        bafs = X[:, 1, k] / total_bb_RD[:, k]

        valid_rdr = base_nb_mean[:, k] > 0

        # NB base_nb_mean is initially non-defined.
        if np.any(valid_rdr):
            rdrs = X[:, 0, k] / base_nb_mean[:, k]
        else:
            rdrs = np.nan * np.ones_like(X[:, 0, k])

        logger.info(
            f"Found median (finite) baf={np.median(bafs[np.isfinite(bafs)]):.3f} for clone {k}."
        )
        logger.info(
            f"Found {len(idx)} spots, mean umis per spot={np.sum(X[:, 0, k]) / len(idx):.3f} and mean snp-covering umis per spot={np.sum(total_bb_RD[:, k]) / len(idx):.3f} for clone {k}"
        )

        if np.any(valid_rdr):
            if not np.isclose(
                np.nansum(X[:, 0, k]),
                np.nansum(base_nb_mean[:, k]),
                rtol=1e-5,
                atol=1e-6,
            ):
                logger.warning(
                    f"Expected consistency between normal baseline normalization total umi for the clone, {np.nansum(X[:, 0, k])} != {np.sum(base_nb_mean[:, k])}"
                )

            logger.info(
                f"Found median umis={np.median(X[:, 0, k])} and median RDR={np.median(rdrs[valid_rdr]):.3f} for clone {k} with {100.0 * np.mean(valid_rdr > 0.0):.3f}% valid."
            )
            logger.info(
                f"Found umi percentiles=\n{np.percentile(X[:, 0, k], percentiles)}\nfor\n{percentiles} [%]."
            )

    if normal_clone_index is not None:
        logger.warning(
            f"Assuming normal_clone_index={normal_clone_index} for pseudobulk baseline expression normalization."
        )

        normal_base_nb_mean = base_nb_mean[:, normal_clone_index].copy()

        for k, idx in enumerate(clone_index):
            new_base_nb_mean = (
                normal_base_nb_mean
                * len(clone_index[k])
                / len(clone_index[normal_clone_index])
            )
            logger.info(
                f"Calculated correction factors={len(clone_index[k]) / len(clone_index[normal_clone_index]):.3f} and {base_nb_mean[:, k].sum() / normal_base_nb_mean.sum():.3f}"
            )
            base_nb_mean[:, k] = new_base_nb_mean

        for k, idx in enumerate(clone_index):
            percentiles = [50, 75, 90, 95, 99, 100]

            bafs = X[:, 1, k] / total_bb_RD[:, k]

            valid_rdr = base_nb_mean[:, k] > 0
            rdrs = X[:, 0, k] / base_nb_mean[:, k]

            logger.info(f"Found median BAF={np.median(bafs):.3f} for clone {k}.")
            logger.info(
                f"Found {len(idx)} spots, mean UMIs per spot={np.sum(X[:, 0, k]) / len(idx):.3f} and mean snp-covering UMIs per spot={np.sum(total_bb_RD[:, k]) / len(idx):.3f} for clone {k}"
            )

            if np.any(valid_rdr):
                if not np.isclose(
                    np.nansum(X[:, 0, k]),
                    np.nansum(base_nb_mean[:, k]),
                    rtol=1e-5,
                    atol=1e-6,
                ):
                    logger.warning(
                        f"Expected consistency between normal baseline normalization total UMI for the clone, {np.nansum(X[:, 0, k])} != {np.sum(base_nb_mean[:, k])}"
                    )

                logger.info(
                    f"Found median UMIs={np.median(X[:, 0, k])} and median RDR={np.median(rdrs[valid_rdr]):.3f} for clone {k} with {100.0 * np.mean(valid_rdr > 0.0):.3f}% valid."
                )
                logger.info(
                    f"Found UMI percentiles=\n{np.percentile(X[:, 0, k], percentiles)}\nfor\n{percentiles} [%]."
                )

    logger.info(f"Merged single_X to pseudobulk of shape {X.shape[2]}.")

    return X, base_nb_mean, total_bb_RD, tumor_prop
