"""`clone_stack_obs`, with the clone-major buffers its consumers walk.

**#234, installed by #259 stage 1.** `cnaster.hmrf_utils.clone_stack_obs`
concatenates clones along the genomic axis and returns
`(n_clones * n_obs, n_comp, 1)`, C-contiguous with strides `(16, 8, 8)`
bytes. **One clone's one channel is therefore strided by two elements**: a
kernel walking it loads a 64-byte line and uses four of its eight doubles.

Three call sites consume it -- `hmrf.py:502`, `hmrf.py:689` and
`hmm_initialize.py:98`, `:685` -- and every one of them re-derives the index
arithmetic `c * n_obs + t` for itself.

## What this returns, and what it does not change

**The same six values, bitwise.** Shapes, dtypes and contents are upstream's,
asserted in `tests/test_hmrf_utils_patch.py` against the function itself, so
the row belongs in `SWAPS` rather than in `NUMERIC_SWAPS`: every consumer
that indexes the result keeps working unchanged.

What it adds is `channels(...)`: the same data as one contiguous
`CloneStack` per channel, so `view()[c]` is exactly
`flat[c * n_obs : (c + 1) * n_obs]` and a per-clone reduction is an axis
reduction rather than a `start_idx` loop. Nothing in `cnaster` calls it yet;
it is the accessor #259 stages 2-5 reach for, and it is here rather than in a
later stage so the layout arrives with the function that produces it.

**This is not a speedup claim.** #238 measured the layout at **1.007x** at
the gate size, under `CLAUDE.md`'s 2x bar, and the module that measured it
says why: halving wasted line-fill helps a bandwidth-bound kernel, and
`_nb_logpmf_1d` calls `lgamma` per element, which is not one. The row stands
on equivalence and on removing the index arithmetic from four call sites.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from cnaster.hmrf_utils import clone_stack_obs as UPSTREAM

from port.patch.clone_stack import CloneStack, channels_of

__all__ = ["UPSTREAM", "channels", "clone_stack_obs"]


def clone_stack_obs(
    X: np.ndarray,
    base_nb_mean: np.ndarray,
    total_bb_RD: np.ndarray,
    lengths: Any,
    log_sitewise_transmat: Any,
    tumor_prop: Any,
) -> tuple[Any, ...]:
    """Upstream's six values, from one transpose-reshape per array.

    `X.transpose(2, 0, 1).reshape(-1, n_comp, 1)` is upstream's own comment's
    claim -- the `flatten("F")` plus `vstack` it replaced, in one C-level
    copy -- and is kept verbatim. The two covariates transpose and reshape
    the same way; `lengths` and the transition tile; `tumor_prop` repeats.

    Returns
    -------
    tuple
        `(X, base_nb_mean, total_bb_RD, lengths, log_sitewise_transmat,
        tumor_prop)`, clone-concatenated, exactly as upstream returns them.
    """
    n_obs, n_comp, n_clones = X.shape

    stacked_X = X.transpose(2, 0, 1).reshape(-1, n_comp, 1)
    stacked_base = base_nb_mean.T.reshape(-1, 1)
    stacked_total = total_bb_RD.T.reshape(-1, 1)

    stacked_lengths = None if lengths is None else np.tile(lengths, n_clones)
    stacked_transmat = (
        None
        if log_sitewise_transmat is None
        else np.tile(log_sitewise_transmat, n_clones)
    )
    stacked_tumor = (
        None if tumor_prop is None else np.repeat(tumor_prop, n_obs).reshape(-1, 1)
    )

    return (
        stacked_X,
        stacked_base,
        stacked_total,
        stacked_lengths,
        stacked_transmat,
        stacked_tumor,
    )


def channels(stacked_X: np.ndarray, n_clones: int) -> tuple[CloneStack, ...]:
    """The stacked observations as one contiguous buffer per channel.

    The accessor the four call sites open-code. `channels(...)[0].view()[c]`
    is clone `c`'s total-count series at unit stride, and `per_clone` takes a
    reduction over it without an index loop.

    Parameters
    ----------
    stacked_X : np.ndarray
        `clone_stack_obs`'s first return.
    n_clones : int
        How many clones it concatenates.
    """
    return channels_of(stacked_X, n_clones)
