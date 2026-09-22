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

## `CloneStack` lives here

The accessor was `port.patch.clone_stack` when #259 installed the function,
and #250's naming rule moved it into the module named for the `cnaster` one it
serves. Both halves are in this file for that reason: the function `cnaster`
defines, and the layout it is read through.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from cnaster.hmrf_utils import clone_stack_obs as UPSTREAM

__all__ = ["UPSTREAM", "CloneStack", "channels", "channels_of", "clone_stack_obs"]


@dataclass(frozen=True)
class CloneStack:
    """A 1D clone-major buffer with its shape, and the two ways to read it.

    `values` is `n_clones * n_obs` long and clone-major, which is what
    `clone_stack_obs` already produces. Holding `n_clones` and `n_obs` beside
    it is the whole of the fix: the arithmetic stops being re-derived at every
    call site, and `view()` makes the per-clone axis a real axis.
    """

    values: np.ndarray
    n_clones: int
    n_obs: int

    def __post_init__(self) -> None:
        expected = self.n_clones * self.n_obs
        size = int(np.asarray(self.values).size)

        if size != expected:
            msg = (
                f"{size} values for {self.n_clones} clones of {self.n_obs} "
                "observations; a clone stack that does not divide is a "
                "different layout, not a shorter one"
            )
            raise ValueError(msg)

    def view(self) -> np.ndarray:
        """`(n_clones, n_obs)`, sharing memory with `values`.

        A reshape rather than a copy, and it stays a view because the buffer
        is clone-major: row `c` **is** `values[c * n_obs : (c + 1) * n_obs]`,
        which `tests/test_clone_stack.py` pins rather than assumes.
        """
        return np.asarray(self.values).reshape(self.n_clones, self.n_obs)

    def clone(self, index: int) -> np.ndarray:
        """One clone's observations, contiguous.

        Negative indices are refused rather than wrapped: `stack.clone(-1)`
        reading the last clone is a convenience that turns an off-by-one into
        silently correct-looking numbers for the wrong clone.
        """
        if not 0 <= index < self.n_clones:
            msg = f"clone {index} is outside 0..{self.n_clones - 1}"
            raise IndexError(msg)

        return np.asarray(self.view()[index])

    def per_clone(self, reduce: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """Apply a reduction along the observation axis, one value per clone.

        `reduce` takes the `(n_clones, n_obs)` view and returns `(n_clones,)`.
        This is the operation `compute_logmu_shifts` writes as a two-pass loop
        over `start_idx`.
        """
        reduced = np.asarray(reduce(self.view()))

        if reduced.shape != (self.n_clones,):
            msg = f"reduction returned {reduced.shape}, expected ({self.n_clones},)"
            raise ValueError(msg)

        return reduced

    def broadcast(self, per_clone_values: np.ndarray) -> np.ndarray:
        """One value per clone, back over that clone's observations.

        The second half of what `compute_logmu_shifts` does -- it assigns
        `logmu_shifts[start:start + clone_len] = shift_val` -- as a repeat.
        """
        values = np.asarray(per_clone_values).reshape(-1)

        if values.size != self.n_clones:
            msg = f"{values.size} values for {self.n_clones} clones"
            raise ValueError(msg)

        return np.repeat(values, self.n_obs)


def channels_of(stacked: np.ndarray, n_clones: int) -> tuple[CloneStack, ...]:
    """Split `clone_stack_obs`'s output into one contiguous buffer per channel.

    `stacked` is `(n_clones * n_obs, n_comp, 1)` or `(n_clones * n_obs,
    n_comp)`. Each returned buffer is C-contiguous, so a clone's series walks
    at unit stride instead of `n_comp`.

    The copy is deliberate and is the point: a view of the interleaved array
    would preserve the stride this exists to remove.
    """
    array = np.asarray(stacked)

    if array.ndim == 3:
        array = array[:, :, 0]

    rows, n_comp = array.shape

    if rows % n_clones:
        msg = f"{rows} rows do not divide into {n_clones} clones"
        raise ValueError(msg)

    n_obs = rows // n_clones

    return tuple(
        CloneStack(np.ascontiguousarray(array[:, channel]), n_clones, n_obs)
        for channel in range(n_comp)
    )


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
