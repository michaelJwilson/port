"""A clone-stacked 1D buffer, addressed as `(n_clones, n_obs)`.

**#234 PR 1.** `cnaster.hmrf_utils.clone_stack_obs` concatenates clones along
the genomic axis, so the index is already `c * n_obs + t`. What it does not
give is an accessor: every consumer re-derives the arithmetic, and
`hmm_nophasing.compute_logmu_shifts` walks it with a hand-rolled `start_idx`
loop performing a per-clone `logsumexp` its own docstring shows the vectorized
form of.

## What the layout costs

Measured at `n_obs = 3,000`, `n_comp = 2`, `n_clones = 4`. `clone_stack_obs`
returns `(n_clones * n_obs, n_comp, 1)`, C-contiguous with strides
`(16, 8, 8)` bytes, so **one clone's one channel is strided by two elements**:
a kernel walking it loads a 64-byte line and uses four of its eight doubles.

| array | contiguous | stride |
| --- | --- | ---: |
| `clone_stack_obs` output, one clone, one channel | no | 2 elements |
| this module's per-channel buffer, viewed | yes | 1 element |

Splitting the channels into their own clone-major buffers makes `view()[c]`
**exactly** `flat[c * n_obs : (c + 1) * n_obs]`, so a per-clone reduction is
an axis reduction rather than a loop.

## What this module does not claim

**That it is faster.** The ticket proposes the layout "assuming this is
faster", and `CLAUDE.md` puts a 2x bar on a speedup claim measured at a stress
size. The honest prior is that this lands under it: halving wasted line-fill
helps a bandwidth-bound kernel, and `_nb_logpmf_1d` calls `lgamma` per element,
which is not. `tests/test_clone_stack_bench.py` reports the ratio and the
verdict either way; below the bar this is a simplification standing on its
evidence of equivalence.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

MIRRORS: tuple[str, ...] = ("cnaster.hmrf_utils",)
"""`hmrf_utils.clone_stack_obs`'s addressing, as an accessor.

The `cnaster` module this stands in for, or `()` where it stands in for
none (#250). Declared rather than inferred: a reader holding a `cnaster`
module open should be able to find `port`'s answer to it, and
`tests/test_module_correspondence.py` reads this to check that every swap
row lands in a module that admits to its target."""

__all__ = ["CloneStack", "channels_of"]


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
