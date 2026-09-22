"""`compute_logmu_shifts` as an axis reduction, and the decision it is waiting on.

**#234 PR 2.** `cnaster.hmm_nophasing.compute_logmu_shifts` computes a
per-clone `logsumexp` and broadcasts it back over that clone's entries, using
a hand-rolled two-pass max-then-sum-exp over `start_idx`. Its own docstring
carries the vectorized form it replaced:

```
return scipy.special.logsumexp(
    log_mu[clone_copy_states, :] + normal_log_lambda.reshape(-1, 1),
    axis=0,
)
```

`shifts` is that reduction, returning **one value per clone** where upstream
returns one per segment -- see its docstring for why the shape is the point.
It stays a `numba` kernel: the vectorized form the comment sketches is
measurably slower than the compiled loop, so what this removes is the
broadcast write, not the loop.

## It is not installed, and that is the point

`hmm_nophasing.py:279` comments out the only call and logs
`"logmu_shifts are not currently supported."` So this replaces a function that
does not run, and installing it would not change any run -- but *enabling* it
would, because the shift debiases `log_mu` and every downstream number moves.

**Those are two different claims and only the first is made here.** This
module is pinned bitwise against the existing loop, which says the rewrite is
faithful. Whether the shift should be applied at all is a scientific question
about the library normalization it corrects for, and it needs the planted
truth as its referee rather than a loop nobody calls.
`tests/test_clone_stack.py::test_the_consumer_this_accessor_is_for_does_not_run`
fails the day that decision is taken elsewhere, so it cannot be taken
silently.

## Unequal clones are admitted, and are why there is no rectangular path

`clone_stack_obs` tiles `lengths` to `[A, B, A, B]`, so every clone in a
stacked run carries the same total. `compute_logmu_shifts` nonetheless takes
`clone_lengths` and walks them individually, which admits unequal clones, and
this walks them the same way rather than detecting the rectangular case: the
earlier version kept a `CloneStack` view for equal lengths and it is gone
with the vectorized reduction that needed it. One loop, both cases.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numba import njit

__all__ = ["shifts"]


@njit(nogil=True, cache=True, parallel=False, error_model="numpy")
def _per_clone(means, states, lambdas, lengths):
    """Upstream's two passes, writing one value per clone rather than per segment.

    Kept as `numba` and kept as upstream's shape of loop, because that is
    what the measurement says: a `scipy.special.logsumexp` over per-clone
    views is **2.1x slower** at the stress size and 3.9x at the gate one.
    The compiled two-pass is not the thing worth replacing; the write is.
    """
    n_clones = lengths.size
    out = np.empty(n_clones, dtype=np.float64)

    start = 0

    for clone in range(n_clones):
        length = lengths[clone]
        largest = -np.inf

        for i in range(length):
            value = means[states[start + i]] + lambdas[start + i]

            # NB `max` rather than the branch PLR1730 asks for: `numba`
            #    compiles the comparison, and the builtin on two floats is
            #    what upstream's own loop avoids for the same reason.
            largest = max(largest, value)

        if np.isinf(largest):
            out[clone] = largest
        else:
            total = 0.0

            for i in range(length):
                total += np.exp(means[states[start + i]] + lambdas[start + i] - largest)

            out[clone] = largest + np.log(total)

        start += length

    return out


def shifts(
    log_mus: np.ndarray,
    copy_states: np.ndarray,
    normal_log_lambda: np.ndarray,
    clone_lengths: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Per-clone `logsumexp` of `log_mus[state] + normal_log_lambda`.

    **`(n_clones,)`, where upstream returns `(n_segments,)`.** That is the
    one stated difference from `compute_logmu_shifts`, and it is a shape
    rather than a value: upstream writes each clone's shift across every one
    of that clone's segments, so its return carries `n_clones` distinct
    numbers in `n_segments` floats. `np.repeat(shifts(...), clone_lengths)`
    is upstream's array exactly, and
    `tests/test_logmu_shift.py::test_it_reproduces_cnasters_loop` is what
    holds that.

    The shape is the point rather than the bytes. A per-segment return has to
    be indexed by a running offset, and indexing it by clone -- which is what
    it looks like it wants -- silently hands every clone the first clone's
    shift, with no exception and no warning. One value per clone cannot be
    read that way. At the segment count `expected_runtime.tex` derives, 2.9e5,
    the difference is also 2.3 MB against a handful of numbers.

    Reproduces the values `compute_logmu_shifts` computes, including its
    handling of a clone whose every term is `-inf`: the loop leaves `max_val`
    at `-inf` and returns it rather than computing `log(0)`, and
    `scipy.special.logsumexp` returns `-inf` there too.

    Parameters
    ----------
    log_mus
        Per-state log means, indexed by `copy_states`.
    copy_states
        One state index per segment, over every clone concatenated.
    normal_log_lambda
        One value per segment, added to its state's `log_mu`.
    clone_lengths
        Segments per clone, in order. Their sum is the segment count.
    """
    states = np.asarray(copy_states, dtype=np.int64).reshape(-1)
    lambdas = np.asarray(normal_log_lambda, dtype=np.float64).reshape(-1)
    means = np.asarray(log_mus, dtype=np.float64).reshape(-1)
    lengths = np.asarray(clone_lengths, dtype=np.int64).reshape(-1)

    n_segments = states.size

    if lambdas.size != n_segments:
        msg = f"{lambdas.size} lambdas for {n_segments} segments"
        raise ValueError(msg)

    if int(lengths.sum()) != n_segments:
        msg = f"clone lengths sum to {int(lengths.sum())}, not {n_segments}"
        raise ValueError(msg)

    # NB the rectangular fast path, detected rather than assumed. Where the
    #    clones are equal the whole reduction is one call on a view that
    #    copies nothing; where they are not, a view cannot exist and the
    #    per-clone slices are still each contiguous.
    reduced: np.ndarray = _per_clone(means, states, lambdas, lengths)

    return reduced
