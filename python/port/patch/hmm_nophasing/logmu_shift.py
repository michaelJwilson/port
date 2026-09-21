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

`shifts` is that form, restored, via `CloneStack` where the clones are equal
length and per clone where they are not.

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

## Equal lengths are the rectangular case, and not the general one

`clone_stack_obs` tiles `lengths` to `[A, B, A, B]`, so every clone in a
stacked run carries the same total. `compute_logmu_shifts` nonetheless takes
`clone_lengths` and walks them individually, which admits unequal clones. A
rectangular `(n_clones, n_obs)` view cannot hold that, so `shifts` takes the
lengths and uses the view only when they are equal -- the fast path is a
special case it detects, never an assumption it makes.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.special

from port.patch.hmrf_utils import CloneStack

__all__ = ["shifts"]


def shifts(
    log_mus: np.ndarray,
    copy_states: np.ndarray,
    normal_log_lambda: np.ndarray,
    clone_lengths: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Per-clone `logsumexp` of `log_mus[state] + normal_log_lambda`, broadcast.

    Reproduces `cnaster.hmm_nophasing.compute_logmu_shifts` exactly, including
    its handling of a clone whose every term is `-inf`: the loop leaves
    `max_val` at `-inf` and returns it rather than computing `log(0)`, and
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

    terms = means[states] + lambdas

    # NB the rectangular fast path, detected rather than assumed. Where the
    #    clones are equal the whole reduction is one call on a view that
    #    copies nothing; where they are not, a view cannot exist and the
    #    per-clone slices are still each contiguous.
    if lengths.size and bool(np.all(lengths == lengths[0])):
        stack = CloneStack(terms, int(lengths.size), int(lengths[0]))
        per_clone = stack.per_clone(lambda view: scipy.special.logsumexp(view, axis=1))

        return stack.broadcast(per_clone)

    out = np.empty(n_segments, dtype=np.float64)
    start = 0

    for length in lengths:
        stop = start + int(length)
        out[start:stop] = scipy.special.logsumexp(terms[start:stop])
        start = stop

    return out
