"""`port`'s axis reduction reproduces `cnaster`'s hand-rolled loop.

**#234 PR 2.** `compute_logmu_shifts` is a per-clone `logsumexp` written as a
two-pass max-then-sum-exp over `start_idx`. `port.patch.logmu_shift.shifts` is
the vectorized form the function's own docstring carries. These pin that the
two agree.

`patch`: this says the rewrite is faithful, not that the shift is right.
Whether it should be applied at all is a different claim with a different
referee, and `cnaster` does not currently apply it.
"""

from __future__ import annotations

import numpy as np
import pytest
from cnaster.hmm_nophasing import compute_logmu_shifts
from port.patch.logmu_shift import shifts

EXACT = 1e-12
"""Two summation orders over the same terms; not a tolerance on the science."""


def _case(
    lengths: list[int], n_states: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    rng = np.random.default_rng(seed)
    n_segments = sum(lengths)

    return (
        rng.normal(size=n_states),
        rng.integers(0, n_states, size=n_segments),
        rng.normal(size=n_segments),
        lengths,
    )


@pytest.mark.patch
@pytest.mark.parametrize(
    "lengths",
    [
        pytest.param([30, 30, 30], id="equal-the-rectangular-fast-path"),
        pytest.param([40, 25, 55], id="unequal-no-view-can-exist"),
        pytest.param([1, 99], id="one-segment-clone"),
        pytest.param([17], id="single-clone"),
    ],
)
def test_it_reproduces_cnasters_loop(lengths: list[int]) -> None:
    """Both paths, against the shipped function.

    The unequal case is the one that matters: with equal lengths the index
    arithmetic is a multiplication and an off-by-one cancels, so a test using
    only those would pass while checking nothing.
    """
    log_mus, copy_states, normal_log_lambda, clone_lengths = _case(lengths, 4, 17)

    theirs = compute_logmu_shifts(
        log_mus, copy_states, normal_log_lambda, clone_lengths
    )
    ours = shifts(log_mus, copy_states, normal_log_lambda, clone_lengths)

    assert ours.shape == theirs.shape
    assert np.allclose(theirs, ours, rtol=0.0, atol=EXACT), (
        f"max |difference| {np.max(np.abs(theirs - ours)):.3e}"
    )


@pytest.mark.patch
def test_a_clone_of_minus_infinities_stays_minus_infinity() -> None:
    """`cnaster` returns `max_val` rather than computing `log(0)`.

    The loop's one branch, and the case a naive `logsumexp` rewrite gets
    wrong by emitting a `nan`. Pinned because it is the difference between
    reproducing the function and reproducing its happy path.
    """
    log_mus = np.array([-np.inf, 0.5])
    copy_states = np.array([0, 0, 1, 1])
    normal_log_lambda = np.array([0.0, 0.0, 0.3, 0.7])
    clone_lengths = [2, 2]

    theirs = compute_logmu_shifts(
        log_mus, copy_states, normal_log_lambda, clone_lengths
    )
    ours = shifts(log_mus, copy_states, normal_log_lambda, clone_lengths)

    # NB one clone per assertion, so a failure names which of the two lost
    #    its `-inf` rather than reporting that the conjunction is false
    #    (`ruff` PT018).
    assert np.isneginf(theirs[0])
    assert np.isneginf(theirs[1])
    assert np.isneginf(ours[0])
    assert np.isneginf(ours[1])
    assert not np.any(np.isnan(ours)), "a nan here would be a silent wrong answer"
    assert np.allclose(theirs[2:], ours[2:], rtol=0.0, atol=EXACT)


@pytest.mark.patch
def test_the_broadcast_is_constant_within_each_clone() -> None:
    """One value per clone, repeated over its segments — the loop's assignment."""
    log_mus, copy_states, normal_log_lambda, clone_lengths = _case([12, 20], 3, 5)

    out = shifts(log_mus, copy_states, normal_log_lambda, clone_lengths)

    assert len(np.unique(out[:12])) == 1
    assert len(np.unique(out[12:])) == 1
    assert out[0] != out[12], "two clones sharing a shift would hide a bug"


@pytest.mark.patch
def test_mismatched_lengths_are_refused() -> None:
    """A length vector that does not sum to the segments is a caller error."""
    log_mus, copy_states, normal_log_lambda, _ = _case([10, 10], 3, 1)

    with pytest.raises(ValueError, match="sum to"):
        shifts(log_mus, copy_states, normal_log_lambda, [10, 5])

    with pytest.raises(ValueError, match="lambdas for"):
        shifts(log_mus, copy_states, normal_log_lambda[:-1], [10, 10])
