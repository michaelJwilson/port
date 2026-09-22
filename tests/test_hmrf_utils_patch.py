"""`port`'s `clone_stack_obs` against `cnaster`'s, on every return.

**#234, installed on #259 stage 1.** The row is in `SWAPS`, so the claim is
bitwise: four call sites index the result and none of them may notice.

`patch`: this says the replacement computes what it replaces. Whether the
clone-major layout is *worth* anything is #238's question and was measured
at 1.007x, under the bar -- so the row stands on this equivalence and on
`channels()` removing the index arithmetic, not on speed.
"""

from __future__ import annotations

import numpy as np
import pytest
from cnaster.hmrf_utils import clone_stack_obs as upstream
from port.patch.hmrf_utils import channels, clone_stack_obs


def _case(
    n_obs: int, n_clones: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    return (
        rng.integers(0, 90, size=(n_obs, 2, n_clones)).astype(float),
        rng.random((n_obs, n_clones)),
        rng.random((n_obs, n_clones)),
        rng.normal(size=n_obs),
        rng.random(n_clones),
    )


@pytest.mark.patch
@pytest.mark.parametrize(
    ("n_obs", "n_clones", "lengths"),
    [
        pytest.param(12, 3, [5, 7], id="uneven-segments"),
        pytest.param(9, 1, [9], id="one-clone"),
        pytest.param(16, 4, [4, 4, 4, 4], id="even-segments"),
    ],
)
def test_it_returns_cnasters_six_values(
    n_obs: int, n_clones: int, lengths: list[int]
) -> None:
    """All six, bitwise, including the three that may be `None`.

    Parameterized over an uneven split because the even one is where an
    off-by-one in the reshape cancels: a test using only `[4, 4, 4, 4]`
    would pass while checking nothing.
    """
    X, base, total, transmat, tumor = _case(n_obs, n_clones, seed=11)

    ours = clone_stack_obs(X, base, total, lengths, transmat, tumor)
    theirs = upstream(X, base, total, lengths, transmat, tumor)

    assert len(ours) == len(theirs) == 6

    for index, (mine, yours) in enumerate(zip(ours, theirs, strict=True)):
        assert np.array_equal(np.asarray(mine), np.asarray(yours)), (
            f"return {index} differs"
        )
        assert np.asarray(mine).shape == np.asarray(yours).shape
        assert np.asarray(mine).dtype == np.asarray(yours).dtype


@pytest.mark.patch
def test_the_optional_arguments_stay_none() -> None:
    """Three call sites pass `None` for all three; upstream returns `None`."""
    X, base, total, _, _ = _case(10, 2, seed=3)

    ours = clone_stack_obs(X, base, total, None, None, None)
    theirs = upstream(X, base, total, None, None, None)

    assert ours[3] is theirs[3] is None
    assert ours[4] is theirs[4] is None
    assert ours[5] is theirs[5] is None


@pytest.mark.patch
def test_the_accessor_is_the_same_data_at_unit_stride() -> None:
    """`channels()` is a view of the same numbers, clone-major and contiguous.

    The layout the row exists to make available: upstream's output strides a
    clone's one channel by `n_comp`, and each of these is contiguous, so a
    per-clone reduction is an axis reduction.
    """
    X, base, total, _, _ = _case(12, 3, seed=5)

    stacked = clone_stack_obs(X, base, total, None, None, None)[0]
    per_channel = channels(stacked, 3)

    assert len(per_channel) == 2

    for channel, buffer in enumerate(per_channel):
        assert buffer.view().shape == (3, 12)
        assert buffer.view().flags["C_CONTIGUOUS"]

        for clone in range(3):
            assert np.array_equal(buffer.view()[clone], X[:, channel, clone])
