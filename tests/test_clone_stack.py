"""The clone-stacked layout the shifted emission reads, and the loop it replaced.

**#234 PR 1**, folded into the `hmm_nophasing` patch by #349. `CountEncoder`
keeps a view of the clone-stacked `X`, `(n_clones * n_obs, 2, 1)`, so one
channel walks at a stride of two elements; `shifted_emission._clone_major`
copies it to one contiguous clone-major buffer and tags each entry with its
clone. It replaced `port.patch.hmrf_utils` (`CloneStack`, `channels_of`),
which no run called. These pin:

- the buffer is contiguous and **is** the channel, clone after clone, for
  unequal clone lengths (`patch`);
- `_triples` built from it is bitwise what the pre-#349 build was (`patch`);
- `compute_logmu_shifts`' `start_idx` walk is the per-clone `logsumexp`
  (`patch`), and upstream still does not call it (`bug`).
"""

from __future__ import annotations

import numpy as np
import pytest
from cnaster.hmm_nophasing import compute_logmu_shifts
from port.patch.hmm_nophasing.shifted_emission import _clone_major, _triples


def _stacked(lengths: tuple[int, ...], seed: int = 17) -> np.ndarray:
    """A clone-stacked `X`, `(sum(lengths), 2, 1)`, as `clone_stack_obs` returns."""
    rng = np.random.default_rng(seed)
    rows = int(sum(lengths))
    stacked = np.empty((rows, 2, 1))
    stacked[:, 0, 0] = rng.poisson(40.0, size=rows)
    stacked[:, 1, 0] = rng.binomial(60, 0.3, size=rows)
    return stacked


@pytest.mark.patch
def test_the_buffer_is_the_channel_contiguous_and_clone_tagged() -> None:
    """Unequal lengths, so an off-by-one in the tiling cannot cancel (#234).

    The encoder's channel is strided by two; the buffer is contiguous, equal
    to it entry for entry, and entry `t` of clone `c`'s block carries `c`.
    """
    lengths = (40, 25, 55)
    stacked = _stacked(lengths)
    channel = stacked[:, 0, :]

    assert channel.strides[0] // channel.itemsize == 2

    values, clones = _clone_major(channel, lengths)

    assert values.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(values, stacked[:, 0, 0])
    np.testing.assert_array_equal(clones, np.repeat([0, 1, 2], lengths))


@pytest.mark.patch
def test_lengths_that_do_not_tile_the_channel_are_refused() -> None:
    with pytest.raises(ValueError, match="tile the array"):
        _clone_major(_stacked((3, 3))[:, 0, :], (3, 4))


@pytest.mark.patch
def test_the_triples_are_bitwise_the_pre_fold_build(cnaster_config: None) -> None:
    """`_triples` on the strided channel, against #276's build verbatim.

    Float counts, so the configured rounding runs on both sides as it does in
    a fit (`CountEncoder` is built on `X`, which is float).
    """
    from cnaster.config import get_global_config

    lengths = (300, 300, 300)
    stacked = _stacked(lengths, seed=5)
    obs, total = stacked[:, 0, :], stacked[:, 1, :]

    clones = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
    counts = np.column_stack(
        [clones.astype(np.float64), obs.reshape(-1), total.reshape(-1)]
    )
    counts = counts.round(decimals=get_global_config().hmm.compression_decimals)
    unique, inverse = np.unique(counts, axis=0, return_inverse=True)
    triples = _triples(obs, total, lengths)

    np.testing.assert_array_equal(triples.obs, unique[:, 1])
    np.testing.assert_array_equal(triples.total, unique[:, 2])
    np.testing.assert_array_equal(triples.inverse, inverse.reshape(-1))
    np.testing.assert_array_equal(
        triples.bounds, np.searchsorted(unique[:, 0], np.arange(len(lengths) + 1))
    )


@pytest.mark.patch
def test_the_per_clone_reduction_reproduces_cnasters_loop() -> None:
    """`logsumexp(axis=1)` against `compute_logmu_shifts`'s `start_idx` walk.

    The clone lengths are deliberately **unequal**. With equal lengths the
    index arithmetic is a multiplication and any off-by-one cancels, so the
    referee would be vacuous — which is the trap #234's plan names.
    """
    import scipy.special

    rng = np.random.default_rng(17)
    clone_lengths = [40, 25, 55]
    n_segments = sum(clone_lengths)
    n_states = 4

    log_mus = rng.normal(size=n_states)
    copy_states = rng.integers(0, n_states, size=n_segments)
    normal_log_lambda = rng.normal(size=n_segments)

    theirs = compute_logmu_shifts(
        log_mus, copy_states, normal_log_lambda, clone_lengths
    )

    # NB the same quantity via one clone at a time because the lengths
    #    differ, which is the case a rectangular view cannot hold.
    ours = np.empty(n_segments)
    start = 0

    for length in clone_lengths:
        block = (
            log_mus[copy_states[start : start + length]]
            + normal_log_lambda[start : start + length]
        )
        ours[start : start + length] = scipy.special.logsumexp(block)
        start += length

    assert np.allclose(theirs, ours, rtol=0.0, atol=1e-12), (
        f"max |difference| {np.max(np.abs(theirs - ours)):.3e}"
    )


@pytest.mark.bug
def test_the_consumer_this_accessor_is_for_does_not_run() -> None:
    """`compute_logmu_shifts` is dead code, and #234 PR 2 has to decide it.

    `hmm_nophasing.py:279` comments out the only call and logs
    `"logmu_shifts are not currently supported."` So the per-clone shift
    upstream defines is computed nowhere in an unpatched run.

    Written as a `bug` pin against `cnaster`'s own contract -- the function is
    defined, documented and unreachable -- so it **fails** the day the call is
    restored, which is when PR 2 must judge the shift against the planted
    truth rather than merely reproducing a loop.
    """
    import inspect

    import cnaster.hmm_nophasing as nophasing

    source = inspect.getsource(nophasing)

    assert "logmu_shifts are not currently supported" in source, (
        "the warning is gone, so the call may be live: #234 PR 2 now owes a "
        "validation of the shift against planted truth, not just equivalence"
    )
    assert "# logmu_shifts = compute_logmu_shifts(" in source, (
        "the call is no longer commented out"
    )
