"""`shifted_emission`'s clone-major buffer against the strided channel (#234, #349).

`compute_logmu_shifts`' loop is reproduced (`patch`); upstream never calls it (`bug`).
"""

from __future__ import annotations

import inspect

import cnaster.hmm_nophasing as nophasing
import numpy as np
import pytest
import scipy.special
from cnaster.config import get_global_config
from cnaster.hmm_nophasing import compute_logmu_shifts
from port.patch.hmm_nophasing.shifted_emission import _clone_major, clone_count_triples


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
    """Buffer is contiguous, equals the channel entry for entry, and is clone-tagged (#234)."""
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
    """`clone_count_triples` is bitwise #276's build on float counts."""

    lengths = (300, 300, 300)
    stacked = _stacked(lengths, seed=5)
    obs, total = stacked[:, 0, :], stacked[:, 1, :]

    clones = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
    counts = np.column_stack(
        [clones.astype(np.float64), obs.reshape(-1), total.reshape(-1)]
    )
    counts = counts.round(decimals=get_global_config().hmm.compression_decimals)
    unique, inverse = np.unique(counts, axis=0, return_inverse=True)
    triples = clone_count_triples(obs, total, lengths)

    np.testing.assert_array_equal(triples.obs, unique[:, 1])
    np.testing.assert_array_equal(triples.total, unique[:, 2])
    np.testing.assert_array_equal(triples.inverse, inverse.reshape(-1))
    np.testing.assert_array_equal(
        triples.bounds, np.searchsorted(unique[:, 0], np.arange(len(lengths) + 1))
    )


@pytest.mark.patch
def test_the_per_clone_reduction_reproduces_cnasters_loop() -> None:
    """`logsumexp(axis=1)` reproduces `compute_logmu_shifts`'s walk at unequal clone lengths."""

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

    # NB per clone, since unequal lengths admit no rectangular view.
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
    """`compute_logmu_shifts` is never called upstream (#234); fails when the call is restored."""

    source = inspect.getsource(nophasing)

    assert "logmu_shifts are not currently supported" in source, (
        "the warning is gone, so the call may be live: #234 PR 2 now owes a "
        "validation of the shift against planted truth, not just equivalence"
    )
    assert "# logmu_shifts = compute_logmu_shifts(" in source, (
        "the call is no longer commented out"
    )
