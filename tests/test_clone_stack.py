"""The clone-stack accessor reproduces what `cnaster` computes by hand.

**#234 PR 1.** `compute_logmu_shifts` performs a per-clone `logsumexp` with a
hand-rolled two-pass loop over `start_idx`, and its own docstring carries the
vectorized form it replaced. `CloneStack.per_clone` is that form. These pin
that the two agree, which is what makes replacing the loop a refactor.

`patch` throughout: this says the accessor and the loop agree, not that either
is the right quantity. Whether `compute_logmu_shifts` is *applied* is open —
on the `port` branch the call is live but its result is never read
(`# TODO fold in logmu_shifts`), which
`test_the_shift_is_computed_and_then_discarded` pins so the day it is folded
in is not silent.
"""

from __future__ import annotations

import numpy as np
import pytest
from cnaster.hmm_nophasing import compute_logmu_shifts
from port.patch.clone_stack import CloneStack, channels_of


@pytest.mark.patch
def test_a_row_is_the_slice_it_claims_to_be() -> None:
    """`view()[c]` **is** `values[c*n_obs:(c+1)*n_obs]`, not merely equal to it.

    Pinned rather than assumed: the whole accessor rests on the buffer being
    clone-major, and a layout change elsewhere would make every row quietly
    wrong rather than loudly broken.
    """
    n_clones, n_obs = 4, 50
    values = np.arange(n_clones * n_obs, dtype=np.float64)
    stack = CloneStack(values, n_clones, n_obs)

    for clone in range(n_clones):
        expected = values[clone * n_obs : (clone + 1) * n_obs]

        assert np.array_equal(stack.clone(clone), expected)
        assert np.shares_memory(stack.view(), values), "view() must not copy"


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

    # NB the same quantity via the accessor, one clone at a time because the
    #    lengths differ -- which is exactly the case a rectangular view cannot
    #    hold, and is why `channels_of` takes one `n_obs` rather than a list.
    #    One value per clone: the `port` branch returns `(n_clones,)` rather
    #    than the segment-broadcast the branch before it returned (#259).
    ours = np.empty(len(clone_lengths))
    start = 0

    for clone, length in enumerate(clone_lengths):
        block = (
            log_mus[copy_states[start : start + length]]
            + normal_log_lambda[start : start + length]
        )
        ours[clone] = scipy.special.logsumexp(block)
        start += length

    assert theirs.shape == ours.shape == (len(clone_lengths),)
    assert np.allclose(theirs, ours, rtol=0.0, atol=1e-12), (
        f"max |difference| {np.max(np.abs(theirs - ours)):.3e}"
    )


@pytest.mark.patch
def test_equal_lengths_are_the_rectangular_case_the_view_handles() -> None:
    """Where every clone is the same length, `per_clone` is one call."""
    import scipy.special

    rng = np.random.default_rng(23)
    n_clones, n_obs = 3, 30
    values = rng.normal(size=n_clones * n_obs)

    stack = CloneStack(values, n_clones, n_obs)
    reduced = stack.per_clone(lambda view: scipy.special.logsumexp(view, axis=1))

    expected = np.array(
        [
            scipy.special.logsumexp(values[c * n_obs : (c + 1) * n_obs])
            for c in range(n_clones)
        ]
    )

    assert np.allclose(reduced, expected, rtol=0.0, atol=1e-12)
    assert np.array_equal(stack.broadcast(reduced), np.repeat(expected, n_obs))


@pytest.mark.patch
def test_channels_are_split_into_contiguous_buffers() -> None:
    """The stride this exists to remove, measured on both sides."""
    n_obs, n_comp, n_clones = 200, 2, 3
    X = np.arange(n_obs * n_comp * n_clones, dtype=np.float64).reshape(
        n_obs, n_comp, n_clones
    )
    stacked = X.transpose(2, 0, 1).reshape(-1, n_comp, 1)

    interleaved = stacked[0:n_obs, 0, 0]

    assert not interleaved.flags["C_CONTIGUOUS"]
    assert interleaved.strides[0] // interleaved.itemsize == n_comp

    channels = channels_of(stacked, n_clones)

    assert len(channels) == n_comp

    for channel_index, channel in enumerate(channels):
        assert channel.n_clones == n_clones
        assert channel.n_obs == n_obs
        assert channel.values.flags["C_CONTIGUOUS"]
        assert channel.view().strides[1] // channel.values.itemsize == 1

        # NB and it is the same data, not merely the same shape.
        assert np.array_equal(channel.clone(0), stacked[0:n_obs, channel_index, 0])


@pytest.mark.patch
def test_a_stack_that_does_not_divide_is_refused() -> None:
    with pytest.raises(ValueError, match="does not divide|values for"):
        CloneStack(np.zeros(7), 2, 3)

    with pytest.raises(IndexError, match="outside"):
        CloneStack(np.zeros(6), 2, 3).clone(-1)


@pytest.mark.bug
def test_the_shift_is_computed_and_then_discarded() -> None:
    """The call is live on the `port` branch; nothing reads what it returns.

    The branch this repository pinned before #259 commented the call out and
    logged `"logmu_shifts are not currently supported."`. The `port` branch
    calls it, inside `for i in range(n_states)`, and drops the result --
    `# TODO fold in logmu_shifts`. So the emission is still computed without
    the per-clone normalizer, and the cost of computing it is now paid
    `n_states` times per spot for nothing.

    Pinned by counting binds against reads in the function's AST rather than
    by matching a comment, so the day the shift is folded in this **fails**,
    which is when #234 PR 2 owes a judgement of the shift against planted
    truth rather than reproduction of a loop. #259 stages 2--4 are what
    clear it.
    """
    import ast
    import inspect

    import cnaster.hmm_nophasing as nophasing

    consumer = "compute_emission_probability_nb_betabinom_coded"
    function = next(
        node
        for node in ast.walk(ast.parse(inspect.getsource(nophasing)))
        if isinstance(node, ast.FunctionDef) and node.name == consumer
    )
    uses = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Name) and node.id == "logmu_shifts"
    ]

    binds = [node for node in uses if isinstance(node.ctx, ast.Store)]
    reads = [node for node in uses if isinstance(node.ctx, ast.Load)]

    assert binds, f"{consumer} no longer computes the shift at all"
    assert not reads, (
        f"{consumer} now reads the shift on line(s) "
        f"{[node.lineno for node in reads]}: #234 PR 2 owes a validation "
        "against planted truth, not just equivalence to a loop"
    )

    # NB and it is recomputed per state, though it does not depend on one.
    #    Loop-invariant, so the waste is a factor of `n_states`.
    over_states = [
        loop
        for loop in ast.walk(function)
        if isinstance(loop, ast.For)
        and ast.unparse(loop.iter) == "range(n_states)"
        and any(
            isinstance(node, ast.Name) and node.id == "logmu_shifts"
            for node in ast.walk(loop)
        )
    ]

    assert len(over_states) == 1, "the per-state recompute moved; re-read the call site"
